# make_jp_runsim: Detailed Process Explanation

## Overview

`make_jp_runsim` creates a JAX-compiled simulation function for neural network models with delayed connectivity and BOLD (Blood Oxygen Level Dependent) signal monitoring. The key innovation is a **memory-efficient hierarchical loop structure** that decimates BOLD sampling during integration, reducing memory usage by a factor of `bold_decimate` (typically 10x).

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csr_weights` | `scipy.sparse.csr_matrix` | - | Sparse connectivity matrix in CSR format |
| `idelays` | `np.ndarray` | - | Delay matrix for connections (shape: num_nodes × num_nodes) |
| `params` | NamedTuple | - | Model parameters (we, wi, wd, ws, sigma_V, sigma_u, etc.) |
| `horizon` | `int` | - | Size of circular buffer for delays |
| `num_item` | `int` | 8 | Batch size (number of parallel simulations) |
| `num_svar` | `int` | 10 | Number of state variables per node |
| `num_time` | `int` | 1000 | Total integration time steps |
| `dt` | `float` | 0.1 | Integration time step (ms) |
| `num_skip` | `int` | 5 | Neural steps between BOLD updates |
| `bold_decimate` | `int` | 10 | BOLD decimation factor |

## Hierarchical Loop Structure

The function implements a **three-level hierarchical loop** for optimal memory efficiency:

```
┌────────────────────────────────────────────────────────────────┐
│  Outer Loop: BOLD Decimation (scan over decimated_indices)     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  Middle Loop: BOLD Updates (fori_loop, bold_decimate)     │ │
│  │  ┌──────────────────────────────────────────────────────┐ │ │
│  │  │  Inner Loop: Neural Integration (for loop, num_skip) │ │ │
│  │  │                                                      │ │ │
│  │  │  - Heun integration steps                            │ │ │
│  │  │  - Circular buffer updates                           │ │ │
│  │  │  - BOLD buffer updates                               │ │ │
│  │  └──────────────────────────────────────────────────────┘ │ │
│  │                                                           │ │
│  │  - Accumulate bold_decimate BOLD updates                  │ │
│  │  - Sample BOLD signal once                                │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                │
│  - Store only decimated BOLD samples                           │
└────────────────────────────────────────────────────────────────┘
```

## Memory Optimization Strategy

### Problem
- **Original approach**: Store all BOLD samples, decimate later
- **Memory usage**: `(num_time // num_skip, num_nodes, num_item)` array
- **Issue**: With large `num_item`, memory consumption becomes prohibitive

### Solution: Decimation During Integration
- **New approach**: Decimate BOLD sampling during the simulation loop
- **Memory usage**: `(num_time // num_skip // bold_decimate, num_nodes, num_item)` array
- **Benefit**: `bold_decimate`× memory reduction (typically 10×)

### Key Insight
Instead of storing every BOLD update, we:
1. Run `bold_decimate` neural integration blocks
2. Update BOLD buffer `bold_decimate` times
3. Sample BOLD signal only once per `bold_decimate` blocks
4. Store only the decimated samples

## Detailed Execution Flow

### 1. Initialization

```python
# Create circular buffer for delayed connectivity
buffer = jp.zeros((num_node, horizon, num_item)) + init_state[0]

# Initialize BOLD monitor
bold_buf0, bold_step, bold_samp = vb.make_bold(
    shape=(num_node, num_item),
    dt=dt_bold,  # dt * num_skip / 1000.0 * 10 (convert to seconds)
    p=vb.bold_default_theta,
)

# Simulation state dictionary
state = {
    'buf': buffer,           # Circular buffer for delays
    'bold': bold_buf0,       # BOLD monitor buffer
    'x': neural_state,       # Neural state variables
    'rng_key': rng_key       # Random number generator key
}
```

### 2. Time Index Calculations

```python
# Total BOLD samples before decimation
num_bold_samples = num_time // num_skip

# Final BOLD samples after decimation
num_decimated_samples = num_bold_samples // bold_decimate

# Indices for outer scan loop
decimated_indices = jp.arange(num_decimated_samples)
```

### 3. Outer Loop: `jax.lax.scan(step_and_sample_bold, init, decimated_indices)`

For each `decim_idx` in `decimated_indices`:

#### 3.1 Middle Loop: `jax.lax.fori_loop(0, bold_decimate, inner_loop_body, state)`

For each `iter_i` from 0 to `bold_decimate-1`:

##### 3.1.1 Inner Loop: Neural Integration Steps

For each `step_i` from 0 to `num_skip-1`:

```python
# Calculate absolute time index
abs_time = step_i + (decim_idx * bold_decimate + iter_i) * num_skip

# Compute delayed connectivity inputs
coupling = cfun(buffer, abs_time)

# Heun integration step
neural_state = heun(neural_state, coupling, rng_keys[step_i])

# Update circular buffer with firing rate
buffer = buffer.at[:, abs_time % horizon].set(neural_state[0])
```

##### 3.1.2 BOLD Buffer Update

```python
# Update BOLD buffer (but don't sample yet)
bold_buf = bold_step(bold_buf, neural_state[0])
```

#### 3.2 BOLD Signal Sampling

After `bold_decimate` iterations of the middle loop:

```python
# Sample BOLD signal once
bold_buf, bold_signal = bold_samp(state['bold'])
state['bold'] = bold_buf  # Reset buffer for next decimation block
```

### 4. Connectivity Function (`cfun`)

```python
def cfun(buffer, abs_time):
    # Extract delayed firing rates from circular buffer
    delayed_rates = buffer[j_indices, (abs_time - idelays2) & horizonm1]
    
    # Weight by connection strengths
    wxij = j_weights.reshape(-1,1) * delayed_rates
    
    # Accumulate inputs per target node
    coupling = jp.zeros((4, num_out_node, num_item))  # Ce, Ci, Cd, Cs
    coupling = coupling.at[:, j_csr_rows].add(wxij)
    
    return coupling
```

### 5. Neural Dynamics (`heun`)

```python
def heun(neural_state, coupling, rng_key):
    # Unpack coupling: excitatory, inhibitory, dopamine, serotonin
    Ce_aff, Ci_aff, Cd_aff, Cs_aff = coupling.reshape(4, num_node, num_item)
    
    # Compute drift term
    dx1 = gm.sigm_d1d2sero_dfun(neural_state, (
        params.we * Ce_aff,
        params.wi * Ci_aff, 
        params.wd * Cd_aff,
        params.ws * Cs_aff
    ), params)
    
    # Generate noise
    noise = jp.zeros((num_svar, num_node, num_item))
    noise = noise.at[1:3].set(jax.random.normal(rng_key, (2, num_node, num_item)))
    noise = noise.at[1].multiply(params.sigma_V)
    noise = noise.at[2].multiply(params.sigma_u)
    
    # Heun integration
    dx2 = gm.sigm_d1d2sero_dfun(neural_state + dt*dx1 + noise, coupling[1], params)
    neural_state_new = neural_state + dt/2*(dx1 + dx2) + noise
    
    # Apply constraints if provided
    if adhoc is not None:
        neural_state_new = adhoc(neural_state_new, params)
    
    return neural_state_new
```

## Mathematical Relationships

### Time Scales
- **Neural integration**: `dt` (0.1 ms)
- **BOLD update**: `dt * num_skip` (0.5 ms)
- **BOLD sampling**: `dt * num_skip * bold_decimate` (5 ms)

### Array Shapes
- **Neural state**: `(num_svar, num_node, num_item)` = `(10, num_node, num_item)`
- **Circular buffer**: `(num_node, horizon, num_item)`
- **BOLD signal**: `(num_decimated_samples, num_node, num_item)`

### Memory Usage
- **Before**: `num_time // num_skip * num_node * num_item * 4` bytes (float32)
- **After**: `(num_time // num_skip // bold_decimate) * num_node * num_item * 4` bytes
- **Reduction**: `bold_decimate`× memory savings

## Performance Benefits

### Memory Efficiency
- **10× memory reduction** with `bold_decimate=10`
- Enables larger batch sizes (`num_item`) for parameter sweeps
- Prevents out-of-memory errors on GPU

### Computational Efficiency
- **JAX optimization**: All loops use `jax.lax.scan` or `jax.lax.fori_loop`
- **GPU acceleration**: Entire computation runs on GPU
- **JIT compilation**: Function is compiled once, runs fast

### Accuracy Preservation
- **No loss of accuracy**: Neural dynamics identical to original
- **Proper delay handling**: Circular buffer maintains temporal relationships
- **BOLD model integrity**: Same BOLD parameters and sampling rate

## Usage Examples

### Basic Usage
```python
run_sim = make_jp_runsim(
    csr_weights=connectivity_matrix,
    idelays=delay_matrix,
    params=model_parameters,
    horizon=buffer_size,
    num_item=100,      # Batch size
    num_time=10000,    # Total steps
    bold_decimate=10   # 10x memory reduction
)

bold_signals = run_sim(rng_key, initial_state)
# Shape: (1000, num_nodes, 100) instead of (10000, num_nodes, 100)
```

### Parameter Sweep
```python
results = run_sweep(
    (parameters, setup_dict),
    seed=42,
    bold_decimate=10
)
```

## Key Functions Summary

| Function | Purpose | Memory Impact |
|----------|---------|---------------|
| `step_and_update_bold` | Neural integration + BOLD update | Updates buffer |
| `step_and_sample_bold` | Multiple updates + single sample | Decimates storage |
| `cfun` | Delayed connectivity computation | No storage |
| `heun` | Stochastic integration | No storage |
| `vb.make_bold` | BOLD model initialization | Buffer management |

## Debugging Tips

1. **Check time indices**: Ensure `abs_time` calculations are correct
2. **Verify buffer bounds**: `abs_time % horizon` should stay within bounds
3. **Monitor memory**: Use `jax.profiler` to track memory usage
4. **Validate outputs**: Compare with `bold_decimate=1` for correctness

## Comparison with Original Implementation

### Original (`bold_decimate=1`)
```python
# Store every BOLD sample
_, bold = jax.lax.scan(step_and_sample_bold, init, all_indices)
# Memory: O(num_time // num_skip)
```

### Optimized (`bold_decimate=10`)
```python
# Store every 10th BOLD sample
_, bold = jax.lax.scan(step_and_sample_bold, init, decimated_indices)
# Memory: O(num_time // num_skip // 10)
```

The optimization maintains identical neural dynamics while dramatically reducing memory footprint, enabling larger-scale simulations.</content>
<parameter name="filePath">/home/ziaee/Desktop/workstation/vbt_clinical/inference_sdde/make_jp_runsim_explanation.md
