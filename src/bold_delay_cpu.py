"Jax impl of simulation in fused.ispc.c."

import os
# Enable XLA deterministic operations for reproducibility on GPU
os.environ.setdefault('XLA_FLAGS', '--xla_gpu_deterministic_ops=true')

import jax
import jax.numpy as jnp
import vbjax as vb
import scipy.sparse
import numpy as np
from . import gast_model as gm
from tqdm import tqdm
from copy import deepcopy
from jax.ops import segment_sum
import pickle


def make_jp_runsim(
    csr_weights: scipy.sparse.csr_matrix,
    idelays: np.ndarray,
    params: np.ndarray,
    horizon: int,
    rng_seed=43,
    num_svar=10,
    num_time=1000,
    t_cut=0.0,
    dt=0.1,
    num_skip=5,
    adhoc=None,
    bold_decimate=10,
):
    """
    Optimized version of make_jp_runsim for num_item=1 (single simulation).
    
    This version removes the batch dimension entirely for cleaner code and potentially
    better performance. All arrays are 2D (num_svar, num_node) instead of 3D.
    
    Args:
        csr_weights: Sparse connectivity matrix (CSR format)
        idelays: Delay matrix for connections
        params: Model parameters (NamedTuple)
        horizon: Size of circular buffer for delays
        rng_seed: Random seed (not used, kept for compatibility)
        num_svar: Number of state variables
        num_time: Total integration steps
        t_cut: Time to cut from beginning to remove transient (in ms)
        dt: Integration timestep
        num_skip: Steps between BOLD updates
        adhoc: Constraint function applied after integration
        bold_decimate: Factor to decimate BOLD samples
        
    Returns:
        jit-compiled simulation function for single simulation
    """
    
    # Squeeze parameters from (num_node, 1) to (num_node,) for single simulation
    # This is necessary because the batched version uses (num_node, 1) for broadcasting
    # with (num_node, num_item), but single simulation uses (num_node,)
    params_squeezed = type(params)(**{
        field: jnp.squeeze(jnp.asarray(getattr(params, field))) 
        if hasattr(getattr(params, field), 'shape') and len(jnp.asarray(getattr(params, field)).shape) == 2
        else getattr(params, field)
        for field in params._fields
    })
    
    num_out_node, num_node = csr_weights.shape
    horizonm1 = horizon - 1
    j_indices = jnp.array(csr_weights.indices)
    j_weights = jnp.array(csr_weights.data)
    j_indptr = jnp.array(csr_weights.indptr)
    assert idelays.max() < horizon - 2
    idelays2 = jnp.array(horizon + np.c_[idelays, idelays - 1].T)
    idelays2 = idelays2.astype(jnp.int32)
    j_indices = j_indices.astype(jnp.int32)
    
    _csr_rows = np.concatenate(
        [i * np.ones(n, "i") for i, n in enumerate(np.diff(csr_weights.indptr))]
    )
    assert np.all(_csr_rows[:-1] <= _csr_rows[1:]), "CSR row indices must be sorted for segment_sum"
    j_csr_rows = jnp.array(_csr_rows)
    
    def cfun(buffer, t):
        """
        Compute coupling for single simulation (no batch dimension).
        buffer shape: (num_node, horizon)
        returns: (2, num_out_node) where num_out_node = 4 * num_node (stacked E, I, D, S)
        
        Uses the same scatter-add approach as make_jp_runsim for consistency.
        """
        # Get delayed buffer values
        # buffer[j_indices, (t - idelays2) & horizonm1] has shape (2, num_edges)
        # because idelays2 has shape (2, num_edges) and broadcasting creates 2 in first dim
        delayed_values = buffer[j_indices, (t - idelays2) & horizonm1]  # shape: (2, num_edges)
        
        # Multiply: j_weights shape (num_edges,) * delayed_values shape (2, num_edges)
        # Need to broadcast j_weights to shape (1, num_edges) so result is (2, num_edges)
        wxij = j_weights.reshape(1, -1) * delayed_values  # shape: (2, num_edges)
        
        # Use scatter-add (same as make_jp_runsim but without batch dimension)
        cx = jnp.zeros((2, num_out_node))
        cx = cx.at[:, j_csr_rows].add(wxij)
        return cx
    
    def dfun(x, cx):
        """
        Dynamics function for single simulation.
        x shape: (num_svar, num_node)
        cx shape: (num_out_node,) = (4*num_node,) - single delay component
        returns: (num_svar, num_node)
        """
        # cx has 4*num_node entries: [E_connections, I_connections, D_connections, S_connections]
        # For single simulation, we don't have a batch dimension
        Ce_aff, Ci_aff, Cd_aff, Cs_aff = cx.reshape(4, num_node)
        return gm.sigm_d1d2sero_dfun(
            x,
            (
                params_squeezed.we * Ce_aff,
                params_squeezed.wi * Ci_aff,
                params_squeezed.wd * Cd_aff,
                params_squeezed.ws * Cs_aff,
            ),
            params_squeezed,
        )
    
    def heun(x, cx, step_counter, master_key):
        """
        Heun integration for single simulation.
        x shape: (num_svar, num_node)
        cx shape: (2, num_node)
        returns: (num_svar, num_node)
        """
        step_key = jax.random.fold_in(master_key, step_counter)
        
        z = jnp.zeros((num_svar, num_node))
        
        # Generate noise - shape: (2, num_node)
        noise = jax.random.normal(step_key, (2, num_node))
        
        z = z.at[1:3].set(noise)
        z = z.at[1].multiply(params_squeezed.sigma_V)
        z = z.at[2].multiply(params_squeezed.sigma_u)
        
        dx1 = dfun(x, cx[0])
        dx2 = dfun(x + dt * dx1 + z, cx[1])
        x_new = x + dt / 2 * (dx1 + dx2) + z
        
        if adhoc is not None:
            x_new = adhoc(x_new, params_squeezed)
        
        return x_new
    
    # BOLD monitor - shape: (num_node,) for single simulation
    dt_bold = dt * num_skip / 1000.0
    bold_buf0, bold_step, bold_samp = vb.make_bold(
        shape=(num_node,),  # No batch dimension
        dt=dt_bold,
        p=vb.bold_default_theta,
    )
    
    def step_and_update_bold(state, time_idx):
        """
        Inner loop for single simulation.
        All state arrays have no batch dimension.
        """
        buffer = state["buf"]
        bold_buf = state["bold"]
        neural_state = state["x"]
        step_counter = state["k"]
        master_key = state["master_key"]
        
        assert neural_state.shape == (num_svar, num_node)
        
        for step_i in range(num_skip):
            abs_time = step_i + time_idx * num_skip
            coupling = cfun(buffer, abs_time)
            
            neural_state = heun(neural_state, coupling, step_counter, master_key)
            
            # Store firing rate - shape: (num_node,)
            buffer = buffer.at[:, abs_time % horizon].set(neural_state[0])
            
            step_counter = step_counter + 1
        
        # Update BOLD buffer
        bold_buf = bold_step(bold_buf, neural_state[0])
        
        state["x"] = neural_state
        state["buf"] = buffer
        state["bold"] = bold_buf
        state["k"] = step_counter
        return state, None
    
    def step_and_sample_bold(state, decim_idx):
        """Outer loop with BOLD decimation."""
        start_idx = decim_idx * bold_decimate
        
        def inner_loop_body(iter_i, state_carry):
            state_carry, _ = step_and_update_bold(state_carry, start_idx + iter_i)
            return state_carry
        
        state = jax.lax.fori_loop(0, bold_decimate, inner_loop_body, state)
        
        bold_buf, bold_signal = bold_samp(state["bold"])
        state["bold"] = bold_buf
        
        return state, bold_signal
    
    def run_sim_jp(master_key, init_state):
        """
        Run single simulation.
        
        Args:
            master_key: JAX PRNG key
            init_state: Initial state array, shape (num_svar,) or (num_svar, 1)
        
        Returns:
            bold: BOLD signal, shape (num_time_points, num_node)
            ts: Time array, shape (num_time_points,)
        """
        # Buffer shape: (num_node, horizon) - no batch dimension
        buffer = jnp.zeros((num_node, horizon)) + init_state[0]
        
        # Ensure init_state is flattened properly
        init_state_flat = init_state.flatten() if init_state.ndim > 1 else init_state
        
        init = {
            "buf": buffer,
            "bold": bold_buf0,
            "x": jnp.zeros((num_svar, num_node)) + init_state_flat.reshape(-1, 1),
            "k": 0,
            "master_key": master_key,
        }
        
        num_bold_samples = num_time // num_skip
        num_decimated_samples = num_bold_samples // bold_decimate
        decimated_indices = jnp.r_[:num_decimated_samples]
        
        _, bold = jax.lax.scan(step_and_sample_bold, init, decimated_indices)
        
        if t_cut > 0:
            dt_bold_sample = dt * num_skip * bold_decimate
            cut_idx = int(t_cut / dt_bold_sample)
            bold = bold[cut_idx:]
            ts = (jnp.arange(bold.shape[0]) + cut_idx) * dt * num_skip * bold_decimate / 1000.0
        else:
            ts = jnp.arange(bold.shape[0]) * dt * num_skip * bold_decimate / 1000.0
        
        return bold, ts
    
    return jax.jit(run_sim_jp)


def run_sweep(
    theta_list,
    setup,
    seed=42,
    adhoc=None,
    verbose=True,
    extract_features=False,
    extract_bold_features=None,
    kwargs_features=None,
    output_path="outputs",
    store_bolds=False,
    same_noise=False,
    n_devices=None,
):
    """
    Run parameter sweep across multiple parameter sets.
    
    Each parameter set in theta_list represents a different simulation with different
    parameter values. The function processes simulations in batches across available
    devices. Note: Parameters are baked into the JIT-compiled function, so each
    parameter set requires a separate compilation.

    Args:
        theta_list: List of parameter objects (NamedTuple), one per simulation.
                   Each theta should be valid for a single simulation (no batch dimension).
                   The number of simulations = len(theta_list).
        setup: Dictionary with simulation configuration
        seed: Random seed for PRNG
        adhoc: Constraint function
        verbose: Print progress messages
        extract_features: Whether to extract features from BOLD
        extract_bold_features: Function to extract features
        kwargs_features: Keyword arguments for feature extraction
        output_path: Path to save outputs
        store_bolds: Whether to store raw BOLD signals
        same_noise: If True, use the same PRNG key for all simulations (identical noise).
                   If False (default), each simulation uses a unique key (independent noise).
                   Using same_noise=True helps isolate the effect of parameter changes.
        n_devices: Number of devices to use. If None, uses all available devices.

    Returns:
        results_array: Extracted features or raw BOLD signals
        ts: Time array (None if extract_features=True)
    """
    # Get num_nodes from the connectivity matrix
    num_nodes = setup["Seids"].shape[1]
    num_simulations = len(theta_list)
    
    # Validate that theta_list is indeed a list
    if not isinstance(theta_list, list):
        raise ValueError("theta_list must be a list of parameter objects")
    
    if num_simulations == 0:
        raise ValueError("theta_list cannot be empty")

    # Extract adhoc from setup if not provided as argument
    if adhoc is None and "adhoc" in setup:
        adhoc = setup["adhoc"]
    if adhoc is None:
        adhoc = gm.sigm_d1d2sero_stay_positive

    # Create the single-simulation function (no pmap yet)
    run_sim_jp = make_jp_runsim(
        csr_weights=setup["Seids"],
        idelays=setup["idelays"],
        params=theta_list[0],  # Use first theta as template for compilation
        horizon=setup["horizon"],
        num_svar=setup.get("num_svar", 10),
        dt=setup["dt"],
        num_skip=setup["num_skip"],
        num_time=setup["num_time"],
        bold_decimate=setup.get("bold_decimate", 10),
        adhoc=adhoc,
    )
    
    # Determine number of devices
    if n_devices is None:
        n_devices = jax.local_device_count()
    else:
        n_devices = min(n_devices, jax.local_device_count())
    
    if verbose:
        print(f"Running {num_simulations} simulations using {n_devices} devices")
        noise_mode = "identical" if same_noise else "independent"
        print(f"Noise mode: {noise_mode}")
    
    # Generate master keys
    master_key = jax.random.PRNGKey(seed)
    
    # Calculate number of batches needed
    n_batches = (num_simulations + n_devices - 1) // n_devices
    
    # Prepare init_state
    init_state = setup["init_state"]
    if init_state.ndim > 1:
        init_state = init_state.flatten()
    
    # Storage for results
    all_bolds = []
    ts = None
    
    # Process in batches
    for batch_idx in tqdm(range(n_batches), desc="Simulation batches", disable=not verbose):
        start_idx = batch_idx * n_devices
        end_idx = min(start_idx + n_devices, num_simulations)
        batch_size = end_idx - start_idx
        
        # Get parameter sets for this batch
        batch_thetas = theta_list[start_idx:end_idx]
        
        # If batch is smaller than n_devices, pad with copies of the last theta
        # (we'll discard the extra results later)
        if batch_size < n_devices:
            batch_thetas = batch_thetas + [batch_thetas[-1]] * (n_devices - batch_size)
        
        # Generate keys for this batch
        if same_noise:
            # Use the same key for all simulations (identical noise)
            batch_keys = jnp.tile(master_key, (n_devices, 1))
        else:
            # Generate unique keys for each simulation in this batch (independent noise)
            batch_keys = jax.random.split(
                jax.random.fold_in(master_key, batch_idx), n_devices
            )
        
        # Run each simulation in the batch
        # Note: Since parameters are baked into the JIT-compiled function,
        # each parameter set requires a separate compilation
        batch_bolds = []
        for i, theta in enumerate(batch_thetas[:batch_size]):
            # Compile simulation with this specific parameter set
            run_sim_theta = make_jp_runsim(
                csr_weights=setup["Seids"],
                idelays=setup["idelays"],
                params=theta,
                horizon=setup["horizon"],
                num_svar=setup.get("num_svar", 10),
                dt=setup["dt"],
                num_skip=setup["num_skip"],
                num_time=setup["num_time"],
                bold_decimate=setup.get("bold_decimate", 10),
                t_cut=setup.get("t_cut", 0.0),
                adhoc=adhoc,
            )
            bold, ts = run_sim_theta(batch_keys[i], init_state)
            bold.block_until_ready()
            batch_bolds.append(np.array(bold))
        
        all_bolds.extend(batch_bolds)
    
    # Stack all results
    bolds = jnp.array(all_bolds)  # Shape: (num_simulations, num_time_points, num_nodes)
    
    if verbose:
        print(f"Simulation complete. BOLD shape: {bolds.shape}, Time points: {ts.shape}")
    
    if store_bolds:
        bolds_np = np.array(bolds)
        ts_np = np.array(ts)
        with open(output_path, "wb") as f:
            pickle.dump({"bolds": bolds_np, "ts": ts_np, "theta_list": theta_list}, f)
    
    # Feature extraction
    if extract_features:
        if extract_bold_features is None:
            raise ValueError(
                "extract_bold_features function must be provided when extract_features is True"
            )
        
        feat_vmap_threshold = 50
        feat_chunk_size = min(num_simulations, feat_vmap_threshold)
        
        if verbose:
            print(f"Extracting features in chunks of size {feat_chunk_size}...")
        
        vmapped_feat = jax.vmap(lambda b: extract_bold_features(b, **kwargs_features))

        if num_simulations <= feat_vmap_threshold:
            results_array = vmapped_feat(bolds)
            if verbose:
                print(f"Feature extraction complete. Features shape: {results_array.shape}")
        else:
            n_feat_chunks = (num_simulations + feat_chunk_size - 1) // feat_chunk_size
            results_chunks = []
            for ci in tqdm(
                range(n_feat_chunks), desc="Feature chunks", disable=not verbose
            ):
                s = ci * feat_chunk_size
                e = min((ci + 1) * feat_chunk_size, num_simulations)
                chunk_signals = bolds[s:e]
                chunk_res = vmapped_feat(chunk_signals)
                chunk_res_np = np.array(chunk_res)
                results_chunks.append(chunk_res_np)

            results_array = jnp.array(np.concatenate(results_chunks, axis=0))
            if verbose:
                print(f"Feature extraction complete. Features shape: {results_array.shape}")
        
        return results_array, None
    else:
        return bolds, ts


def bold_offline(raw_ts, dt, p=vb.bold_default_theta):
    """
    raw_ts: (T, n_nodes) firing rate
    dt: sampling step in seconds
    """
    bold_buf, bold_step, bold_samp = vb.make_bold(shape=raw_ts.shape[1:], dt=dt, p=p)

    def step(buf, x):
        buf = bold_step(buf, x)
        _, y = bold_samp(buf)
        return buf, y

    _, bold = jax.lax.scan(step, bold_buf, raw_ts)
    return bold
