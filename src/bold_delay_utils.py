"Jax impl of simulation in fused.ispc.c."

import jax
import jax.numpy as jnp
import vbjax as vb
import scipy.sparse
import numpy as np
from . import gast_model as gm
from tqdm import tqdm
from copy import deepcopy
import pickle


def make_jp_runsim(
    csr_weights: scipy.sparse.csr_matrix,
    idelays: np.ndarray,
    params: np.ndarray,
    horizon: int,
    rng_seed=43,
    num_item=8,
    num_svar=10,
    num_time=1000,
    t_cut=0.0,
    dt=0.1,
    num_skip=5,
    adhoc=None,
    bold_decimate=10,  # New parameter: decimate BOLD by this factor
    same_noise=False,  # New parameter: whether to use same noise across items
):
    """
    Create a JAX-compiled simulation function with delayed connectivity and BOLD monitoring.

    Memory-efficient BOLD sampling strategy:
    - Hierarchical loop structure (similar to helpers_dopa.py make_run_v):
      1. Innermost: num_skip neural integration steps → update BOLD buffer once
      2. Middle: bold_decimate BOLD updates → sample BOLD signal once
      3. Result: Only stores BOLD at decimated rate, reducing memory by factor of bold_decimate

    PRNG handling (inspired by helpers_dopa.py):
    - Uses master_key + fold_in strategy for reproducible noise generation
    - Each integration step uses jax.random.fold_in(master_key, step_counter)
    - Ensures reproducibility: same master_key → same noise sequence
    - Enables same_noise mode: identical noise across simulations for parameter effect isolation

    Args:
        csr_weights: Sparse connectivity matrix (CSR format)
        idelays: Delay matrix for connections
        params: Model parameters (NamedTuple)
        horizon: Size of circular buffer for delays
        rng_seed: Random seed (not used, kept for compatibility)
        num_item: Batch size (number of parallel simulations)
        num_svar: Number of state variables
        num_time: Total integration steps
        t_cut: Time to cut from beginning to remove transient (in same units as dt, typically ms)
        dt: Integration timestep
        num_skip: Steps between BOLD updates (typically 10)
        adhoc: Constraint function applied after integration
        bold_decimate: Factor to decimate BOLD samples (default 10)
                      Final BOLD samples = (num_time // num_skip // bold_decimate)
        same_noise: If True, use identical noise across all items (simulations).
                   Noise shape becomes (num_node, 1) and broadcasts to (num_node, num_item).
                   If False (default), each item gets independent noise with shape (num_node, num_item).

    Returns:
        jit-compiled simulation function
    """

    num_out_node, num_node = csr_weights.shape
    horizonm1 = horizon - 1
    j_indices = jnp.array(csr_weights.indices)
    j_weights = jnp.array(csr_weights.data)
    j_indptr = jnp.array(csr_weights.indptr)
    assert idelays.max() < horizon - 2
    idelays2 = jnp.array(horizon + np.c_[idelays, idelays - 1].T)

    _csr_rows = np.concatenate(
        [i * np.ones(n, "i") for i, n in enumerate(np.diff(csr_weights.indptr))]
    )
    j_csr_rows = jnp.array(_csr_rows)

    def cfun(buffer, t):
        wxij = j_weights.reshape(-1, 1) * buffer[j_indices, (t - idelays2) & horizonm1]
        cx = jnp.zeros((2, num_out_node, num_item))
        cx = cx.at[:, j_csr_rows].add(wxij)
        return cx

    def dfun(x, cx):
        # cx.shape = (3*num_node, num_node, ...)
        Ce_aff, Ci_aff, Cd_aff, Cs_aff = cx.reshape(4, num_node, num_item)
        return gm.sigm_d1d2sero_dfun(
            x,
            (
                params.we * Ce_aff,
                params.wi * Ci_aff,
                params.wd * Cd_aff,
                params.ws * Cs_aff,
            ),
            params,
        )

    def heun(x, cx, step_counter, master_key):
        """
        Heun integration step with noise generation using fold_in strategy.
        
        Args:
            x: Current state
            cx: Coupling terms
            step_counter: Global step counter (jnp.uint32)
            master_key: Master PRNG key for reproducible noise
        """
        # Generate step-specific key using fold_in (like helpers_dopa.py)
        step_key = jax.random.fold_in(master_key, step_counter)
        
        z = jnp.zeros((num_svar, num_node, num_item))
        
        # Generate noise: if same_noise=True, use shape (2, num_node, 1) to broadcast
        # if same_noise=False, use shape (2, num_node, num_item) for independent noise
        noise_shape = (2, num_node, 1) if same_noise else (2, num_node, num_item)
        noise = jax.random.normal(step_key, noise_shape)
        
        z = z.at[1:3].set(noise)
        z = z.at[1].multiply(params.sigma_V)
        z = z.at[2].multiply(params.sigma_u)
        dx1 = dfun(x, cx[0])
        dx2 = dfun(x + dt * dx1 + z, cx[1])
        x_new = x + dt / 2 * (dx1 + dx2) + z

        # Apply adhoc constraint function if provided
        if adhoc is not None:
            x_new = adhoc(x_new, params)

        return x_new

    # --- BOLD monitor ---
    dt_bold = dt * num_skip / 1000.0  # convert to seconds
    bold_buf0, bold_step, bold_samp = vb.make_bold(
        shape=(num_node, num_item),
        dt=dt_bold,
        p=vb.bold_default_theta,
    )

    def step_and_update_bold(state, time_idx):
        """
        Inner loop: neural integration steps + BOLD update (no sampling).
        
        Now uses step counter and master key instead of splitting keys.
        """
        buffer = state["buf"]
        bold_buf = state["bold"]
        neural_state = state["x"]
        step_counter = state["k"]
        master_key = state["master_key"]
        
        assert neural_state.shape == (num_svar, num_node, num_item)

        # Run num_skip neural integration steps
        for step_i in range(num_skip):
            abs_time = step_i + time_idx * num_skip
            coupling = cfun(buffer, abs_time)
            
            # Use fold_in with step counter (like helpers_dopa.py)
            neural_state = heun(neural_state, coupling, step_counter, master_key)
            
            # Store firing rate in buffer
            # If same_noise=True, buffer has shape (num_node, horizon, 1), 
            # store only first item's state with shape (num_node, 1)
            # If same_noise=False, buffer has shape (num_node, horizon, num_item), 
            # store all items with shape (num_node, num_item)
            if same_noise:
                state_to_store = neural_state[0, :, 0:1]  # Shape: (num_node, 1)
            else:
                state_to_store = neural_state[0]  # Shape: (num_node, num_item)
            buffer = buffer.at[:, abs_time % horizon].set(state_to_store)
            
            # Increment step counter (use same dtype as step_counter)
            step_counter = step_counter + 1

        # Update BOLD buffer (but don't sample yet)
        bold_buf = bold_step(bold_buf, neural_state[0])

        state["x"] = neural_state
        state["buf"] = buffer
        state["bold"] = bold_buf
        state["k"] = step_counter
        return state, None  # Return None - no BOLD sampling yet

    def step_and_sample_bold(state, decim_idx):
        """Outer loop: multiple BOLD updates + sample once (decimation)"""
        # Run bold_decimate iterations of neural dynamics + BOLD updates
        start_idx = decim_idx * bold_decimate

        def inner_loop_body(iter_i, state_carry):
            state_carry, _ = step_and_update_bold(state_carry, start_idx + iter_i)
            return state_carry

        state = jax.lax.fori_loop(0, bold_decimate, inner_loop_body, state)

        # Now sample the BOLD signal after bold_decimate updates
        bold_buf, bold_signal = bold_samp(state["bold"])
        state["bold"] = bold_buf

        return state, bold_signal

    def run_sim_jp(master_key, init_state):
        """
        Run simulation with master_key for reproducible noise generation.
        
        Args:
            master_key: JAX PRNG key - will be used with fold_in for all noise
            init_state: Initial state array
        """
        # If same_noise=True, use buffer with shape (num_node, horizon, 1) for broadcasting
        # Otherwise, each item gets its own buffer (num_node, horizon, num_item)
        buffer_shape = (num_node, horizon, 1 if same_noise else num_item)
        buffer = jnp.zeros(buffer_shape) + init_state[0]
        
        init = {
            "buf": buffer,
            "bold": bold_buf0,
            "x": jnp.zeros((num_svar, num_node, num_item))
            + init_state.reshape(-1, 1, 1),
            "k": 0,  # Step counter for fold_in (will use JAX's default int type)
            "master_key": master_key,  # Store master key, not split keys
        }

        # Calculate number of decimated samples
        num_bold_samples = num_time // num_skip
        num_decimated_samples = num_bold_samples // bold_decimate
        decimated_indices = jnp.r_[:num_decimated_samples]

        # Use the decimated sampling function
        _, bold = jax.lax.scan(step_and_sample_bold, init, decimated_indices)

        if t_cut > 0:
            # Calculate number of samples to cut from beginning
            dt_bold_sample = dt * num_skip * bold_decimate  # in ms
            cut_idx = int(t_cut / dt_bold_sample)
            # Remove initial transient
            bold = bold[cut_idx:]
            ts = (
                (jnp.arange(bold.shape[0]) + cut_idx)
                * dt
                * num_skip
                * bold_decimate
                / 1000.0
            )  # convert to seconds
        else:
            ts = (
                jnp.arange(bold.shape[0]) * dt * num_skip * bold_decimate / 1000.0
            )  # convert to seconds

        return bold, ts

    return jax.jit(run_sim_jp)


def check_input_parameters(theta, num_item, num_nodes):
    """
    Check that input parameters have correct dimensions.

    Parameters:
    -----------
    theta : NamedTuple
        Parameter object containing model parameters
    num_item : int
        Number of parallel simulations (batch size for parameter sweep)
    num_nodes : int
        Number of nodes in the network

    Raises:
    -------
    ValueError
        If parameter dimensions are incorrect
    """
    # Scalar parameters: should be either a scalar or have length equal to num_item
    scalar_params = ["we", "wi", "wd", "ws", "Zs", "Zd1", "Zd2"]

    for param_name in scalar_params:
        if hasattr(theta, param_name):
            param_value = getattr(theta, param_name)

            # Convert to numpy/jax array if needed
            if isinstance(param_value, (int, float)):
                # Scalar value is OK - will be broadcast
                continue

            param_array = jnp.asarray(param_value)

            # Check if it's a scalar (0D array)
            if param_array.ndim == 0:
                continue

            # If it's 1D, check length matches num_item
            if param_array.ndim == 1:
                if param_array.shape[0] != num_item:
                    raise ValueError(
                        f"Scalar parameter '{param_name}' has length {param_array.shape[0]}, "
                        f"but expected length {num_item} (num_item)"
                    )
            else:
                raise ValueError(
                    f"Scalar parameter '{param_name}' should be 0D or 1D, "
                    f"but has shape {param_array.shape}"
                )

    # Vector parameters: should be (num_nodes, 1) or (num_nodes, num_item)
    vector_params = ["Jdopa", "Rd1", "Rd2", "Rs"]

    for param_name in vector_params:
        if hasattr(theta, param_name):
            param_value = getattr(theta, param_name)

            # Convert to numpy/jax array if needed
            if isinstance(param_value, (int, float)):
                # Scalar value is OK - will be broadcast
                continue

            param_array = jnp.asarray(param_value)

            # Check if it's a scalar (0D array)
            if param_array.ndim == 0:
                continue

            # If 1D, should have length num_nodes
            if param_array.ndim == 1:
                if param_array.shape[0] != num_nodes:
                    raise ValueError(
                        f"Vector parameter '{param_name}' has shape {param_array.shape}, "
                        f"but expected ({num_nodes},) or ({num_nodes}, 1) or ({num_nodes}, {num_item})"
                    )
            # If 2D, first dimension should be num_nodes, second should be 1 or num_item
            elif param_array.ndim == 2:
                if param_array.shape[0] != num_nodes:
                    raise ValueError(
                        f"Vector parameter '{param_name}' has shape {param_array.shape}, "
                        f"but first dimension should be {num_nodes} (num_nodes)"
                    )
                if param_array.shape[1] not in [1, num_item]:
                    raise ValueError(
                        f"Vector parameter '{param_name}' has shape {param_array.shape}, "
                        f"but second dimension should be 1 or {num_item} (num_item)"
                    )
            else:
                raise ValueError(
                    f"Vector parameter '{param_name}' should be 0D, 1D, or 2D, "
                    f"but has shape {param_array.shape}"
                )


def run_sweep(
    theta,
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
):
    """
    Run parameter sweep with optional identical noise across simulations.

    Args:
        theta: Parameter object (NamedTuple)
        setup: Dictionary with simulation configuration
        seed: Random seed for PRNG
        adhoc: Constraint function
        verbose: Print progress messages
        extract_features: Whether to extract features from BOLD
        extract_bold_features: Function to extract features
        kwargs_features: Keyword arguments for feature extraction
        output_path: Path to save outputs
        store_bolds: Whether to store raw BOLD signals
        same_noise: If True, use identical noise realizations across all simulations.
                   This isolates parameter effects by keeping stochastic dynamics constant.
                   If False (default), each simulation batch uses independent noise.

    Returns:
        results_array: Extracted features or raw BOLD signals
        ts: Time array (None if extract_features=True)
    """

    # Get num_nodes from the connectivity matrix
    num_nodes = setup["Seids"].shape[1]
    num_item = setup["num_item"]

    # Validate input parameters
    theta = deepcopy(theta)  # Avoid modifying original theta
    check_input_parameters(theta, num_item, num_nodes)

    # Extract adhoc from setup if not provided as argument
    if adhoc is None and "adhoc" in setup:
        adhoc = setup["adhoc"]
    if adhoc is None:
        adhoc = gm.sigm_d1d2sero_stay_positive

    run_sim_jp = make_jp_runsim(
        csr_weights=setup["Seids"],
        idelays=setup["idelays"],
        params=theta,
        horizon=setup["horizon"],
        num_item=num_item,
        dt=setup["dt"],
        num_skip=setup["num_skip"],
        num_time=setup["num_time"],
        bold_decimate=setup.get("bold_decimate", 10),
        t_cut=setup.get("t_cut", 0.0),
        adhoc=adhoc,
        same_noise=same_noise,  # Pass same_noise to make_jp_runsim
    )

    # Generate master key for simulation
    # If same_noise=True, all batches use the same key
    # If same_noise=False, this is just the master key for this call
    master_key = jax.random.PRNGKey(seed)
    
    if verbose:
        noise_mode = "identical" if same_noise else "independent"
        print(f"Running simulation with {noise_mode} noise across {num_item} parameter sets...")
    
    bolds, ts = run_sim_jp(master_key, setup["init_state"])
    _ = bolds.block_until_ready()
    
    if verbose:
        print(f"Simulation complete. BOLD shape: {bolds.shape}, Time points: {ts.shape}")
        
    if store_bolds:
        # store bolds
        bolds_np = np.array(bolds)
        ts_np = np.array(ts)
        # store in pickle file with keys "bolds" and "ts" and theta
        with open(output_path, "wb") as f:
            pickle.dump({"bolds": bolds_np, "ts": ts_np, "theta": theta}, f)
    
    # bolds shape is (num_bold_samples, num_node, num_item)
    # reshape to (num_item, num_bold_samples, num_node) for feature extraction

    if extract_features:
        
        bolds = jnp.transpose(bolds, (2, 0, 1))
        
        if extract_bold_features is None:
            raise ValueError(
                "extract_bold_features function must be provided when extract_features is True"
            )
        feat_vmap_threshold = 50  # Reduced from 201 to avoid memory issues
        feat_chunk_size = min(num_item, feat_vmap_threshold)
        if verbose:
            print(f"Extracting features in chunks of size {feat_chunk_size}...")
        vmapped_feat = jax.vmap(lambda b: extract_bold_features(b, **kwargs_features))

        if num_item <= feat_vmap_threshold:
            # Single vmap call for all simulations
            results_array = vmapped_feat(bolds)
            if verbose:
                print(f"Feature extraction complete. Features shape: {results_array.shape}")
        else:
            # Process features in chunks to avoid OOM
            n_feat_chunks = (num_item + feat_chunk_size - 1) // feat_chunk_size
            results_chunks = []
            for ci in tqdm(
                range(n_feat_chunks), desc="Feature chunks", disable=not verbose
            ):
                s = ci * feat_chunk_size
                e = min((ci + 1) * feat_chunk_size, num_item)
                chunk_signals = bolds[s:e]
                chunk_res = vmapped_feat(chunk_signals)

                # Move chunk to host (numpy) to free device memory before next chunk
                chunk_res_np = np.array(chunk_res)
                results_chunks.append(chunk_res_np)

            # Concatenate chunks on host and convert back to jax array
            results_array = jnp.array(np.concatenate(results_chunks, axis=0))
            if verbose:
                print(
                    f"Feature extraction complete. Features shape: {results_array.shape}"
                )
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
