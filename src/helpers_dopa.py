# =========================
# Standard library
# =========================
import os
import time
import warnings
from os.path import exists, join
from itertools import product
from typing import Optional, Tuple, Sequence

# =========================
# Scientific / numerical
# =========================
import numpy as np
import pandas as pd

# =========================
# JAX ecosystem
# =========================
import jax
import jax.numpy as jnp
import jax.tree_util

# =========================
# Machine learning / stats
# =========================
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import torch
import sbi.utils as utils
from sbi.analysis import pairplot

# =========================
# Progress / utilities
# =========================
from tqdm import tqdm

# =========================
# Domain-specific (VB / simulations)
# =========================
import vbjax as vb
from vbjax import make_sdde
import lib.gast_model as gm
from lib.inference import Inference
from vbt_trial.analysis.analyse_simulation import (
    compute_fcd,
    find_ALFF_of_roi,
)

# from vbt_trial.simulations import gast_model as gm

# =========================
# Plotting
# =========================
import matplotlib.pyplot as plt
import matplotlib.axes as maxes


warnings.filterwarnings("ignore", category=RuntimeWarning)

# from vbt_trial.simulations.simulations_utils import make_run

# TODO write a test function to ensure get_bold_feature_labels and extract_bold_features are consistent


def make_sde_v(dt, dfun, gfun, adhoc=None, return_euler=False):
    """
    Create SDE stepper with internal noise generation.
    """
    sqrt_dt = jnp.sqrt(dt)
    if not hasattr(gfun, "__call__"):
        sig = gfun
        gfun = lambda *_: sig

    def step(x, k, p, master_key):
        k = jnp.asarray(k, dtype=jnp.uint32)
        step_key = jax.random.fold_in(master_key, k)
        z_t = jax.random.normal(step_key, shape=x.shape)
        noise = gfun(x, p) * sqrt_dt * z_t
        return vb.heun_step(
            x, dfun, dt, p, add=noise, adhoc=adhoc, return_euler=return_euler
        )

    return step, None


def make_run_v(init, tavg_period, tr, total_time, dt, sigma, adhoc, store_var=False):
    """
    Returns the run function using hierarchical time loops.
    Factory function that creates and configures the run function based on the provided parameters.

    Hierarchical structure:
    - Innermost: SDE integration steps (dt) with time averaging
    - Middle: Sample averaged state and update BOLD (every tavg_period)
    - Outer: Sample BOLD signal (at bold sampling rate)

    Args:
        init: Initial state array
        tavg_period: Time averaging period in ms
        tr: BOLD repetition time in ms
        total_time: Total simulation time in ms
        dt: Integration time step in ms
        sigma: Noise intensity scaling
        adhoc: Adhoc function for state constraints
        store_var: Whether to store time-averaged variables (default: True).
                   Set to False to reduce memory usage and only return BOLD.
    """
    n_nodes = init.shape[1]
    n_vars = init.shape[0]

    # Number of integration steps per time average
    n_steps_per_avg = int(tavg_period / dt)
    steps_per_avg = jnp.r_[:n_steps_per_avg]

    # BOLD sampling period (same as tavg_period for now)
    n_avgs_per_bold = int(tr / (n_steps_per_avg * dt))
    avgs_per_bold = jnp.r_[:n_avgs_per_bold]

    # Noise intensity
    noise_intensity = (
        jnp.array([0.0, 0.1, 0.01, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1) * sigma
    )

    # Define noise function for make_sde_v2
    def gfun(x, p):
        return noise_intensity

    # Create SDE stepper
    step, _ = make_sde_v(dt=dt, dfun=gm.dopa_net, gfun=gfun, adhoc=adhoc)

    # Setup time averaging monitor
    ta_buf, ta_update, ta_sample = vb.make_timeavg((n_vars, n_nodes))

    # Setup BOLD monitor
    bold_buf, bold_update, bold_sample = vb.make_bold(
        shape=init[0].shape,  # only r
        dt=tavg_period / 1e3,  # convert to seconds
        p=vb.bold_default_theta,
    )

    def run(params, key=jax.random.PRNGKey(42)):
        """
        Run the simulation with hierarchical time loops.

        Args:
            params: (Ci, Ce, Cd, theta) tuple for dopa_net
            key: JAX random key

        Returns:
            If store_var=True:
                ts: Time array (at tavg_period sampling rate)
                ta_y: Time-averaged state (shape: (n_samples, n_vars, n_nodes))
                bold: BOLD signal
            If store_var=False:
                ts: None
                ta_y: None
                bold: BOLD signal (reduced memory usage)
        """
        # Number of BOLD samples for total time
        n_bold_samples = int(total_time / (n_avgs_per_bold * n_steps_per_avg * dt))
        bold_sample_indices = jnp.r_[:n_bold_samples]

        # Initialize state
        state = {
            "master_key": key,
            "k": jnp.array(0, dtype=jnp.uint32),
            "x": init,
            "tavg": ta_buf,
            "bold": bold_buf,
        }

        # Innermost step: SDE integration with time averaging
        def step_tavg(state, _):
            state["x"] = step(state["x"], state["k"], params, state["master_key"])
            state["k"] = state["k"] + jnp.array(1, dtype=jnp.uint32)
            state["tavg"] = ta_update(state["tavg"], state["x"])
            return state, None

        # Middle loop: sample averaged state and update BOLD
        def step_bold(state, _):
            state, _ = jax.lax.scan(step_tavg, state, steps_per_avg)
            state["tavg"], ta_value = ta_sample(state["tavg"])
            state["bold"] = bold_update(
                state["bold"], ta_value[0]
            )  # Update BOLD with averaged r
            return state, ta_value

        # Outer loop: sample BOLD signal (step_bold_decimate)
        if store_var:
            # Store both ta_values and bold_val
            def step_bold_decimate(state, _):
                state, ta_values = jax.lax.scan(step_bold, state, avgs_per_bold)
                state["bold"], bold_val = bold_sample(state["bold"])
                return state, (ta_values, bold_val)

        else:
            # Only store bold_val to reduce memory
            def step_bold_decimate(state, _):
                state, _ = jax.lax.scan(step_bold, state, avgs_per_bold)
                state["bold"], bold_val = bold_sample(state["bold"])
                return state, bold_val

        @jax.jit
        def compiled_scan(initial_state):
            _, results = jax.lax.scan(
                step_bold_decimate, initial_state, bold_sample_indices
            )
            return results

        results = compiled_scan(state)

        if store_var:
            ta_values, bold_signal = results
            # Flatten time-averaged results to get all samples at tavg_period rate
            n_ta_samples = n_bold_samples * n_avgs_per_bold
            ta_y = ta_values.reshape(n_ta_samples, n_vars, n_nodes)
            # Time points
            ts = jnp.arange(n_ta_samples) * tavg_period
            return ts, ta_y, bold_signal
        else:
            # Return None for ts and ta_y to save memory
            bold_signal = results
            return None, None, bold_signal

    return run


def run_for_parameters_v(
    p,
    key=jax.random.PRNGKey(0),
    total_time=30e3,
    dt=5e-1,
    tavg_period=1.0,
    tr=300.0,
    t_cut=10e3,  # in ms
    store_var=False,
    verbose=True,
):
    """
    Run simulation for given parameters and return time-averaged variables and BOLD signal.

    Takes a tuple of parameters and a random key, simulates the neural model,
    and returns time-averaged state variables and downsampled BOLD signal.

    Args:
        p: Tuple of (Ci, Ce, Cd, theta, sigma) where:
           - Ci: Inhibitory connectivity matrix
           - Ce: Excitatory connectivity matrix
           - Cd: Dopamine connectivity matrix
           - theta: Global coupling parameter
           - sigma: Noise intensity scaling
        key: JAX random key for noise generation
        total_time: Total simulation time in ms (default: 30000.0)
        dt: Integration time step in ms (default: 0.5)
        tavg_period: Time averaging period in ms (default: 1.0)
        tr: BOLD repetition time in ms (default: 300.0)
        t_cut: Time cutoff in ms to skip initial transient (default: 10000.0)
        store_var: Whether to store time-averaged variables (default: True).
                   Set to False to reduce memory usage and only return BOLD.
        verbose: Whether to print simulation time (default: True)

    Returns:
        ts: Time array at tavg_period sampling rate (None if store_var=False)
        ta_y: Time-averaged state variables (shape: (n_samples, n_vars, n_nodes), None if store_var=False)
        bold_t: Time array for BOLD signal in ms
        bold_ds: Downsampled BOLD signal
    """
    start = time.time()

    Ci, Ce, Cd, theta, sigma = p
    n_nodes = Ce.shape[0]
    init = jnp.array(
        [
            0.01,
            -50.0,
            0.0,
            0.04,
            0.0,
            0.0,
            0.0,
        ]
    )  # Inizializzazione of the variables
    init = jnp.outer(init, jnp.ones(n_nodes))

    run = make_run_v(
        init,
        tavg_period=tavg_period,  # 5ms or 200Hz
        tr=tr,
        total_time=total_time,
        dt=dt,
        sigma=sigma,
        adhoc=gm.dopa_stay_positive,
        store_var=store_var,
    )
    ts, ta_y, bold = run((Ci, Ce, Cd, theta), key)
    bold_t = jnp.arange(bold.shape[0]) * tr  # in ms
    end = time.time() - start
    if verbose:
        print("Simulating BOLD took ", end, " seconds")

    # Calculate the sample index corresponding to t_cut based on TR
    cut_idx = int(t_cut / tr)
    nt = bold.shape[0]
    bold_ds = bold[cut_idx:]
    bold_t = bold_t[cut_idx:]
    step = max(1, int(nt // 300))  # Ensure step is at least 1
    bold_t = bold_t[::step]
    bold_ds = bold_ds[::step]

    if store_var:
        ta_y = ta_y[int(t_cut / tavg_period) :, :, :]
        ts = ts[int(t_cut / tavg_period) :]
    return ts, ta_y, bold_t, bold_ds


def setup_connectome(atlas, data_path="data"):
    Ce_mask = pd.read_csv(join(data_path, f"{atlas}_exc_mask.csv"), index_col=0).values
    Ci_mask = pd.read_csv(join(data_path, f"{atlas}_inh_mask.csv"), index_col=0).values
    Cd_mask = pd.read_csv(join(data_path, f"{atlas}_dopa_mask.csv"), index_col=0).values
    weights = pd.read_csv(
        join(data_path, f"{atlas}_weights_with_dopa.csv"), index_col=0
    )
    trac_length = pd.read_csv(
        join(data_path, "lengths_with_sero_and_dopa.csv"), index_col=0
    )
    # remove R.RN and L.RN
    trac_length = trac_length.drop(index=["R.RN", "L.RN"], columns=["R.RN", "L.RN"])
    weights = weights / np.max(weights)
    regions_names = list(weights.index)

    weights = weights.values
    trac_length = trac_length.values

    Ce = jnp.array(weights * Ce_mask)
    Cd = jnp.array(weights * Cd_mask)
    Ci = jnp.array(weights * Ci_mask)

    return regions_names, Ce, Cd, Ci, weights, trac_length


def setup_specific_params(sub_name, atlas, data_path="data"):
    """
    Preparing the array of parameters Rd, I, Ja and Jg
    """
    regions_names, Ce, Cd, Ci, weights, trac_len = setup_connectome(atlas, data_path)
    params_df = pd.DataFrame(
        data=np.zeros((len(regions_names), 4)), columns=["Rd", "I", "Ja", "Jg"]
    )
    params_df["x_labels"] = regions_names
    params_df = params_df.set_index("x_labels")

    receptor_list = pd.read_csv(
        join(data_path, f"{atlas}_D1receptor_data.csv"), index_col="x_labels"
    )

    for i, row in receptor_list.iterrows():
        params_df.loc[row.name, "Rd"] = row["numbers_of_receptors"] / (
            5 * np.max(receptor_list["numbers_of_receptors"])
        )

    params_df["I"] = 1.0

    changed_names = {
        "dk": {
            "left pallidum": ["L.PA"],
            "right pallidum": ["R.PA"],
            "left nigra": ["L.SN"],
            "right nigra": ["R.SN"],
            "left amygdala": ["L.AM"],
            "right amygdala": ["R.AM"],
        },
        "schaefer": {
            "left pallidum": ["Left-Pallidum"],
            "right pallidum": ["Right-Pallidum"],
            "left nigra": ["Left-SN"],
            "right nigra": ["Right-SN"],
            "left amygdala": ["Left-Amygdala"],
            "right amygdala": ["Right-Amygdala"],
        },
        "aal2": {
            "left pallidum": ["Pallidum_L"],
            "right pallidum": ["Pallidum_R"],
            "left nigra": ["Nigra_L"],
            "right nigra": ["Nigra_R"],
            "left amygdala": ["Amygdala_L"],
            "right amygdala": ["Amygdala_R"],
        },
    }

    params_df.loc[changed_names[atlas]["right pallidum"], "I"] = 1.3
    params_df.loc[changed_names[atlas]["left pallidum"], "I"] = 1.3

    params_df["Ja"] = 1.0
    params_df.loc[[regions_names[i] for i in np.unique(np.where(Ci != 0)[1])], "Ja"] = (
        0.0
    )
    params_df.loc[changed_names[atlas]["left nigra"], "Ja"] = 1.5
    params_df.loc[changed_names[atlas]["right nigra"], "Ja"] = 1.5
    params_df.loc[changed_names[atlas]["left amygdala"], "Ja"] = 1.0
    params_df.loc[changed_names[atlas]["right amygdala"], "Ja"] = 1.0

    # Enhance cerebellar nodes in aal2
    if atlas == "aal2":
        cereb_regions = [
            "Cerebelum_Crus1_L",
            "Cerebelum_Crus1_R",
            "Cerebelum_Crus2_L",
            "Cerebelum_Crus2_R",
            "Cerebelum_3_L",
            "Cerebelum_3_R",
            "Cerebelum_4_5_L",
            "Cerebelum_4_5_R",
            "Cerebelum_6_L",
            "Cerebelum_6_R",
            "Cerebelum_7b_L",
            "Cerebelum_7b_R",
            "Cerebelum_8_L",
            "Cerebelum_8_R",
            "Cerebelum_9_L",
            "Cerebelum_9_R",
            "Cerebelum_10_L",
            "Cerebelum_10_R",
            "Vermis_1_2",
            "Vermis_3",
            "Vermis_4_5",
            "Vermis_6",
            "Vermis_7",
            "Vermis_8",
            "Vermis_9",
            "Vermis_10",
        ]
        for region in cereb_regions:
            params_df.loc[region, "I"] = 1.06

    return params_df


def sweep_parameters(
    p_base,
    param_labels,
    theta,
    key=jax.random.PRNGKey(42),
    total_time=30e3,
    dt=5e-1,
    tavg_period=1.0,
    tr=300.0,
    t_cut=10e3,
    extract_features: bool = True,
    same_noise: bool = False,
    verbose: bool = True,
    extract_bold_features=None,
    kwargs_features={},
    store_var: bool = False,
):
    """
    Sweep over multiple parameters using vmap for simulations, then sequential feature extraction.

    This function separates the simulation (run in parallel with vmap) from feature extraction
    (run sequentially) to optimize memory usage. The simulations run on GPU in parallel, but
    feature extraction happens one at a time to avoid holding all BOLD signals in memory.

    Args:
        p_base: Tuple of (Ci, Ce, Cd, theta, sigma) - base parameters
        param_labels: List of parameter names to sweep (e.g., ["we", "wi"])
        theta: 2D array of parameter values with shape (n_simulations, n_parameters)
               Each row corresponds to one simulation, each column to one parameter
        key: JAX random key
        total_time: Total simulation time in ms
        dt: Integration time step in ms
        tavg_period: Time averaging period in ms
        tr: BOLD repetition time in ms
        t_cut: Time cutoff in ms to skip initial transient
        extract_features: If True, extract features from BOLD. If False, return raw BOLD.
        same_noise: If True, use the same noise realization for all simulations.
        verbose: If True, print progress information.
        extract_bold_features: Function to extract features from BOLD signals. If extract_features is True, this function must be provided.
        kwargs_features: Keyword arguments to pass to extract_bold_features
        store_var: Whether to store time-averaged variables in run_for_parameters_v

    Returns:
        results_array: 2D array of shape (n_simulations, n_features) if extract_features=True,
                      or (n_simulations, n_timepoints, n_regions) if extract_features=False
    """
    Ci, Ce, Cd, theta_base, sigma = p_base

    # Validate parameter names
    theta_fields = set(theta_base._fields)
    invalid_params = [name for name in param_labels if name not in theta_fields]
    if invalid_params:
        raise ValueError(
            f"Invalid parameter name(s): {invalid_params}. "
            f"Valid parameters are: {sorted(theta_fields)}"
        )

    # Convert to JAX array and require 2D input
    theta = jnp.asarray(theta)
    if theta.ndim != 2:
        # Be strict: require explicit 2D shape (n_simulations, n_parameters)
        raise ValueError(
            "theta must be a 2D array with shape (n_simulations, n_parameters). "
            f"Got shape {theta.shape}. If you have a single simulation, reshape to (1, n_parameters)."
        )

    n_simulations = theta.shape[0]

    # Validate shape (number of columns must match number of parameter names)
    if theta.shape[1] != len(param_labels):
        raise ValueError(
            f"theta has {theta.shape[1]} columns but "
            f"{len(param_labels)} parameter names were provided"
        )

    if verbose:
        print(f"Running {n_simulations} simulations in parallel with vmap...")

    # STEP 1: Run all simulations in parallel using vmap (GPU efficient)
    def wrapper_bold(param_row, sim_key):
        """Run simulation with a row of parameter values - returns BOLD signal only."""
        # Create dictionary mapping parameter names to values
        param_dict = {name: param_row[i] for i, name in enumerate(param_labels)}

        # Create modified theta with updated parameters
        theta_modified = theta_base._replace(**param_dict)

        # Run simulation - ONLY return BOLD, no feature extraction here
        ts, ta_y, bold_t, bold_ds = run_for_parameters_v(
            (Ci, Ce, Cd, theta_modified, sigma),
            key=sim_key,
            total_time=total_time,
            dt=dt,
            tavg_period=tavg_period,
            tr=tr,
            t_cut=t_cut,
            store_var=store_var,
            verbose=False,  # Disable verbose in vmap
        )

        return bold_ds

    # Handle noise: generate different keys for each simulation or use the same key
    if not same_noise:
        # Create different keys for each simulation
        keys_array = jax.random.split(key, n_simulations)

        # Apply vmap with different keys to run simulations in parallel
        vmapped_func = jax.vmap(wrapper_bold)
        bold_signals = vmapped_func(theta, keys_array)
    else:
        # Use the same key for all simulations
        def run_single_sim_same_key(param_row):
            return wrapper_bold(param_row, key)

        # Apply vmap with same key
        vmapped_func = jax.vmap(run_single_sim_same_key)
        bold_signals = vmapped_func(theta)

    if verbose:
        print(f"Simulations complete. BOLD signals shape: {bold_signals.shape}")

    # STEP 2: Extract features (use vmap, chunk if too large)
    if extract_features:
        if extract_bold_features is None:
            raise ValueError(
                "extract_bold_features function must be provided when extract_features=True"
            )

        feat_vmap_threshold = 201  # if n_simulations > threshold, use chunking
        feat_chunk_size = 200

        if verbose:
            print(
                f"Extracting features (vmap). Will chunk if simulations > {feat_vmap_threshold}..."
            )

        # Parameters for chunking feature extraction to limit memory use

        # vmapped feature extractor (maps over leading axis of bold signals)
        vmapped_feat = jax.vmap(lambda b: extract_bold_features(b, **kwargs_features))

        if n_simulations <= feat_vmap_threshold:
            # Single vmap call for all simulations
            results_array = vmapped_feat(bold_signals)
            if verbose:
                print(
                    f"Feature extraction complete. Features shape: {results_array.shape}"
                )
        else:
            # Process features in chunks to avoid OOM
            n_feat_chunks = (n_simulations + feat_chunk_size - 1) // feat_chunk_size
            results_chunks = []
            for ci in tqdm(
                range(n_feat_chunks), desc="Feature chunks", disable=not verbose
            ):
                s = ci * feat_chunk_size
                e = min((ci + 1) * feat_chunk_size, n_simulations)
                chunk_signals = bold_signals[s:e]
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
    else:
        # Return raw BOLD signals if no feature extraction requested
        results_array = bold_signals

    return results_array



# import networkx as nx
# G = nx.from_numpy_array(np.array(Ci))
# print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
# print(f"Is the graph connected? {nx.is_connected(G)}")
# density = nx.density(G)
# print(f"Graph density: {density}")
# exit(0)
