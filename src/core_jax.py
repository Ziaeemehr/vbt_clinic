"""
JAX-based implementation of Functional Connectivity Dynamics (FCD) computation.

This module provides GPU-accelerated functions for computing FCD matrices
using JAX with JIT compilation. The implementation is optimized for performance
and supports automatic differentiation.

Functions
---------
get_windows : Create sliding windows from time series
get_ut_fc : Compute upper triangular functional connectivity
get_fcd : Compute Functional Connectivity Dynamics matrix
get_fc : Compute functional connectivity matrix
fc_correlation_cost : Calculate FC correlation cost comparing empirical and simulated data

Notes
-----
JAX must be installed to use this module. For multi-device parallelization
examples using vmap and pmap, see the examples directory.
"""
import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=["w"])
def get_windows(x, w):
    """
    Create sliding windows from time series data using JAX.
    
    Parameters
    ----------
    x : jax.numpy.ndarray
        Time series data of shape (n_nodes, n_samples)
    w : int
        Window size (number of time points per window)
    
    Returns
    -------
    jax.numpy.ndarray
        Windowed data of shape (n_windows, n_nodes, w)
        where n_windows = n_samples - w + 1
    
    Notes
    -----
    This function is JIT-compiled for performance.
    Uses maximum overlap (leave-1-out sliding windows).
    """
    ws = x.shape[1] - w + 1
    f = jnp.r_[:ws]
    slc = lambda f: jax.lax.dynamic_slice(x, (0, f), (x.shape[0], w))
    wd = jax.vmap(slc)(f)
    return wd


@partial(jax.jit, static_argnames=["positive"])
def get_ut_fc(x, positive=True):
    """
    Compute upper triangular functional connectivity from time series.
    
    Calculates the correlation matrix and extracts the upper triangular
    part (excluding diagonal). Optionally sets negative correlations to zero.
    
    Parameters
    ----------
    x : jax.numpy.ndarray
        Time series data of shape (n_nodes, n_samples)
    positive : bool, optional
        If True, negative correlations are set to zero (default: True)
    
    Returns
    -------
    jax.numpy.ndarray
        Upper triangular functional connectivity vector of shape
        (n_nodes * (n_nodes - 1) / 2,)
    
    Notes
    -----
    This function is JIT-compiled for performance.
    """
    fc = jnp.corrcoef(x, rowvar=True)
    fc = jax.lax.cond(positive, lambda fc: fc * (fc > 0), lambda fc: fc, fc)
    ut_fc = fc[jnp.triu_indices(fc.shape[0], k=1)]
    return ut_fc


def get_fcd(x, window_size, positive=False):
    """
    Compute Functional Connectivity Dynamics (FCD) matrix using JAX.
    
    The FCD matrix represents the similarity between functional connectivity
    patterns across different time windows. It is computed by:
    1. Creating sliding windows from the time series
    2. Computing functional connectivity (FC) for each window
    3. Computing correlations between FC patterns across windows
    
    Parameters
    ----------
    x : jax.numpy.ndarray
        Time series data of shape (n_nodes, n_samples)
        Typically brain regions x time points
    window_size : int
        Window size (number of time points per window)
    positive : bool, optional
        If True, negative correlations in FC are set to zero (default: False)
    
    Returns
    -------
    jax.numpy.ndarray
        FCD matrix of shape (n_windows, n_windows)
        where n_windows = n_samples - w + 1
        
    Notes
    -----
    - Uses maximum overlap (leave-1-out sliding windows)
    - The diagonal of the FCD matrix represents correlation of each window with itself
    - This implementation uses JAX for GPU acceleration and JIT compilation
    
    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> from fcdtools.core_jax import get_fcd
    >>> # Generate random time series: 10 brain regions, 100 time points
    >>> ts = jnp.array(np.random.randn(10, 100))
    >>> fcd = get_fcd(ts, window_size=30)
    >>> print(fcd.shape)  # (71, 71) since 100 - 30 + 1 = 71 windows
    """
    n_overlap = window_size - 1  # leave-1-out overlappping windows
    k = n_overlap // (window_size - n_overlap) + 1
    wd = get_windows(x, window_size)
    ut_fcs = jax.vmap(get_ut_fc, (0, None))(wd, positive)
    fcd = jnp.corrcoef(ut_fcs, rowvar=True)
    # ut_fcd = fcd[jnp.triu_indices(fcd.shape[0], k=k)]
    return fcd #, ut_fcd


def get_fc(x, positive=False):
    """
    Compute functional connectivity (FC) matrix using JAX.
    
    Parameters
    ----------
    x : jax.numpy.ndarray
        Time series data of shape (n_nodes, n_samples)
    positive : bool, optional
        If True, negative correlations are set to zero (default: False)
    
    Returns
    -------
    jax.numpy.ndarray
        FC matrix of shape (n_nodes, n_nodes)
    
    Notes
    -----
    This function is JIT-compiled for performance.
    """
    fc = jnp.corrcoef(x, rowvar=True)
    fc = jax.lax.cond(positive, lambda fc: fc * (fc > 0), lambda fc: fc, fc)
    return fc
    

@jax.jit
def fc_correlation_cost(
    fc_empirical: jax.numpy.ndarray,
    timeseries: jax.numpy.ndarray
):
    """
    Calculate FC correlation cost comparing empirical FC and simulated timeseries using JAX.
    
    This function is JIT-compiled for maximum performance and is compatible with vmap
    for batch processing. It computes the FC from the simulated timeseries and calculates
    the correlation with the pre-computed empirical FC.
    
    Parameters
    ----------
    fc_empirical : jax.numpy.ndarray
        Empirical FC matrix of shape (n_nodes, n_nodes)
        Pre-computed functional connectivity matrix from empirical data
    timeseries : jax.numpy.ndarray
        Simulated BOLD signal of shape (n_nodes, n_timesamples)
        Must be 2D array with same n_nodes as fc_empirical
        
    Returns
    -------
    jax.numpy.ndarray (scalar)
        FC correlation cost. Values range from 0 (perfect match) to ~2 (worst case).
        Returns NaN if computation fails due to NaN values or constant signals.
    
    Notes
    -----
    - This function is JIT-compiled and vmap-compatible
    - NaN values in input will propagate to output (returns NaN)
    - Shapes are validated at trace time (compile time)
    - For validation with error messages, validate inputs before calling this function
                  
    Example
    -------
    >>> import jax.numpy as jnp
    >>> import numpy as np
    >>> from fcdtools.core_jax import get_fc, fc_correlation_cost
    >>> # Compute empirical FC once
    >>> bold_emp = jnp.array(np.random.randn(68, 1200))
    >>> fc_emp = get_fc(bold_emp)
    >>> # Compute cost for single simulation
    >>> bold_sim = jnp.array(np.random.randn(68, 1200))
    >>> cost = fc_correlation_cost(fc_emp, bold_sim)
    >>> 
    >>> # Batch processing with vmap
    >>> bold_sims = jnp.array(np.random.randn(100, 68, 1200))  # 100 simulations
    >>> costs = jax.vmap(lambda ts: fc_correlation_cost(fc_emp, ts))(bold_sims)
    """
    
    # Get number of nodes from fc_empirical
    n_nodes = fc_empirical.shape[0]
    
    # Compute simulated FC
    sim_fc = jnp.corrcoef(timeseries, rowvar=True)
    
    # Extract upper triangular parts using explicit indices (JIT-compatible)
    # Boolean masking is not JIT-compatible, so we use triu_indices
    i, j = jnp.triu_indices(n_nodes, k=1)
    emp_fc_vector = fc_empirical[i, j]
    sim_fc_vector = sim_fc[i, j]
    
    # Apply Fisher z-transformation
    # Clip values to avoid arctanh of exactly ±1
    emp_fc_z = jnp.arctanh(jnp.clip(emp_fc_vector, -0.999999, 0.999999))
    sim_fc_z = jnp.arctanh(jnp.clip(sim_fc_vector, -0.999999, 0.999999))
    
    # Calculate correlation between empirical and simulated FC vectors
    correlation = jnp.corrcoef(jnp.stack([emp_fc_z, sim_fc_z]))[0, 1]
    
    # Convert to cost: 1 - correlation (so 0 = perfect, 2 = worst case)
    cost = 1 - correlation
    
    return cost


def prepare_fcd_cumsum(bold_ref, window_size=30, n_bins=10000, fcd_range=(-1.0, 1.0)):
    """
    Helper function to prepare reference FCD cumulative distribution.
    Parameters
    ----------
    bold_ref : jax.numpy.ndarray
        Reference BOLD timeseries, shape [n_nodes, n_timesamples].
    window_size : int, optional
        Window length in samples (default: 30).
    n_bins : int, optional
        Number of bins for histogram (default: 10000).
    fcd_range : tuple, optional
        Range for histogram as (min, max) (default: (-1.0, 1.0)).
        
    Returns
    -------
    fcd_ref_cdf : jax.numpy.ndarray
        Pre-computed cumulative distribution of reference FCD, shape [n_bins,].
    
    """
    fcd_ref = get_fcd(bold_ref, window_size=window_size)
    n_windows = fcd_ref.shape[0]
    i, j = jnp.triu_indices(n_windows, k=1)
    fcd_ref_triu = fcd_ref[i, j]
    hist, _ = jnp.histogram(fcd_ref_triu, bins=n_bins, range=fcd_range)
    fcd_cumsum = jnp.cumsum(hist) / hist.sum()
    return fcd_cumsum


@partial(jax.jit, static_argnames=["win_len", "n_bins", "fcd_range"])
def fcd_ks_cost(fcd_ref_cdf, timeseries, win_len=30, n_bins=10000, fcd_range=(-1.0, 1.0)):
    """
    Calculate FCD KS statistics cost between reference and simulated data.
    
    This is a JIT-compiled version that can be used with jax.vmap for efficient
    batch processing. It computes the Kolmogorov-Smirnov statistic comparing
    the distribution of FCD values.
    
    Parameters
    ----------
    fcd_ref_cdf : jax.numpy.ndarray
        Pre-computed cumulative distribution function (CDF) of reference FCD, shape [n_bins,].
        This should be computed once from reference data (e.g., empirical or target data) and reused.
    timeseries : jax.numpy.ndarray
        Simulated BOLD timeseries, shape [n_nodes, n_timesamples].
    win_len : int, optional
        Window length in samples (default: 30). Must be static for JIT compilation.
    n_bins : int, optional
        Number of bins for histogram (default: 10000). Must be static for JIT compilation.
    fcd_range : tuple, optional
        Range for histogram as (min, max) (default: (-1.0, 1.0)).
        Must be static for JIT compilation.
        
    Returns
    -------
    cost : float
        The KS statistic (maximum difference between CDFs), range [0, 1].
        Returns NaN if FCD cannot be computed.
        
    Notes
    -----
    - All parameters except fcd_ref_cdf and timeseries must be static
    - Shape validation happens at JIT compile time
    - NaN values propagate naturally through JAX operations
    - Compatible with jax.vmap for batch processing multiple simulations
    - For efficiency, pass pre-computed fcd_ref_cdf rather than raw FCD matrix
    
    Examples
    --------
    >>> # Prepare reference data once
    >>> bold_ref = jnp.array(np.random.randn(68, 1200))
    >>> fcd_ref = get_fcd(bold_ref, window_size=30)
    >>> # Extract upper triangular and compute cumsum
    >>> n_windows = fcd_ref.shape[0]
    >>> i, j = jnp.triu_indices(n_windows, k=1)
    >>> fcd_ref_triu = fcd_ref[i, j]
    >>> hist, _ = jnp.histogram(fcd_ref_triu, bins=10000, range=(-1.0, 1.0))
    >>> fcd_ref_cdf = jnp.cumsum(hist) / hist.sum()
    >>> 
    >>> # Single simulation
    >>> bold_sim = jnp.array(np.random.randn(68, 1200))
    >>> cost = fcd_ks_cost(fcd_ref_cdf, bold_sim, win_len=30)
    >>> 
    >>> # Batch processing with vmap
    >>> bold_sims = jnp.array(np.random.randn(100, 68, 1200))
    >>> cost_fn = lambda ts: fcd_ks_cost(fcd_ref_cdf, ts, win_len=30)
    >>> costs = jax.vmap(cost_fn)(bold_sims)
    """
    # Compute FCD for simulated data
    fcd = get_fcd(timeseries, window_size=win_len)
    
    # Extract upper triangular part (excluding diagonal) using explicit indices
    n_windows = fcd.shape[0]
    i, j = jnp.triu_indices(n_windows, k=1)
    fcd_triu = fcd[i, j]
    
    # Compute histogram for simulated FCD
    hist, _ = jnp.histogram(fcd_triu, bins=n_bins, range=fcd_range)
    
    # Normalize to get cumulative distribution
    sim_fcd_cumsum = jnp.cumsum(hist) / hist.sum()
    
    # Calculate KS statistic: maximum absolute difference between CDFs
    ks_stat = jnp.abs(fcd_ref_cdf - sim_fcd_cumsum).max()
    
    return ks_stat
