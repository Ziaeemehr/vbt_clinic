# =========================
# Feature extraction functions (moved from utils.py)
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
# from lib.core_jax import get_fcd
# from lib.core_np import get_fcd

# =========================
# Machine learning / stats
# =========================
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
import scipy
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
import src.gast_model as gm
from src.inference import Inference
# from vbt_trial.analysis.analyse_simulation import (
#     compute_fcd,
#     find_ALFF_of_roi,
# )

# =========================
# Plotting
# =========================
import matplotlib.pyplot as plt
import matplotlib.axes as maxes

# Rich imports for beautiful console output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


warnings.filterwarnings("ignore", category=RuntimeWarning)


def get_bold_feature_labels(**kwargs):
    """
    Get feature labels for extract_bold_features output.

    This function returns the labels corresponding to features extracted by
    extract_bold_features. Call this once with the same kwargs to get labels,
    then use extract_bold_features in vmap for efficient batch processing.

    Args:
        **kwargs: Same parameters as extract_bold_features:
            - k: Upper triangular offset (default: 1)
            - eigenvalues: Whether eigenvalue features are computed (default: True)
            - quantiles: Quantiles computed (default: [0.05, 0.25, 0.5, 0.75, 0.95])
            - features: List of statistical features (default: ["sum", "max", "min", "mean", "std", "skew", "kurtosis"])
            - cortex_idx: Cortical region indices (optional, for ALFF features)
            - subcortex_idx: Subcortical region indices (optional, for ALFF features)

    Returns:
        labels: List of feature labels, one for each feature in extract_bold_features output

    Example:
        >>> labels = get_bold_feature_labels()
        >>> features = extract_bold_features(bold_signal)
        >>> print(f"{labels[0]}: {features[0]}")
    """
    k = kwargs.get("k", 1)
    eigenvalues = kwargs.get("eigenvalues", True)
    quantiles = kwargs.get("quantiles", jnp.array([0.05, 0.25, 0.5, 0.75, 0.95]))
    features = kwargs.get(
        "features", ["sum", "max", "min", "mean", "std", "skew", "kurtosis"]
    )
    cortex_idx = kwargs.get("cortex_idx", None)
    subcortex_idx = kwargs.get("subcortex_idx", None)

    if quantiles is not None:
        quantiles = jnp.asarray(quantiles)

    labels = []

    # Generate labels for both FC and FCD
    for matrix_type in ["FC", "FCD"]:
        # Quantile labels
        if quantiles is not None and len(quantiles) > 0:
            for q in quantiles:
                labels.append(f"{matrix_type}_q{float(q):.2f}")

        # Eigenvalue feature labels
        if eigenvalues:
            for feat in features:
                labels.append(f"{matrix_type}_eig_{feat}")

        # Upper triangular feature labels
        for feat in features:
            labels.append(f"{matrix_type}_ut_{feat}")

        # Off-diagonal sum
        labels.append(f"{matrix_type}_offdiag_sum")
    
    # Add ALFF feature labels
    if cortex_idx is not None and subcortex_idx is not None:
        labels.extend([
            "ALFF_mean_ctx",
            "fALFF_mean_ctx",
            "ALFF_mean_subcort",
            "fALFF_mean_subcort",
            "ALFF_std_ctx",
            "fALFF_std_ctx",
            "ALFF_std_subcort",
            "fALFF_std_subcort"
        ])
    else:
        labels.extend([
            "ALFF_mean_global",
            "fALFF_mean_global",
            "ALFF_std_global",
            "fALFF_std_global"
        ])

    return labels


def extract_bold_features(bold, **kwargs):
    """
    Extract features from BOLD signal using matrix_stat_jax and ALFF analysis.

    Computes comprehensive statistical features from both FC (functional connectivity),
    FCD (functional connectivity dynamics) matrices, and ALFF (Amplitude of Low 
    Frequency Fluctuations) measures. Returns NaN values if the input BOLD signal 
    contains NaN, making it compatible with JAX's vmap and scan.

    Args:
        bold: BOLD signal array (time_points, n_regions)
        regions_names: List of region names (optional)
        **kwargs: Additional parameters
            - tr: Repetition time in ms (default: 300.0)
            - window_length: Window length for FCD in ms (default: 6000)
            - overlap: Overlap for FCD windows in ms (default: 5700)
            - k: Upper triangular offset for matrix_stat_jax (default: 1)
            - eigenvalues: Whether to compute eigenvalue features (default: True)
            - quantiles: Quantiles to compute (default: [0.05, 0.25, 0.5, 0.75, 0.95])
            - features: List of statistical features (default: ["sum", "max", "min", "mean", "std", "skew", "kurtosis"])
            - cortex_idx: List of cortical region indices (optional, for regional ALFF)
            - subcortex_idx: List of subcortical region indices (optional, for regional ALFF)

    Returns:
        features: JAX array of shape (n_features,) containing concatenated features from:
                  - FC matrix statistics
                  - FCD matrix statistics  
                  - ALFF measures (global or separated by cortical/subcortical)
                  Returns NaN array if input contains NaN values

    Note:
        - Use get_bold_feature_labels(**kwargs) to get feature labels corresponding to the output.
        - ALFF features are computed using jax.lax.scan for memory efficiency.
        - If cortex_idx and subcortex_idx are provided, returns separate ALFF stats for each region type.
    """
    tr = kwargs.get("tr", 300.0)
    window_length = kwargs.get("window_length", 50 * tr)
    overlap = kwargs.get("overlap", 49 * tr)
    k = kwargs.get("k", 1)
    eigenvalues = kwargs.get("eigenvalues", True)
    quantiles = kwargs.get("quantiles", jnp.array([0.05, 0.25, 0.5, 0.75, 0.95]))
    features = kwargs.get(
        "features", ["sum", "max", "min", "mean", "std", "skew", "kurtosis"]
    )
    cortex_idx = kwargs.get("cortex_idx", None)
    subcortex_idx = kwargs.get("subcortex_idx", None)

    # Check if BOLD signal contains NaN using JAX operations
    has_nan = jnp.any(jnp.isnan(bold))

    # Compute FC matrix
    fc = jnp.corrcoef(bold.T)
    fc = jnp.nan_to_num(fc, nan=0.0)

    # Compute FCD matrix
    fcd = compute_fcd_jax(bold, tr, window_length=window_length, overlap=overlap)
    fcd = jnp.nan_to_num(fcd, nan=0.0)

    # Extract features from FC matrix
    fc_features = matrix_stat_jax(
        fc, k=k, eigenvalues=eigenvalues, quantiles=quantiles, features=features
    )

    # Extract features from FCD matrix
    fcd_features = matrix_stat_jax(
        fcd, k=k, eigenvalues=eigenvalues, quantiles=quantiles, features=features
    )
    
    # Add features from ALFF analysis for cortical and subcortical regions
    # Process all ROIs using scan for memory efficiency
    def process_roi(carry, roi_timeseries):
        """Process one ROI and return ALFF, fALFF"""
        ALFF, fALFF = find_ALFF_of_roi_jax(roi_timeseries, dt=tr)
        return carry, jnp.array([ALFF, fALFF])
    
    # Transpose to get (n_regions, n_timepoints) for scanning over regions
    bold_transposed = bold.T
    _, alff_results = jax.lax.scan(process_roi, None, bold_transposed)
    # alff_results shape: (n_regions, 2) where columns are [ALFF, fALFF]
    
    ALFF_all = alff_results[:, 0]  # Shape: (n_regions,)
    fALFF_all = alff_results[:, 1]  # Shape: (n_regions,)
    
    # Compute statistics on ALFF and fALFF across regions
    if cortex_idx is not None and subcortex_idx is not None:
        # Separate cortical and subcortical regions
        ALFF_ctx = ALFF_all[jnp.array(cortex_idx)]
        fALFF_ctx = fALFF_all[jnp.array(cortex_idx)]
        ALFF_subcort = ALFF_all[jnp.array(subcortex_idx)]
        fALFF_subcort = fALFF_all[jnp.array(subcortex_idx)]
        
        # Compute mean for each group
        alff_features = jnp.array([
            jnp.mean(ALFF_ctx), jnp.mean(fALFF_ctx),
            jnp.mean(ALFF_subcort), jnp.mean(fALFF_subcort),
            jnp.std(ALFF_ctx), jnp.std(fALFF_ctx),
            jnp.std(ALFF_subcort), jnp.std(fALFF_subcort),
        ])
    else:
        # Just compute global mean if indices not provided
        alff_features = jnp.array([jnp.mean(ALFF_all), jnp.mean(fALFF_all),
                                   jnp.std(ALFF_all), jnp.std(fALFF_all)
        ])
    

    # Concatenate FC, FCD, and ALFF features
    all_features = jnp.concatenate([fc_features, fcd_features, alff_features])

    # Use where to return NaN if input had NaN, otherwise return computed features
    # This avoids jax.lax.cond and is more efficient
    nan_array = jnp.full_like(all_features, jnp.nan)
    all_features = jnp.where(has_nan, nan_array, all_features)

    return all_features


def compute_fcd_jax(ts, tr, window_length=6000, overlap=5700):
    """
    JAX-compatible version of compute_fcd for use within vmap/jit contexts.

    Args:
        ts: Time series array (n_samples, n_regions)
        tr: TR (repetition time) in milliseconds
        window_length: Window length in milliseconds
        overlap: Overlap between windows in milliseconds

    Returns:
        FCD: Functional Connectivity Dynamics matrix
    """
    # Use floor and astype instead of int() for JAX compatibility
    window_length = jnp.floor(window_length / tr).astype(jnp.int32)
    overlap = jnp.floor(overlap / tr).astype(jnp.int32)
    n_samples, n_regions = ts.shape

    window_steps_size = window_length - overlap
    n_windows = jnp.floor((n_samples - window_length) / window_steps_size + 1).astype(
        jnp.int32
    )

    # upper triangle indices
    triu_indices = jnp.triu_indices(n_regions, k=1)
    n_connections = len(triu_indices[0])

    # Compute FC for each window using vmap for efficiency
    def compute_window_fc(i):
        start_idx = window_steps_size * i
        # Use dynamic_slice for JAX compatibility with vmap
        window_data = jax.lax.dynamic_slice(
            ts,
            (start_idx, 0),  # start indices
            (window_length, n_regions),  # slice sizes
        )

        # Compute correlation for this window
        fc_window = jnp.corrcoef(window_data.T)
        fc_window = jnp.nan_to_num(fc_window, nan=0.0)

        # Extract upper triangle
        return fc_window[triu_indices]

    # Vectorize over all windows
    FC_t = jax.vmap(compute_window_fc)(jnp.arange(n_windows))
    FC_t = FC_t.T  # Shape: (n_connections, n_windows)

    # Compute FCD by correlating the FCs with each other
    FCD = jnp.corrcoef(FC_t.T)
    # FCD = jnp.nan_to_num(FCD, nan=0.0)

    return FCD

# def compute_fcd_jax(ts, tr=None, window_length=60, overlap=None):
#     from lib.core_jax import get_fcd
#     # ts = jnp.asarray(ts).T 
#     # return get_fcd(ts, window_size=window_length)
#     return get_fcd(ts, window_size=window_length)
    
    


def compute_fc_jax(ts):
    """
    Compute functional connectivity (FC) matrix from time series using JAX.

    Args:
        ts: Time series array (n_samples, n_regions)
    Returns:
        FC: Functional connectivity matrix (n_regions x n_regions)
    """
    fc = jnp.corrcoef(ts.T)
    fc = jnp.nan_to_num(fc, nan=0.0)
    return fc


def matrix_stat_jax(
    A,
    k: int = 1,
    eigenvalues: bool = True,
    quantiles=jnp.array([0.05, 0.25, 0.5, 0.75, 0.95]),
    features=["sum", "max", "min", "mean", "std", "skew", "kurtosis"],
):
    """
    JAX-compatible version of matrix_stat that is jittable.

    Calculate comprehensive statistics from a matrix (typically connectivity matrices).
    This function extracts various statistical features from a matrix including
    basic statistics, eigenvalue properties, and quantiles.

    Parameters
    ----------
    A : jax array [n x n]
        Input matrix, typically a square connectivity or correlation matrix
    k : int, optional
        Upper triangular matrix offset. Only elements above the k-th diagonal
        are considered (default: 1, excludes main diagonal)
    eigenvalues : bool, optional
        Whether to compute eigenvalue-based features (default: True)
    quantiles : jax array, optional
        Array of quantiles to compute (default: [0.05, 0.25, 0.5, 0.75, 0.95])
        Set to None to skip quantile computation
    features : list of str, optional
        List of statistical features to compute from matrix values
        Options: ["sum", "max", "min", "mean", "std", "skew", "kurtosis"]

    Returns
    -------
    values : jax array
        Concatenated array of all computed feature values
    """
    # Convert inputs to JAX arrays
    A = jnp.asarray(A)
    if quantiles is not None:
        quantiles = jnp.asarray(quantiles)

    # Get upper triangular indices and values
    ut_idx = jnp.triu_indices_from(A, k=k)
    A_ut = A[ut_idx[0], ut_idx[1]]

    values_list = []

    # Compute quantiles
    if quantiles is not None and len(quantiles) > 0:
        q = jnp.quantile(A, quantiles)
        values_list.append(q)

    # Compute eigenvalue features
    if eigenvalues:
        eigen_vals = jnp.linalg.eigh(A)[0]  # Use eigh for symmetric matrices
        # Exclude the last eigenvalue (often near zero for correlation matrices)
        eigen_vals_real = eigen_vals[:-1]
        eig_stats = compute_stats_jax(eigen_vals_real, features)
        values_list.append(eig_stats)

    # Compute upper triangular statistics
    ut_stats = compute_stats_jax(A_ut, features)
    values_list.append(ut_stats)

    # Compute off-diagonal absolute sum
    off_diag_sum = jnp.sum(jnp.abs(A)) - jnp.trace(jnp.abs(A))
    values_list.append(jnp.array([off_diag_sum]))

    # Concatenate all values
    if len(values_list) > 0:
        values = jnp.concatenate(values_list)
    else:
        values = jnp.array([])

    return values


def compute_stats_jax(x, features):
    """
    Compute basic statistics on array x using JAX operations.

    Parameters
    ----------
    x : jax array
        Input array
    features : list of str
        Statistics to compute: ["sum", "max", "min", "mean", "std", "skew", "kurtosis"]

    Returns
    -------
    stats : jax array
        Array of computed statistics
    """
    stats_list = []

    for feature in features:
        if feature == "sum":
            stats_list.append(jnp.sum(x))
        elif feature == "max":
            stats_list.append(jnp.max(x))
        elif feature == "min":
            stats_list.append(jnp.min(x))
        elif feature == "mean":
            stats_list.append(jnp.mean(x))
        elif feature == "std":
            stats_list.append(jnp.std(x))
        elif feature == "skew":
            # Skewness = mean((x - mean)^3) / std^3
            mean_x = jnp.mean(x)
            std_x = jnp.std(x)
            skew_val = jnp.mean(((x - mean_x) / std_x) ** 3)
            stats_list.append(skew_val)
        elif feature == "kurtosis":
            # Kurtosis = mean((x - mean)^4) / std^4 - 3
            mean_x = jnp.mean(x)
            std_x = jnp.std(x)
            kurt_val = jnp.mean(((x - mean_x) / std_x) ** 4) - 3
            stats_list.append(kurt_val)

    return jnp.array(stats_list)


def calculate_fcd_safe(bold_signal, window_size=44, overlap=0.9):
    """
    Calculate functional connectivity dynamics (FCD) safely.

    Args:
        bold_signal: Array of shape (time_points, n_regions)
        window_size: Size of sliding window (in timepoints)
        overlap: Overlap between windows (0-1)

    Returns:
        fcd: FCD matrix (n_windows x n_windows) - correlation between FC patterns
    """
    bold_signal = np.asarray(bold_signal)
    n_timepoints = bold_signal.shape[0]

    # Calculate step size from overlap
    step = max(1, int(window_size * (1 - overlap)))

    # Get FC for each window
    fc_list = []
    for start in range(0, n_timepoints - window_size + 1, step):
        window = bold_signal[start : start + window_size, :]

        # Calculate FC for this window
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered")
            fc_window = np.corrcoef(window.T)
            fc_window = np.nan_to_num(fc_window, nan=0.0)

        # Flatten upper triangle (excluding diagonal)
        fc_flat = fc_window[np.triu_indices_from(fc_window, k=1)]
        fc_list.append(fc_flat)

    if len(fc_list) < 2:
        raise ValueError(
            f"Not enough windows for FCD. Need at least 2, got {len(fc_list)}"
        )

    fc_array = np.array(fc_list)

    # Calculate correlation between FC patterns
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered")
        fcd = np.corrcoef(fc_array)
        fcd = np.nan_to_num(fcd, nan=0.0)

    return fcd


def power_spectral_analysis_jax(sign, dt=1000.0):
    """
    JAX-compatible power spectral analysis using Fast Fourier Transform.

    This function computes the power spectrum of a signal using FFT and identifies
    the dominant frequency component. Compatible with jax.jit and jax.vmap.

    Parameters
    ----------
    sign : jax.Array
        Input signal array. Should be 1D for vmap compatibility.
    dt : float, optional
        Time step between samples in milliseconds (default: 1000.0).

    Returns
    -------
    freqs : jax.Array
        Array of frequency values corresponding to the FFT bins.
    power_spectrum : jax.Array
        Power spectrum of the input signal (magnitude squared of FFT).
    dominant_frequency : float
        Frequency with the highest power in the spectrum.

    Notes
    -----
    - This is a JAX implementation that can be jitted and vmapped.
    - The dominant frequency is determined by finding the frequency bin with
      maximum power, which may not be suitable for all signal types.
    """
    sign = jnp.asarray(sign)
    fft_output = jnp.fft.fft(sign)
    freqs = jnp.fft.fftfreq(len(sign), dt / 1000.0)
    power_spectrum = jnp.abs(fft_output) ** 2
    dominant_frequency = freqs[jnp.argmax(power_spectrum)]

    return freqs, power_spectrum, dominant_frequency


def compute_fALFF_jax(freqs, ampl, low_freq_range=(0.01, 0.08), high_freq_range=(0.0, 0.25)):
    """
    JAX-compatible computation of fractional Amplitude of Low Frequency Fluctuations (fALFF).

    fALFF is a measure of the relative contribution of low-frequency oscillations
    to the total frequency power within a specified frequency range.

    Parameters
    ----------
    freqs : jax.Array
        Array of frequency values.
    ampl : jax.Array
        Array of amplitude/power values corresponding to the frequencies.
    low_freq_range : tuple of float, optional
        Low frequency range (min, max) in Hz (default: (0.01, 0.08)).
    high_freq_range : tuple of float, optional
        High frequency range (min, max) in Hz (default: (0.0, 0.25)).

    Returns
    -------
    ALFF : float
        Amplitude of Low Frequency Fluctuations (sum of power in given low range in Hz).
    fALFF : float
        Fractional ALFF (ALFF divided by total power in given high range).

    Notes
    -----
    - This is a JAX implementation that can be jitted and vmapped.
    - Returns 0 for fALFF if total amplitude in high frequency range is zero.
    """
    freqs = jnp.asarray(freqs)
    ampl = jnp.asarray(ampl)
    
    LF_idx = (freqs > low_freq_range[0]) & (freqs < low_freq_range[1])
    HF_idx = (freqs > high_freq_range[0]) & (freqs < high_freq_range[1])

    ALFF = jnp.sum(ampl * LF_idx)
    total_ampl = jnp.sum(ampl * HF_idx)

    # Use where to avoid division by zero in a JAX-friendly way
    fALFF = jnp.where(total_ampl != 0, ALFF / total_ampl, 0.0)

    return ALFF, fALFF


def find_ALFF_of_roi_jax(bold_roi_timeseries, dt=1000.0):
    """
    JAX-compatible computation of ALFF and fALFF for a single ROI time series.

    This function performs power spectral analysis on a single ROI time series
    and computes ALFF and fALFF measures. Compatible with jax.jit, jax.vmap, and jax.lax.scan.

    Parameters
    ----------
    bold_roi_timeseries : jax.Array
        BOLD signal array for a single ROI (1D array of time_points).
    dt : float, optional
        Time step between samples in milliseconds (default: 1000.0).

    Returns
    -------
    ALFF : float
        Amplitude of Low Frequency Fluctuations for the ROI.
    fALFF : float
        Fractional ALFF for the ROI.

    Notes
    -----
    - This is a JAX implementation that can be jitted, vmapped, or used in scan.
    - Only positive frequencies are considered in the analysis.
    - For multiple ROIs, use jax.lax.scan for memory efficiency (recommended) or jax.vmap.
    
    Examples
    --------
    Single ROI:
        >>> roi_idx = region_names.index("Region_3")
        >>> ALFF, fALFF = find_ALFF_of_roi_jax(bold_data[:, roi_idx], dt=100.0)
    
    All ROIs using scan (memory efficient, recommended):
        >>> def process_roi(carry, roi_ts):
        ...     ALFF, fALFF = find_ALFF_of_roi_jax(roi_ts, dt=100.0)
        ...     return carry, jnp.array([ALFF, fALFF])
        >>> bold_transposed = bold_data.T  # (n_regions, n_timepoints)
        >>> _, results = jax.lax.scan(process_roi, None, bold_transposed)
        >>> ALFF_all, fALFF_all = results[:, 0], results[:, 1]
    
    All ROIs using vmap (faster but uses more memory):
        >>> process_all_rois = jax.vmap(lambda x: find_ALFF_of_roi_jax(x, 100.0), in_axes=1)
        >>> ALFF_all, fALFF_all = process_all_rois(jnp.array(bold_data))
        
    Original numpy version:
        >>> ALFF, fALFF = find_ALFF_of_roi(bold_data, region_names, "Region_3", dt=100)
    """
    f, p, domf = power_spectral_analysis_jax(bold_roi_timeseries, dt)
    
    # Filter for positive frequencies
    positive_freqs_idx = f > 0
    f_pos = jnp.where(positive_freqs_idx, f, 0.0)
    p_pos = jnp.where(positive_freqs_idx, p, 0.0)

    ALFF, fALFF = compute_fALFF_jax(f_pos, p_pos)
    return ALFF, fALFF


def power_spectral_analysis(sign, dt=1000):
    """
    Perform power spectral analysis using Fast Fourier Transform.

    This function computes the power spectrum of a signal using FFT and identifies
    the dominant frequency component.

    Parameters
    ----------
    sign : array_like
        Input signal array. Can be 1D or 2D (with time along axis 0).
    dt : float, optional
        Time step between samples in milliseconds (default: 1000).

    Returns
    -------
    freqs : ndarray
        Array of frequency values corresponding to the FFT bins.
    power_spectrum : ndarray
        Power spectrum of the input signal (magnitude squared of FFT).
    dominant_frequency : float
        Frequency with the highest power in the spectrum.

    Notes
    -----
    The dominant frequency is determined by finding the frequency bin with
    maximum power, which may not be suitable for all signal types.
    """
    fft_output = np.fft.fft(sign, axis=0)
    freqs = np.fft.fftfreq(len(sign), dt / 1000)
    power_spectrum = np.abs(fft_output) ** 2
    dominant_frequency = freqs[np.argmax(power_spectrum)]

    return freqs, power_spectrum, dominant_frequency


def compute_fALFF(freqs, ampl, low_freq_range=(0.01, 0.08), high_freq_range=(0, 0.25)):
    """
    Compute fractional Amplitude of Low Frequency Fluctuations (fALFF).

    fALFF is a measure of the relative contribution of low-frequency oscillations
    to the total frequency power within a specified frequency range.

    Parameters
    ----------
    freqs : array_like
        Array of frequency values.
    ampl : array_like
        Array of amplitude/power values corresponding to the frequencies.

    Returns
    -------
    ALFF : float
        Amplitude of Low Frequency Fluctuations (sum of power in given low range in Hz).
    fALFF : float
        Fractional ALFF (ALFF divided by total power in given high range).

    Notes
    -----
    - Returns 0 for fALFF if total amplitude in high frequency range is zero.
    """
    LF_idx = (freqs > low_freq_range[0]) & (freqs < low_freq_range[1])
    HF_idx = (freqs > high_freq_range[0]) & (freqs < high_freq_range[1])

    ALFF = np.sum(ampl[LF_idx])  # or np.mean() depending on convention
    total_ampl = np.sum(ampl[HF_idx])

    fALFF = ALFF / total_ampl if total_ampl != 0 else 0

    return float(ALFF), float(fALFF)


def find_ALFF_of_roi(bold_sch, regions_names_general, roi, dt=1000):
    """
    Compute ALFF and fALFF for a specific region of interest (ROI).

    This function extracts the time series for a given ROI from BOLD data,
    performs power spectral analysis, and computes ALFF and fALFF measures.

    Parameters
    ----------
    bold_sch : ndarray
        BOLD signal array of shape (time_points, n_regions).
    regions_names_general : list
        List of region names corresponding to columns in bold_sch.
    roi : str
        Name of the region of interest to analyze.
    dt : float, optional
        Time step between samples in milliseconds (default: 1000).

    Returns
    -------
    ALFF : float
        Amplitude of Low Frequency Fluctuations for the ROI.
    fALFF : float
        Fractional ALFF for the ROI.

    Raises
    ------
    ValueError
        If the specified ROI is not found in regions_names_general.

    Notes
    -----
    Only positive frequencies are considered in the analysis.
    """
    f, p, domf = power_spectral_analysis(bold_sch[:, regions_names_general.index(roi)], dt)
    positive_freqs_idx = f > 0
    f = f[positive_freqs_idx]
    p = p[positive_freqs_idx]

    ALFF, fALFF = compute_fALFF(f, p)
    return ALFF, fALFF


# =========================
# Feature Pruning Module
# =========================

def prune_features(
    df: pd.DataFrame,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    pca_variance_ratio: Optional[float] = None,
    sort_indices: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Prune features from an unlabeled dataset using unsupervised methods.
    
    This function applies multiple feature selection techniques suitable for
    unsupervised learning:
    1. Variance threshold: Remove low-variance features
    2. Correlation-based: Remove highly correlated features
    3. PCA-based (optional): Keep features explaining desired variance
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe where each column represents a feature.
        Rows are samples, columns are features.
    variance_threshold : float, optional
        Minimum variance threshold. Features with variance below this value
        are removed. Default is 0.01.
    correlation_threshold : float, optional
        Maximum correlation threshold. If two features have correlation above
        this value, one is removed. Default is 0.95.
    pca_variance_ratio : float, optional
        If provided (between 0 and 1), performs PCA and keeps only features
        that cumulatively explain this ratio of variance. If None, PCA-based
        filtering is skipped. Default is None.
    sort_indices : bool, optional
        If True, sorts the retained indices in ascending order and reorders
        the dataframe columns accordingly. If False, features are ordered by
        importance (most important first) when PCA is used. Default is True.
    verbose : bool, optional
        If True, prints information about the pruning process. Default is True.
    
    Returns
    -------
    pruned_df : pd.DataFrame
        Dataframe with pruned features. Row order and NaN values are preserved.
    retained_indices : np.ndarray
        Array of column indices that were retained from the original dataframe.
    
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # Create sample data
    >>> data = np.random.randn(100, 20)
    >>> df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(20)])
    >>> 
    >>> # Prune features
    >>> pruned_df, retained_idx = prune_features(df, variance_threshold=0.1)
    >>> print(f"Original features: {df.shape[1]}, Retained: {pruned_df.shape[1]}")
    >>> print(f"Retained column indices: {retained_idx}")
    >>> 
    >>> # With NaN values and theta parameters
    >>> theta = np.random.randn(100, 5)  # Parameters corresponding to each row
    >>> pruned_df, retained_idx = prune_features(df, variance_threshold=0.1)
    >>> # Drop NaN rows after pruning if needed (keeps row order)
    >>> valid_mask = ~pruned_df.isna().any(axis=1)
    >>> pruned_df_clean = pruned_df[valid_mask]
    >>> theta_clean = theta[valid_mask]  # Filter theta to match
    
    Notes
    -----
    - **Row Order**: Row order is ALWAYS preserved. Original pandas indices are maintained.
    - **NaN Handling**: 
      * NaN values are ALWAYS kept in the output (never replaced/imputed)
      * For calculations only, NaNs are temporarily filled with column means
      * To drop NaN rows after pruning: use `df[~df.isna().any(axis=1)]`
      * To keep theta aligned: `theta[~df.isna().any(axis=1)]`
    - Features are pruned in the following order: variance → correlation → PCA
    - For correlation-based pruning, when two features are highly correlated,
      the one appearing later in the dataframe is removed
    - This function is designed for unlabeled data and does not use target labels
    - If `sort_indices=True` (default), returned indices are in ascending order
      for easier application to new data
    """
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.decomposition import PCA
    
    if verbose:
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                f"[bold cyan]Feature Pruning Started[/bold cyan]\n"
                f"Original features: {df.shape[1]}\n"
                f"Original samples: {df.shape[0]}",
                title="Feature Selection"
            ))
        else:
            print(f"Feature Pruning Started")
            print(f"Original features: {df.shape[1]}")
            print(f"Original samples: {df.shape[0]}")
    
    # Store original column names and indices
    original_columns = df.columns.tolist()
    retained_indices = np.arange(len(original_columns))
    
    # Create a copy to avoid modifying the original
    df_pruned = df.copy()
    
    # Step 1: Remove features with low variance
    if variance_threshold > 0:
        selector = VarianceThreshold(threshold=variance_threshold)
        
        # Handle NaN values temporarily for calculations only
        df_filled = df_pruned.fillna(df_pruned.mean())
        
        try:
            selector.fit(df_filled)
            variance_mask = selector.get_support()
            
            df_pruned = df_pruned.loc[:, variance_mask]
            retained_indices = retained_indices[variance_mask]
            
            if verbose:
                removed_count = np.sum(~variance_mask)
                if RICH_AVAILABLE:
                    console.print(f"[yellow]Variance threshold ({variance_threshold}):[/yellow] "
                                f"Removed {removed_count} low-variance features. "
                                f"Remaining: {df_pruned.shape[1]}")
                else:
                    print(f"Variance threshold ({variance_threshold}): "
                          f"Removed {removed_count} low-variance features. "
                          f"Remaining: {df_pruned.shape[1]}")
        except Exception as e:
            if verbose:
                print(f"Warning: Variance filtering failed: {e}")
    
    # Step 2: Remove highly correlated features
    if correlation_threshold < 1.0 and df_pruned.shape[1] > 1:
        # Compute correlation matrix
        corr_matrix = df_pruned.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper_corr = corr_matrix.where(upper_triangle)
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper_corr.columns 
                   if any(upper_corr[column] > correlation_threshold)]
        
        if len(to_drop) > 0:
            # Get indices to keep
            keep_mask = ~df_pruned.columns.isin(to_drop)
            df_pruned = df_pruned.loc[:, keep_mask]
            
            # Update retained indices
            retained_indices = retained_indices[keep_mask]
            
            if verbose:
                if RICH_AVAILABLE:
                    console.print(f"[yellow]Correlation threshold ({correlation_threshold}):[/yellow] "
                                f"Removed {len(to_drop)} highly correlated features. "
                                f"Remaining: {df_pruned.shape[1]}")
                else:
                    print(f"Correlation threshold ({correlation_threshold}): "
                          f"Removed {len(to_drop)} highly correlated features. "
                          f"Remaining: {df_pruned.shape[1]}")
    
    # Step 3: PCA-based feature selection (optional)
    if pca_variance_ratio is not None and df_pruned.shape[1] > 1:
        if not (0 < pca_variance_ratio <= 1):
            raise ValueError("pca_variance_ratio must be between 0 and 1")
        
        # Standardize the features
        df_filled = df_pruned.fillna(df_pruned.mean())
        df_standardized = (df_filled - df_filled.mean()) / df_filled.std()
        
        try:
            # Fit PCA
            pca = PCA(n_components=min(df_pruned.shape))
            pca.fit(df_standardized)
            
            # Calculate cumulative variance explained
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumulative_variance >= pca_variance_ratio) + 1
            
            # Get feature importances from PCA components
            # Use the absolute sum of loadings across selected components
            feature_importance = np.abs(pca.components_[:n_components, :]).sum(axis=0)
            
            # Select top features based on importance
            n_features_to_keep = min(n_components * 2, df_pruned.shape[1])
            # Get indices of most important features, sorted by importance (descending)
            top_features_idx = np.argsort(feature_importance)[-n_features_to_keep:]
            # Reverse to get most important first when sort_indices=False
            top_features_idx = top_features_idx[::-1]
            
            # Filter dataframe
            df_pruned = df_pruned.iloc[:, top_features_idx]
            retained_indices = retained_indices[top_features_idx]
            
            if verbose:
                if RICH_AVAILABLE:
                    console.print(f"[yellow]PCA-based selection ({pca_variance_ratio:.1%} variance):[/yellow] "
                                f"Kept {n_features_to_keep} most important features. "
                                f"Remaining: {df_pruned.shape[1]}")
                else:
                    print(f"PCA-based selection ({pca_variance_ratio:.1%} variance): "
                          f"Kept {n_features_to_keep} most important features. "
                          f"Remaining: {df_pruned.shape[1]}")
        except Exception as e:
            if verbose:
                print(f"Warning: PCA-based filtering failed: {e}")
    
    if verbose:
        if RICH_AVAILABLE:
            console.print(Panel.fit(
                f"[bold green]Feature Pruning Complete[/bold green]\n"
                f"Original: {len(original_columns)} → Final: {df_pruned.shape[1]}\n"
                f"Reduction: {(1 - df_pruned.shape[1]/len(original_columns)):.1%}",
                title="Results"
            ))
        else:
            print(f"\nFeature Pruning Complete")
            print(f"Original: {len(original_columns)} → Final: {df_pruned.shape[1]}")
            print(f"Reduction: {(1 - df_pruned.shape[1]/len(original_columns)):.1%}")
    
    # Sort indices if requested (maintains original column order)
    if sort_indices:
        sort_order = np.argsort(retained_indices)
        retained_indices = retained_indices[sort_order]
        df_pruned = df_pruned.iloc[:, sort_order]
    
    return df_pruned, retained_indices

