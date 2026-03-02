"""
NumPy-based implementation of Functional Connectivity Dynamics (FCD) computation.

This module provides CPU-based functions for computing FCD matrices using NumPy.
It includes support for configurable window overlap and TR (repetition time) parameters.

Functions
---------
get_fc : Compute functional connectivity matrix
get_fcd : Compute Functional Connectivity Dynamics matrix with configurable overlap
fcd_ks_cost : Calculate FCD KS statistics cost comparing empirical and simulated data
fc_correlation_cost : Calculate FC correlation cost comparing empirical and simulated data

Notes
-----
This is the default CPU implementation. For GPU-accelerated computation,
see core_jax.py or core_torch.py modules.
"""
import numpy as np
from copy import deepcopy


def get_fc(ts, positive=False, fc_fucntion="corrcoef"):
    """
    calculate the functional connectivity matrix

    Parameters
    ----------
    ts : numpy.ndarray [n_regions, n_timepoints]
        Input signal
    positive : bool, optional
        If True, only keep positive correlations (default: False)
    fc_fucntion : str, optional
        Function to use for computing FC: "corrcoef" or "cov" (default: "corrcoef")

    Returns
    -------
    FC : numpy.ndarray
        functional connectivity matrix
    """

    from numpy import corrcoef, cov

    fc = eval(fc_fucntion)(ts)
    
    if positive:
        fc = fc * (fc > 0)
    
    # Remove diagonal
    # FC = FC - np.diag(np.diagonal(FC))

    return fc


def get_fcd(
    ts,
    TR=1,
    win_len=30,
    positive=False,
    overlap=None,
):
    """
    Compute dynamic functional connectivity.

    Parameters
    ----------

    ts: numpy.ndarray [n_regions, n_timepoints]
        Input signal
    win_len: int
        sliding window length in samples, default is 30
    TR: int
        repetition time. It refers to the amount of time that
        passes between consecutive acquired brain volumes during
        functional magnetic resonance imaging (fMRI) scans.
    positive: bool
        if True, only positive values of FC are considered.
        default is False
    overlap: float, optional
        Overlap between consecutive windows as fraction (0-1).
        If None (default), uses maximum overlap (1-sample shift).
        0.0 = no overlap, 0.9 = 90% overlap, etc.

    Returns
    -------
        FCD: ndarray
            matrix of functional connectivity dynamics
    """
    if not isinstance(ts, np.ndarray):
        ts = np.array(ts)

    ts = ts.T
    n_samples, n_nodes = ts.shape
    win_len_samples = int(win_len / TR)
    
    # check if lenght of the time series is enough
    if n_samples < 2 * win_len_samples:
        raise ValueError(
            f"get_fcd: Length of the time series should be at least 2 times of win_len. n_samples: {n_samples}, win_len: {win_len_samples}"
        )

    # Calculate step size based on overlap
    if overlap is None:
        # Maximum overlap: 1-sample shift
        step = 1
    else:
        if not (0.0 <= overlap < 1.0):
            raise ValueError(f"overlap must be between 0.0 and <1.0, got {overlap}")
        # step = win_len * (1 - overlap)
        step = max(1, int(win_len_samples * (1 - overlap)))
    
    # Create windows manually with specified step
    n_windows = (n_samples - win_len_samples) // step + 1
    windowed_data = np.array([
        ts[i*step:i*step+win_len_samples, :] 
        for i in range(n_windows)
    ])
    
    fc_stream = np.asarray(
        [np.corrcoef(windowed_data[i, :, :], rowvar=False) for i in range(n_windows)]
    )

    if positive:
        fc_stream *= fc_stream > 0

    # Extract upper triangular part of FC matrices (excluding diagonal)
    mask_full = np.ones((n_nodes, n_nodes))
    mask = np.triu(mask_full, k=1)
    nonzero_idx = np.nonzero(mask)
    fc_stream_masked = fc_stream[:, nonzero_idx[0], nonzero_idx[1]]
    fcd = np.corrcoef(fc_stream_masked, rowvar=True)

    return fcd


def fcd_ks_cost(
    bold_empirical,
    bold_simulated,
    window_size=30,
    TR=1,
    n_bins=10000,
    fcd_range=(-1.0, 1.0),
    overlap=None,
    verbose=True,
):
    """
    Calculate FCD KS statistics cost comparing empirical and simulated BOLD signals.

    This function computes functional connectivity dynamics (FCD) matrices using sliding
    windows, then calculates the Kolmogorov-Smirnov statistic to compare the distributions
    of FCD values between empirical and simulated data. NaN values in simulated data are
    handled by assigning np.nan to affected sets, allowing you to filter them out later.

    Parameters
    ----------
    bold_empirical : numpy.ndarray
        Empirical BOLD signal of shape [n_nodes, n_timesamples]
    bold_simulated : numpy.ndarray
        Simulated BOLD signal of shape [n_nodes, n_timesamples] or [n_sets, n_nodes, n_timesamples]
        If 3D, each set along axis 0 is evaluated separately
    window_size : int, optional
        Size of sliding window in TR units (default: 30)
    TR : int, optional
        Repetition time (default: 1)
    n_bins : int, optional
        Number of bins for FCD histogram (default: 10000)
    fcd_range : tuple, optional
        Range for FCD histogram as (min, max) tuple (default: (-1.0, 1.0))
    overlap : float, optional
        Overlap between consecutive windows as fraction (0-1).
        If None (default), uses maximum overlap (1-sample shift)
    verbose : bool, optional
        Whether to print timing information (default: True)

    Returns
    -------
    ks_cost : float or numpy.ndarray
        KS statistics cost. If bold_simulated is 2D, returns a single float.
        If 3D with n_sets, returns array of shape [n_sets,] with one cost per set.
        Values are normalized KS distances between [0, 1], where 0 = perfect match.
        Sets with NaN values or failed computations are assigned np.nan.

    Example
    -------
    >>> import numpy as np
    >>> bold_emp = np.random.randn(68, 1200)  # 68 nodes, 1200 timepoints
    >>> bold_sim = np.random.randn(68, 1200)  # Single simulated set
    >>> cost = fcd_ks_cost(bold_emp, bold_sim, window_size=30)
    >>> # Or with multiple sets:
    >>> bold_sim_multi = np.random.randn(100, 68, 1200)  # 100 different parameter sets
    >>> costs = fcd_ks_cost(bold_emp, bold_sim_multi, window_size=30)
    >>> costs.shape
    (100,)
    """
    import time
    
    if verbose:
        fcd_timestart = time.time()

    # Validate input shapes
    if bold_empirical.ndim != 2:
        raise ValueError(
            f"bold_empirical must be 2D [n_nodes, n_timesamples], got shape {bold_empirical.shape}"
        )

    # Check if simulated is 2D or 3D
    if bold_simulated.ndim == 2:
        # Single simulation, add set dimension
        bold_simulated = bold_simulated[np.newaxis, :, :]
        single_set = True
    elif bold_simulated.ndim == 3:
        single_set = False
    else:
        raise ValueError(
            f"bold_simulated must be 2D [n_nodes, n_timesamples] or 3D [n_sets, n_nodes, n_timesamples], got shape {bold_simulated.shape}"
        )

    n_nodes = bold_empirical.shape[0]
    n_timesamples = bold_empirical.shape[1]
    n_sets = bold_simulated.shape[0]

    if bold_simulated.shape[1] != n_nodes:
        raise ValueError(
            f"Node dimension mismatch: empirical has {n_nodes}, simulated has {bold_simulated.shape[1]}"
        )

    # Handle time dimension mismatch
    sim_timesamples = bold_simulated.shape[2]
    if sim_timesamples != n_timesamples:
        min_timesamples = min(n_timesamples, sim_timesamples)
        if verbose:
            print(
                f"Warning: Time dimension mismatch - empirical has {n_timesamples}, "
                f"simulated has {sim_timesamples}. Using minimum length {min_timesamples}."
            )
        bold_empirical = bold_empirical[:, :min_timesamples]
        bold_simulated = bold_simulated[:, :, :min_timesamples]

    # Check for NaN in empirical data
    if np.isnan(bold_empirical).any():
        raise ValueError("bold_empirical contains NaN values")

    # Compute empirical FCD
    try:
        emp_fcd = get_fcd(bold_empirical, TR=TR, win_len=window_size, overlap=overlap)
    except Exception as e:
        raise ValueError(f"Failed to compute empirical FCD: {e}")

    # Check if empirical FCD contains NaN
    if np.isnan(emp_fcd).any():
        raise ValueError("Empirical FCD contains NaN values")

    # Extract upper triangular values (excluding diagonal) from empirical FCD
    emp_fcd_triu = emp_fcd[np.triu_indices_from(emp_fcd, k=1)]
    
    # Create histogram of empirical FCD values
    emp_hist, bin_edges = np.histogram(emp_fcd_triu, bins=n_bins, range=fcd_range)
    emp_cumsum = np.cumsum(emp_hist) / emp_hist.sum()  # Normalize to [0, 1]

    # Compute KS cost for each simulated set
    ks_costs = np.zeros(n_sets)
    
    for i in range(n_sets):
        # Check for NaN in this simulated set
        if np.isnan(bold_simulated[i]).any():
            if verbose:
                print(f"Warning: Set {i} contains NaN values, assigning cost of np.nan")
            ks_costs[i] = np.nan
            continue
            
        try:
            # Compute FCD for this simulation
            sim_fcd = get_fcd(bold_simulated[i, :, :], TR=TR, win_len=window_size, overlap=overlap)
            
            # Check if simulated FCD contains NaN
            if np.isnan(sim_fcd).any():
                if verbose:
                    print(f"Warning: FCD for set {i} contains NaN values, assigning cost of np.nan")
                ks_costs[i] = np.nan
                continue
            
            # Extract upper triangular values
            sim_fcd_triu = sim_fcd[np.triu_indices_from(sim_fcd, k=1)]
            
            # Create histogram
            sim_hist, _ = np.histogram(sim_fcd_triu, bins=n_bins, range=fcd_range)
            
            # Check if histogram is empty
            if sim_hist.sum() == 0:
                if verbose:
                    print(f"Warning: Empty histogram for set {i}, assigning cost of np.nan")
                ks_costs[i] = np.nan
                continue
                
            sim_cumsum = np.cumsum(sim_hist) / sim_hist.sum()  # Normalize to [0, 1]
            
            # Calculate KS statistic: max absolute difference between CDFs
            ks_stat = np.abs(emp_cumsum - sim_cumsum).max()
            ks_costs[i] = ks_stat
            
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to compute FCD for set {i}: {e}")
            ks_costs[i] = np.nan  # Assign NaN for failed computation

    if verbose:
        fcd_elapsed = time.time() - fcd_timestart
        print(f"Time for calculating FCD KS statistics cost: {fcd_elapsed:.3f}s")

    # Return scalar if single set, array otherwise
    if single_set:
        return ks_costs[0]
    else:
        return ks_costs


def fc_correlation_cost(
    bold_empirical,
    bold_simulated,
    verbose=True,
):
    """
    Calculate FC correlation cost comparing empirical and simulated BOLD signals.
    
    This function computes functional connectivity (FC) matrices from empirical and
    simulated BOLD signals, then calculates the correlation between them. NaN values
    are handled by assigning np.nan to affected sets.

    Parameters
    ----------
    bold_empirical : numpy.ndarray
        Empirical BOLD signal of shape [n_nodes, n_timesamples]
    bold_simulated : numpy.ndarray
        Simulated BOLD signal of shape [n_nodes, n_timesamples] or [n_sets, n_nodes, n_timesamples]
        If 3D, each set along axis 0 is evaluated separately
    verbose : bool, optional
        Whether to print timing information (default: True)
        
    Returns
    -------
    corr_cost : float or numpy.ndarray
        FC correlation cost for each parameter set. If bold_simulated is 2D, returns a single float.
        If 3D with n_sets, returns array of shape [n_sets,].
        Values range from 0 (perfect match) to ~2 (worst case).
        Sets with NaN values are assigned np.nan.
                  
    Example
    -------
    >>> import numpy as np
    >>> bold_emp = np.random.randn(68, 1200)  # 68 nodes, 1200 timepoints
    >>> bold_sim = np.random.randn(68, 1200)  # Single simulated set
    >>> cost = fc_correlation_cost(bold_emp, bold_sim)
    >>> # Or with multiple sets:
    >>> bold_sim_multi = np.random.randn(100, 68, 1200)  # 100 different parameter sets
    >>> costs = fc_correlation_cost(bold_emp, bold_sim_multi)
    >>> costs.shape
    (100,)
    """
    import time
    
    if verbose:
        fc_timestart = time.time()
    
    # Validate input shapes
    if bold_empirical.ndim != 2:
        raise ValueError(
            f"bold_empirical must be 2D [n_nodes, n_timesamples], got shape {bold_empirical.shape}"
        )
    
    # Check if simulated is 2D or 3D
    if bold_simulated.ndim == 2:
        # Single simulation, add set dimension
        bold_simulated = bold_simulated[np.newaxis, :, :]
        single_set = True
    elif bold_simulated.ndim == 3:
        single_set = False
    else:
        raise ValueError(
            f"bold_simulated must be 2D [n_nodes, n_timesamples] or 3D [n_sets, n_nodes, n_timesamples], got shape {bold_simulated.shape}"
        )
    
    n_nodes = bold_empirical.shape[0]
    n_timesamples = bold_empirical.shape[1]
    n_sets = bold_simulated.shape[0]
    
    if bold_simulated.shape[1] != n_nodes:
        raise ValueError(
            f"Node dimension mismatch: empirical has {n_nodes}, simulated has {bold_simulated.shape[1]}"
        )
    
    # Handle time dimension mismatch
    sim_timesamples = bold_simulated.shape[2]
    if sim_timesamples != n_timesamples:
        min_timesamples = min(n_timesamples, sim_timesamples)
        if verbose:
            print(
                f"Warning: Time dimension mismatch - empirical has {n_timesamples}, "
                f"simulated has {sim_timesamples}. Using minimum length {min_timesamples}."
            )
            if min_timesamples < 50:
                print(f"Warning: Time series length ({min_timesamples}) is less than 50, which may affect correlation quality.")
        bold_empirical = bold_empirical[:, :min_timesamples]
        bold_simulated = bold_simulated[:, :, :min_timesamples]
        n_timesamples = min_timesamples
    elif verbose and n_timesamples < 50:
        print(f"Warning: Time series length ({n_timesamples}) is less than 50, which may affect correlation quality.")
    
    # Check for NaN in empirical data
    if np.isnan(bold_empirical).any():
        raise ValueError("bold_empirical contains NaN values")
    
    # Compute empirical FC
    emp_fc = np.corrcoef(bold_empirical)
    
    # Check if empirical FC contains NaN
    if np.isnan(emp_fc).any():
        raise ValueError("Empirical FC contains NaN values")
    
    # Create upper triangular mask for vectorizing FC matrices (excluding diagonal)
    fc_mask = np.triu(np.ones((n_nodes, n_nodes), dtype=bool), k=1)
    vect_len = int(n_nodes * (n_nodes - 1) / 2)
    
    # Vectorize empirical FC
    emp_fc_vector = emp_fc[fc_mask]
    
    # Compute costs for each simulated set
    corr_costs = np.zeros(n_sets)
    
    for i in range(n_sets):
        # Check for NaN in this simulated set
        if np.isnan(bold_simulated[i]).any():
            if verbose:
                print(f"Warning: Set {i} contains NaN values, assigning cost of np.nan")
            corr_costs[i] = np.nan
            continue
        
        try:
            # Compute simulated FC
            sim_fc = np.corrcoef(bold_simulated[i, :, :])
            
            # Check if simulated FC contains NaN
            if np.isnan(sim_fc).any():
                if verbose:
                    print(f"Warning: FC for set {i} contains NaN values, assigning cost of np.nan")
                corr_costs[i] = np.nan
                continue
            
            # Vectorize simulated FC
            sim_fc_vector = sim_fc[fc_mask]
            
            # Apply Fisher z-transformation
            # Clip values to avoid arctanh of exactly ±1
            emp_fc_z = np.arctanh(np.clip(emp_fc_vector, -0.999999, 0.999999))
            sim_fc_z = np.arctanh(np.clip(sim_fc_vector, -0.999999, 0.999999))
            
            # Calculate correlation between empirical and simulated FC vectors
            correlation = np.corrcoef(emp_fc_z, sim_fc_z)[0, 1]
            
            # Check if correlation is NaN (can happen with constant vectors)
            if np.isnan(correlation):
                if verbose:
                    print(f"Warning: Correlation for set {i} is NaN, assigning cost of np.nan")
                corr_costs[i] = np.nan
                continue
            
            # Convert to cost: 1 - correlation (so 0 = perfect, 2 = worst case)
            corr_costs[i] = 1 - correlation
            
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to compute FC correlation for set {i}: {e}")
            corr_costs[i] = np.nan
    
    if verbose:
        fc_elapsed = time.time() - fc_timestart
        print(f"Time for calculating FC correlation cost: {fc_elapsed:.3f}s")
    
    # Return scalar if single set, array otherwise
    if single_set:
        return corr_costs[0]
    else:
        return corr_costs
