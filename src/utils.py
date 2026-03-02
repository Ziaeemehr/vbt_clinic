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
import lib.gast_model as gm
from lib.inference import Inference
from vbt_trial.analysis.analyse_simulation import (
    compute_fcd,
    find_ALFF_of_roi,
)

# Import analysis functions for backward compatibility
from lib.analysis import (
    get_bold_feature_labels,
    extract_bold_features,
    compute_fcd_jax,
    compute_fc_jax,
    matrix_stat_jax,
    compute_stats_jax,
    calculate_fcd_safe,
)

# from vbt_trial.simulations import gast_model as gm

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


def format_time_duration(seconds, message=None, print_output=False, success=True):
    """
    Format time duration in seconds to hours:minutes:seconds format.
    Optionally print a beautiful message with the duration.
    
    Args:
        seconds (float): Time duration in seconds
        message (str, optional): Custom message to display with the duration
        print_output (bool): Whether to print the message (default: False)
        success (bool): Whether this is a success message (affects styling)
        
    Returns:
        str: Formatted time string (e.g., "2h 15m 30s" or "45m 23s" or "12s")
    """
    if seconds < 0:
        duration = "0s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            duration = f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            duration = f"{minutes}m {secs}s"
        else:
            duration = f"{secs}s"
    
    if print_output and message:
        if RICH_AVAILABLE:
            # Create beautiful rich output
            text = Text()
            if success:
                text.append("✅ ", style="bold green")
                border_style = "green"
                title = "[bold green]🎉 Success"
            else:
                text.append("ℹ️  ", style="bold blue")
                border_style = "blue"
                title = "[bold blue]ℹ️  Info"
            
            text.append(message, style="bold cyan")
            text.append(" in ", style="white")
            text.append(duration, style="bold yellow")
            
            panel = Panel(
                text,
                title=title,
                border_style=border_style,
                padding=(0, 1)
            )
            console.print(panel)
        else:
            # Fallback to regular print with emojis
            emoji = "✅" if success else "ℹ️"
            print(f"{emoji} {message} in {duration}")
    
    return duration


# TODO write a test function to ensure get_bold_feature_labels and extract_bold_features are consistent


def plot_matrix(
    A,
    ax,
    xlabel="",
    ylabel="",
    title="",
    # xticks=[], yticks=[],
    **kwargs,
):

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    im = ax.imshow(A, **kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # ax.set_xticks(xticks)
    # ax.set_yticks(yticks)


def calculate_fc_safe(bold_signal, remove_nan=True):
    """
    Calculate functional connectivity (FC) safely, handling numerical issues.

    The RuntimeWarning about division when computing correlation occurs when
    some regions have very low or zero variance. This function handles that.

    Args:
        bold_signal: Array of shape (time_points, n_regions)
        remove_nan: If True, suppress warnings and return FC even if NaN present

    Returns:
        fc: Functional connectivity matrix (n_regions x n_regions)
    """
    bold_signal = np.asarray(bold_signal)

    if remove_nan:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered")
            fc = np.corrcoef(bold_signal.T)
            # Replace any NaN values with 0
            fc = np.nan_to_num(fc, nan=0.0)
    else:
        fc = np.corrcoef(bold_signal.T)

    return fc


def plot_kde_1d(
    samples: np.ndarray,
    xlim: Tuple[float, float],
    ax: maxes.Axes,
    *,
    bandwidth: float = 0.03,
    n_grid: int = 1000,
    color: str = "#f41e49",
    prior: Optional[np.ndarray] = None,
    x_true: Optional[float] = None,
    kde_color: str = "#f0f0f0",
    prior_color: str = "blue",
    truth_color: str = "green",
    alpha: float = 1.0,
) -> None:
    """
    Plot a 1D kernel density estimate (KDE) with optional prior and ground truth.

    Parameters
    ----------
    samples : np.ndarray
        1D array of posterior samples.
    xlim : (float, float)
        Plot range.
    ax : matplotlib.axes.Axes
        Target axes.
    bandwidth : float, optional
        KDE bandwidth.
    n_grid : int, optional
        Number of evaluation points.
    color : str, optional
        Fill color for posterior KDE.
    prior : np.ndarray, optional
        1D array of prior samples.
    x_true : float, optional
        Ground truth value.
    """

    samples = np.asarray(samples).ravel()
    x_grid = np.linspace(*xlim, n_grid)

    # Posterior KDE
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(samples[:, None])
    log_pdf = kde.score_samples(x_grid[:, None])
    pdf = np.exp(log_pdf)

    ax.plot(x_grid, pdf, color=kde_color, lw=1)
    # ax.fill_between(x_grid, pdf, color=color, alpha=alpha)

    # Prior KDE
    if prior is not None:
        prior = np.asarray(prior).ravel()
        kde.fit(prior[:, None])
        prior_pdf = np.exp(kde.score_samples(x_grid[:, None]))
        ax.plot(x_grid, prior_pdf, color=prior_color, lw=1)

    # Ground truth
    if x_true is not None:
        ax.axvline(x_true, color=truth_color, lw=3, alpha=0.6)

    # Style cleanup
    ax.set_xlim(xlim)
    ax.set_yticks([])
    ax.patch.set_alpha(0.0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_joint_kde(
    samples: np.ndarray,
    limits: Sequence[Tuple[float, float]],
    ax: maxes.Axes,
    *,
    cmap: str = "hot",
    grid_size: int = 60,
    point: Optional[Tuple[float, float]] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    show_corr: bool = True,
    corr_loc: Tuple[float, float] = (0.05, 0.85),
    label_size: int = 22,
) -> None:
    """
    Plot a 2D joint KDE using imshow.

    Parameters
    ----------
    samples : np.ndarray
        Shape (N, 2) array of samples.
    limits : [(xmin, xmax), (ymin, ymax)]
        Plot limits for each dimension.
    ax : matplotlib.axes.Axes
        Target axes.
    """

    samples = np.asarray(samples)
    assert samples.shape[1] == 2, "samples must have shape (N, 2)"

    kde = gaussian_kde(samples.T, bw_method="scott")

    x = np.linspace(*limits[0], grid_size)
    y = np.linspace(*limits[1], grid_size)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = kde(positions).reshape(X.shape)

    ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=(*limits[0], *limits[1]),
    )

    # Optional reference point
    if point is not None:
        ax.scatter(
            point[0],
            point[1],
            s=150,
            marker="*",
            color="#6cf086",
            edgecolors="k",
            zorder=10,
        )

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_size + 2, labelpad=-10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_size + 2, rotation=0, labelpad=-10)

    # Correlation annotation
    if show_corr:
        rho = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
        ax.text(
            corr_loc[0],
            corr_loc[1],
            rf"$\rho = {rho:.2f}$",
            transform=ax.transAxes,
            fontsize=12,
            color="white",
        )


def plot_theta_obs(theta_obs, param_labels):
    """
    Plot the observation parameters theta_obs.

    Args:
        theta_obs (np.ndarray): Array of observation parameters.
        param_labels (list): List of parameter labels.
    """
    n_params = len(param_labels)
    fig = plt.figure()
    if n_params == 3:
        ax = fig.add_subplot(projection="3d")
        ax.scatter(theta_obs[:, 0], theta_obs[:, 1], theta_obs[:, 2], s=20)
        ax.set_xlabel(param_labels[0])
        ax.set_ylabel(param_labels[1])
        ax.set_zlabel(param_labels[2])
    elif n_params == 2:
        plt.scatter(theta_obs[:, 0], theta_obs[:, 1], s=5)
        plt.xlabel(param_labels[0])
        plt.ylabel(param_labels[1])
        plt.title("Interior chess-grid sampling")
    plt.show()


def analyze_graph_from_mask(mask, name="Graph"):
    """
    Analyze a graph defined by a binary mask matrix.

    Args:
        mask (np.ndarray): Binary adjacency matrix where non-zero entries indicate edges.
        name (str): Name of the graph for reporting.

    Prints:
        - Number of nodes
        - Number of edges
        - Average degree
        - Density
        - Graph type (directed/undirected)
    """
    num_nodes = mask.shape[0]
    
    # Check if the graph is symmetric (undirected)
    is_symmetric = np.allclose(mask, mask.T)
    
    if is_symmetric:
        # Undirected graph
        num_edges = np.sum(mask != 0) // 2  # Count each edge once
        avg_degree = np.sum(mask != 0) / num_nodes
        density = num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
        graph_type = "Undirected"
    else:
        # Directed graph
        num_edges = np.sum(mask != 0)  # Count all directed edges
        avg_out_degree = np.sum(mask != 0, axis=1).mean()
        avg_in_degree = np.sum(mask != 0, axis=0).mean()
        avg_degree = avg_out_degree  # For directed graphs, we report out-degree
        density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        graph_type = "Directed"

    if RICH_AVAILABLE:
        from rich.table import Table
        
        # Create a table for the graph statistics
        table = Table(title=f"📊 {name} Analysis", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan", width=20)
        table.add_column("Value", style="green", justify="right")
        
        # Add rows
        if is_symmetric:
            graph_emoji = "🔄 "
            graph_style = "bold blue"
        else:
            graph_emoji = "➡️ "
            graph_style = "bold yellow"
        
        table.add_row("Graph Type", f"{graph_emoji} [bold]{graph_type}[/bold]")
        table.add_row("Number of Nodes", f"[bold white]{num_nodes}[/bold white]")
        table.add_row("Number of Edges", f"[bold white]{num_edges}[/bold white]")
        
        if not is_symmetric:
            table.add_row("Avg Out-Degree", f"[bold cyan]{avg_out_degree:.2f}[/bold cyan]")
            table.add_row("Avg In-Degree", f"[bold cyan]{avg_in_degree:.2f}[/bold cyan]")
        else:
            table.add_row("Average Degree", f"[bold cyan]{avg_degree:.2f}[/bold cyan]")
        
        # Color code density
        if density < 0.1:
            density_color = "red"
            density_desc = "(sparse)"
        elif density < 0.5:
            density_color = "yellow"
            density_desc = "(moderate)"
        else:
            density_color = "green"
            density_desc = "(dense)"
        
        table.add_row("Density", f"[bold {density_color}]{density:.4f}[/bold {density_color}] {density_desc}")
        
        console.print(table)
        console.print("\n")
        
    else:
        # Fallback to regular print
        print(f"\n{'='*50}")
        print(f"{name} Statistics:")
        print(f"{'='*50}")
        print(f"Graph type: {graph_type}")
        print(f"Number of nodes: {num_nodes}")
        print(f"Number of edges: {num_edges}")
        if not is_symmetric:
            print(f"Average out-degree: {avg_out_degree:.2f}")
            print(f"Average in-degree: {avg_in_degree:.2f}")
        else:
            print(f"Average degree: {avg_degree:.2f}")
        print(f"Density: {density:.4f}")
        print(f"{'='*50}\n")

def load_simulation_data(data_dir="data", verbose=False):
    """
    Load and process simulation data from CSV files.

    Args:
        data_dir (str): Directory containing the data files.

    Returns:
        Ceids (np.ndarray): Excitatory connections.
        Leids (np.ndarray): Lengths for connections.
        Seids (scipy.sparse.csr_matrix): Sparse matrix of connections.
        Rd1 (np.ndarray): D1 receptor data.
        Rd2 (np.ndarray): D2 receptor data.
        Rsero (np.ndarray): Serotonin receptor data.
    """
    W = pd.read_csv(os.path.join(data_dir, "weights_with_sero_and_dopa.csv"), index_col=0)
    L = pd.read_csv(os.path.join(data_dir, "lengths_with_sero_and_dopa.csv"), index_col=0)
    Ce_mask = pd.read_csv(os.path.join(data_dir, "dk_sero_exc_mask.csv"), index_col=0)
    Ci_mask = pd.read_csv(os.path.join(data_dir, "dk_sero_inh_mask.csv"), index_col=0)
    Cd_mask = pd.read_csv(os.path.join(data_dir, "dk_sero_dopa_mask.csv"), index_col=0)
    Cs_mask = pd.read_csv(os.path.join(data_dir, "dk_sero_sero_mask.csv"), index_col=0)
    Ce = W.values * Ce_mask.values
    Ci = W.values * Ci_mask.values
    Cd = W.values * Cd_mask.values
    Cs = W.values * Cs_mask.values
    region_names = W.index.tolist()

    Ceids = np.vstack([Ce, Ci, Cd, Cs])
    Leids = np.vstack([L, L, L, L])
    Seids = scipy.sparse.csr_matrix(Ceids)
    
    if verbose:
        analyze_graph_from_mask(Cs_mask.values, name="Serotonin Receptor Graph")
        analyze_graph_from_mask(Ce_mask.values, name="Excitatory Connection Graph")
        analyze_graph_from_mask(Ci_mask.values, name="Inhibitory Connection Graph")
        analyze_graph_from_mask(Cd_mask.values, name="Dopamine Connection Graph")

    Rdf = pd.read_csv(os.path.join(data_dir, "dk_D1_D2_5HT2A_receptor_data.csv"))
    Rd1 = Rdf["D1_number"].values
    Rd2 = Rdf["D2_number"].values
    Rsero = Rdf["5HT2A_number"].values
    Rd2 = 1.5 * Rd2 / (5 * Rd2.max())
    Rd1 = 1.5 * Rd1 / (5 * Rd1.max())
    Rsero = 1.5 * Rsero / (5 * Rsero.max())
    Rd1 = Rd1.reshape(-1, 1)
    Rd2 = Rd2.reshape(-1, 1)
    Rsero = Rsero.reshape(-1, 1)

    return Ceids, Leids, Seids, Rd1, Rd2, Rsero, region_names


def plot_features_vs_params(theta_np, X, theta_obs, x_obs, param_labels, kwargs_features, output_dir, verbose=False, max_sim_plot=1000, max_obs_plot=100):
    if verbose:
        print("Plotting features vs parameters...")

    theta_obs_arr = np.array(theta_obs)
    xo_arr = np.array(x_obs)
    n_labels = len(param_labels)
    if theta_obs_arr.shape[1] != n_labels:
        raise ValueError(
            f"theta_obs length {theta_obs_arr.shape[1]} != expected {n_labels}"
        )

    num_sim = theta_np.shape[0]
    if theta_obs_arr.shape[0] > max_obs_plot:
        obs_indices = np.random.choice(theta_obs_arr.shape[0], max_obs_plot, replace=False)
        theta_obs_plot = theta_obs_arr[obs_indices]
        xo_plot = xo_arr[obs_indices]
    else:
        theta_obs_plot = theta_obs_arr
        xo_plot = xo_arr

    if num_sim > max_sim_plot:
        indices = np.random.choice(num_sim, max_sim_plot, replace=False)
        X_plot = X[indices]
        theta_plot = theta_np[indices]
    else:
        X_plot = X
        theta_plot = theta_np

    labels = param_labels
    n_labels = len(labels)

    for ii in range(n_labels):
        xvalues = theta_plot[:, ii]
        feature_labels = get_bold_feature_labels(**kwargs_features)
        n_features = X_plot.shape[1]
        feature_indices = list(range(0, n_features, 2))
        num_plots = len(feature_indices)
        n_cols = 5
        n_rows = (num_plots + n_cols - 1) // n_cols  # Ceiling division
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
        axs = axs.flatten()
        for idx, i in enumerate(feature_indices):
            axs[idx].scatter(xvalues, X_plot[:, i], alpha=0.4, s=10, color="gray")
            colors = plt.cm.viridis(np.linspace(0, 1, theta_obs_plot.shape[0]))
            for jj in range(theta_obs_plot.shape[0]):
                axs[idx].scatter(
                    theta_obs_plot[jj, ii],
                    xo_plot[jj, i],
                    color=colors[jj],
                    s=30,
                    marker="*",
                )
            axs[idx].set_xlabel(labels[ii])
            axs[idx].set_ylabel(feature_labels[i])
        # Hide unused subplots
        for idx in range(num_plots, len(axs)):
            axs[idx].set_visible(False)
        plt.tight_layout()
        plt.savefig(join(output_dir, f"{labels[ii]}_features.png"), dpi=300)
        plt.close()


def get_subcortical_regions(atlas):
    """
    Return subcortical regions of the corresponding atlas
    """
    if atlas == 'dk':
        return ['L.CER', 'L.TH', 'L.CA', 'L.PU', 'L.PA', 'L.HI', 'L.AM', 'L.AC',
                'R.TH', 'R.CA', 'R.PU', 'R.PA', 'R.HI', 'R.AM', 'R.AC', 'R.CER',
                'L.SN', 'L.VTA', 'R.SN', 'R.VTA']
    elif atlas == 'schaefer':
        return ['Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate', 'Left-Putamen',
                'Left-Pallidum', 'Left-Hippocampus', 'Left-Amygdala', 'Left-Accumbens-area',
                'Right-Cerebellum-Cortex', 'Right-Thalamus-Proper', 'Right-Caudate',
                'Right-Putamen', 'Right-Pallidum', 'Right-Hippocampus', 'Right-Amygdala',
                'Right-Accumbens-area', 'Left-Nigra', 'Left-VTA', 'Right-Nigra', 'Right-VTA']
    elif atlas == 'aal2':
        return ['Caudate_L', 'Caudate_R', 'Putamen_L', 'Putamen_R', 'Pallidum_L', 'Pallidum_R',
                'Thalamus_L', 'Thalamus_R', 'Amygdala_L', 'Amygdala_R',
                'Cerebelum_Crus1_L', 'Cerebelum_Crus1_R', 'Cerebelum_Crus2_L', 'Cerebelum_Crus2_R',
                'Cerebelum_3_L', 'Cerebelum_3_R', 'Cerebelum_4_5_L', 'Cerebelum_4_5_R',
                'Cerebelum_6_L', 'Cerebelum_6_R', 'Cerebelum_7b_L', 'Cerebelum_7b_R',
                'Cerebelum_8_L', 'Cerebelum_8_R', 'Cerebelum_9_L', 'Cerebelum_9_R',
                'Cerebelum_10_L', 'Cerebelum_10_R',
                'Vermis_1_2', 'Vermis_3', 'Vermis_4_5', 'Vermis_6', 'Vermis_7', 'Vermis_8',
                'Vermis_9', 'Vermis_10', 'Nigra_L', 'VTA_L', 'Nigra_R', 'VTA_R']
    else:
        raise ValueError(f"Atlas {atlas} not recognized")


def expand_jdopa_params(par_dict, num_nodes, num_sim, region_names):
    """
    Expand Jdopa_cortex and Jdopa_subcortex parameters into full Jdopa vector.
    After updating the Jdopa in par_dict, remove extra labels of Jdoba_...
    
    Args:
        par_dict: Dictionary of parameters (may contain Jdopa_cortex and Jdopa_subcortex)
        num_nodes: Total number of nodes
        num_sim: Number of simulations (num_item in batch)
        region_names: region names
    
    Returns:
        Updated par_dict with Jdopa as (num_nodes, num_sim) array
        
    Note:
        The shape must be (num_nodes, num_sim) to match the expected format where:
        - axis 0 = num_nodes (spatial dimension - different brain regions)
        - axis 1 = num_sim (batch dimension - different parameter sets)
        This is required because state variables have shape (num_svar, num_nodes, num_sim)
        and coupling terms have shape (num_nodes, num_sim).
    """
    
    subcortical = get_subcortical_regions("dk")
    # Find indices of subcortical regions in region_names
    subcortical_indices = [i for i, name in enumerate(region_names) if name in subcortical]
    ctx_indices = [i for i in range(len(region_names)) if i not in subcortical_indices]

    if 'Jdopa_cortex' in par_dict and 'Jdopa_subcortex' in par_dict:
        # Create full Jdopa array with shape (num_nodes, num_sim)
        Jdopa = jnp.zeros((num_nodes, num_sim))
        
        # Fill cortical values: broadcast (num_sim,) to (num_ctx_nodes, num_sim)
        Jdopa = Jdopa.at[ctx_indices, :].set(
            par_dict['Jdopa_cortex'].reshape(1, -1)  # Shape: (1, num_sim)
        )
        
        # Fill subcortical values: broadcast (num_sim,) to (num_subcort_nodes, num_sim)
        Jdopa = Jdopa.at[subcortical_indices, :].set(
            par_dict['Jdopa_subcortex'].reshape(1, -1)  # Shape: (1, num_sim)
        )
        
        # Remove the separate parameters and add combined Jdopa
        par_dict = {k: v for k, v in par_dict.items() 
                   if k not in ['Jdopa_cortex', 'Jdopa_subcortex']}
        par_dict['Jdopa'] = Jdopa
    
    return par_dict
