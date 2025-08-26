# plotting.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import awkward as ak
from pathlib import Path

def load_and_preprocess_data(tree_orig, tree_comp, variables, branch_prefix):
    """Load and preprocess data from ROOT trees."""
    original_data = {}
    compressed_data = {}
    abs_diffs = {}
    signed_diffs = {}
    
    for var, label in variables.items():
        orig_array = ak.flatten(tree_orig[f"{branch_prefix}.{var}"].array()).to_numpy()
        comp_array = ak.flatten(tree_comp[f"{branch_prefix}.{var}"].array()).to_numpy()

        mask = np.isfinite(orig_array) & np.isfinite(comp_array) & (np.abs(orig_array) > 1e-13)
        orig = np.abs(orig_array[mask])
        comp = np.abs(comp_array[mask])
        
        original_data[var] = orig
        compressed_data[var] = comp
        abs_diffs[var] = np.abs(orig - comp)
        signed_diffs[var] = comp - orig
        
    return original_data, compressed_data, abs_diffs, signed_diffs

def plot_histogram(data, varnames=(), log=True, n_bins=100, save_path=None):
    """
    Plot histogram(s) of data with optional log scaling.
    
    Args:
        data: Single array or tuple of arrays to plot
        varnames: Tuple of variable names corresponding to data arrays
        log: Whether to use logarithmic bins and scaling
    """
    # Convert single array to tuple for unified processing
    if not isinstance(data, tuple):
        data = (data,)
        varnames = (varnames,) if varnames else ('',)

    # Validate input lengths
    if varnames and len(data) != len(varnames):
        raise ValueError("Length of data tuple must match length of varnames")

    # Get positive values only (for log scale)
    data_pos = [d[d > 0] for d in data]

    # Calculate global min/max for consistent bins
    if log:
        global_min = min(np.min(d) for d in data_pos if len(d) > 0)
        global_max = max(np.max(d) for d in data_pos if len(d) > 0)
        bins = np.logspace(np.log10(global_min), np.log10(global_max), n_bins)
    else:
        bins = n_bins

    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Plot each dataset
    colors = plt.cm.tab10.colors  # Use colormap for consistent colors
    for i, (d, d_pos) in enumerate(zip(data, data_pos)):
        label = varnames[i] if varnames else f'Dataset {i+1}'
        plt.hist(d_pos if log else d, 
                bins=bins,
                alpha=0.6,
                label=label,
                histtype='stepfilled',
                color=colors[i % len(colors)])

    # Formatting
    if log:
        plt.xscale('log')
        plt.xlabel('Value [log scale]')
    else:
        plt.xlabel('Value')
        
    plt.ylabel('Counts')
    title = 'Distribution Comparison' + (' (log-scaled)' if log else '')
    plt.title(title)
    
    # if len(data) > 1:  # Only show legend for multiple datasets
    if varnames:
        plt.legend()
        
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(Path(save_path) / "histogram_output.png")
        print(f"Histogram saved to {Path(save_path) / 'histogram_output.png'}")
        plt.close()
    else:
        plt.show()

def plot_absolute_differences(abs_diffs, variables, colors, save_path=None):
    """Plot absolute differences with peak detection."""
    plt.figure(figsize=(10, 6))
    log_peak_positions = {}
    
    for var, label in variables.items():
        diff = abs_diffs[var]
        diff = diff[np.isfinite(diff) & (diff > 0)]
        
        if len(diff) == 0:
            continue

        bins = np.logspace(np.log10(np.max(np.min(diff), 1e-13)), 
                          np.log10(np.max(diff)), 100)
        hist_vals, bin_edges = np.histogram(diff, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        smoothed = gaussian_filter1d(hist_vals, sigma=1)
        
        peaks, _ = find_peaks(smoothed, prominence=0.01*np.max(smoothed), distance=3)
        log_peak_positions[label] = np.log10(bin_centers[peaks])
        
        plt.plot(bin_centers, smoothed, color=colors[var], 
                label=f"|Δ {label}| (smoothed)", alpha=0.7)
        plt.scatter(bin_centers[peaks], smoothed[peaks], 
                   color=colors[var], edgecolor='black', zorder=5)

    plt.xscale("log")
    plt.xlabel("Absolute Difference (log scale)")
    plt.ylabel("Number of electrons (smoothed)")
    plt.title("Absolute Differences with Peak Detection")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(Path(save_path) / "absolute_differences.png")
        print(f"Absolute differences plot saved to {Path(save_path) / 'absolute_differences.png'}")
        plt.close()
    else:
        plt.show()
    return log_peak_positions

def analyze_peak_spacing(log_peaks_dict):
    """Analyze spacing between peaks in log scale."""
    print("\n🔍 Log-Scale Peak Spacing Analysis:")
    all_diffs = []
    equidistant_vars = []

    for var, log_peaks in log_peaks_dict.items():
        if len(log_peaks) < 2:
            print(f"  • {var}: Less than 2 peaks found.")
            continue

        diffs = np.diff(log_peaks)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        all_diffs.extend(diffs)

        is_equidistant = std_diff < 0.05 * mean_diff
        status = "✅ Approx. equidistant" if is_equidistant else "❌ Not equidistant"
        if is_equidistant:
            equidistant_vars.append(var)

        print(f"  • {var}: mean Δlog10 = {mean_diff:.3f}, std = {std_diff:.3f} → {status}")

    if all_diffs:
        global_mean = np.mean(all_diffs)
        global_std = np.std(all_diffs)
        print(f"\n📊 Global mean Δlog10 spacing = {global_mean:.3f}, std = {global_std:.3f}")
    else:
        print("\n⚠️ Not enough peak data to compute global spacing.")
        
    return global_mean, global_std

def plot_signed_differences_hist(signed_diffs, variables, colors, thr=1e-5, save_path=None):
    """Plot signed differences using symlog scale."""
    all_signed = np.concatenate([signed_diffs[var] for var in variables])
    all_signed = all_signed[np.isfinite(all_signed)]
    max_abs = max(thr, np.max(np.abs(all_signed)))
    
    bins_pos = np.logspace(np.log10(thr), np.log10(max_abs), 50)
    common_bins = np.concatenate((-bins_pos[::-1], [0], bins_pos))
    
    plt.figure(figsize=(10, 6))
    for var, label in variables.items():
        diff = signed_diffs[var]
        plt.hist(diff, bins=common_bins, histtype='stepfilled', alpha=0.4,
                color=colors[var], label=f"Δ {label}")

    plt.xscale('symlog', linthresh=thr)
    plt.xlabel("Signed Difference (compressed − original)")
    plt.ylabel("Number of electrons")
    plt.title("Signed Differences (Common Bins)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(Path(save_path) / "signed_differences_hist.png")
        print(f"Signed differences histogram saved to {Path(save_path) / 'signed_differences_hist.png'}")
        plt.close()
    else:
        plt.show()

def plot_signed_vs_original(original_data, compressed_data, variables, colors, save_path=None):
    """Scatter plot of signed differences vs original values."""
    plt.figure(figsize=(10, 6))
    for var, label in variables.items():
        orig = original_data[var]
        diff = compressed_data[var] - orig
        plt.scatter(orig, diff, s=1, alpha=0.3, label=f"Δ {label}", color=colors[var])
    
    plt.xscale('log')
    plt.xlabel("Original Value")
    plt.ylabel("Signed Difference (Compressed - Original)")
    plt.title("Signed Difference vs. Original Value")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(Path(save_path) / "signed_vs_original.png")
        print(f"Signed vs original plot saved to {Path(save_path) / 'signed_vs_original.png'}")
        plt.close()
    else:
        plt.show()

def plot_large_differences_hist(signed_diffs, variables, threshold=1e-2, xlim=(-100, 100), save_path=None):
    """Histogram of large signed differences."""
    diff_all = np.concatenate([signed_diffs[v] for v in variables.keys()])
    large_diffs = diff_all[np.abs(diff_all) > threshold]
    
    plt.figure(figsize=(10, 6))
    plt.hist(large_diffs, bins=100)
    plt.title(f"Large Signed Differences (|Δ| > {threshold})")
    plt.xlabel("Δ (Compressed - Original)")
    plt.ylabel("Frequency")
    plt.xlim(xlim)
    plt.grid(True)
    if save_path is not None:
        plt.savefig(Path(save_path) / "large_differences_hist.png")
        print(f"Large differences histogram saved to {Path(save_path) / 'large_differences_hist.png'}")
        plt.close()
    else:
        plt.show()

def plot_log_residual_contour(x_true, x_recon, gmm=None, varname="pt", m=10, offset=0.3, save_path=None):
    # Calculate residuals and transform to log scale
    residual = x_true - x_recon
    log_x_true = np.log10((x_recon) + 1e-12)
    log_residual = np.sign(residual) * np.log10(np.abs(residual) + 1e-12)

    # Filter finite entries
    mask = np.isfinite(log_x_true) & np.isfinite(log_residual)
    log_x_true = log_x_true[mask]
    log_residual = log_residual[mask]
    
    # Exit early if array is empty
    if len(log_x_true) == 0 or len(log_residual) == 0:
        print(f"[WARNING] No finite values to plot for {varname}")
        return
    
    # Create grid for contours
    x_min, x_max = np.percentile(log_x_true, [1, 99.9])
    y_min, y_max = np.percentile(log_residual, [1, 99.9])
    
    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(
        log_x_true, log_residual, 
        bins=100, 
        range=[[x_min, x_max], [y_min, y_max]],
        density=True
    )
    
    # Smooth the histogram
    hist_smooth = gaussian_filter1d(hist, sigma=1)
    
    # Create meshgrid for plotting
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centers, y_centers)
    
    # Plot
    plt.figure(figsize=(12, 8))

    # Plot theoretical bounds
    C = -m * np.log10(2) + offset
    delta_x = np.log10(2)
    x_vals = np.linspace(x_min, x_max, 1000)
    x_step = np.floor(x_vals / delta_x) * delta_x
    y_upper = x_vals + C
    y_lower = -x_vals - C
    plt.plot(x_step, y_upper, 'r--', linewidth=2, label=f'$y = \log_{{10}}|x| - {m}\log_{{10}}2$')
    plt.plot(x_step, y_lower, 'b--', linewidth=2, label=f'$y = -\log_{{10}}|x| + {m}\log_{{10}}2$')
    
    # Contour plot of actual data
    levels = np.linspace(0, hist_smooth.max(), 50)
    cs = plt.contourf(xx, yy, hist_smooth.T, levels=levels, cmap='viridis', alpha=0.7)
    plt.colorbar(cs, label='Density')
    
    # Plot GMM components if provided
    if gmm is not None:
        # Create evaluation grid
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        scores = gmm.score_samples(grid)
        scores = np.exp(scores).reshape(xx.shape)
        
        # Plot GMM contours
        gmm_levels = np.linspace(0, scores.max(), 10)
        plt.contour(xx, yy, scores, levels=gmm_levels, colors='red', linewidths=1, alpha=0.7)
        
        # Plot component means
        plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], 
                   c='red', s=100, marker='x', label='GMM Means')
        plt.scatter(gmm.means_[:, 0], -gmm.means_[:, 1], 
                   c='blue', s=100, marker='o', label='Mirrored Means')
    
    # Formatting
    plt.title(f'Log-Scale Residual vs True Value: {varname}')
    plt.xlabel('log10(|True Value|)')
    plt.ylabel('Signed log10(|Residual|)')
    plt.grid(True, 'both', alpha=0.3)
    plt.legend()
    
    # Add marginal distributions
    ax = plt.gca()
    ax_top = ax.inset_axes([0, 1.05, 1, 0.2], sharex=ax)
    ax_right = ax.inset_axes([1.05, 0, 0.2, 1], sharey=ax)
    
    # Top marginal (log|x_true| distribution)
    ax_top.hist(log_x_true, bins=x_centers, density=True, color='gray', alpha=0.7)
    ax_top.set_yticks([])
    
    # Right marginal (log|residual| distribution)
    ax_right.hist(log_residual, bins=y_centers, density=True, 
                 orientation='horizontal', color='gray', alpha=0.7)
    ax_right.set_xticks([])
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        print(f"Log residual contour plot saved to {save_path}")
        plt.close()
    else:
        plt.show()