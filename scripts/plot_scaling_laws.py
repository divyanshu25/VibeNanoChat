#!/usr/bin/env python3
"""
Script to visualize scaling laws from training logs.
Extracts validation BPB at each step and plots against total training FLOPs.
Fits a power law scaling curve: L(C) = a*C^b + c
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from typing import List, Tuple, Dict
import argparse


def parse_log_file(log_path: Path) -> Dict:
    """
    Parse a single log file to extract training metadata and validation results.
    
    Returns:
        Dict with keys: depth, flops_per_token, batch_size, steps, val_bpb
    """
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract depth from filename (e.g., scaling_laws_N8_R10.log -> 8)
    depth_match = re.search(r'N(\d+)_R\d+', log_path.name)
    depth = int(depth_match.group(1)) if depth_match else None
    
    # Extract FLOPs per token
    flops_match = re.search(r'💫 FLOPs per token:\s+([\d.e+]+)', content)
    flops_per_token = float(flops_match.group(1)) if flops_match else None
    
    # Extract batch size
    batch_match = re.search(r'🎯 Total batch size:\s+([\d,]+)\s+tokens/step', content)
    if batch_match:
        batch_size = int(batch_match.group(1).replace(',', ''))
    else:
        batch_size = None
    
    # Extract validation results (Step, BPB)
    validation_pattern = r'📊 VALIDATION \| Step\s+(\d+)\s+\|.*?\| BPB:\s+([\d.]+)'
    validation_matches = re.findall(validation_pattern, content)
    
    steps = []
    val_bpb = []
    for step, bpb in validation_matches:
        steps.append(int(step))
        val_bpb.append(float(bpb))
    
    return {
        'depth': depth,
        'flops_per_token': flops_per_token,
        'batch_size': batch_size,
        'steps': np.array(steps),
        'val_bpb': np.array(val_bpb),
    }


def calculate_training_flops(steps: np.ndarray, batch_size: int, flops_per_token: float) -> np.ndarray:
    """
    Calculate cumulative training FLOPs at each step.
    
    FLOPs = step * batch_size * flops_per_token
    Note: Adds 1 to step to avoid FLOPs=0 at step 0
    """
    # Add small offset to avoid zero FLOPs
    return (steps + 1) * batch_size * flops_per_token


def power_law(C, a, b, c):
    """
    Power law scaling function: L(C) = a*C^b + c
    
    Args:
        C: Training FLOPs (compute)
        a: Scale parameter
        b: Power law exponent (typically negative, e.g., -0.05)
        c: Irreducible loss offset
    """
    return a * np.power(C, b) + c


def fit_scaling_law(flops: np.ndarray, losses: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Fit a power law to the scaling data.
    
    Returns:
        (fitted_params, fit_info)
    """
    # Filter out any zero or negative FLOPs
    valid_mask = flops > 0
    flops = flops[valid_mask]
    losses = losses[valid_mask]
    
    if len(flops) < 3:
        print("Warning: Not enough valid data points for fitting")
        return None, None
    
    # Better initial guess for parameters based on data range
    flops_range = flops.max() / flops.min()
    loss_range = losses.max() - losses.min()
    
    # a: scale parameter (typical: loss_range * flops_min^(-b))
    # b: exponent (typical: -0.05 to -0.15 for scaling laws)
    # c: asymptotic loss (approximate as min loss)
    a_init = loss_range * np.power(flops.min(), 0.05)
    b_init = -0.05
    c_init = losses.min() * 0.9  # Slightly below min
    
    p0 = [a_init, b_init, c_init]
    
    # Set reasonable bounds
    bounds = (
        [0, -1.0, 0],  # Lower bounds: a>0, b>-1, c>0
        [np.inf, 0, losses.max()]  # Upper bounds: a<inf, b<0, c<max_loss
    )
    
    try:
        # Fit the curve
        popt, pcov = curve_fit(
            power_law, flops, losses, 
            p0=p0, 
            bounds=bounds,
            maxfev=50000,
            method='trf'
        )
        
        # Calculate R-squared
        residuals = losses - power_law(flops, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((losses - np.mean(losses))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        fit_info = {
            'params': popt,
            'covariance': pcov,
            'r_squared': r_squared,
        }
        
        return popt, fit_info
    except Exception as e:
        print(f"Warning: Curve fitting failed: {e}")
        return None, None


def plot_scaling_laws(data_list: List[Dict], save_path: str = None):
    """
    Create visualization of scaling laws with fitted curve.
    Plots only the final validation BPB for each model.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Collect only final data points (last step for each model)
    all_flops = []
    all_bpb = []
    depths = []
    colors = plt.cm.viridis(np.linspace(0, 1, len(data_list)))
    
    # Plot individual runs - only final point
    for i, data in enumerate(data_list):
        depth = data['depth']
        # Calculate FLOPs for the last step
        final_step = data['steps'][-1]
        final_flops = calculate_training_flops(
            np.array([final_step]), 
            data['batch_size'], 
            data['flops_per_token']
        )[0]
        final_bpb = data['val_bpb'][-1]
        
        all_flops.append(final_flops)
        all_bpb.append(final_bpb)
        depths.append(depth)
        
        # Plot 1: Log-log scale
        ax1.plot(final_flops, final_bpb, 'o', label=f'Depth {depth}', 
                color=colors[i], markersize=12, alpha=0.8)
        
        # Plot 2: Linear scale  
        ax2.plot(final_flops, final_bpb, 'o', label=f'Depth {depth}', 
                color=colors[i], markersize=12, alpha=0.8)
    
    # Convert to arrays for fitting
    all_flops = np.array(all_flops)
    all_bpb = np.array(all_bpb)
    
    # Sort by FLOPs for better visualization
    sort_idx = np.argsort(all_flops)
    all_flops = all_flops[sort_idx]
    all_bpb = all_bpb[sort_idx]
    
    # Fit scaling law
    popt, fit_info = fit_scaling_law(all_flops, all_bpb)
    
    if popt is not None:
        # Generate smooth curve for plotting
        flops_smooth = np.logspace(np.log10(all_flops.min()), 
                                   np.log10(all_flops.max()), 200)
        bpb_fitted = power_law(flops_smooth, *popt)
        
        # Plot fitted curve
        ax1.plot(flops_smooth, bpb_fitted, 'r--', linewidth=2.5, 
                label=f'Fit: L(C) = {popt[0]:.2e}·C^{popt[1]:.4f} + {popt[2]:.4f}\n$R^2$ = {fit_info["r_squared"]:.4f}',
                alpha=0.9)
        ax2.plot(flops_smooth, bpb_fitted, 'r--', linewidth=2.5,
                label=f'Fit: L(C) = {popt[0]:.2e}·C^{popt[1]:.4f} + {popt[2]:.4f}\n$R^2$ = {fit_info["r_squared"]:.4f}',
                alpha=0.9)
        
        # Print fit parameters
        print("\n" + "="*70)
        print("SCALING LAW FIT RESULTS")
        print("="*70)
        print(f"Power law form: L(C) = a·C^b + c")
        print(f"  a = {popt[0]:.6e}")
        print(f"  b = {popt[1]:.6f}")
        print(f"  c = {popt[2]:.6f}")
        print(f"  R² = {fit_info['r_squared']:.6f}")
        print()
        print(f"Inverse form: C ∝ (L - c)^(1/b)")
        print(f"  Exponent (1/b) = {1/popt[1]:.6f}")
        print(f"  This means: C (FLOPs) ∝ (val_bpb - {popt[2]:.4f})^{1/popt[1]:.4f}")
        print()
        print(f"To achieve a target BPB L*, you need approximately:")
        print(f"  C ≈ {popt[0]:.3e} × (L* - {popt[2]:.4f})^{1/popt[1]:.4f} FLOPs")
        print("="*70 + "\n")
    
    # Configure plot 1 (log-log)
    ax1.set_xlabel('Total Training FLOPs (C)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation BPB (bits per byte)', fontsize=12, fontweight='bold')
    ax1.set_title('Scaling Laws: Validation BPB vs Training FLOPs\n(Log-Log Scale)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=9, loc='best')
    
    # Configure plot 2 (linear)
    ax2.set_xlabel('Total Training FLOPs (C)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation BPB (bits per byte)', fontsize=12, fontweight='bold')
    ax2.set_title('Scaling Laws: Validation BPB vs Training FLOPs\n(Linear Scale)', 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')
    
    plt.tight_layout()
    
    # Save figure
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot saved to: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize scaling laws from training logs'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='/mnt/localssd/VibeNanoChat/logs',
        help='Directory containing log files'
    )
    parser.add_argument(
        '--depths',
        type=int,
        nargs='+',
        default=[8, 10, 12, 14, 16, 18],
        help='List of depths to include (e.g., 8 10 12 14 16 18)'
    )
    parser.add_argument(
        '--ratio',
        type=int,
        default=10,
        help='Data:param ratio (R value in log filename)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='scripts/graphs/scaling_laws_plot.png',
        help='Output filename for the plot (relative to project root)'
    )
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    
    print("\n" + "="*70)
    print("SCALING LAWS VISUALIZATION")
    print("="*70)
    print(f"Log directory: {log_dir}")
    print(f"Depths: {args.depths}")
    print(f"Data:param ratio: {args.ratio}")
    print("="*70 + "\n")
    
    # Parse all log files
    data_list = []
    for depth in args.depths:
        log_file = log_dir / f"scaling_laws_N{depth}_R{args.ratio}.log"
        
        if not log_file.exists():
            print(f"⚠️  Warning: Log file not found: {log_file}")
            continue
        
        print(f"📊 Parsing: {log_file.name}")
        data = parse_log_file(log_file)
        
        if data['steps'].size > 0:
            data_list.append(data)
            print(f"   ✓ Depth {data['depth']}: {len(data['steps'])} validation points")
            print(f"   ✓ FLOPs/token: {data['flops_per_token']:.3e}")
            print(f"   ✓ Batch size: {data['batch_size']:,}")
            print(f"   ✓ Final val BPB: {data['val_bpb'][-1]:.4f}")
        else:
            print(f"   ✗ No validation data found")
        print()
    
    if not data_list:
        print("❌ No valid data found to plot!")
        return
    
    # Create visualization
    print("📈 Creating visualization...")
    output_path = Path(args.output)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_scaling_laws(data_list, save_path=str(output_path))
    
    print(f"\n✅ Done! Plot saved to: {output_path.absolute()}")


if __name__ == '__main__':
    main()
