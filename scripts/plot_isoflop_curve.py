#!/usr/bin/env python3
"""
Plot ISOFlop curves as seen in the Chinchilla paper.
This script visualizes the relationship between model size and training tokens
for a fixed compute budget (ISOFlop curve).
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import os
import re
from collections import defaultdict
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar


def parse_log_file(log_path):
    """Parse a log file to extract key metrics."""
    with open(log_path, "r") as f:
        content = f.read()

    # Extract parameters - updated pattern for new log format
    param_match = re.search(r"Model parameters:\s*([\d,]+)", content)
    if not param_match:
        return None
    params = int(param_match.group(1).replace(",", ""))

    # Extract number of iterations (steps) - updated pattern
    iter_match = re.search(
        r"Calculated number of iterations from target FLOPs:\s*([\d,]+)", content
    )
    if not iter_match:
        return None
    iterations = int(iter_match.group(1).replace(",", ""))

    # Extract batch size - updated pattern for "Total batch size"
    batch_match = re.search(r"Total batch size:\s*([\d,]+)", content)
    if not batch_match:
        return None
    batch_size = int(batch_match.group(1).replace(",", ""))

    # Calculate tokens
    tokens = iterations * batch_size

    # Extract final validation BPB - updated pattern
    val_matches = re.findall(r"VALIDATION.*?BPB:\s*([\d.]+)", content)
    final_val_bpb = float(val_matches[-1]) if val_matches else None

    # Extract final step - updated pattern
    step_matches = re.findall(r"VALIDATION.*?Step\s+(\d+)", content)
    final_step = int(step_matches[-1]) if step_matches else iterations

    return {
        "params": params,
        "tokens": tokens,
        "iterations": iterations,
        "batch_size": batch_size,
        "final_val_bpb": final_val_bpb,
        "final_step": final_step,
    }


# Define FLOP budgets to analyze
FLOP_BUDGETS = ["1e18", "2e18", "3e18", "6e18"]  # Add or modify as needed (3e18 and 6e18 still running)

# Auto-discover log files matching the pattern: scaling_laws_N<depth>_F<FLOPBudget>
log_dir = "/mnt/localssd/VibeNanoChat/logs"
log_files = []

print("Discovering log files...")
for budget_str in FLOP_BUDGETS:
    pattern = os.path.join(log_dir, f"scaling_laws_N*_F{budget_str}.log")
    matching_files = sorted(glob(pattern))
    log_files.extend(matching_files)
    print(f"  Found {len(matching_files)} files for budget {budget_str}")

log_files = sorted(log_files)
print(f"\nTotal log files found: {len(log_files)}")

# Group data by budget
data_by_budget = defaultdict(dict)

print("\nParsing log files...")
for log_file in log_files:
    if not os.path.exists(log_file):
        print(f"Warning: {log_file} not found, skipping...")
        continue

    # Extract n_layers and budget from filename: scaling_laws_N<depth>_F<FLOPBudget>
    match = re.search(r"scaling_laws_N(\d+)_F([\de]+)\.log", log_file)
    if not match:
        print(f"Warning: Could not parse filename {log_file}, skipping...")
        continue

    n_layers = int(match.group(1))
    budget_str = match.group(2)
    budget = float(budget_str.replace("e", "e"))

    print(f"  Parsing {log_file} (n={n_layers}, budget={budget:.1e})...")

    metrics = parse_log_file(log_file)
    if metrics:
        model_name = f"n{n_layers}"
        metrics["flops"] = budget
        metrics["flops_per_token"] = budget / metrics["tokens"]
        data_by_budget[budget][model_name] = metrics
        bpb_str = (
            f"{metrics['final_val_bpb']:.4f}" if metrics["final_val_bpb"] else "N/A"
        )
        print(
            f"    ✓ {model_name}: {metrics['params']/1e6:.1f}M params, {metrics['tokens']/1e9:.2f}B tokens, BPB: {bpb_str}"
        )
    else:
        print("    ✗ Failed to parse metrics")

print(f"\nFound data for {len(data_by_budget)} different budgets")

# Consolidate into single data structure
data = {}
for budget, models in data_by_budget.items():
    for model_name, metrics in models.items():
        key = f"{model_name}_b{budget:.0e}"
        data[key] = metrics

# ============================================================================
# PLOT 1: ISOFlop Curves for Multiple Budgets
# ============================================================================

# Create figure with good styling
fig, ax = plt.subplots(figsize=(12, 8))

# Color schemes for different budgets - dynamically generate colors
available_colors = [
    "#FF6B6B",
    "#4ECDC4",
    "#45B7D1",
    "#96CEB4",
    "#FFEAA7",
    "#DFE6E9",
    "#A29BFE",
    "#FD79A8",
]
available_markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]

budget_colors = {}
budget_markers = {}
for i, budget in enumerate(sorted(data_by_budget.keys())):
    budget_colors[budget] = available_colors[i % len(available_colors)]
    budget_markers[budget] = available_markers[i % len(available_markers)]

# Sort budgets for consistent plotting
sorted_budgets = sorted(data_by_budget.keys())

# Find overall parameter range for theoretical curves
all_params = []
for budget in sorted_budgets:
    models = data_by_budget[budget]
    params = np.array([models[k]["params"] for k in models.keys()])
    all_params.extend(params)

param_min = min(all_params) * 0.7
param_max = max(all_params) * 1.3

# Plot for each budget
for budget in sorted_budgets:
    models = data_by_budget[budget]
    if not models:
        continue

    model_names = sorted(models.keys(), key=lambda x: int(x[1:]))
    params = np.array([models[k]["params"] for k in model_names])
    tokens = np.array([models[k]["tokens"] for k in model_names])

    # Convert to millions/billions
    params_M = params / 1e6
    tokens_B = tokens / 1e9

    color = budget_colors.get(budget, "#45B7D1")
    marker = budget_markers.get(budget, "D")

    # Plot the data points
    ax.scatter(
        params_M,
        tokens_B,
        s=200,
        c=color,
        marker=marker,
        alpha=0.8,
        edgecolors="black",
        linewidth=2,
        label=f"Budget = {budget:.1e} FLOPs",
        zorder=3,
    )

    # Annotate each point
    for i, name in enumerate(model_names):
        ax.annotate(
            name.upper(),
            (params_M[i], tokens_B[i]),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=color,
                alpha=0.8,
            ),
        )

    # Plot theoretical ISOFlop curve
    # For a fixed compute budget C: C = 6 * N * D
    N_range = np.linspace(param_min, param_max, 100)
    D_theoretical = budget / (6 * N_range)

    ax.plot(
        N_range / 1e6,
        D_theoretical / 1e9,
        "--",
        linewidth=2,
        alpha=0.6,
        color=color,
        zorder=2,
    )

# Formatting
ax.set_xlabel("Model Parameters (Millions) [log scale]", fontsize=14, fontweight="bold")
ax.set_ylabel("Training Tokens (Billions)", fontsize=14, fontweight="bold")
ax.set_title(
    "ISOFlop Curves: Model Size vs. Training Data\nComparing Different Compute Budgets",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

# Set log scale for x-axis
ax.set_xscale("log")

# Grid
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

# Legend
ax.legend(fontsize=12, loc="upper right", framealpha=0.9)

# Add text box with key information
info_text = f"Budgets: {', '.join([f'{b:.1e}' for b in sorted_budgets])} FLOPs\n"
info_text += f"Models: {len(data)} total configurations\n"
info_text += "Dashed lines: Theoretical ISOFlop curves (C = 6×N×D)"
ax.text(
    0.02,
    0.98,
    info_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
)

# Tight layout
plt.tight_layout()

# Save figure
output_path = "/mnt/localssd/VibeNanoChat/scripts/graphs/isoflop_curve.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\n✅ ISOFlop curve plot saved to: {output_path}")

plt.close()

# ============================================================================
# PLOT 2: Validation BPB vs Model Parameters (with Curve Fitting)
# ============================================================================

# Create second figure
fig, ax = plt.subplots(figsize=(12, 8))

# Store optimal points from curve fitting
fitted_optima = {}

# Plot for each budget
for budget in sorted_budgets:
    models = data_by_budget[budget]
    if not models:
        continue

    # Extract models with validation data
    models_with_val = {
        k: v for k, v in models.items() if v["final_val_bpb"] is not None
    }
    if not models_with_val:
        continue

    model_names = sorted(models_with_val.keys(), key=lambda x: int(x[1:]))
    val_params = np.array([models_with_val[k]["params"] for k in model_names])
    val_bpb = np.array([models_with_val[k]["final_val_bpb"] for k in model_names])

    # Convert to millions
    val_params_M = val_params / 1e6

    color = budget_colors.get(budget, "#45B7D1")
    marker = budget_markers.get(budget, "D")

    # Plot the data points
    ax.scatter(
        val_params_M,
        val_bpb,
        s=250,
        c=color,
        marker=marker,
        alpha=0.8,
        edgecolors="black",
        linewidth=2,
        label=f"Data: Budget = {budget:.1e} FLOPs",
        zorder=3,
    )

    # Annotate each point
    for i, name in enumerate(model_names):
        ax.annotate(
            name.upper(),
            (val_params_M[i], val_bpb[i]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=color,
                alpha=0.9,
            ),
        )

    # Fit a smooth curve to the data (in log space for better fitting)
    # Use log(params) as x-axis for more stable fitting
    if len(val_params_M) >= 4:  # Need at least 4 points for good fit
        log_params = np.log(val_params_M)

        # Use UnivariateSpline with smoothing
        # s parameter controls smoothing (lower = closer fit, higher = smoother)
        spline = UnivariateSpline(log_params, val_bpb, k=3, s=0.0001)

        # Create dense points for smooth curve
        log_params_dense = np.linspace(log_params.min(), log_params.max(), 500)
        params_dense = np.exp(log_params_dense)
        bpb_dense = spline(log_params_dense)

        # Plot the fitted curve
        ax.plot(
            params_dense,
            bpb_dense,
            "-",
            linewidth=2.5,
            alpha=0.7,
            color=color,
            label=f"Fit: Budget = {budget:.1e} FLOPs",
            zorder=2,
        )

        # Find minimum of the fitted curve
        result = minimize_scalar(
            spline, bounds=(log_params.min(), log_params.max()), method="bounded"
        )
        optimal_log_params = result.x
        optimal_params_M = np.exp(optimal_log_params)
        optimal_bpb = spline(optimal_log_params)

        # Calculate optimal tokens from FLOP budget: C = 6 * N * D => D = C / (6 * N)
        optimal_tokens = budget / (6 * optimal_params_M * 1e6)
        optimal_tokens_per_param = optimal_tokens / (optimal_params_M * 1e6)

        # Store optimal point
        fitted_optima[budget] = {
            "params_M": optimal_params_M,
            "bpb": optimal_bpb,
            "tokens": optimal_tokens,
            "tokens_per_param": optimal_tokens_per_param,
        }

        # Mark the optimal point with a star
        ax.scatter(
            [optimal_params_M],
            [optimal_bpb],
            s=600,
            c=color,
            marker="*",
            alpha=1.0,
            edgecolors="black",
            linewidth=3,
            zorder=5,
        )

        # Store optimal info for later positioning on the right side
        fitted_optima[budget]["color"] = color
        fitted_optima[budget]["optimal_params_M"] = optimal_params_M
        fitted_optima[budget]["optimal_bpb"] = optimal_bpb

# Formatting
ax.set_xlabel("Model Parameters (Millions) [log scale]", fontsize=14, fontweight="bold")
ax.set_ylabel("Validation BPB (Bits Per Byte)", fontsize=14, fontweight="bold")
ax.set_title(
    "Scaling Law: Validation BPB vs Model Size (with Curve Fitting)\nComparing Different Compute Budgets",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

# Set log scale for x-axis
ax.set_xscale("log")

# Grid
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

# Legend - positioned outside the plot area at the top right
ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.9, ncol=1)

# Add optimal model annotations on the right side of the plot
# Generate vertical positions dynamically based on the number of budgets
num_budgets = len(sorted_budgets)
vertical_positions = [0.9 - (i * 0.3) for i in range(num_budgets)]
for i, budget in enumerate(sorted_budgets):
    if budget in fitted_optima:
        opt = fitted_optima[budget]
        color = opt["color"]
        optimal_params_M = opt["optimal_params_M"]
        optimal_bpb = opt["optimal_bpb"]
        
        # Create annotation text
        annotation_text = f"Optimal @ {budget:.1e} FLOPs\n"
        annotation_text += f"{opt['params_M']:.1f}M params\n"
        annotation_text += f"{opt['tokens_per_param']:.1f} tok/param\n"
        annotation_text += f"BPB: {opt['bpb']:.4f}"
        
        # Add annotation on the right side (without arrow)
        ax.text(
            1.02,
            vertical_positions[i],
            annotation_text,
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=color,
                edgecolor="black",
                alpha=0.95,
                linewidth=2.5,
            ),
            ha="left",
            va="center",
        )

# Tight layout with extra space for the legend and text box
plt.tight_layout(rect=[0, 0, 0.75, 1])

# Save figure
bpb_output_path = "/mnt/localssd/VibeNanoChat/scripts/graphs/validation_bpb_curve.png"
plt.savefig(bpb_output_path, dpi=300, bbox_inches="tight")
print(f"✅ BPB plot saved to: {bpb_output_path}")

plt.close()

# Print fitted optima to console
print("\n" + "=" * 80)
print("FITTED CURVE OPTIMA")
print("=" * 80)
for budget in sorted_budgets:
    if budget in fitted_optima:
        opt = fitted_optima[budget]
        print(f"\nBudget {budget:.1e} FLOPs:")
        print(f"  Optimal Parameters:  {opt['params_M']:.2f}M")
        print(f"  Tokens/Param Ratio:  {opt['tokens_per_param']:.2f}")
        print(f"  Optimal Tokens:      {opt['tokens']/1e9:.2f}B")
        print(f"  Optimal BPB:         {opt['bpb']:.4f}")
print("=" * 80 + "\n")

# ============================================================================
# PLOT 3: Optimal Model Parameters and Tokens vs FLOPs (Using Fitted Curves)
# ============================================================================

# Collect optimal models for each budget from fitted curves
optimal_data = []
for budget in sorted_budgets:
    if budget in fitted_optima:
        opt = fitted_optima[budget]
        optimal_data.append(
            {
                "budget": budget,
                "params": opt["params_M"] * 1e6,  # Convert back to raw params
                "tokens": opt["tokens"],
                "val_bpb": opt["bpb"],
                "model_name": f"Fitted_{opt['params_M']:.0f}M",
                "tokens_per_param": opt["tokens_per_param"],
            }
        )

if optimal_data:
    # Sort by budget
    optimal_data.sort(key=lambda x: x["budget"])

    budgets = np.array([d["budget"] for d in optimal_data])
    optimal_params = np.array([d["params"] for d in optimal_data])
    optimal_tokens = np.array([d["tokens"] for d in optimal_data])
    model_names = [d["model_name"] for d in optimal_data]

    # Convert to millions/billions
    optimal_params_M = optimal_params / 1e6
    optimal_tokens_B = optimal_tokens / 1e9

    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # ========== LEFT PLOT: Optimal Parameters vs FLOPs ==========

    # Plot the data points
    ax1.scatter(
        budgets,
        optimal_params_M,
        s=300,
        c="#FF6B6B",
        marker="D",
        alpha=0.8,
        edgecolors="black",
        linewidth=2.5,
        label="Optimal Model",
        zorder=3,
    )

    # Fit a power law curve if there are multiple budgets: N ∝ C^a
    if len(optimal_data) > 1:
        # Fit power law in log space: log(N) = a*log(C) + b
        log_budgets = np.log(budgets)
        log_params = np.log(optimal_params_M)
        
        # Linear fit in log space
        coeffs = np.polyfit(log_budgets, log_params, 1)
        a_params = coeffs[0]  # exponent
        b_params = coeffs[1]  # intercept
        
        # Generate fitted curve
        budgets_dense = np.logspace(np.log10(budgets.min()), np.log10(budgets.max()), 200)
        params_fit = np.exp(b_params) * (budgets_dense ** a_params)
        
        # Plot fitted curve
        ax1.plot(
            budgets_dense,
            params_fit,
            "-",
            linewidth=3,
            alpha=0.6,
            color="#FF6B6B",
            label=f"Power Law Fit: N ∝ C^{a_params:.3f}",
            zorder=2,
        )

    # Annotate each point with params and tokens/param ratio
    for i, opt in enumerate(optimal_data):
        label_text = f"{optimal_params_M[i]:.1f}M params\n{opt['tokens_per_param']:.1f} tok/param\nBPB: {opt['val_bpb']:.4f}"
        ax1.annotate(
            label_text,
            (budgets[i], optimal_params_M[i]),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="#FF6B6B",
                alpha=0.9,
                linewidth=2,
            ),
            ha="left",
        )

    # Formatting
    ax1.set_xlabel("Compute Budget (FLOPs) [log scale]", fontsize=14, fontweight="bold")
    ax1.set_ylabel(
        "Optimal Model Parameters (Millions) [log scale]", fontsize=14, fontweight="bold"
    )
    ax1.set_title(
        "Compute-Optimal Model Size\nOptimal Parameters vs Compute Budget",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )

    # Use log scale for both axes to show power law as straight line
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    # Grid
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Legend
    ax1.legend(fontsize=11, loc="upper left", framealpha=0.9)

    # Add info text
    info_text = "Fitted Optimal Models:\n"
    info_text += "Optimal points found by\n"
    info_text += "fitting smooth curves to\n"
    info_text += "validation BPB data\n\n"
    info_text += f"Total Budgets: {len(optimal_data)}\n"
    info_text += (
        f"Param Range:\n{optimal_params_M.min():.1f}M - {optimal_params_M.max():.1f}M"
    )

    ax1.text(
        0.98,
        0.02,
        info_text.strip(),
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
    )

    # ========== RIGHT PLOT: Optimal Tokens vs FLOPs ==========

    # Plot the data points
    ax2.scatter(
        budgets,
        optimal_tokens_B,
        s=300,
        c="#4ECDC4",
        marker="D",
        alpha=0.8,
        edgecolors="black",
        linewidth=2.5,
        label="Optimal Training Tokens",
        zorder=3,
    )

    # Fit a power law curve if there are multiple budgets: D ∝ C^b
    if len(optimal_data) > 1:
        # Fit power law in log space: log(D) = b*log(C) + c
        log_budgets = np.log(budgets)
        log_tokens = np.log(optimal_tokens_B)
        
        # Linear fit in log space
        coeffs_tokens = np.polyfit(log_budgets, log_tokens, 1)
        a_tokens = coeffs_tokens[0]  # exponent
        b_tokens = coeffs_tokens[1]  # intercept
        
        # Generate fitted curve
        budgets_dense = np.logspace(np.log10(budgets.min()), np.log10(budgets.max()), 200)
        tokens_fit = np.exp(b_tokens) * (budgets_dense ** a_tokens)
        
        # Plot fitted curve
        ax2.plot(
            budgets_dense,
            tokens_fit,
            "-",
            linewidth=3,
            alpha=0.6,
            color="#4ECDC4",
            label=f"Power Law Fit: D ∝ C^{a_tokens:.3f}",
            zorder=2,
        )

    # Annotate each point with tokens and tokens/param ratio
    for i, opt in enumerate(optimal_data):
        label_text = f"{optimal_tokens_B[i]:.2f}B tokens\n{opt['tokens_per_param']:.1f} tok/param\nBPB: {opt['val_bpb']:.4f}"
        ax2.annotate(
            label_text,
            (budgets[i], optimal_tokens_B[i]),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="#4ECDC4",
                alpha=0.9,
                linewidth=2,
            ),
            ha="left",
        )

    # Formatting
    ax2.set_xlabel("Compute Budget (FLOPs) [log scale]", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Optimal Training Tokens (Billions) [log scale]", fontsize=14, fontweight="bold")
    ax2.set_title(
        "Compute-Optimal Training Data\nOptimal Tokens vs Compute Budget",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )

    # Use log scale for both axes to show power law as straight line
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    # Grid
    ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Legend
    ax2.legend(fontsize=11, loc="upper right", framealpha=0.9)

    # Add info text
    info_text2 = "Fitted Optimal Ratios:\n"
    for opt in optimal_data:
        info_text2 += (
            f"  {opt['budget']:.1e}: {opt['tokens_per_param']:.1f} tok/param\n"
        )
    info_text2 += "\nChinchilla suggests:\n"
    info_text2 += "  20 tok/param\n"
    info_text2 += "\nOur optima are much lower!"

    ax2.text(
        0.98,
        0.02,
        info_text2.strip(),
        transform=ax2.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
    )

    # Tight layout
    plt.tight_layout()

    # Save figure
    optimal_output_path = (
        "/mnt/localssd/VibeNanoChat/scripts/graphs/optimal_model_vs_flops.png"
    )
    plt.savefig(optimal_output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Optimal model vs FLOPs plot saved to: {optimal_output_path}")
else:
    print("⚠️  No optimal model data available to plot (no models with validation BPB)")

plt.close()

# ============================================================================
# Print Summary Statistics
# ============================================================================
print("\n" + "=" * 80)
print("ISOFLOP CURVE ANALYSIS - SUMMARY")
print("=" * 80)

for budget in sorted_budgets:
    print(f"\n{'='*80}")
    print(f"BUDGET: {budget:.1e} FLOPs")
    print(f"{'='*80}")

    models = data_by_budget[budget]
    model_names = sorted(models.keys(), key=lambda x: int(x[1:]))

    for name in model_names:
        d = models[name]
        ratio = d["tokens"] / d["params"]
        print(f"\n  {name.upper()}:")
        print(f"    Parameters:      {d['params']/1e6:.2f}M")
        print(f"    Training Tokens: {d['tokens']/1e9:.2f}B")
        print(f"    Total FLOPs:     {d['flops']:.2e}")
        print(f"    Tokens/Param:    {ratio:.2f}")
        if d["final_val_bpb"] is not None:
            print(f"    Final Val BPB:   {d['final_val_bpb']:.4f}")
        else:
            print("    Final Val BPB:   N/A (still running)")

    # Find best model in this budget
    models_with_val = {
        k: v for k, v in models.items() if v["final_val_bpb"] is not None
    }
    if models_with_val:
        best_model = min(models_with_val.items(), key=lambda x: x[1]["final_val_bpb"])
        print(
            f"\n  ⭐ BEST MODEL: {best_model[0].upper()} with BPB = {best_model[1]['final_val_bpb']:.4f}"
        )

print("\n" + "=" * 80)

# Calculate optimal point (from Chinchilla scaling laws)
print("\nCHINCHILLA SCALING INSIGHT:")
print("=" * 80)
print("For optimal performance at fixed compute, Chinchilla suggests:")
print("tokens ≈ 20 × parameters")
print("\nYour models show:")

for budget in sorted_budgets:
    print(f"\n  Budget {budget:.1e}:")
    models = data_by_budget[budget]
    model_names = sorted(models.keys(), key=lambda x: int(x[1:]))
    for name in model_names:
        d = models[name]
        ratio = d["tokens"] / d["params"]
        print(f"    {name.upper()}: {ratio:.2f} tokens/param")

print("\n" + "=" * 80)
