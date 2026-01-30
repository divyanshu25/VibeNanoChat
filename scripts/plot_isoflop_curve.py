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
FLOP_BUDGETS = ["1e18"]  # Add or modify as needed (3e18 and 6e18 still running)

# Auto-discover log files matching the pattern: scaling_laws_N<depth>_F<FLOPBudget>
log_dir = "/mnt/localssd/NanoGPT/logs"
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
ax.set_xlabel("Model Parameters (Millions)", fontsize=14, fontweight="bold")
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
output_path = "/mnt/localssd/NanoGPT/scripts/graphs/isoflop_curve.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\n✅ ISOFlop curve plot saved to: {output_path}")

plt.close()

# ============================================================================
# PLOT 2: Validation BPB vs Model Parameters (Multiple Budgets)
# ============================================================================

# Create second figure
fig, ax = plt.subplots(figsize=(12, 8))

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
        label=f"Budget = {budget:.1e} FLOPs",
        zorder=3,
    )

    # Connect points with a line
    ax.plot(
        val_params_M, val_bpb, "--", linewidth=1.5, alpha=0.5, color=color, zorder=2
    )

    # Annotate each point
    for i, name in enumerate(model_names):
        ax.annotate(
            name.upper(),
            (val_params_M[i], val_bpb[i]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=color,
                alpha=0.9,
            ),
        )

# Formatting
ax.set_xlabel("Model Parameters (Millions)", fontsize=14, fontweight="bold")
ax.set_ylabel("Validation BPB (Bits Per Byte)", fontsize=14, fontweight="bold")
ax.set_title(
    "Scaling Law: Validation BPB vs Model Size\nComparing Different Compute Budgets",
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

# Find best models for each budget
info_text = "Best Models per Budget:\n"
for budget in sorted_budgets:
    models = data_by_budget[budget]
    models_with_val = {
        k: v for k, v in models.items() if v["final_val_bpb"] is not None
    }
    if models_with_val:
        best_model = min(models_with_val.items(), key=lambda x: x[1]["final_val_bpb"])
        info_text += f"  {budget:.1e}: {best_model[0].upper()} ({best_model[1]['final_val_bpb']:.4f} BPB)\n"

ax.text(
    0.02,
    0.98,
    info_text.strip(),
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.6),
)

# Tight layout
plt.tight_layout()

# Save figure
bpb_output_path = "/mnt/localssd/NanoGPT/scripts/graphs/validation_bpb_curve.png"
plt.savefig(bpb_output_path, dpi=300, bbox_inches="tight")
print(f"✅ BPB plot saved to: {bpb_output_path}")

plt.close()

# ============================================================================
# PLOT 3: Optimal Model Parameters and Tokens vs FLOPs
# ============================================================================

# Collect optimal models for each budget
optimal_data = []
for budget in sorted_budgets:
    models = data_by_budget[budget]
    models_with_val = {
        k: v for k, v in models.items() if v["final_val_bpb"] is not None
    }
    if models_with_val:
        # Find the model with lowest validation BPB for this budget
        best_model_name, best_model_data = min(
            models_with_val.items(), key=lambda x: x[1]["final_val_bpb"]
        )
        optimal_data.append(
            {
                "budget": budget,
                "params": best_model_data["params"],
                "tokens": best_model_data["tokens"],
                "val_bpb": best_model_data["final_val_bpb"],
                "model_name": best_model_name,
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

    # Connect points with a line if there are multiple budgets
    if len(optimal_data) > 1:
        ax1.plot(
            budgets,
            optimal_params_M,
            "-",
            linewidth=2.5,
            alpha=0.7,
            color="#FF6B6B",
            zorder=2,
        )

    # Annotate each point with model name and BPB
    for i, opt in enumerate(optimal_data):
        label_text = f"{opt['model_name'].upper()}\nBPB: {opt['val_bpb']:.4f}"
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
    ax1.set_xlabel("Compute Budget (FLOPs)", fontsize=14, fontweight="bold")
    ax1.set_ylabel(
        "Optimal Model Parameters (Millions)", fontsize=14, fontweight="bold"
    )
    ax1.set_title(
        "Compute-Optimal Model Size\nOptimal Parameters vs Compute Budget",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )

    # Use log scale for x-axis to better show different budget scales
    ax1.set_xscale("log")

    # Grid
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Legend
    ax1.legend(fontsize=11, loc="upper left", framealpha=0.9)

    # Add info text
    info_text = "Optimal Model Definition:\n"
    info_text += "For each compute budget,\n"
    info_text += "the model with the lowest\n"
    info_text += "validation BPB\n\n"
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

    # Connect points with a line if there are multiple budgets
    if len(optimal_data) > 1:
        ax2.plot(
            budgets,
            optimal_tokens_B,
            "-",
            linewidth=2.5,
            alpha=0.7,
            color="#4ECDC4",
            zorder=2,
        )

    # Annotate each point with model name and tokens
    for i, opt in enumerate(optimal_data):
        label_text = f"{opt['model_name'].upper()}\n{optimal_tokens_B[i]:.2f}B tokens"
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
    ax2.set_xlabel("Compute Budget (FLOPs)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Optimal Training Tokens (Billions)", fontsize=14, fontweight="bold")
    ax2.set_title(
        "Compute-Optimal Training Data\nOptimal Tokens vs Compute Budget",
        fontsize=15,
        fontweight="bold",
        pad=15,
    )

    # Use log scale for x-axis to better show different budget scales
    ax2.set_xscale("log")

    # Grid
    ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Legend
    ax2.legend(fontsize=11, loc="upper right", framealpha=0.9)

    # Add info text
    info_text2 = "Chinchilla suggests:\n"
    info_text2 += "tokens ≈ 20 × parameters\n\n"
    info_text2 += f"Total Budgets: {len(optimal_data)}\n"
    info_text2 += (
        f"Token Range:\n{optimal_tokens_B.min():.2f}B - {optimal_tokens_B.max():.2f}B"
    )

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
        "/mnt/localssd/NanoGPT/scripts/graphs/optimal_model_vs_flops.png"
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
