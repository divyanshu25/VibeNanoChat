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

import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path):
    """Parse a log file to extract key metrics."""
    with open(log_path, "r") as f:
        content = f.read()

    # Extract parameters
    param_match = re.search(
        r"Num decay parameter tensors.*?with ([\d,]+) parameters", content
    )
    if not param_match:
        return None
    params = int(param_match.group(1).replace(",", ""))

    # Extract number of iterations (steps)
    iter_match = re.search(
        r"Calculated number of iterations from target FLOPs: ([\d,]+)", content
    )
    if not iter_match:
        return None
    iterations = int(iter_match.group(1).replace(",", ""))

    # Extract batch size
    batch_match = re.search(r"Total batch size: ([\d,]+)", content)
    if not batch_match:
        return None
    batch_size = int(batch_match.group(1).replace(",", ""))

    # Calculate tokens
    tokens = iterations * batch_size

    # Extract final validation BPB
    val_matches = re.findall(r"VALIDATION.*?BPB: ([\d.]+)", content)
    final_val_bpb = float(val_matches[-1]) if val_matches else None

    # Extract final step
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


# Parse all log files
log_files = [
    "/mnt/localssd/NanoGPT/logs/scaling_law_n6_b1e18.log",
    "/mnt/localssd/NanoGPT/logs/scaling_law_n6_b3e18.log",
    "/mnt/localssd/NanoGPT/logs/scaling_law_n8_b1e18.log",
    "/mnt/localssd/NanoGPT/logs/scaling_law_n10_b1e18.log",
    "/mnt/localssd/NanoGPT/logs/scaling_law_n10_b3e18.log",
    "/mnt/localssd/NanoGPT/logs/scaling_law_n12_b1e18.log",
    "/mnt/localssd/NanoGPT/logs/scaling_law_n12_b3e18.log",
    "/mnt/localssd/NanoGPT/logs/scaling_law_n14_3e18.log",
    "/mnt/localssd/NanoGPT/logs/scaling_law_n14_b1e18.log",
    "/mnt/localssd/NanoGPT/logs/scaling_law_n16_b1e18.log",
    "/mnt/localssd/NanoGPT/logs/scaling_law_n16_b3e18.log",
    "/mnt/localssd/NanoGPT/logs/scaling_law_n18_b1e18.log",
    "/mnt/localssd/NanoGPT/logs/scaling_law_n18_b3e18.log",
]

# Group data by budget
data_by_budget = defaultdict(dict)

print("Parsing log files...")
for log_file in log_files:
    if not os.path.exists(log_file):
        print(f"Warning: {log_file} not found, skipping...")
        continue

    # Extract n_layers and budget from filename
    match = re.search(r"scaling_law_n(\d+)_b?([\de]+)\.log", log_file)
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

# Color schemes for different budgets
budget_colors = {1e18: "#FF6B6B", 3e18: "#4ECDC4"}
budget_markers = {1e18: "o", 3e18: "s"}

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
