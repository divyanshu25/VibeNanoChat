#!/usr/bin/env python3
"""
Plot training time vs model size for scaling law experiments.
Parses time_taken from logs and creates visualization.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_file):
    """
    Parse the log file and extract parameters, time, and MFU values.
    Returns a dictionary with all metrics.
    """
    times = []
    mfus = []
    params = None

    with open(log_file, "r") as f:
        content = f.read()

        # Extract parameters
        param_match = re.search(
            r"Num decay parameter tensors.*?with ([\d,]+) parameters", content
        )
        if param_match:
            params = int(param_match.group(1).replace(",", ""))

        # Extract time and MFU from step logs
        for line in content.split("\n"):
            # Look for step logs with format: [Epoch ...] | Time: X.XXs | MFU: XX.XX%
            if "[Epoch" in line and "| Time:" in line:
                # Extract the time value using regex
                time_match = re.search(r"\| Time:\s+([\d.]+)s", line)
                if time_match:
                    times.append(float(time_match.group(1)))

                # Extract MFU value
                mfu_match = re.search(r"\| MFU:\s+([\d.]+)%", line)
                if mfu_match:
                    mfus.append(float(mfu_match.group(1)))

    if not times or params is None:
        return None

    total_seconds = sum(times)
    total_hours = total_seconds / 3600
    avg_mfu = np.mean(mfus) if mfus else 0

    return {
        "params": params,
        "total_hours": total_hours,
        "total_minutes": total_hours * 60,
        "num_steps": len(times),
        "avg_time_per_step": np.mean(times),
        "avg_mfu": avg_mfu,
    }


# Log files to parse
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

print("=" * 80)
print("PARSING TRAINING TIMES FROM LOGS")
print("=" * 80)
print()

for log_file in log_files:
    if not os.path.exists(log_file):
        print(f"⚠️  {log_file} not found, skipping...")
        continue

    # Extract n_layers and budget from filename
    match = re.search(r"scaling_law_n(\d+)_b?([\de]+)\.log", log_file)
    if not match:
        print(f"⚠️  Could not parse filename {log_file}, skipping...")
        continue

    n_layers = int(match.group(1))
    budget_str = match.group(2)
    budget = float(budget_str.replace("e", "e"))

    print(f"📊 {Path(log_file).name} (n={n_layers}, budget={budget:.1e})")

    metrics = parse_log_file(log_file)
    if metrics:
        model_name = f"n{n_layers}"
        metrics["budget"] = budget
        data_by_budget[budget][model_name] = metrics

        print(f"   Params: {metrics['params']/1e6:.1f}M")
        print(f"   Steps: {metrics['num_steps']}")
        print(
            f"   Total time: {metrics['total_minutes']:.1f} min ({metrics['total_hours']:.3f}h)"
        )
        print(f"   Avg time/step: {metrics['avg_time_per_step']:.3f}s")
        print(f"   Avg MFU: {metrics['avg_mfu']:.2f}%")
    else:
        print("   ✗ Failed to parse metrics")
    print()

print("=" * 80)
print(f"Found data for {len(data_by_budget)} different budgets")
print("=" * 80)
print()

# ============================================================================
# PLOT 1: Training Time and MFU vs Model Size (Multiple Budgets)
# ============================================================================

fig, ax1 = plt.subplots(figsize=(14, 8))

# Color schemes for different budgets
budget_colors = {1e18: "#FF6B6B", 3e18: "#4ECDC4"}
budget_markers = {1e18: "o", 3e18: "s"}

# Sort budgets for consistent plotting
sorted_budgets = sorted(data_by_budget.keys())

# Track all times and MFUs for axis limits
all_times = []
all_mfus = []

lines_time = []
lines_mfu = []

# Create secondary y-axis for MFU
ax2 = ax1.twinx()

# Plot for each budget
for budget in sorted_budgets:
    models = data_by_budget[budget]
    if not models:
        continue

    model_names = sorted(models.keys(), key=lambda x: int(x[1:]))
    params = np.array([models[k]["params"] for k in model_names])
    times_minutes = np.array([models[k]["total_minutes"] for k in model_names])
    avg_mfus = np.array([models[k]["avg_mfu"] for k in model_names])

    params_M = params / 1e6

    all_times.extend(times_minutes)
    all_mfus.extend(avg_mfus)

    color = budget_colors.get(budget, "#45B7D1")
    marker = budget_markers.get(budget, "D")

    # Plot training time on primary axis
    scatter1 = ax1.scatter(
        params_M,
        times_minutes,
        s=250,
        c=color,
        alpha=0.8,
        edgecolors="black",
        linewidth=2,
        zorder=3,
        marker=marker,
    )

    line1 = ax1.plot(
        params_M,
        times_minutes,
        "-",
        linewidth=2,
        alpha=0.6,
        color=color,
        zorder=2,
        label=f"Time ({budget:.1e} FLOPs)",
    )[0]
    lines_time.append(line1)

    # Plot MFU on secondary axis
    scatter2 = ax2.scatter(
        params_M,
        avg_mfus,
        s=200,
        c=color,
        alpha=0.6,
        edgecolors="darkred",
        linewidth=1.5,
        zorder=4,
        marker="^",
    )

    line2 = ax2.plot(
        params_M,
        avg_mfus,
        "--",
        linewidth=1.5,
        alpha=0.5,
        color=color,
        zorder=2,
        label=f"MFU ({budget:.1e} FLOPs)",
    )[0]
    lines_mfu.append(line2)

    # Annotate each point
    for i, name in enumerate(model_names):
        ax1.annotate(
            name.upper(),
            (params_M[i], times_minutes[i]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=color,
                alpha=0.8,
            ),
        )

# Formatting for left y-axis (Training Time)
ax1.set_xlabel("Model Parameters (Millions)", fontsize=14, fontweight="bold")
ax1.set_ylabel(
    "Training Time (Minutes)", fontsize=14, fontweight="bold", color="darkblue"
)
ax1.tick_params(axis="y", labelcolor="darkblue")

# Formatting for right y-axis (MFU)
ax2.set_ylabel("Average MFU (%)", fontsize=14, fontweight="bold", color="darkred")
ax2.tick_params(axis="y", labelcolor="darkred")

# Title
ax1.set_title(
    "Training Time and MFU vs Model Size\nComparing Different Compute Budgets",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

# Grid
ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

# Combined legend
all_lines = lines_time + lines_mfu
all_labels = [line.get_label() for line in all_lines]
ax1.legend(all_lines, all_labels, fontsize=10, loc="upper left", framealpha=0.9, ncol=2)

# Add text box with insights
info_text = f"Budgets: {', '.join([f'{b:.1e}' for b in sorted_budgets])} FLOPs\n"
info_text += f"Time range: {min(all_times):.1f} - {max(all_times):.1f} min\n"
info_text += f"MFU range: {min(all_mfus):.1f}% - {max(all_mfus):.1f}%"

ax1.text(
    0.02,
    0.30,
    info_text,
    transform=ax1.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7),
)

plt.tight_layout()

# Save figure
output_path = "/mnt/localssd/NanoGPT/scripts/graphs/training_time_vs_model_size.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"\n✅ Training time plot saved to: {output_path}")

plt.close()

# ============================================================================
# PLOT 2: Average Time per Step vs Model Size (Multiple Budgets)
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

# Track all step times for axis limits
all_step_times = []

# Plot for each budget
for budget in sorted_budgets:
    models = data_by_budget[budget]
    if not models:
        continue

    model_names = sorted(models.keys(), key=lambda x: int(x[1:]))
    params = np.array([models[k]["params"] for k in model_names])
    avg_times = np.array([models[k]["avg_time_per_step"] for k in model_names])

    params_M = params / 1e6
    all_step_times.extend(avg_times)

    color = budget_colors.get(budget, "#45B7D1")
    marker = budget_markers.get(budget, "D")

    # Plot the data points
    scatter = ax.scatter(
        params_M,
        avg_times,
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
    ax.plot(params_M, avg_times, "--", linewidth=2, alpha=0.5, color=color, zorder=2)

    # Annotate each point
    for i, name in enumerate(model_names):
        ax.annotate(
            name.upper(),
            (params_M[i], avg_times[i]),
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
ax.set_ylabel("Average Time per Step (Seconds)", fontsize=14, fontweight="bold")
ax.set_title(
    "Training Step Time vs Model Size\n2 x H100 GPUs, Batch Size = 524K tokens",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

# Grid
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

# Legend
ax.legend(fontsize=11, loc="upper left", framealpha=0.9)

# Add text box with insights
info_text = "Hardware: 2 x H100 80GB\n"
info_text += "Total batch: 524,288 tokens\n"
info_text += (
    f"Step time range: {min(all_step_times):.2f}s - {max(all_step_times):.2f}s\n"
)
slowdown = ((max(all_step_times) / min(all_step_times)) - 1) * 100
info_text += f"Max slowdown: +{slowdown:.1f}%"

ax.text(
    0.02,
    0.98,
    info_text,
    transform=ax.transAxes,
    fontsize=11,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.7),
)

plt.tight_layout()

# Save figure
output_path2 = "/mnt/localssd/NanoGPT/scripts/graphs/step_time_vs_model_size.png"
plt.savefig(output_path2, dpi=300, bbox_inches="tight")
print(f"✅ Step time plot saved to: {output_path2}")

plt.close()

# ============================================================================
# Print Summary Tables
# ============================================================================
print("\n" + "=" * 90)
print("TRAINING TIME SUMMARY")
print("=" * 90)

for budget in sorted_budgets:
    print(f"\n{'='*90}")
    print(f"BUDGET: {budget:.1e} FLOPs")
    print(f"{'='*90}")
    print(
        f"{'Model':<8} {'Params (M)':<12} {'Steps':<8} {'Total Time':<18} {'Avg/Step':<12} {'Avg MFU':<10}"
    )
    print("-" * 90)

    models = data_by_budget[budget]
    model_names = sorted(models.keys(), key=lambda x: int(x[1:]))

    for name in model_names:
        d = models[name]
        print(
            f"{name.upper():<8} {d['params']/1e6:>10.1f}M  {d['num_steps']:>6}   "
            f"{d['total_minutes']:>7.1f} min ({d['total_hours']:>5.2f}h)  {d['avg_time_per_step']:>8.3f}s   "
            f"{d['avg_mfu']:>7.2f}%"
        )

    # Find best MFU in this budget
    best_mfu_model = max(models.items(), key=lambda x: x[1]["avg_mfu"])
    fastest_model = min(models.items(), key=lambda x: x[1]["total_minutes"])

    print("\n  📊 Budget Analysis:")
    print(
        f"     • Best MFU: {best_mfu_model[1]['avg_mfu']:.2f}% ({best_mfu_model[0].upper()})"
    )
    print(
        f"     • Fastest training: {fastest_model[1]['total_minutes']:.1f} min ({fastest_model[0].upper()})"
    )

print("\n" + "=" * 90)

# Overall efficiency analysis
print("\n📊 OVERALL EFFICIENCY ANALYSIS:")
print("=" * 90)
print(
    "  • Larger models take proportionally longer per step due to increased computation"
)
print(
    "  • Higher compute budgets (3e18) result in more training steps → longer total time"
)
print(
    "  • Smaller models (n6, n8) generally achieve higher MFU due to better hardware utilization"
)

all_mfus_flat = []
all_times_flat = []
for budget in sorted_budgets:
    models = data_by_budget[budget]
    for model_data in models.values():
        all_mfus_flat.append(model_data["avg_mfu"])
        all_times_flat.append(model_data["total_minutes"])

print(f"  • Overall MFU range: {min(all_mfus_flat):.2f}% - {max(all_mfus_flat):.2f}%")
print(
    f"  • Overall time range: {min(all_times_flat):.1f} - {max(all_times_flat):.1f} minutes"
)
print("=" * 90)
