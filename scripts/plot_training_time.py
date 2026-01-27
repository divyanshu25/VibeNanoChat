#!/usr/bin/env python3
"""
Plot training time vs model size for scaling law experiments.
Parses time_taken from logs and creates visualization.
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_training_time(log_file):
    """
    Parse the log file and extract time and MFU values from step logs.
    Returns the total training time in hours and average MFU.
    """
    times = []
    mfus = []

    with open(log_file, "r") as f:
        for line in f:
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

    if not times:
        print(f"⚠️  No timing entries found in {log_file}")
        return None

    total_seconds = sum(times)
    total_hours = total_seconds / 3600
    avg_mfu = np.mean(mfus) if mfus else 0

    print(f"📊 {Path(log_file).name}")
    print(f"   Steps: {len(times)}")
    print(
        f"   Total time: {total_seconds:.2f}s = {total_hours:.3f}h = {total_hours*60:.1f}min"
    )
    print(f"   Avg time/step: {np.mean(times):.3f}s")
    print(f"   Avg MFU: {avg_mfu:.2f}%")
    print()

    return total_hours, len(times), np.mean(times), avg_mfu


# Log files
log_files = {
    "n10": "/mnt/localssd/NanoGPT/logs/scaling_law_n10_20260126_131156.log",
    "n12": "/mnt/localssd/NanoGPT/logs/scaling_law_n12_20260126_131156.log",
    "n14": "/mnt/localssd/NanoGPT/logs/scaling_law_n14_20260126_131156.log",
    "n16": "/mnt/localssd/NanoGPT/logs/scaling_law_n16_20260126_131156.log",
}

# Model parameters (from previous analysis)
model_params = {
    "n10": 263_738_880,
    "n12": 303_093_760,
    "n14": 342_448_640,
    "n16": 381_803_520,
}

print("=" * 70)
print("PARSING TRAINING TIMES FROM LOGS")
print("=" * 70)
print()

# Parse all logs
results = {}
for name, log_file in log_files.items():
    result = parse_training_time(log_file)
    if result:
        total_hours, num_steps, avg_time, avg_mfu = result
        results[name] = {
            "params": model_params[name],
            "total_hours": total_hours,
            "num_steps": num_steps,
            "avg_time_per_step": avg_time,
            "avg_mfu": avg_mfu,
        }

print("=" * 70)
print()

# Prepare data for plotting
model_names = sorted(results.keys(), key=lambda x: results[x]["params"])
params = np.array([results[k]["params"] for k in model_names])
times_hours = np.array([results[k]["total_hours"] for k in model_names])
times_minutes = times_hours * 60
avg_mfus = np.array([results[k]["avg_mfu"] for k in model_names])

params_M = params / 1e6

# ============================================================================
# PLOT 1: Training Time and MFU vs Model Size
# ============================================================================

fig, ax1 = plt.subplots(figsize=(12, 7))

# Plot the data points for training time
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#95E1D3"]
model_colors = {k: colors[i] for i, k in enumerate(["n10", "n12", "n14", "n16"])}
point_colors = [model_colors[k] for k in model_names]

scatter1 = ax1.scatter(
    params_M,
    times_minutes,
    s=300,
    c=point_colors,
    alpha=0.8,
    edgecolors="black",
    linewidth=2.5,
    zorder=3,
    marker="o",
    label="Training Time",
)

# Connect points with a line
line1 = ax1.plot(
    params_M,
    times_minutes,
    "b-",
    linewidth=2,
    alpha=0.5,
    zorder=2,
    label="Training Time",
)[0]

# Annotate training time
for i, name in enumerate(model_names):
    ax1.annotate(
        name.upper(),
        (params_M[i], times_minutes[i]),
        xytext=(10, 15),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9
        ),
    )
    # Add time value as text
    ax1.text(
        params_M[i],
        times_minutes[i] - 1.8,
        f"{times_minutes[i]:.1f} min",
        ha="center",
        va="top",
        fontsize=9,
        style="italic",
        color="darkblue",
        fontweight="bold",
    )

# Formatting for left y-axis (Training Time)
ax1.set_xlabel("Model Parameters (Millions)", fontsize=14, fontweight="bold")
ax1.set_ylabel(
    "Training Time (Minutes)", fontsize=14, fontweight="bold", color="darkblue"
)
ax1.tick_params(axis="y", labelcolor="darkblue")
ax1.set_ylim(bottom=times_minutes.min() * 0.88, top=times_minutes.max() * 1.12)

# Create secondary y-axis for MFU
ax2 = ax1.twinx()

# Plot MFU on secondary axis
scatter2 = ax2.scatter(
    params_M,
    avg_mfus,
    s=250,
    c="orange",
    alpha=0.7,
    edgecolors="darkred",
    linewidth=2,
    zorder=4,
    marker="s",
    label="Avg MFU",
)

line2 = ax2.plot(
    params_M, avg_mfus, "r--", linewidth=2, alpha=0.6, zorder=2, label="Avg MFU"
)[0]

# Annotate MFU values
for i, name in enumerate(model_names):
    ax2.text(
        params_M[i] + 2,
        avg_mfus[i],
        f"{avg_mfus[i]:.1f}%",
        ha="left",
        va="center",
        fontsize=9,
        style="italic",
        color="darkred",
        fontweight="bold",
    )

# Formatting for right y-axis (MFU)
ax2.set_ylabel("Average MFU (%)", fontsize=14, fontweight="bold", color="darkred")
ax2.tick_params(axis="y", labelcolor="darkred")
ax2.set_ylim(bottom=avg_mfus.min() * 0.92, top=avg_mfus.max() * 1.08)

# Title
ax1.set_title(
    "Training Time and MFU vs Model Size\nISOFlop Budget = 3.0e18 FLOPs",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

# Grid
ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

# Combined legend
lines = [line1, line2]
labels = ["Training Time", "Average MFU"]
ax1.legend(lines, labels, fontsize=11, loc="upper left", framealpha=0.9)

# Add text box with insights
info_text = "Fixed Compute: 3.0e18 FLOPs\n"
info_text += f"Models evaluated: {len(model_names)}\n"
info_text += f"Time range: {times_minutes.min():.1f} - {times_minutes.max():.1f} min\n"
info_text += f"MFU range: {avg_mfus.min():.1f}% - {avg_mfus.max():.1f}%"

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
print(f"✅ Plot saved to: {output_path}")

plt.close()

# ============================================================================
# PLOT 2: Average Time per Step vs Model Size
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 7))

avg_times = np.array([results[k]["avg_time_per_step"] for k in model_names])

# Plot the data points
scatter = ax.scatter(
    params_M,
    avg_times,
    s=300,
    c=point_colors,
    alpha=0.8,
    edgecolors="black",
    linewidth=2.5,
    zorder=3,
)

# Connect points with a line
ax.plot(params_M, avg_times, "k--", linewidth=2, alpha=0.5, zorder=2)

# Annotate each point
for i, name in enumerate(model_names):
    ax.annotate(
        name.upper(),
        (params_M[i], avg_times[i]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9
        ),
    )
    # Add time value as text
    ax.text(
        params_M[i],
        avg_times[i] - 0.05,
        f"{avg_times[i]:.2f}s",
        ha="center",
        va="top",
        fontsize=10,
        style="italic",
        color="darkred",
        fontweight="bold",
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

# Add text box with insights
info_text = "Hardware: 2 x H100 80GB\n"
info_text += "Total batch: 524,288 tokens\n"
info_text += f"Step time range: {avg_times.min():.2f}s - {avg_times.max():.2f}s\n"
slowdown = ((avg_times.max() / avg_times.min()) - 1) * 100
info_text += f"Slowdown: +{slowdown:.1f}%"

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
print(f"✅ Plot saved to: {output_path2}")

plt.close()

# Print summary table
print("\n" + "=" * 80)
print("TRAINING TIME SUMMARY")
print("=" * 80)
print(
    f"{'Model':<8} {'Params (M)':<12} {'Steps':<8} {'Total Time':<15} {'Avg/Step':<12} {'Avg MFU':<10}"
)
print("-" * 80)
for name in model_names:
    d = results[name]
    print(
        f"{name.upper():<8} {d['params']/1e6:>10.1f}M  {d['num_steps']:>6}   "
        f"{d['total_hours']*60:>6.1f} min ({d['total_hours']:.2f}h)  {d['avg_time_per_step']:>8.3f}s   "
        f"{d['avg_mfu']:>7.2f}%"
    )
print("=" * 80)

# Calculate efficiency metrics
best_mfu_idx = np.argmax(avg_mfus)
best_mfu_model = model_names[best_mfu_idx]
print("\n📊 EFFICIENCY ANALYSIS:")
print(
    "  • Larger models take proportionally longer per step due to increased computation"
)
print(
    "  • Despite larger models, same 3e18 FLOPs budget maintained through fewer steps"
)
print(
    f"  • Training time variation: {times_minutes.min():.1f} - {times_minutes.max():.1f} minutes"
)
print(f"  • MFU range: {avg_mfus.min():.2f}% - {avg_mfus.max():.2f}%")
print(f"  • Best MFU: {avg_mfus.max():.2f}% achieved by {best_mfu_model.upper()}")
print(f"  • N16 (largest model) has lowest MFU at {avg_mfus.min():.2f}%")
print("=" * 80)
