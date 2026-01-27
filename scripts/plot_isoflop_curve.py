#!/usr/bin/env python3
"""
Plot ISOFlop curves as seen in the Chinchilla paper.
This script visualizes the relationship between model size and training tokens
for a fixed compute budget (ISOFlop curve).
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np

# Data extracted from scaling law logs
data = {
    "n10": {
        "params": 263_738_880,
        "tokens": 2_006_974_464,
        "flops": 3.0e18,
        "flops_per_token": 1.495e9,
        "final_val_bpb": 1.0338,  # Step 3827
    },
    "n12": {
        "params": 303_093_760,
        "tokens": 1_672_478_720,
        "flops": 3.0e18,
        "flops_per_token": 1.794e9,
        "final_val_bpb": 1.0477,  # Step 3189
    },
    "n14": {
        "params": 342_448_640,
        "tokens": 1_433_403_392,
        "flops": 3.0e18,
        "flops_per_token": 2.093e9,
        "final_val_bpb": 1.06189,  # Step 2733
    },
    "n16": {
        "params": 381_803_520,
        "tokens": 1_254_096_896,
        "flops": 3.0e18,
        "flops_per_token": 2.392e9,
        "final_val_bpb": 1.07513,  # Step 2391
    },
}

# Extract data for plotting
model_names = list(data.keys())
params = np.array([data[k]["params"] for k in model_names])
tokens = np.array([data[k]["tokens"] for k in model_names])
flops_values = np.array([data[k]["flops"] for k in model_names])

# Convert to millions/billions for better readability
params_M = params / 1e6
tokens_B = tokens / 1e9

# Create figure with good styling
plt.figure(figsize=(10, 7))
ax = plt.gca()

# Plot the data points
scatter = plt.scatter(
    params_M,
    tokens_B,
    s=200,
    c=["#FF6B6B", "#4ECDC4", "#45B7D1", "#95E1D3"],
    alpha=0.8,
    edgecolors="black",
    linewidth=2,
    zorder=3,
)

# Annotate each point
for i, name in enumerate(model_names):
    plt.annotate(
        name.upper(),
        (params_M[i], tokens_B[i]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8
        ),
    )

# Theoretical ISOFlop curve
# For a fixed compute budget C: C = 6 * N * D
# where N = parameters, D = tokens
# Rearranging: D = C / (6 * N)
C = 3.0e18  # Fixed compute budget
N_range = np.linspace(params_M.min() * 0.8, params_M.max() * 1.2, 100) * 1e6
D_theoretical = C / (6 * N_range)

plt.plot(
    N_range / 1e6,
    D_theoretical / 1e9,
    "k--",
    linewidth=2,
    alpha=0.6,
    label=f"ISOFlop Curve (C = {C:.1e} FLOPs)",
    zorder=2,
)

# Formatting
plt.xlabel("Model Parameters (Millions)", fontsize=14, fontweight="bold")
plt.ylabel("Training Tokens (Billions)", fontsize=14, fontweight="bold")
plt.title(
    "ISOFlop Curve: Model Size vs. Training Data\nFixed Compute Budget = 3.0e18 FLOPs",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

# Grid
plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

# Legend
plt.legend(fontsize=11, loc="upper right", framealpha=0.9)

# Add text box with key metrics
info_text = f"Compute Budget: {C:.1e} FLOPs\n"
info_text += f"Model range: {params_M.min():.1f}M - {params_M.max():.1f}M params\n"
info_text += f"Token range: {tokens_B.min():.2f}B - {tokens_B.max():.2f}B tokens"
plt.text(
    0.02,
    0.98,
    info_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

# Tight layout
plt.tight_layout()

# Save figure
output_path = "/mnt/localssd/NanoGPT/scripts/graphs/isoflop_curve.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"✅ Plot saved to: {output_path}")

# Close the first plot
plt.close()

# ============================================================================
# PLOT 2: Validation BPB vs Model Parameters
# ============================================================================

# Extract models with validation data
models_with_val = {k: v for k, v in data.items() if v["final_val_bpb"] is not None}
val_model_names = list(models_with_val.keys())
val_params = np.array([models_with_val[k]["params"] for k in val_model_names])
val_bpb = np.array([models_with_val[k]["final_val_bpb"] for k in val_model_names])

# Convert to millions
val_params_M = val_params / 1e6

# Create second figure
fig, ax = plt.subplots(figsize=(10, 7))

# Plot the data points
colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#95E1D3"]
model_colors = {k: colors[i] for i, k in enumerate(data.keys())}
scatter_colors = [model_colors[k] for k in val_model_names]

scatter = ax.scatter(
    val_params_M,
    val_bpb,
    s=250,
    c=scatter_colors,
    alpha=0.8,
    edgecolors="black",
    linewidth=2,
    zorder=3,
)

# Connect points with a line
ax.plot(val_params_M, val_bpb, "k--", linewidth=1.5, alpha=0.4, zorder=2)

# Annotate each point
for i, name in enumerate(val_model_names):
    ax.annotate(
        name.upper(),
        (val_params_M[i], val_bpb[i]),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=12,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9
        ),
    )
    # Add BPB value as text
    ax.text(
        val_params_M[i],
        val_bpb[i] - 0.003,
        f"{val_bpb[i]:.4f}",
        ha="center",
        va="top",
        fontsize=10,
        style="italic",
        color="darkred",
    )

# Formatting
ax.set_xlabel("Model Parameters (Millions)", fontsize=14, fontweight="bold")
ax.set_ylabel("Validation BPB (Bits Per Byte)", fontsize=14, fontweight="bold")
ax.set_title(
    "Scaling Law: Validation BPB vs Model Size\nFixed Compute Budget = 3.0e18 FLOPs",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

# Grid
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

# Add text box with insights
info_text = f"Models evaluated: {len(val_model_names)}\n"
info_text += f"Compute Budget: {C:.1e} FLOPs\n"
info_text += (
    f"Best BPB: {val_bpb.min():.4f} ({val_model_names[np.argmin(val_bpb)].upper()})"
)

ax.text(
    0.02,
    0.98,
    info_text,
    transform=ax.transAxes,
    fontsize=11,
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

# Print summary statistics
print("\n" + "=" * 60)
print("ISOFLOP CURVE ANALYSIS")
print("=" * 60)
for name in model_names:
    d = data[name]
    ratio = d["tokens"] / d["params"]
    print(f"\n{name.upper()}:")
    print(f"  Parameters:     {d['params']/1e6:.2f}M")
    print(f"  Training Tokens: {d['tokens']/1e9:.2f}B")
    print(f"  Total FLOPs:     {d['flops']:.2e}")
    print(f"  Tokens/Param:    {ratio:.2f}")
    if d["final_val_bpb"] is not None:
        print(f"  Final Val BPB:   {d['final_val_bpb']:.4f}")
    else:
        print("  Final Val BPB:   N/A (still running)")
print("\n" + "=" * 60)

# Calculate optimal point (from Chinchilla scaling laws)
print("\nCHINCHILLA SCALING INSIGHT:")
print("For optimal performance at fixed compute, Chinchilla suggests:")
print("tokens ≈ 20 × parameters")
print("\nYour models show:")
for name in model_names:
    d = data[name]
    ratio = d["tokens"] / d["params"]
    print(f"  {name.upper()}: {ratio:.2f} tokens/param")
print("=" * 60)
