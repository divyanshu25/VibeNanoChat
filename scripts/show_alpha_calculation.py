#!/usr/bin/env python3
"""
Demonstrate how the empirical exponent α is calculated from two data points.
"""

import numpy as np

print("=" * 80)
print("HOW TO COMPUTE THE EMPIRICAL EXPONENT α")
print("=" * 80)
print()

print("Given a power law: D = K × C^α")
print()
print("Taking log₁₀ of both sides:")
print("  log(D) = log(K) + α × log(C)")
print()
print("This is a linear equation: y = b + m × x")
print("  where y = log(D), x = log(C), m = α (slope), b = log(K)")
print()
print("For two points (C₁, D₁) and (C₂, D₂), the slope is:")
print("  α = [log(D₂) - log(D₁)] / [log(C₂) - log(C₁)]")
print()
print("=" * 80)
print()

# Data points
C1 = 1.0e18
D1 = 50.90e6
model1 = "N8"

C2 = 3.0e18
D2 = 81.32e6
model2 = "N10"

print("📊 DATA POINTS:")
print("-" * 80)
print(f"Point 1 ({model1}):")
print(f"  C₁ = {C1:.2e} FLOPs")
print(f"  D₁ = {D1/1e6:.2f}M parameters = {D1:.2e} parameters")
print()
print(f"Point 2 ({model2}):")
print(f"  C₂ = {C2:.2e} FLOPs")
print(f"  D₂ = {D2/1e6:.2f}M parameters = {D2:.2e} parameters")
print()

print("=" * 80)
print("STEP-BY-STEP CALCULATION:")
print("=" * 80)
print()

# Step 1: Calculate logarithms
log_C1 = np.log10(C1)
log_C2 = np.log10(C2)
log_D1 = np.log10(D1)
log_D2 = np.log10(D2)

print("Step 1: Calculate logarithms (base 10)")
print(f"  log₁₀(C₁) = log₁₀({C1:.2e}) = {log_C1:.6f}")
print(f"  log₁₀(C₂) = log₁₀({C2:.2e}) = {log_C2:.6f}")
print(f"  log₁₀(D₁) = log₁₀({D1:.2e}) = {log_D1:.6f}")
print(f"  log₁₀(D₂) = log₁₀({D2:.2e}) = {log_D2:.6f}")
print()

# Step 2: Calculate differences
delta_log_C = log_C2 - log_C1
delta_log_D = log_D2 - log_D1

print("Step 2: Calculate differences")
print("  Δlog(C) = log₁₀(C₂) - log₁₀(C₁)")
print(f"          = {log_C2:.6f} - {log_C1:.6f}")
print(f"          = {delta_log_C:.6f}")
print()
print("  Δlog(D) = log₁₀(D₂) - log₁₀(D₁)")
print(f"          = {log_D2:.6f} - {log_D1:.6f}")
print(f"          = {delta_log_D:.6f}")
print()

# Step 3: Calculate alpha
alpha = delta_log_D / delta_log_C

print("Step 3: Calculate slope α")
print("  α = Δlog(D) / Δlog(C)")
print(f"    = {delta_log_D:.6f} / {delta_log_C:.6f}")
print(f"    = {alpha:.6f}")
print()

# Step 4: Calculate K
K = 10 ** (log_D1 - alpha * log_C1)

print("Step 4: Calculate constant K")
print("  From D₁ = K × C₁^α:")
print("  K = D₁ / C₁^α")
print(f"    = {D1:.2e} / ({C1:.2e})^{alpha:.4f}")
print(f"    = {K:.6f}")
print()

# Alternative calculation using logarithms
log_K = log_D1 - alpha * log_C1
print("  Alternatively, using log(K) = log(D₁) - α × log(C₁):")
print(f"  log(K) = {log_D1:.6f} - {alpha:.4f} × {log_C1:.6f}")
print(f"         = {log_K:.6f}")
print(f"  K = 10^{log_K:.6f} = {10**log_K:.6f}")
print()

print("=" * 80)
print("RESULT:")
print("=" * 80)
print()
print(f"✅ Empirical Power Law: D = {K:.6f} × C^{alpha:.4f}")
print()
print("Or in scientific notation:")
print(f"   D = {K:.4e} × C^{alpha:.4f}")
print()

# Verify the fit
print("=" * 80)
print("VERIFICATION:")
print("=" * 80)
print()
print(f"Using D = {K:.4f} × C^{alpha:.4f}:")
print()

D1_pred = K * (C1**alpha)
D2_pred = K * (C2**alpha)

error1 = abs(D1_pred - D1) / D1 * 100
error2 = abs(D2_pred - D2) / D2 * 100

print(f"Point 1 ({model1} @ {C1:.2e}):")
print(f"  Actual D₁:    {D1/1e6:.4f}M")
print(f"  Predicted D₁: {D1_pred/1e6:.4f}M")
print(f"  Error:        {error1:.6f}% ≈ 0% ✓")
print()

print(f"Point 2 ({model2} @ {C2:.2e}):")
print(f"  Actual D₂:    {D2/1e6:.4f}M")
print(f"  Predicted D₂: {D2_pred/1e6:.4f}M")
print(f"  Error:        {error2:.6f}% ≈ 0% ✓")
print()

print("(Errors are effectively zero because we fit the line through these two points)")
print()

# Show why α=0.49 gives errors
print("=" * 80)
print("COMPARISON: Why α=0.49 has ~3.5% error")
print("=" * 80)
print()

alpha_fixed = 0.49

# Calculate K for each point with α=0.49
K1_049 = D1 / (C1**alpha_fixed)
K2_049 = D2 / (C2**alpha_fixed)
K_avg_049 = (K1_049 + K2_049) / 2

print("If we fix α = 0.49 (closer to theoretical 0.5):")
print()
print(f"From point 1: K₁ = D₁/C₁^0.49 = {K1_049:.6f}")
print(f"From point 2: K₂ = D₂/C₂^0.49 = {K2_049:.6f}")
print(f"Average:      K_avg = {K_avg_049:.6f}")
print()
print(f"The two K values differ by {abs(K2_049-K1_049)/K_avg_049*100:.2f}%")
print("This mismatch causes prediction errors when using K_avg.")
print()

D1_pred_049 = K_avg_049 * (C1**alpha_fixed)
D2_pred_049 = K_avg_049 * (C2**alpha_fixed)

error1_049 = abs(D1_pred_049 - D1) / D1 * 100
error2_049 = abs(D2_pred_049 - D2) / D2 * 100

print(f"Predictions with D = {K_avg_049:.4f} × C^0.49:")
print(f"  Point 1: {error1_049:.2f}% error")
print(f"  Point 2: {error2_049:.2f}% error")
print()

print("=" * 80)
print("CONCLUSION:")
print("=" * 80)
print()
print(f"• Empirical α = {alpha:.4f} gives PERFECT fit (0% error)")
print("• Fixed α = 0.49 gives GOOD fit (~3.5% average error)")
print("• Fixed α = 0.50 gives FAIR fit (~4% average error)")
print()
print("The empirical α is lower than 0.5 because:")
print("  1. We only have 2 data points (limited sample)")
print("  2. Models are discrete (can't test all sizes)")
print("  3. True optimal might be between tested depths")
print()
print("α = 0.49 is a good practical choice - very close to theory,")
print("better fit than 0.50, and easier to work with than 0.4265!")
print()
print("=" * 80)
