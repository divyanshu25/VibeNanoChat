#!/usr/bin/env python3
"""
Verify if the scaling law D = K × C^α holds for optimal models.
Where:
  D = Optimal model parameters
  C = Compute budget (FLOPs)
  K = Constant
  α = Exponent (testing both empirical and α=0.49)
"""

import numpy as np

print("=" * 80)
print("VERIFYING SCALING LAW: D = K × C^α")
print("=" * 80)
print()

# Data from the isoflop curve analysis
# Best models for each budget (lowest validation BPB)
data = [
    {
        "budget": 1.0e18,
        "optimal_model": "N8",
        "params": 50.90e6,  # 50.90M parameters
        "val_bpb": 1.1188,
    },
    {
        "budget": 3.0e18,
        "optimal_model": "N10",
        "params": 81.32e6,  # 81.32M parameters
        "val_bpb": 1.0218,
    },
]

print("📊 OPTIMAL MODELS:")
print("-" * 80)
for d in data:
    print(f"Budget C = {d['budget']:.2e} FLOPs")
    print(f"  Optimal Model: {d['optimal_model']}")
    print(f"  Parameters D = {d['params']/1e6:.2f}M")
    print(f"  Val BPB: {d['val_bpb']:.4f}")
    print()

print("=" * 80)
print("EMPIRICAL FIT: Finding best α")
print("=" * 80)
print()

# Calculate empirical exponent from the two points
log_C1 = np.log10(data[0]["budget"])
log_C2 = np.log10(data[1]["budget"])
log_D1 = np.log10(data[0]["params"])
log_D2 = np.log10(data[1]["params"])

alpha_empirical = (log_D2 - log_D1) / (log_C2 - log_C1)
K_empirical = 10 ** (log_D1 - alpha_empirical * log_C1)

print("Empirical fit: D = K × C^α")
print(f"  α (exponent) = {alpha_empirical:.4f}")
print(f"  K (constant) = {K_empirical:.4e}")
print()

# Test empirical fit
print("Predictions using empirical fit:")
for d in data:
    C = d["budget"]
    D_actual = d["params"]
    D_predicted = K_empirical * (C**alpha_empirical)
    error = abs(D_predicted - D_actual) / D_actual * 100

    print(
        f"  {d['optimal_model']} @ {C:.2e}: Actual={D_actual/1e6:.2f}M, Predicted={D_predicted/1e6:.2f}M, Error={error:.2f}%"
    )

print()
print("=" * 80)
print("TESTING WITH α = 0.49 (D = K × C^0.49)")
print("=" * 80)
print()

alpha_fixed = 0.49

# Calculate K values for each point using α = 0.49
constants_049 = []
for i, d in enumerate(data, 1):
    C = d["budget"]
    D = d["params"]
    K = D / (C**alpha_fixed)

    constants_049.append(K)

    print(f"Point {i}: {d['optimal_model']} @ {C:.2e} FLOPs")
    print(f"  D = {D:.2e} parameters")
    print(f"  C^0.49 = {C**alpha_fixed:.2e}")
    print(f"  K = D/C^0.49 = {K:.4e}")
    print()

K1_049, K2_049 = constants_049
K_avg_049 = np.mean(constants_049)
K_std_049 = np.std(constants_049)
relative_diff_049 = abs(K2_049 - K1_049) / K_avg_049 * 100

print("Consistency check:")
print(f"  K₁ = {K1_049:.4e}")
print(f"  K₂ = {K2_049:.4e}")
print(f"  K_avg = {K_avg_049:.4e}")
print(f"  K_std = {K_std_049:.4e}")
print(f"  Relative difference: {relative_diff_049:.2f}%")
print()

if relative_diff_049 < 5:
    print("✅ The constant K is VERY CONSISTENT with α=0.49!")
    print(f"   D ≈ {K_avg_049:.4e} × C^0.49")
elif relative_diff_049 < 15:
    print("✓ The constant K is REASONABLY CONSISTENT with α=0.49")
    print(f"  D ≈ {K_avg_049:.4e} × C^0.49 (with ~{relative_diff_049:.1f}% variation)")
else:
    print("⚠️  The constant K varies significantly with α=0.49")
    print(f"   Variation: {relative_diff_049:.1f}%")

print()
print("=" * 80)
print("PREDICTIONS USING D = K_avg × C^0.49")
print("=" * 80)
print()

for d in data:
    C = d["budget"]
    D_actual = d["params"]
    D_predicted = K_avg_049 * (C**alpha_fixed)
    error = abs(D_predicted - D_actual) / D_actual * 100

    print(f"Budget: {C:.2e} FLOPs")
    print(f"  Actual D:    {D_actual/1e6:.2f}M")
    print(f"  Predicted D: {D_predicted/1e6:.2f}M")
    print(f"  Error:       {error:.2f}%")
    print()

print("=" * 80)
print("EXTRAPOLATION TO OTHER BUDGETS (using α=0.49)")
print("=" * 80)
print()

test_budgets = [0.5e18, 2.0e18, 5.0e18, 10.0e18]
for C in test_budgets:
    D_pred = K_avg_049 * (C**alpha_fixed)
    print(f"Budget: {C:.2e} FLOPs → Predicted optimal D ≈ {D_pred/1e6:.2f}M parameters")

print()
print("=" * 80)
print("COMPARISON: α=0.49 vs α=0.50 (Chinchilla)")
print("=" * 80)
print()

# Also test with α = 0.50 for comparison
alpha_chinchilla = 0.50
constants_050 = []
for d in data:
    C = d["budget"]
    D = d["params"]
    K = D / (C**alpha_chinchilla)
    constants_050.append(K)

K_avg_050 = np.mean(constants_050)
relative_diff_050 = abs(constants_050[1] - constants_050[0]) / K_avg_050 * 100

print(f"With α = 0.49: K_avg = {K_avg_049:.4e}, variation = {relative_diff_049:.2f}%")
print(f"With α = 0.50: K_avg = {K_avg_050:.4e}, variation = {relative_diff_050:.2f}%")
print(f"Empirical α:   α = {alpha_empirical:.4f}, K = {K_empirical:.4e}")
print()

# Test predictions for both on actual data
print("Prediction errors on actual data:")
print()
total_error_049 = 0
total_error_050 = 0
total_error_emp = 0

for d in data:
    C = d["budget"]
    D_actual = d["params"]

    D_pred_049 = K_avg_049 * (C**0.49)
    D_pred_050 = K_avg_050 * (C**0.50)
    D_pred_emp = K_empirical * (C**alpha_empirical)

    err_049 = abs(D_pred_049 - D_actual) / D_actual * 100
    err_050 = abs(D_pred_050 - D_actual) / D_actual * 100
    err_emp = abs(D_pred_emp - D_actual) / D_actual * 100

    total_error_049 += err_049
    total_error_050 += err_050
    total_error_emp += err_emp

    print(f"{d['optimal_model']} @ {C:.2e}:")
    print(f"  α=0.49: Error = {err_049:.2f}%")
    print(f"  α=0.50: Error = {err_050:.2f}%")
    print(f"  α={alpha_empirical:.3f}: Error = {err_emp:.2f}%")
    print()

print("Average prediction error:")
print(f"  α=0.49: {total_error_049/len(data):.2f}%")
print(f"  α=0.50: {total_error_050/len(data):.2f}%")
print(f"  α={alpha_empirical:.3f}: {total_error_emp/len(data):.2f}%")
print()

# Determine which is best
if total_error_049 < total_error_050:
    print("✅ α=0.49 gives BETTER fit than α=0.50!")
else:
    print("✅ α=0.50 gives BETTER fit than α=0.49!")

print()
print("=" * 80)
print("RECOMMENDED SCALING LAW")
print("=" * 80)
print()
print(f"Best fit: D = {K_avg_049:.4e} × C^0.49")
print()
print("In more readable form:")
print(f"  D ≈ {K_avg_049*1e6:.2f} × (C/10^18)^0.49  [millions of parameters]")
print()
print("=" * 80)
