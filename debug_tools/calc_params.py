"""Calculate parameter counts for different model depths."""

import sys

sys.path.insert(0, "/mnt/localssd/VibeNanoChat/src")

from gpt_2.config import GPTConfig
from gpt_2.gpt2_model import GPT
from gpt_2.utils import get_scaling_params

# Standard depth configurations following nanochat conventions
# d_model = 64 * depth (tuned for optimal width-to-depth ratio)
depths = [12, 14, 16, 17, 18, 20]

print("=" * 80)
print("PARAMETER COUNTS BY DEPTH")
print("=" * 80)
print()

# Collect all data
all_data = []
for depth in depths:
    # Calculate n_embed following nanochat's convention: 64 * depth
    n_embed = 64 * depth
    n_head = depth  # One head per layer (standard)

    config = GPTConfig(
        n_layer=depth,
        n_head=n_head,
        n_embed=n_embed,
        vocab_size=50257,
        block_size=2048,
    )

    model = GPT(config)
    param_counts = model.num_scaling_params()
    scaling_params = get_scaling_params(model)

    all_data.append(
        {
            "depth": depth,
            "n_embed": n_embed,
            "wte": param_counts["wte"],
            "value_embeds": param_counts["value_embeds"],
            "lm_head": param_counts["lm_head"],
            "transformer_matrices": param_counts["transformer_matrices"],
            "scalars": param_counts["scalars"],
            "total": param_counts["total"],
            "scaling_params": scaling_params,
        }
    )

    print(f"Depth {depth} (n_embed={n_embed}, n_head={n_head}):")
    print(f"  {'wte (embeddings):':<30} {param_counts['wte']:>15,}")
    print(f"  {'value_embeds:':<30} {param_counts['value_embeds']:>15,}")
    print(f"  {'lm_head:':<30} {param_counts['lm_head']:>15,}")
    print(
        f"  {'transformer_matrices:':<30} {param_counts['transformer_matrices']:>15,}"
    )
    print(f"  {'scalars:':<30} {param_counts['scalars']:>15,}")
    print(f"  {'-' * 30} {'-' * 15}")
    print(f"  {'TOTAL:':<30} {param_counts['total']:>15,}")
    print(f"  {'Scaling params (tm + lm):':<30} {scaling_params:>15,}")
    print()

print("=" * 80)
print()

# Print clean summary table
print("=" * 140)
print("PARAMETER COUNT SUMMARY TABLE")
print("=" * 140)
header = f"{'Depth':<7} {'n_emb':<7} {'WTE':>12} {'Val_Embs':>12} {'LM_Head':>12} {'Xformer':>12} {'Scalars':>9} {'Total':>13} {'Scaling':>13}"
print(header)
print("-" * 140)

for data in all_data:
    row = (
        f"{data['depth']:<7} "
        f"{data['n_embed']:<7} "
        f"{data['wte']:>12,} "
        f"{data['value_embeds']:>12,} "
        f"{data['lm_head']:>12,} "
        f"{data['transformer_matrices']:>12,} "
        f"{data['scalars']:>9,} "
        f"{data['total']:>13,} "
        f"{data['scaling_params']:>13,}"
    )
    print(row)

print("=" * 140)
print()
print("Column descriptions:")
print("  WTE      = Word token embeddings")
print("  Val_Embs = Value embeddings (ResFormer-style)")
print("  LM_Head  = Language model output head")
print("  Xformer  = Transformer block matrices (attention + MLP)")
print("  Scalars  = Per-layer scalars (resid_lambdas + x0_lambdas)")
print("  Total    = All parameters")
print("  Scaling  = Scaling params used for training (Xformer + LM_Head)")
print()
