"""Generates docs/training_pipeline.excalidraw for the VibeNanoChat training pipeline.
Font families: 1=Virgil(hand), 2=Helvetica(clean), 3=Cascadia(mono)
"""
import json, os

elements = []
_seed = 1

def sid():
    global _seed; _seed += 1
    return str(_seed).zfill(8)

# ── primitives ────────────────────────────────────────────────────────────────

def rect(x, y, w, h, bg, stroke, radius=True):
    e = {"id": sid(), "type": "rectangle",
         "x": x, "y": y, "width": w, "height": h,
         "angle": 0, "strokeColor": stroke, "backgroundColor": bg,
         "fillStyle": "solid", "strokeWidth": 1.5, "strokeStyle": "solid",
         "roughness": 0, "opacity": 100, "groupIds": [],
         "roundness": {"type": 3} if radius else None,
         "seed": _seed, "version": 1, "versionNonce": 1,
         "isDeleted": False, "boundElements": [], "updated": 1,
         "link": None, "locked": False}
    elements.append(e); return e["id"]

def text(x, y, w, content, size=13, color="#1e293b", family=2, align="left"):
    """family: 1=Virgil(hand-drawn), 2=Helvetica(clean), 3=Cascadia(mono)"""
    lines = content.count("\n") + 1
    h = size * 1.6 * lines + 4
    e = {"id": sid(), "type": "text",
         "x": x, "y": y, "width": w, "height": h,
         "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
         "fillStyle": "solid", "strokeWidth": 1, "strokeStyle": "solid",
         "roughness": 0, "opacity": 100, "groupIds": [],
         "roundness": None, "seed": _seed, "version": 1, "versionNonce": 1,
         "isDeleted": False, "boundElements": [], "updated": 1,
         "link": None, "locked": False,
         "text": content, "fontSize": size, "fontFamily": family,
         "textAlign": align, "verticalAlign": "top",
         "containerId": None, "originalText": content, "lineHeight": 1.6}
    elements.append(e); return e["id"]

def pill(cx, y, w, lbl, size=13.5, bg="#ffffff", color="#1e293b"):
    pw = w - 32
    rect(cx - pw//2, y, pw, int(size * 2.0), bg, "#cbd5e1", radius=True)
    text(cx - pw//2 + 10, y + int(size * 0.35), pw - 20, lbl,
         size=size, color=color, family=2, align="center")
    return int(size * 2.0) + 8

def section(x, y, w, heading):
    """Uppercase section heading in Helvetica."""
    text(x, y, w, heading, size=9.5, color="#64748b", family=2)
    return 15

def bullets(x, y, w, items, size=12, color="#334155", mono=False):
    content = "\n".join(f"· {i}" for i in items)
    fam = 3 if mono else 2
    text(x, y, w, content, size=size, color=color, family=fam)
    return len(items) * size * 1.6 + 2

def code_block(x, y, w, lines, size=11, color="#1e40af"):
    """Monospace code block."""
    content = "\n".join(lines)
    text(x, y, w, content, size=size, color=color, family=3)
    return len(lines) * size * 1.6 + 2

def hr(x, y, w, color="#cbd5e1"):
    e = {"id": sid(), "type": "line",
         "x": x, "y": y, "width": w, "height": 0,
         "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
         "fillStyle": "solid", "strokeWidth": 1, "strokeStyle": "solid",
         "roughness": 0, "opacity": 55, "groupIds": [],
         "roundness": None, "seed": _seed, "version": 1, "versionNonce": 1,
         "isDeleted": False, "boundElements": [], "updated": 1,
         "link": None, "locked": False,
         "points": [[0, 0], [w, 0]], "lastCommittedPoint": None,
         "startBinding": None, "endBinding": None,
         "startArrowhead": None, "endArrowhead": None}
    elements.append(e)

def arrow_v(cx, y1, y2, lbl=""):
    e = {"id": sid(), "type": "arrow",
         "x": cx, "y": y1, "width": 2, "height": y2 - y1,
         "angle": 0, "strokeColor": "#94a3b8", "backgroundColor": "transparent",
         "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
         "roughness": 0, "opacity": 100, "groupIds": [],
         "roundness": {"type": 2}, "seed": _seed, "version": 1, "versionNonce": 1,
         "isDeleted": False, "boundElements": [], "updated": 1,
         "link": None, "locked": False,
         "points": [[0, 0], [0, y2 - y1]],
         "lastCommittedPoint": None, "startBinding": None, "endBinding": None,
         "startArrowhead": None, "endArrowhead": "arrow"}
    elements.append(e)
    if lbl:
        text(cx + 12, y1 + (y2 - y1)//2 - 10, 320, lbl, size=11,
             color="#64748b", family=2)

def sub(x, y, w, h):
    rect(x, y, w, h, "#ffffff", "#e2e8f0", radius=True)

# ── palette ───────────────────────────────────────────────────────────────────
C = {
    "purple": ("#ede9fe", "#a78bfa"),
    "blue":   ("#dbeafe", "#60a5fa"),
    "rose":   ("#ffe4e6", "#fb7185"),
    "teal":   ("#ccfbf1", "#2dd4bf"),
    "slate":  ("#f1f5f9", "#94a3b8"),
    "green":  ("#dcfce7", "#4ade80"),
    "amber":  ("#fef3c7", "#f59e0b"),
    "violet": ("#f5f3ff", "#8b5cf6"),
    "indigo": ("#e0e7ff", "#6366f1"),
}

P   = 20      # inner padding
GAP = 52      # arrow gap between rows
CX  = 680     # canvas centre x
W3  = 400     # 3-column card width
W2  = 460     # 2-column card width
WFULL = 1260  # full-width card

x1 = CX - W3*3//2 - 20
x2 = CX - W3//2
x3 = CX + W3//2 + 20

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1 — Entry / DDP
# ══════════════════════════════════════════════════════════════════════════════
y = 40
cw, ch = 560, 300
cx0 = CX - cw//2
rect(cx0, y, cw, ch, C["purple"][0], C["purple"][1])
dy = y + P
dy += pill(CX, dy, cw, "🖥️  ddp.py  —  Entry Point")
dy += section(cx0+P, dy, cw-P*2, "CLI ARGUMENTS")
dy += bullets(cx0+P, dy, cw-P*2, [
    "--mode  [pretrain | sft]",
    "--depth  (e.g. 8 / 12 / 18 / 22)  ·  --aspect_ratio  (default 64)  ·  --head_dim  (default 64)",
    "--target_flops  or  --param_data_ratio  (default 10 tokens/param)",
    "--checkpoint_path  (required for sft, optional resume for pretrain)",
], mono=True, size=11.5)
hr(cx0+P, dy+3, cw-P*2, C["purple"][1])
dy += 10
dy += section(cx0+P, dy, cw-P*2, "SETUP_DISTRIBUTED()")
dy += bullets(cx0+P, dy, cw-P*2, [
    "Reads RANK / LOCAL_RANK / WORLD_SIZE env vars  →  detects multi-GPU vs single process",
    "Multi-GPU: init_process_group(backend='nccl')  ·  set_device(LOCAL_RANK)  ·  master = (rank==0)",
    "Single: auto-detect cuda → mps → cpu  ·  master = True",
    "torch.manual_seed(42) + cuda.manual_seed(42)  →  reproducible init across all ranks",
    "Dispatches: run_pretraining()  or  run_sft()  →  finally destroy_process_group()",
], size=11.5)

arrow_v(CX, y+ch, y+ch+GAP, "Trainer(sft_training, checkpoint_path, depth_override, …)")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2 — Trainer Init: 3 setup cards
# ══════════════════════════════════════════════════════════════════════════════
y = y + ch + GAP
ch = 390

# ── Card 1: setup_model ──────────────────────────────────────────────────────
rect(x1, y, W3, ch, C["blue"][0], C["blue"][1])
dy = y + P
dy += pill(x1+W3//2, dy, W3, "1️⃣  setup_model()")
dy += section(x1+P, dy, W3-P*2, "GPTCONFIG  (key defaults)")
dy += bullets(x1+P, dy, W3-P*2, [
    "block_size = 2048  ·  vocab_size = 50,266",
    "head_dim = 64  ·  window_pattern = 'SSSL'",
    "logit_softcap = 15.0  ·  weight_decay = 0.1",
    "grad_clip_norm = 1.0  ·  target_param_data_ratio = 10",
], size=11.5)
dy += section(x1+P, dy, W3-P*2, "DEPTH MODE  (__post_init__)")
dy += code_block(x1+P, dy, W3-P*2, [
    "n_embed = ceil(depth × 64 / 64) × 64",
    "n_layer = depth",
    "n_head  = n_embed / 64",
    "# e.g. depth=18 → 1152-dim, 18L, 18H",
])
hr(x1+P, dy+3, W3-P*2, C["blue"][1])
dy += 10
dy += section(x1+P, dy, W3-P*2, "GPT(CONFIG)")
dy += bullets(x1+P, dy, W3-P*2, [
    "Token embed (wte)  +  RoPE cos/sin buffers",
    "N × Block → CausalSelfAttention  (Flash Attn 3/2/SDPA)  + MLP",
    "Value embeds at alternating layers  ·  per-layer resid/x0 scalars",
    "RMSNorm (functional, no params)  ·  lm_head  ·  logit soft-cap",
    "cast_embeddings_to_bfloat16()  ·  torch.compile(dynamic=False)",
], size=11.5)

# ── Card 2: setup_hyperparameters ────────────────────────────────────────────
rect(x2, y, W3, ch, C["rose"][0], C["rose"][1])
dy = y + P
dy += pill(x2+W3//2, dy, W3, "2️⃣  setup_hyperparameters()")
dy += section(x2+P, dy, W3-P*2, "BASE LEARNING RATES")
dy += code_block(x2+P, dy, W3-P*2, [
    "embedding_lr   = 0.3",
    "matrix_lr      = 0.02",
    "scalar_lr      = 0.5",
    "unembedding_lr = 0.004",
    "adam β1=0.8  β2=0.95",
])
dy += section(x2+P, dy, W3-P*2, "BATCH SCALING  (Power Lines paper)")
dy += code_block(x2+P, dy, W3-P*2, [
    "B_ref = 524,288 tokens (2¹⁹)",
    "Bopt  = B_ref × (D / D_ref)^0.383",
    "η_scale = √(B / B_ref)",
    "λ_scale = √(B/B_ref) × (D_ref/D)",
    "τ_epoch = B/(n·D) = const  ← key invariant",
])
hr(x2+P, dy+3, W3-P*2, C["rose"][1])
dy += 10
dy += section(x2+P, dy, W3-P*2, "LR SCHEDULE  (pretrain)")
dy += code_block(x2+P, dy, W3-P*2, [
    "warmup  : step/warmup_iters  (ratio=0.0)",
    "plateau : lrm = 1.0",
    "cooldown: linear → 0  (ratio=0.4)",
    "SFT     : 1.0 − step/total_steps",
])
dy += section(x2+P, dy, W3-P*2, "DERIVED VALUES")
dy += bullets(x2+P, dy, W3-P*2, [
    "val_eval: max(100, steps/10)  cap=500",
    "CORE eval: max(250, steps/4)  cap=steps/2",
    "grad_accum = total_B / (B×T×world_size)",
], size=11.5)

# ── Card 3: setup_dataloaders + evaluators ───────────────────────────────────
rect(x3, y, W3, ch, C["teal"][0], C["teal"][1])
dy = y + P
dy += pill(x3+W3//2, dy, W3, "3️⃣  setup_dataloaders()  +  evaluators()")
sub(x3+P, dy, W3-P*2, 92)
ddy = dy+8
ddy += section(x3+P+8, ddy, W3-P*2-16, "PRETRAIN  —  FinewebEduParquetBOS")
bullets(x3+P+8, ddy, W3-P*2-16, [
    "Parquet files, sharded by ddp_rank",
    "BOS-aligned best-fit packing, zero-pad  ·  buffer=4096 docs",
    "persistent_workers=True  ·  prefetch",
], size=11)
dy += 98
sub(x3+P, dy, W3-P*2, 80)
ddy = dy+8
ddy += section(x3+P+8, ddy, W3-P*2-16, "SFT  —  MultiplexDataset")
bullets(x3+P+8, ddy, W3-P*2-16, [
    "ARC-Easy/Challenge · GSM8K · SmolTalk · SpellingBee",
    "sampling_strategy='proportional'",
    "collate_fn: pad_token_id, max_len=block_size",
    "workers=4, prefetch_factor=2",
], size=11)
dy += 86
sub(x3+P, dy, W3-P*2, 100)
ddy = dy+8
ddy += section(x3+P+8, ddy, W3-P*2-16, "EVALUATORS")
bullets(x3+P+8, ddy, W3-P*2-16, [
    "TrainingEvaluator  — val loss + BPB",
    "  val_loss_tokens = 10,485,760 (~10.5M)",
    "CoreEvaluator  — likelihood MC ranking",
    "ChatCoreEvaluator  — generative + tool-use",
], size=11)

arrow_v(CX, y+ch, y+ch+GAP)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3 — Optimizer
# ══════════════════════════════════════════════════════════════════════════════
y = y + ch + GAP
cw, ch = 680, 190
cx0 = CX - cw//2
rect(cx0, y, cw, ch, C["slate"][0], C["slate"][1])
dy = y + P
dy += pill(CX, dy, cw, "4️⃣  configure_optimizers()  —  GPT.configure_optimizers()")
sw = cw//2 - P - 10
sub(cx0+P, dy, sw, 120)
ddy = dy+8
ddy += section(cx0+P+8, ddy, sw-16, "MUON  (2D weight matrices)")
code_block(cx0+P+8, ddy, sw-16, [
    "All 2D transformer weights",
    "Nesterov momentum + orthogonalized update",
    "DistMuonAdamW  →  gradient sync across GPUs",
    "lr = matrix_lr × batch_lr_scale",
])
sub(cx0+cw//2+10, dy, sw, 120)
ddy = dy+8
ddy += section(cx0+cw//2+18, ddy, sw-16, "ADAMW  (all other params)")
code_block(cx0+cw//2+18, ddy, sw-16, [
    "Embeddings  (lr = embedding_lr × scale)",
    "lm_head     (lr = unembedding_lr × scale)",
    "Biases, scalars, norms  (lr = scalar_lr)",
    "β1=0.8  β2=0.95  weight_decay=0.1×wd_scale",
])

arrow_v(CX, y+ch, y+ch+GAP, "trainer.train()  →  torch.set_float32_matmul_precision('high')")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — Training Loop
# ══════════════════════════════════════════════════════════════════════════════
y = y + ch + GAP
cw, ch = WFULL, 330
cx0 = CX - cw//2
rect(cx0, y, cw, ch, C["green"][0], C["green"][1])
dy = y + P
dy += pill(CX, dy, cw, "🔁  Training Loop  —  for epoch in range(num_epochs):  for step in range(max_steps):")

# 5 step sub-boxes
sw = (cw - P*2 - 16) // 5
steps = [
    ("① ZERO GRADS", [
        "optimizer.zero_grad()  × 2",
        "loss_accumulator = 0",
        "num_active_tokens = 0",
    ]),
    ("② GRAD ACCUM", [
        "for micro_step in range(accum):",
        "  x,y = next(train_dataloader)",
        "  with autocast(bfloat16):",
        "    loss = model(x,y) / accum",
        "  loss.backward()",
    ]),
    ("③ DDP SYNC", [
        "all_reduce(loss, AVG)",
        "all_reduce(tokens, SUM)",
        "(only when ddp=True)",
        "Average across all ranks",
    ]),
    ("④ LR SCHEDULE", [
        "lrm = get_lr_multiplier(step)",
        "muon_mom = get_momentum(step)",
        "muon_wd  = get_weight_decay()",
        "clip_grad_norm_(model, 1.0)",
        "_update_learning_rates(lrm)",
    ]),
    ("⑤ OPTIMIZER", [
        "Muon.step()",
        "AdamW.step()",
        "global_step += 1",
        "ckpt every 5000 steps",
    ]),
]
for i, (title, items) in enumerate(steps):
    bx = cx0 + P + i * (sw + 4)
    sub(bx, dy, sw, 155)
    ddy = dy + 8
    ddy += section(bx+8, ddy, sw-16, title)
    code_block(bx+8, ddy, sw-16, items, size=10.5)

# triggers row
dy += 163
sub(cx0+P, dy, cw-P*2, 100)
ddy = dy + 8
ddy += section(cx0+P+8, ddy, cw-P*2-16, "⑥ PERIODIC TRIGGERS")
tw = (cw - P*2 - 32) // 4
trigger_items = [
    ("VALIDATION  (every run_evals_after steps)",
     ["est. val loss over ~10.5M tokens", "BPB on eval split", "sample_from_model() → log text"]),
    ("CORE BENCHMARK  (every core_evals_after steps)",
     ["MMLU · HellaSwag · ARC · LAMBADA", "likelihood-ranking, centered score", "core_score = mean(centered)"]),
    ("CHATCORE  (end of each epoch)",
     ["GSM8K · HumanEval · ARC · MMLU", "prefill + autoregressive decode", "tool-use: <|python|> calculator"]),
    ("CHECKPOINT  (every 5000 / 100 steps)",
     ["model + optimizer state_dicts", "epoch, step, global_step", "master_process only (rank 0)"]),
]
for i, (t, its) in enumerate(trigger_items):
    tx = cx0+P+8 + i*(tw+4)
    text(tx, ddy, tw, t, size=10, color="#475569", family=2)
    bullets(tx, ddy+16, tw, its, size=10.5)

arrow_v(CX, y+ch, y+ch+GAP)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 5 — Evaluations (3 cards)
# ══════════════════════════════════════════════════════════════════════════════
y = y + ch + GAP
ch = 350

eval_cards = [
    (x1, "📉  TrainingEvaluator", C["amber"], [
        ("estimate_validation_loss()", [
            ("code", [
                "model.eval()  ·  reset dataloader",
                "for batch in range(val_loss_steps):",
                "  _, loss = model(X, Y, loss_reduction='none')",
                "  accumulate_bpb(loss, valid_tokens)",
                "all_reduce(total_loss, SUM) across ranks",
                "val_loss = total_loss / total_valid_tokens",
                "bpb     = total_nats / (8 × total_bytes)",
                "model.train()  →  log to wandb",
            ]),
        ]),
        ("sample_from_model()", [
            ("bullets", [
                "Prefill context tokens → KV cache",
                "Autoregressively decode N sequences",
                "Uses KV cache → 5-10× faster",
                "Log to generation_log_file + wandb",
            ]),
        ]),
    ]),
    (x2, "🎯  CoreEvaluator", C["violet"], [
        ("tasks  (loaded from core.yaml)", [
            ("bullets", ["MMLU · HellaSwag · ARC-Easy/Challenge · LAMBADA · SQuAD"]),
        ]),
        ("evaluate_all_tasks()", [
            ("code", [
                "model.eval()",
                "for task in core_tasks:",
                "  for example in task.data:",
                "    losses = forward_model(choices)",
                "    pred = argmin(losses)  # likelihood",
                "  acc = correct / total",
                "  centered = (acc−rand)/(1−rand)",
                "core_score = mean(centered_results)",
                "log to wandb  ·  model.train()",
            ]),
        ]),
    ]),
    (x3, "💬  ChatCoreEvaluator", C["rose"], [
        ("tasks", [
            ("bullets", ["GSM8K (math)  ·  HumanEval (code)  ·  ARC  ·  MMLU"]),
        ]),
        ("generate_completion()", [
            ("code", [
                "# PREFILL phase",
                "kv_cache = KVCache(config)",
                "prefill_prompt(model, tokens, kv_cache)",
                "# DECODE loop",
                "while len(out) < max_tokens:",
                "  logits = forward(next_tok, kv_cache)",
                "  if detect <|python|>:",
                "    result = use_calculator(expr)",
                "    inject result tokens",
                "  next = sample(logits, temp, top_k)",
                "  if next == EOS: break",
                "all_reduce(correct, total) → accuracy",
            ]),
        ]),
    ]),
]

for xi, title, color, sections in eval_cards:
    rect(xi, y, W3, ch, color[0], color[1])
    dy = y + P
    dy += pill(xi+W3//2, dy, W3, title)
    for (sec_title, content_list) in sections:
        dy += section(xi+P, dy, W3-P*2, sec_title.upper())
        for (kind, items) in content_list:
            if kind == "code":
                dy += code_block(xi+P, dy, W3-P*2, items, size=10.5)
            else:
                dy += bullets(xi+P, dy, W3-P*2, items, size=11.5)
        hr(xi+P, dy+2, W3-P*2, color[1])
        dy += 8

arrow_v(CX, y+ch, y+ch+GAP)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 6 — Checkpoint + Logging
# ══════════════════════════════════════════════════════════════════════════════
y = y + ch + GAP
ch = 185
for xi, title, color, items in [
    (CX - W2 - 16, "💾  Checkpointing", C["indigo"], [
        "raw_model.state_dict()  — weights only (not compiled)",
        "optimizer[0].state_dict()  (Muon)",
        "optimizer[1].state_dict()  (AdamW)",
        "epoch, step, global_step  — for exact resume",
        "Scenario detection: resume_pretrain / rollover / resume_sft",
        "Interval: every 5000 steps (pretrain)  ·  100 steps (SFT)",
        "Only master_process (rank 0) writes to disk",
    ]),
    (CX + 16, "📊  Wandb Logging", C["indigo"], [
        "train_loss  ·  val_loss  ·  BPB (bits per byte)",
        "lr_multiplier  ·  gradient_norm  ·  MFU",
        "tokens_per_second  ·  flops_so_far",
        "core_score  ·  chatcore_score",
        "Per-task accuracies (MMLU, HellaSwag, …)",
        "Generated text samples",
        "Only master_process logs  ·  wandb.finish() at end",
    ]),
]:
    rect(xi, y, W2, ch, color[0], color[1])
    dy = y + P
    dy += pill(xi+W2//2, dy, W2, title)
    bullets(xi+P, dy, W2-P*2, items, size=11.5)

arrow_v(CX, y+ch, y+ch+GAP)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 7 — Done
# ══════════════════════════════════════════════════════════════════════════════
y = y + ch + GAP
cw, ch = 460, 105
cx0 = CX - cw//2
rect(cx0, y, cw, ch, C["green"][0], C["green"][1])
dy = y + P
dy += pill(CX, dy, cw, "✅  Training Complete")
bullets(cx0+P, dy, cw-P*2, [
    "destroy_process_group()  — teardown NCCL if DDP",
    "wandb.finish()  — flush all metrics",
    "Returns: final checkpoint path (str)",
], size=12)

# ── write ─────────────────────────────────────────────────────────────────────
doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {
        "gridSize": 20,
        "gridColor": {"Bold": "#C9C9C9", "Regular": "#EDEDED"},
        "viewBackgroundColor": "#f8fafc",
    },
    "files": {}
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_pipeline.excalidraw")
with open(out, "w") as f:
    json.dump(doc, f, indent=2)

print(f"✓  {len(elements)} elements → {out}")
