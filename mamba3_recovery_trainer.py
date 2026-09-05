"""mamba3_recovery_trainer.py — Two-Phase Mamba-3 Recovery Trainer
==================================================================
Restores capability after Mamba -> Mamba-3 conversion:

Phase A (default 500 steps):
  Freeze transplanted weights, train only new Mamba-3 gate parameters
  (B/C biases, norms, and dt_bias) at LR=1e-4.

Phase B (default 1000 steps):
  Inject non-destructive forward-hook LoRA adapters onto out_proj and lm_head,
  training adapters + gate parameters at LR=1e-5 to stay within 12GB VRAM
  without triggering C++/CUDA kernel AttributeError.

Usage:
    python mamba3_recovery_trainer.py --model_dir ./converted_model
    python mamba3_recovery_trainer.py --smoke_test
    python mamba3_recovery_trainer.py --resume checkpoints/mamba3_recovery/mamba3_recovery_best.pt
"""

import os
import time
import json
import math
import random
import argparse
import contextlib
from typing import Optional, Tuple

# Expandable segments to reduce memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer

from mamba3_engine import Mamba3LMModel, RMSNorm, CausalOutput

# ─── Training defaults ────────────────────────────────────────────────────────
PHASE_A_STEPS = 500
TOTAL_STEPS   = 1500
BATCH         = 4

LR_GATES = 1e-4     # Phase A: new gate params only
LR_CORE  = 1e-5     # Phase B: LoRA adapters + gates
LR_HEAD  = 3e-5     # Phase B: LM head adapter

LOG_EVERY  = 50
CKPT_EVERY = 200
STOP_ACC   = 0.30
STOP_AFTER = 800

LORA_RANK  = 8
LORA_ALPHA = 16.0

GENERAL_RATIO = 0.50
ANSWER_OPEN   = "<answer>"
ANSWER_CLOSE  = "</answer>"

# ─── Dataset ─────────────────────────────────────────────────────────────────
_MC = [
    ("What color is the sky on a clear day?",           "Blue"),
    ("How many sides does a triangle have?",             "3"),
    ("Which planet is closest to the sun?",              "Mercury"),
    ("What is 2 plus 2?",                                "4"),
    ("What is the largest ocean on Earth?",              "Pacific"),
    ("At what temperature in Celsius does water boil?",  "100"),
    ("How many days are in a week?",                     "7"),
    ("What is the capital of France?",                   "Paris"),
    ("What is the chemical formula for water?",          "H2O"),
    ("How many continents are on Earth?",                "7"),
    ("What is the square root of 16?",                   "4"),
    ("How many seconds are in one minute?",              "60"),
    ("What is the largest planet in our solar system?",  "Jupiter"),
    ("How many legs does a spider have?",                "8"),
    ("What is the capital of Japan?",                    "Tokyo"),
]
_TF = [
    ("The sun is a star.",                               "True"),
    ("Fish can breathe underwater.",                     "True"),
    ("The Earth is flat.",                               "False"),
    ("Ice is hotter than steam.",                        "False"),
    ("The moon produces its own light.",                 "False"),
    ("Gold is a metal.",                                 "True"),
    ("Sound travels faster than light.",                 "False"),
    ("Dogs are mammals.",                                "True"),
    ("Penguins live at the North Pole.",                 "False"),
    ("The heart pumps blood.",                           "True"),
]
_FILL = [
    ("The capital of Japan is ___.",                     "Tokyo"),
    ("Water freezes at ___ degrees Celsius.",            "0"),
    ("A dozen equals ___ items.",                        "12"),
    ("A triangle has ___ sides.",                        "3"),
    ("One minute has ___ seconds.",                      "60"),
    ("The square root of 9 is ___.",                     "3"),
]
_QA = [
    ("What is 7 plus 5?",                                "12"),
    ("How many legs does a spider have?",                "8"),
    ("What is 4 times 4?",                               "16"),
    ("How many hours are in a day?",                     "24"),
    ("What is 10 minus 3?",                              "7"),
    ("What is the first letter of the alphabet?",        "A"),
]


def _make_chain(rng: random.Random) -> Tuple[str, str]:
    hops = rng.randint(2, 5)
    val  = str(rng.randint(1, 9999))
    parts = [f"V1={val}."]
    for i in range(2, hops + 1):
        parts.append(f"V{i}=V{i-1}.")
    parts.append(f"What is V{hops}?")
    return " ".join(parts), val


def make_sample(idx: int) -> Tuple[str, str]:
    rng = random.Random(idx * 31337 + 7)
    if rng.random() < GENERAL_RATIO:
        fmt = rng.randint(0, 3)
        if fmt == 0:
            q, a = rng.choice(_MC)
            return f"[LOGIC] {q}\nSolution: ", a
        elif fmt == 1:
            stmt, a = rng.choice(_TF)
            return f"[LOGIC] True or False: {stmt}\nSolution: ", a
        elif fmt == 2:
            tmpl, a = rng.choice(_FILL)
            return f"[LOGIC] Complete the following: {tmpl}\nSolution: ", a
        else:
            q, a = rng.choice(_QA)
            return f"[LOGIC] {q}\nSolution: ", a
    else:
        q, a = _make_chain(rng)
        return f"[LOGIC] {q}\nSolution: ", a


def build_ids(tokenizer: object, prompt: str, answer: str, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    answer_text = f"{ANSWER_OPEN}{answer}{ANSWER_CLOSE}"
    p_ids = tokenizer.encode(prompt, add_special_tokens=False)
    a_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    full = p_ids + a_ids
    input_ids = torch.tensor([full], dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels[0, :len(p_ids)] = -100
    return input_ids, labels


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_mamba3_model(
    model_dir: str,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
    smoke_test: bool = False,
) -> Tuple[Mamba3LMModel, dict]:
    """Load Mamba-3 model dynamically from model_dir config and weights."""
    cfg_path = os.path.join(model_dir, "config.json")
    config = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            config = json.load(f)

    d_model = config.get("hidden_size", config.get("d_model", 1024 if smoke_test else 2560))
    n_layer = config.get("num_hidden_layers", config.get("n_layer", 2 if smoke_test else 48))
    vocab_size = config.get("vocab_size", 50288)
    headdim = config.get("head_dim", 64)
    d_state = config.get("state_size", config.get("d_state", 128 if smoke_test else 64))

    print(f"[INIT] Instantiating Mamba3LMModel (d_model={d_model}, n_layer={n_layer}, vocab_size={vocab_size})...")
    model = Mamba3LMModel(
        d_model=d_model,
        n_layer=n_layer,
        vocab_size=vocab_size,
        d_state=d_state,
        headdim=headdim,
        dtype=dtype,
        device=device,
    )

    sf_path = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(sf_path):
        print(f"[INIT] Loading weights from {sf_path}...")
        sd = load_file(sf_path, device=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  [WARN] {len(missing)} missing keys in checkpoint.")
        if unexpected:
            print(f"  [INFO] {len(unexpected)} unexpected keys ignored.")
    else:
        if not smoke_test:
            raise FileNotFoundError(f"Checkpoint not found at {sf_path}. Please check --model_dir.")
        print("  [SMOKE_TEST] Initialized random shell weights for verification.")

    model = model.to(device)
    return model, config


# ─── Freezing and Forward-Hook LoRA ───────────────────────────────────────────

def freeze_transplanted(model: Mamba3LMModel) -> None:
    """Freeze all base parameters except new Mamba-3 gates for Phase A."""
    for p in model.parameters():
        p.requires_grad = False

    for block in model.backbone.layers:
        mx = block.mixer
        for attr in ("B_bias", "C_bias", "dt_bias"):
            if hasattr(mx, attr):
                getattr(mx, attr).requires_grad = True
        for attr in ("B_norm", "C_norm"):
            if hasattr(mx, attr):
                for p in getattr(mx, attr).parameters():
                    p.requires_grad = True


def _make_lora_hook(lora_A: nn.Parameter, lora_B: nn.Parameter, scale: float):
    """Non-destructive LoRA forward hook leaving base nn.Linear untouched."""
    def _hook(module, inputs, output):
        # inputs[0]: [batch, seq_len, in_features]
        delta = F.linear(inputs[0], lora_A) @ lora_B.t() * scale
        return output + delta
    return _hook


def inject_lora(model: Mamba3LMModel, rank: int = LORA_RANK, alpha: float = LORA_ALPHA) -> None:
    """Register forward-hook LoRA on out_proj and lm_head for Phase B.
    
    Prevents C++/CUDA AttributeError: 'LoRALinear' object has no attribute 'weight'.
    """
    for p in model.parameters():
        p.requires_grad = False

    scale = alpha / rank
    dtype = model.lm_head.weight.dtype
    device = model.lm_head.weight.device

    for block in model.backbone.layers:
        mx = block.mixer
        d_out, d_in = mx.out_proj.weight.shape
        mx.lora_A = nn.Parameter(torch.empty(rank, d_in, device=device, dtype=dtype))
        mx.lora_B = nn.Parameter(torch.zeros(d_out, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(mx.lora_A, a=math.sqrt(5))
        mx.out_proj.register_forward_hook(_make_lora_hook(mx.lora_A, mx.lora_B, scale))

    d_out, d_in = model.lm_head.weight.shape
    model.lora_A = nn.Parameter(torch.empty(rank, d_in, device=device, dtype=dtype))
    model.lora_B = nn.Parameter(torch.zeros(d_out, rank, device=device, dtype=dtype))
    nn.init.kaiming_uniform_(model.lora_A, a=math.sqrt(5))
    model.lm_head.register_forward_hook(_make_lora_hook(model.lora_A, model.lora_B, scale))

    # Keep Phase A gate parameters trainable alongside LoRA
    for block in model.backbone.layers:
        mx = block.mixer
        for attr in ("B_bias", "C_bias", "dt_bias"):
            if hasattr(mx, attr):
                getattr(mx, attr).requires_grad = True
        for attr in ("B_norm", "C_norm"):
            if hasattr(mx, attr):
                for p in getattr(mx, attr).parameters():
                    p.requires_grad = True


def build_phase_a_optimizer(model: Mamba3LMModel) -> torch.optim.AdamW:
    gate_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(gate_params, lr=LR_GATES, weight_decay=0.01)


def build_phase_b_optimizer(model: Mamba3LMModel) -> torch.optim.AdamW:
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=LR_CORE, weight_decay=0.01)


# ─── Training Step ────────────────────────────────────────────────────────────

def train_step(
    model: Mamba3LMModel,
    optimizer: torch.optim.AdamW,
    tokenizer: object,
    step: int,
    criterion: nn.CrossEntropyLoss,
    device: str,
    batch_size: int = BATCH,
) -> Tuple[float, float]:
    """Execute one training step with per-sample micro-backward."""
    model.train()
    optimizer.zero_grad()

    total_loss = 0.0
    batch_correct = 0
    batch_valid = 0

    for b in range(batch_size):
        prompt, answer = make_sample(step * batch_size + b)
        input_ids, labels = build_ids(tokenizer, prompt, answer, device)
        if input_ids.shape[1] < 2:
            continue

        use_autocast = device != "cpu" and torch.cuda.is_available()
        autocast_ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if use_autocast else contextlib.nullcontext()

        with autocast_ctx:
            out = model(input_ids)
            sl = out.logits[:, :-1, :].contiguous()
            tl = labels[:, 1:].contiguous()
            loss = criterion(sl.view(-1, sl.size(-1)), tl.view(-1))
            loss = loss / batch_size

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        total_loss += loss.item() * batch_size
        batch_valid += 1

        mask_pos = (tl[0] != -100).nonzero(as_tuple=True)[0]
        if len(mask_pos) > 0:
            tag_len = len(tokenizer.encode(ANSWER_OPEN, add_special_tokens=False))
            ans_pos = mask_pos[0].item() + tag_len
            if ans_pos < sl.shape[1]:
                pred_tok = sl[0, ans_pos].detach().argmax().item()
                ans_toks = tokenizer.encode(answer, add_special_tokens=False)
                if ans_toks and pred_tok == ans_toks[0]:
                    batch_correct += 1

    if batch_valid == 0:
        return 0.0, 0.0

    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad and p.grad is not None],
        max_norm=1.0,
    )
    optimizer.step()
    return total_loss / batch_valid, batch_correct / batch_valid


# ─── Main Routine ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Mamba-3 Recovery Trainer")
    parser.add_argument("--model_dir", type=str, default="./converted_mamba3",
                        help="Directory containing converted Mamba-3 checkpoint")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/mamba3_recovery",
                        help="Output directory for recovery checkpoints")
    parser.add_argument("--log_path", type=str, default="mamba3_recovery.log",
                        help="Path to training log file")
    parser.add_argument("--phase_a_steps", type=int, default=PHASE_A_STEPS)
    parser.add_argument("--total_steps", type=int, default=TOTAL_STEPS)
    parser.add_argument("--device", type=str, default=None,
                        help="Compute device: 'cuda', 'rocm', or 'cpu'")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run rapid smoke test (2 steps Phase A, 2 steps Phase B)")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if args.smoke_test:
        args.phase_a_steps = 2
        args.total_steps = 4

    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"\n{'='*68}")
    print("  MAMBA-3 RECOVERY TRAINER")
    print(f"  Model Dir: {args.model_dir}")
    print(f"  Phase A:   steps 0 → {args.phase_a_steps} (new gates only, LR={LR_GATES})")
    print(f"  Phase B:   steps {args.phase_a_steps} → {args.total_steps} (LoRA + gates, LR={LR_CORE})")
    print(f"  Device:    {device.upper()}")
    print(f"{'='*68}\n")

    print("[INIT] Loading tokenizer…")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b", use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model, _ = load_mamba3_model(args.model_dir, device=device, dtype=dtype, smoke_test=args.smoke_test)

    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location=device)
        model.load_state_dict(sd, strict=False)
        print(f"[INIT] Resumed from {args.resume}")

    total_p = sum(p.numel() for p in model.parameters())
    print(f"[INIT] Total parameters: {total_p:,}\n")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    best_acc = 0.0
    rolling = []

    # ══════════════════════════════════════════════════════════════════════
    # PHASE A
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*68}")
    print("  PHASE A — training new Mamba-3 gates (transplanted frozen)")
    print(f"{'─'*68}\n")

    freeze_transplanted(model)
    optim_a = build_phase_a_optimizer(model)
    trainable_a = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable_a:,} / {total_p:,} ({100*trainable_a/total_p:.2f}%)\n")

    t0 = time.time()
    for step in range(args.phase_a_steps):
        loss, acc = train_step(model, optim_a, tokenizer, step, criterion, device)
        rolling.append(acc)
        if len(rolling) > 100:
            rolling.pop(0)
        roll = sum(rolling) / len(rolling)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{args.ckpt_dir}/mamba3_recovery_best.pt")

        if step % LOG_EVERY == 0 or args.smoke_test:
            print(f"[A] {step:5d} | loss {loss:.4f} | acc {acc:.2f} | roll {roll:.2f} | {time.time()-t0:.1f}s", flush=True)

    # ══════════════════════════════════════════════════════════════════════
    # PHASE B
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*68}")
    print("  PHASE B — full recovery (LoRA forward-hook adapters + gates)")
    print(f"{'─'*68}\n")

    inject_lora(model)
    optim_b = build_phase_b_optimizer(model)
    trainable_b = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable_b:,} / {total_p:,} ({100*trainable_b/total_p:.2f}%)\n")

    t0 = time.time()
    for step in range(args.phase_a_steps, args.total_steps):
        loss, acc = train_step(model, optim_b, tokenizer, step, criterion, device)
        rolling.append(acc)
        if len(rolling) > 100:
            rolling.pop(0)
        roll = sum(rolling) / len(rolling)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{args.ckpt_dir}/mamba3_recovery_best.pt")

        if step % LOG_EVERY == 0 or args.smoke_test:
            print(f"[B] {step:5d} | loss {loss:.4f} | acc {acc:.2f} | roll {roll:.2f} | {time.time()-t0:.1f}s", flush=True)

    final_ckpt = f"{args.ckpt_dir}/mamba3_recovery_final.pt"
    torch.save(model.state_dict(), final_ckpt)
    print(f"\n{'='*68}")
    print(f"  🏁 Recovery complete! Final checkpoint: {final_ckpt}")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
