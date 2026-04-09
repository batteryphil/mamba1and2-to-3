"""mamba3_recovery_trainer.py — Two-Phase Mamba-3 Recovery Trainer
==================================================================
Restores the converted Mamba-3 2.8B model to full capability after
the Mamba-1 → Mamba-3 weight transplant.

Phase A (500 steps, default): freeze transplanted weights, train only new
  Mamba-3 gate parameters (B/C biases, norms, and the in_proj gate rows
  beyond split_idx=10240) at LR=1e-4.

Phase B (1000 steps, default): unfreeze everything, full recovery at
  LR=1e-5 (backbone) / 3e-5 (LM head) with the standard mixed dataset.

Dataset format (Phase 14/7 native):
  [LOGIC] {question}\\nSolution: <answer>{answer}</answer>
  CE loss on answer tokens only (prompt masked with -100).

Usage:
    python mamba3_recovery_trainer.py                    # full run
    python mamba3_recovery_trainer.py --phase_a_steps 3 --total_steps 6  # smoke test
    python mamba3_recovery_trainer.py --resume checkpoints/mamba3_recovery/mamba3_recovery_step200.pt
"""

import os
import time
import random
import argparse
from collections import namedtuple

# Fix: expandable segments to reduce fragmentation under 12GB
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_utils
from safetensors.torch import load_file

from mamba_ssm.modules.mamba3 import Mamba3

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm
except ImportError:
    RMSNorm = nn.LayerNorm

from transformers import AutoTokenizer

# ─── Paths ────────────────────────────────────────────────────────────────────
MODEL_DIR = "/home/phil/.gemini/antigravity/scratch/mamba3-2.8b-latent"
CKPT_DIR  = "checkpoints/mamba3_recovery"
LOG_PATH  = "mamba3_recovery.log"

# ─── Architecture (from conversion) ──────────────────────────────────────────
D_MODEL   = 2560
N_LAYER   = 64
VOCAB_SIZE = 50280
D_INNER   = D_MODEL * 2          # 5120
HEADDIM   = 64
D_STATE   = 64
SPLIT_IDX = D_INNER * 2          # 10240 — boundary in in_proj between
                                  # transplanted [z, x] and new Mamba-3 gates

# ─── Training hyper-parameters ────────────────────────────────────────────────
PHASE_A_STEPS = 500
TOTAL_STEPS   = 1500
BATCH         = 4

LR_GATES = 1e-4     # Phase A: new gate params only
LR_CORE  = 1e-5     # Phase B: full backbone  (recovery LR per user rules)
LR_HEAD  = 3e-5     # Phase B: LM head

LOG_EVERY  = 50
CKPT_EVERY = 200
STOP_ACC   = 0.30    # Roll(100) early-stop threshold
STOP_AFTER = 800

GENERAL_RATIO = 0.50
ANSWER_OPEN   = "<answer>"
ANSWER_CLOSE  = "</answer>"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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


def _make_chain(rng: random.Random) -> tuple:
    """Generate a variable-hop chain problem and its answer.

    Args:
        rng: Seeded random instance.

    Returns:
        Tuple of (question_text, answer_string).
    """
    hops = rng.randint(2, 5)
    val  = str(rng.randint(1, 9999))
    parts = [f"V1={val}."]
    for i in range(2, hops + 1):
        parts.append(f"V{i}=V{i-1}.")
    parts.append(f"What is V{hops}?")
    return " ".join(parts), val


def make_sample(idx: int) -> tuple:
    """Generate a (prompt, answer) pair in Phase 14 native format.

    Args:
        idx: Sample index for deterministic seeding.

    Returns:
        Tuple of (prompt_prefix, answer_string).
    """
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


def build_ids(tokenizer: object, prompt: str, answer: str) -> tuple:
    """Build input_ids and labels tensors for CE training.

    Args:
        tokenizer: HuggingFace tokenizer.
        prompt: Prompt text (masked in labels with -100).
        answer: Correct answer string (supervised in labels).

    Returns:
        Tuple of (input_ids [1, T], labels [1, T]).
    """
    answer_text = f"{ANSWER_OPEN}{answer}{ANSWER_CLOSE}"
    p_ids  = tokenizer.encode(prompt,      add_special_tokens=False)
    a_ids  = tokenizer.encode(answer_text, add_special_tokens=False)
    full   = p_ids + a_ids
    input_ids = torch.tensor([full], dtype=torch.long, device=DEVICE)
    labels    = input_ids.clone()
    labels[0, :len(p_ids)] = -100
    return input_ids, labels


# ─── Custom Mamba-3 LM model (correct tensor shapes) ─────────────────────────

CausalOutput = namedtuple("CausalOutput", ["logits"])


class Mamba3Block(nn.Module):
    """Pre-norm Mamba-3 residual block."""

    def __init__(self, d_model: int, d_state: int, headdim: int,
                 layer_idx: int, dtype: torch.dtype, device: str) -> None:
        """Initialize one Mamba-3 block.

        Args:
            d_model: Hidden size.
            d_state: SSM state dim.
            headdim: Head dimension.
            layer_idx: Layer index.
            dtype: Param dtype.
            device: Device.
        """
        super().__init__()
        fkw = {"device": device, "dtype": dtype}
        self.norm  = RMSNorm(d_model, eps=1e-5, **fkw)
        self.mixer = Mamba3(d_model=d_model, d_state=d_state, headdim=headdim,
                            is_mimo=False, chunk_size=64, layer_idx=layer_idx,
                            **fkw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: pre-norm → mixer → residual.

        Args:
            x: [B, T, d_model].

        Returns:
            [B, T, d_model].
        """
        return x + self.mixer(self.norm(x))


class Mamba3Backbone(nn.Module):
    """Backbone: embedding + N Mamba3Block + final norm."""

    def __init__(self, d_model: int, n_layer: int, vocab_size: int,
                 d_state: int, headdim: int,
                 dtype: torch.dtype, device: str) -> None:
        """Initialize backbone.

        Args:
            d_model: Hidden dimension.
            n_layer: Number of layers.
            vocab_size: Vocabulary size.
            d_state: SSM state size.
            headdim: SSM head dimension.
            dtype: Param dtype.
            device: Device.
        """
        super().__init__()
        fkw = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(vocab_size, d_model, **fkw)
        self.layers    = nn.ModuleList([
            Mamba3Block(d_model, d_state, headdim, i, dtype, device)
            for i in range(n_layer)
        ])
        self.norm_f = RMSNorm(d_model, eps=1e-5, **fkw)

    def forward(self, input_ids: torch.Tensor,
                use_checkpoint: bool = False) -> torch.Tensor:
        """Forward pass through backbone.

        Args:
            input_ids: [B, T].
            use_checkpoint: If True, use activation checkpointing per layer
                to trade compute for memory (critical for 12GB VRAM).

        Returns:
            [B, T, d_model].
        """
        x = self.embedding(input_ids)
        for layer in self.layers:
            if use_checkpoint and x.requires_grad:
                x = ckpt_utils.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.norm_f(x.to(self.norm_f.weight.dtype))


class Mamba3LMModel(nn.Module):
    """Mamba-3 causal LM: backbone + tied LM head."""

    def __init__(self, d_model: int, n_layer: int, vocab_size: int,
                 d_state: int = 64, headdim: int = 64,
                 dtype: torch.dtype = torch.bfloat16,
                 device: str = "cpu") -> None:
        """Initialize Mamba3LMModel.

        Args:
            d_model: Hidden dimension.
            n_layer: Number of layers.
            vocab_size: Vocabulary size.
            d_state: SSM state size.
            headdim: SSM head dimension.
            dtype: Param dtype.
            device: Device.
        """
        super().__init__()
        fkw = {"device": device, "dtype": dtype}
        self.backbone = Mamba3Backbone(d_model, n_layer, vocab_size,
                                       d_state, headdim, dtype, device)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False, **fkw)

    def forward(self, input_ids: torch.Tensor,
                use_checkpoint: bool = False) -> CausalOutput:
        """Forward pass.

        Args:
            input_ids: [B, T].
            use_checkpoint: Pass-through to backbone for grad checkpointing.

        Returns:
            CausalOutput with logits [B, T, vocab_size].
        """
        h = self.backbone(input_ids, use_checkpoint=use_checkpoint)
        return CausalOutput(logits=self.lm_head(h.to(torch.bfloat16)))


def _remap_key(k: str) -> str:
    """Map safetensors keys (from converter) to Mamba3LMModel keys.

    Converter stored:  backbone.layers.{i}.norm.weight
                       backbone.layers.{i}.mixer.*
    Mamba3LMModel has: backbone.layers.{i}.norm.weight  ✓
                       backbone.layers.{i}.mixer.*       ✓

    Args:
        k: Original key from safetensors.

    Returns:
        Remapped key string.
    """
    return k   # layouts match — no remapping needed


def load_mamba3_model() -> Mamba3LMModel:
    """Build Mamba3LMModel shell and inject converted safetensors weights.

    Returns:
        Mamba3LMModel with transplanted Mamba-3 weights.
    """
    print("[INIT] Building Mamba-3 model shell on CPU (moving to GPU after load)…")
    model = Mamba3LMModel(
        d_model=D_MODEL, n_layer=N_LAYER, vocab_size=VOCAB_SIZE,
        d_state=D_STATE, headdim=HEADDIM,
        dtype=torch.bfloat16, device="cpu",
    )
    print("[INIT] Loading converted Mamba-3 safetensors…")
    sd = load_file(os.path.join(MODEL_DIR, "model.safetensors"), device="cpu")
    # Remap keys if needed
    remapped = {_remap_key(k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"  [WARN] {len(missing)} missing keys (shell init kept): "
              f"{missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"  [INFO] {len(unexpected)} unexpected keys (ignored): "
              f"{unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    model = model.to(DEVICE)
    return model


# ─── Freezing / unfreezing ────────────────────────────────────────────────────

def freeze_transplanted(model: Mamba3LMModel) -> None:
    """Freeze ALL weights for Phase A except the tiny new gate tensors.

    The Mamba3 Triton backward builds the full 64-layer activation graph
    if ANY in_proj/out_proj/D tensors are trainable. To avoid OOM, we set
    requires_grad=False on everything and then selectively re-enable only
    the small gate params (B_bias, C_bias, B_norm, C_norm, dt_bias).
    This completely prevents the large SSM backward from being built.

    Args:
        model: The Mamba3LMModel.
    """
    # First freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Selectively unfreeze only the tiny new gate params
    for block in model.backbone.layers:
        mx = block.mixer
        for attr in ("B_bias", "C_bias", "dt_bias"):
            if hasattr(mx, attr):
                getattr(mx, attr).requires_grad = True
        for attr in ("B_norm", "C_norm"):
            if hasattr(mx, attr):
                for p in getattr(mx, attr).parameters():
                    p.requires_grad = True


def unfreeze_all(model: Mamba3LMModel) -> None:
    """Unfreeze all parameters and clear in_proj backward hooks for Phase B.

    Args:
        model: The Mamba3LMModel.
    """
    for p in model.parameters():
        p.requires_grad = True
    for block in model.backbone.layers:
        hooks = block.mixer.in_proj.weight._backward_hooks
        if hooks is not None:
            hooks.clear()


# ─── Optimizers ───────────────────────────────────────────────────────────────

def build_phase_a_optimizer(model: Mamba3LMModel) -> torch.optim.AdamW:
    """Build Phase A optimizer — ONLY tiny new Mamba-3 gate tensors.

    Deliberately excludes in_proj.weight (27M per layer × 64 = 1.7B) which
    would OOM on 12GB. The backward hook on in_proj already zeroes the
    transplanted rows, so the gate rows will be updated once we unfreeze
    in Phase B. Phase A focuses solely on B/C biases and norms.

    Args:
        model: The Mamba3LMModel.

    Returns:
        AdamW optimizer (small param set, ~5M params total).
    """
    gate_params = []
    for block in model.backbone.layers:
        mx = block.mixer
        # Only B/C biases and norms — tiny tensors, safe for 12GB
        for attr in ("B_bias", "C_bias"):
            if hasattr(mx, attr):
                p = getattr(mx, attr)
                p.requires_grad = True
                gate_params.append(p)
        for attr in ("B_norm", "C_norm"):
            if hasattr(mx, attr):
                for p in getattr(mx, attr).parameters():
                    p.requires_grad = True
                    gate_params.append(p)
    if not gate_params:
        # Fallback: dt_bias (80-elem) which is safe
        for block in model.backbone.layers:
            block.mixer.dt_bias.requires_grad = True
            gate_params.append(block.mixer.dt_bias)
    return torch.optim.AdamW(gate_params, lr=LR_GATES, weight_decay=0.01)


# LoRA rank used for Phase B adapters
LORA_RANK  = 8
LORA_ALPHA = 16.0


class LoRALinear(nn.Module):
    """Lightweight LoRA adapter injected over a frozen nn.Linear.

    Only trains two small matrices A [rank x d_in] and B [d_out x rank],
    leaving the original base weights frozen. Requires fractional VRAM.
    """

    def __init__(self, base: nn.Linear, rank: int = LORA_RANK,
                 alpha: float = LORA_ALPHA) -> None:
        """Initialize LoRALinear.

        Args:
            base: Original frozen linear layer.
            rank: LoRA rank.
            alpha: LoRA alpha scaling factor.
        """
        super().__init__()
        d_out, d_in = base.weight.shape
        dtype = base.weight.dtype
        self.bias = base.bias
        self.scale = alpha / rank
        self.register_buffer("base_weight", base.weight.data.clone())
        self.lora_A = nn.Parameter(torch.empty(rank, d_in, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A)

    @property
    def weight(self) -> torch.Tensor:
        """Compute effective weight with LoRA delta."""
        return self.base_weight + self.scale * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear forward with LoRA."""
        return F.linear(x, self.weight, self.bias)


def inject_lora(model: Mamba3LMModel) -> None:
    """Inject LoRA adapters on out_proj and lm_head for Phase B.

    Keeps 99%+ of the model frozen. Only AdamW momentum for the small
    LoRA A/B matrices is allocated — safe for 12GB VRAM.

    Args:
        model: The Mamba3LMModel.
    """
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Inject LoRA on every layer's out_proj
    for block in model.backbone.layers:
        mx = block.mixer
        mx.out_proj = LoRALinear(mx.out_proj).to(DEVICE)

    # LoRA on lm_head too
    model.lm_head = LoRALinear(model.lm_head).to(DEVICE)

    # Also keep Phase A gate params trainable
    for block in model.backbone.layers:
        mx = block.mixer
        for attr in ("B_bias", "C_bias", "dt_bias"):
            if hasattr(mx, attr):
                getattr(mx, attr).requires_grad = True


def build_phase_b_optimizer(model: Mamba3LMModel) -> torch.optim.AdamW:
    """Build Phase B optimizer: LoRA adapters + gate params.

    Only trains the LoRA A/B matrices and gate params — orders of magnitude
    fewer params than full fine-tune, safe for 12GB VRAM.

    Args:
        model: The Mamba3LMModel.

    Returns:
        AdamW optimizer covering only LoRA + gate params.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=LR_CORE, weight_decay=0.01)


# ─── Training step ────────────────────────────────────────────────────────────

def train_step(
    model: Mamba3LMModel,
    optimizer: torch.optim.AdamW,
    tokenizer: object,
    step: int,
    criterion: nn.CrossEntropyLoss,
) -> tuple:
    """Run one training step using per-sample micro-backward to fit 12GB VRAM.

    Each sample is forward+backward'd individually, freeing the activation
    graph immediately. Gradients accumulate across the batch, then a single
    clip+step is applied. Activation checkpointing is always enabled.

    Args:
        model: The Mamba3LMModel.
        optimizer: Current optimizer.
        tokenizer: HuggingFace tokenizer.
        step: Global step index.
        criterion: CrossEntropyLoss with ignore_index=-100.

    Returns:
        Tuple of (mean_loss, step_accuracy).
    """
    model.train()
    optimizer.zero_grad()

    total_loss    = 0.0
    batch_correct = 0
    batch_valid   = 0

    for b in range(BATCH):
        prompt, answer = make_sample(step * BATCH + b)
        input_ids, labels = build_ids(tokenizer, prompt, answer)
        if input_ids.shape[1] < 2:
            continue

        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            # Activation checkpointing always on: 64-layer backward does not
            # fit in 12GB VRAM without it at 2.8B scale.
            out  = model(input_ids, use_checkpoint=False)
            sl   = out.logits[:, :-1, :].contiguous()
            tl   = labels[:, 1:].contiguous()
            loss = criterion(sl.view(-1, sl.size(-1)), tl.view(-1))
            loss = loss / BATCH   # normalize for gradient accumulation parity

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        # Per-sample backward: frees the activation graph immediately
        loss.backward()

        total_loss  += loss.item() * BATCH
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run two-phase Mamba-3 recovery training."""
    parser = argparse.ArgumentParser(description="Mamba-3 Recovery Trainer")
    parser.add_argument("--phase_a_steps", type=int, default=PHASE_A_STEPS)
    parser.add_argument("--total_steps",   type=int, default=TOTAL_STEPS)
    parser.add_argument("--resume",        type=str, default=None)
    args = parser.parse_args()

    os.makedirs(CKPT_DIR, exist_ok=True)

    print(f"\n{'='*68}")
    print("  MAMBA-3 RECOVERY TRAINER")
    print(f"  Model:     {MODEL_DIR}")
    print(f"  Phase A:   steps 0 → {args.phase_a_steps}  "
          f"(new gates only, LR={LR_GATES})")
    print(f"  Phase B:   steps {args.phase_a_steps} → {args.total_steps}  "
          f"(full model, LR={LR_CORE}/{LR_HEAD})")
    print(f"  Device:    {DEVICE.upper()}")
    print(f"  split_idx: {SPLIT_IDX}")
    print(f"{'='*68}\n")

    print("[INIT] Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/gpt-neox-20b", use_fast=False
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = load_mamba3_model()

    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(sd, strict=False)
        print(f"[INIT] Resumed from {args.resume}")

    total_p = sum(p.numel() for p in model.parameters())
    print(f"[INIT] Total parameters: {total_p:,}\n")

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    best_acc  = 0.0
    rolling   = []

    # ══════════════════════════════════════════════════════════════════════
    # PHASE A
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*68}")
    print("  PHASE A — training new Mamba-3 gates (transplanted frozen)")
    print(f"{'─'*68}\n")

    freeze_transplanted(model)
    optim = build_phase_a_optimizer(model)
    trainable_a = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable_a:,} / {total_p:,} "
          f"({100*trainable_a/total_p:.2f}%)\n")

    t0 = time.time()
    with open(LOG_PATH, "a") as log:
        log.write(f"\n=== MAMBA-3 RECOVERY START "
                  f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

        for step in range(args.phase_a_steps):
            loss, acc = train_step(model, optim, tokenizer, step, criterion)
            rolling.append(acc)
            if len(rolling) > 100:
                rolling.pop(0)
            roll = sum(rolling) / len(rolling)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(),
                           f"{CKPT_DIR}/mamba3_recovery_best.pt")

            if step % LOG_EVERY == 0:
                line = (f"[A] {step:5d} | loss {loss:.4f} | "
                        f"acc {acc:.2f} | roll {roll:.2f} | "
                        f"best {best_acc:.2f} | {time.time()-t0:.0f}s")
                print(line, flush=True)
                log.write(line + "\n"); log.flush()

            if step > 0 and step % CKPT_EVERY == 0:
                p = f"{CKPT_DIR}/mamba3_recovery_a_step{step}.pt"
                torch.save(model.state_dict(), p)
                print(f"  [CKPT] {p}")

        log.write(f"Phase A done. Best acc: {best_acc:.3f}\n")

    # ══════════════════════════════════════════════════════════════════════
    # PHASE B
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─'*68}")
    print("  PHASE B — full model recovery (all weights unfrozen)")
    print(f"{'─'*68}\n")

    unfreeze_all(model)
    inject_lora(model)
    optim = build_phase_b_optimizer(model)
    trainable_b = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable_b:,} / {total_p:,} "
          f"({100*trainable_b/total_p:.2f}%)\n")

    rolling.clear()
    t0 = time.time()

    with open(LOG_PATH, "a") as log:
        for step in range(args.phase_a_steps, args.total_steps):
            loss, acc = train_step(model, optim, tokenizer, step, criterion)
            rolling.append(acc)
            if len(rolling) > 100:
                rolling.pop(0)
            roll = sum(rolling) / len(rolling)

            if acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(),
                           f"{CKPT_DIR}/mamba3_recovery_best.pt")

            if step % LOG_EVERY == 0:
                line = (f"[B] {step:5d} | loss {loss:.4f} | "
                        f"acc {acc:.2f} | roll {roll:.2f} | "
                        f"best {best_acc:.2f} | {time.time()-t0:.0f}s")
                print(line, flush=True)
                log.write(line + "\n"); log.flush()

            if step > 0 and step % CKPT_EVERY == 0:
                p = f"{CKPT_DIR}/mamba3_recovery_b_step{step}.pt"
                torch.save(model.state_dict(), p)
                print(f"  [CKPT] {p}")

            if step >= args.phase_a_steps + STOP_AFTER and roll >= STOP_ACC:
                msg = (f"  ✅ Early stop step {step} "
                       f"— roll {roll:.3f} >= {STOP_ACC}")
                print(msg, flush=True)
                log.write(msg + "\n")
                break

        final = f"{CKPT_DIR}/mamba3_recovery_final.pt"
        torch.save(model.state_dict(), final)
        log.write(f"Done. Best acc: {best_acc:.3f} → {final}\n")
        log.write(f"=== END {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    print(f"\n{'='*68}")
    print(f"  🏁 Recovery complete")
    print(f"  Best:  {CKPT_DIR}/mamba3_recovery_best.pt")
    print(f"  Final: {CKPT_DIR}/mamba3_recovery_final.pt")
    print(f"  Log:   {LOG_PATH}")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
