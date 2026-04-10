# Mamba 1 & 2 → Mamba 3 Architectural Conversion Guide

This repository documents the methodology, scripts, and **hard-won lessons** for transplanting weights from Mamba-1/Mamba-2 architectures into Mamba-3. It is a field guide — including what works, what fails, and the definitive recommendation for practitioners.

---

## ⚠️ Critical Finding — Updated April 2026

> **The weight transplant produces a structurally valid but linguistically dead model.**

After extensive testing across multiple recovery strategies on a 2.8B parameter model, the conclusion is:

**The Mamba-2 → Mamba-3 weight transplant cannot be recovered through fine-tuning alone.**

The root cause is not the mathematical mismatches (which the converter handles correctly). The problem is deeper: the Mamba-3 SSM's internal state evolution kernels (A, B, C matrices + gate routing) operate in a fundamentally different representation space than Mamba-2. Even after correct weight remapping, the hidden state manifold is incompatible. The model outputs coherent-looking tensors but generates incoherent token sequences (repetition loops, medical/scientific jargon tokens, unicode garbage).

**Recovery training — even with 30,000+ real samples over 5,000 steps with LoRA — does not fix this.** The CE loss plateaus around 8–10 and the output probes never produce real English.

### The correct path forward is to **start from the stock `state-spaces/mamba-2.8b-slimpj` base directly**, without conversion.

---

## The Conversion — What It Does (and What It Cannot Fix)

### 1. `[x, z]` → `[z, x]` Sequence Inversion

Mamba-1's `in_proj` splits into `[x, z]` (main branch, then gate). Mamba-3 expects `[z, x]`. Blind-copying produces a physically reversed network.

**Fix:** Slice `in_proj` weight at `d_inner`, swap upper and lower halves before injection.

```python
w_top, w_bot = weight[:d_inner], weight[d_inner:]
weight_remapped = torch.cat([w_bot, w_top], dim=0)
```

### 2. Dimensionality Collapse (`dt_bias`, `D`)

Mamba-1 scales `D` and `dt_bias` across full sequence length. Mamba-3 pools these into `nheads` header groups.

**Fix:** Average-pool from sequence-length dimension to `nheads`:
```python
dt_bias_remapped = dt_bias.view(nheads, -1).mean(dim=1)
```

### 3. Inverse-Softplus Reparameterization

Mamba-3 kernel variables pass through Triton softplus. Raw bias values require inverse mapping to preserve scale equivalence.

**Fix:**
```python
dt_bias_remapped = torch.log(torch.exp(dt_bias_pooled) - 1.0)
```

---

## 12GB VRAM Techniques

These remain valid for any Mamba-3 fine-tuning regardless of conversion.

### Per-Sample Micro-Backward
Instead of accumulating a full batch graph:
```python
for sample in batch:
    loss = criterion(model(sample)) / BATCH
    loss.backward()   # graph freed immediately per sample
clip_grad_norm_(params, 1.0)
optimizer.step()
```

### LoRA via Forward Hooks (Not Layer Replacement)

> **Do NOT replace `mixer.out_proj` with a LoRALinear wrapper.**

The Mamba CUDA/Triton kernel accesses `out_proj.weight` as a raw C++ buffer. Replacing the layer with a Python wrapper causes:
```
AttributeError: 'LoRALinear' object has no attribute 'weight'
```
at the C++ dispatch level, even if `.weight` is a Python property.

**Correct approach — use a post-forward hook:**
```python
def _make_lora_hook(lora_A, lora_B, scale):
    def _hook(module, inputs, output):
        delta = F.linear(inputs[0], lora_A) @ lora_B.t() * scale
        return output + delta
    return _hook

mixer.out_proj.register_forward_hook(
    _make_lora_hook(lora_A, lora_B, alpha / rank)
)
```
Store `lora_A` / `lora_B` as `nn.Parameter` directly on the mixer object. This keeps the CUDA kernel path intact while adding LoRA gradient signal.

### Selective Freezing Strategy

Train only what is necessary to stay within 12GB VRAM:

| Phase | Trainable | Approx params |
|---|---|---|
| Warmup | `B_bias`, `C_bias`, `dt_bias`, norms | ~835K |
| LoRA | + hook LoRA on `out_proj` | ~5M |
| Deep tune | + `embedding`, `lm_head` | ~260M |

Unfreezing `embedding` and `lm_head` **from the start** is critical if the model's language output is broken — not delayed until Phase C.

---

## Recovery Training — What Fails and Why

### Symptom: CE plateau at 8–10, zero coherent output

If you see this pattern after transplant:
```
[A]   150/5000 | CE: 18.01 | acc: 0.00
[B]   500/5000 | CE: 14.44 | acc: 0.00
[B]  1500/5000 | CE:  8.11 | acc: 0.00
```
...and probes still show `LRQLRQ nanomaterials encoun encoun` — the SSM state space is fundamentally misaligned. More steps will not fix it.

### Why LoRA on `out_proj` alone fails

Training the output projection cannot fix broken intermediate SSM states. The A/B/C matrices determine what the hidden state *represents*. If those representations are randomized by transplant, no output remapping can decode them.

### Why training `embedding` + `lm_head` also fails

Even with the full input/output path trainable, the 64-layer SSM middle is a black box operating in the wrong representation space. Gradients from the CE loss cannot propagate deep enough through 64 frozen Mamba-3 layers to fix the internal dynamics.

### What would theoretically work (but is impractical)

Full fine-tuning of all weights on a large-scale language corpus (100B+ tokens). This is equivalent to training from scratch — at which point the transplant provided no benefit.

---

## Definitive Recommendation

| Goal | Recommended approach |
|---|---|
| Use Mamba-3 with intact language | Fine-tune `state-spaces/mamba-2.8b-slimpj` (Mamba-2) with LoRA — no conversion |
| Prototype Mamba-3 architecture | Train a small Mamba-3 from scratch on domain data |
| Preserve specific fine-tuned behavior | Export and freeze only the layers that differ; retrain the rest |

The conversion scripts in this repository remain useful for understanding the architectural delta between generations and for cold-start initialization experiments, but **should not be relied upon as a production upgrade path**.

---

## File Reference

| File | Purpose |
|---|---|
| `mamba1_to_mamba3_converter.py` | Weight transplant: handles `[x,z]` inversion, dt_bias pooling, inverse-softplus |
| `mamba3_recovery_trainer.py` | Two-phase recovery trainer (Phase A: gate warmup, Phase B: LoRA). See caveats above. |

---

## Lessons Learned Chronology

| Date | Finding |
|---|---|
| Initial | Converter produces structurally valid checkpoint — 0 missing keys |
| Week 1 | Phase A gate warmup reduces CE rapidly on small QA pools |
| Week 2 | Phase B LoRA on `out_proj` crashes with `AttributeError` — CUDA kernel issue |
| Week 2 | Hook-based LoRA fix resolves crash but CE plateaus at 8–10 |
| Week 3 | 30K-sample dataset tried; CE drops faster but probes still incoherent |
| Week 3 | Embedding + lm_head unlocked from step 0 — marginal improvement only |
| Week 3 | **Root cause identified**: SSM state manifold is fundamentally incompatible post-transplant regardless of surface fine-tuning |
| Week 3 | **Resolution**: Abandoned transplant lineage; stock Mamba-2 base used directly — immediate language coherence, CE ~2.3 from step 0 |
