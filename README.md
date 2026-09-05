# Mamba 1 & 2 → Mamba 3 Architectural Conversion Guide

This repository contains the architecture engine, weight converters, recovery trainers, and **empirical findings** for transplanting weights from Mamba-1 and Mamba-2 checkpoints into Mamba-3 SISO.

---

## 🎯 Architectural Diagnosis & Core Findings

### 1. Why the Initial Mamba-1 → Mamba-3 Transplant Collapsed
In early experiments, converting a 2.8B parameter Mamba-1 checkpoint resulted in a "structurally valid but linguistically dead model" with Cross-Entropy (CE) loss stuck at 8–10 and garbled output tokens.

**The Mathematical Root Cause:**
- Mamba-1 computes recurrence via a 1D convolution (`conv1d`), input-dependent projection (`x_proj` creating $B, C, \Delta$ dynamically), and a 2D continuous parameter matrix $A \in \mathbb{R}^{D \times 16}$.
- Attempting to transplant Mamba-1 directly into Mamba-3 dropped `conv1d`, `A_log`, and `x_proj`.
- Dropping these tensors stripped away **100% of the recurrent memory dynamics**. The 64 intermediate SSM layers became randomly initialized. Passing token representations through 64 random recurrent transformations completely destroyed the latent representation space, which surface fine-tuning (LoRA or gate training) could not recover.

### 2. The Solution: Mamba-2 → Mamba-3 via State Space Duality (SSD)
Mamba-2 and Mamba-3 share the same underlying mathematical formulation: **State Space Duality (SSD)**.
- Both use multi-head scalar decay parameters $A \in \mathbb{R}^{H}$ (`A_log`).
- Both use head-partitioned $B$ and $C$ matrices.
- Both compute sequential recurrence using chunked semi-separable matrix multiplications $(M \circ (CB^T))X$.
- Both use unified projections for $(z, x, B, C, \Delta)$.

By converting from **Mamba-2 to Mamba-3**, 100% of the SSD parameters (`in_proj`, `conv1d`, `A_log`, `dt_bias`, `D`, `norm`, `out_proj`) are mapped directly. The new Mamba-3 gating parameters ($B_{\text{bias}}, C_{\text{bias}}, B_{\text{norm}}, C_{\text{norm}}$) are initialized to identity/zero. At step 0, the converted Mamba-3 model retains the original language representations and fluency without representation collapse.

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/batteryphil/mamba1and2-to-3.git
cd mamba1and2-to-3
pip install -r requirements.txt
```

### 1. Converting a Checkpoint to Mamba-3
The converter automatically detects whether the input checkpoint is Mamba-1 or Mamba-2, extracts weights, remaps parameters, and outputs a complete Mamba-3 checkpoint with `config.json` and tokenizer files.

```bash
# Convert a Mamba-2 checkpoint (Recommended — preserves full SSD manifold)
python mamba_to_mamba3_converter.py \
  --input /path/to/mamba2_model \
  --output ./converted_mamba3 \
  --model_type mamba2 \
  --precision bf16

# Convert a Mamba-1 checkpoint (Legacy — with diagnostic warnings)
python mamba_to_mamba3_converter.py \
  --input /path/to/mamba1_model \
  --output ./converted_mamba3 \
  --model_type mamba1
```

### 2. Recovery Training (Two-Phase Pipeline)
Run the two-phase recovery trainer to tune the new Mamba-3 gating parameters and apply parameter-efficient LoRA adapters:

```bash
# Full training run
python mamba3_recovery_trainer.py --model_dir ./converted_mamba3

# Fast smoke test on CPU/GPU
python mamba3_recovery_trainer.py --smoke_test --device cpu
```

### 3. Automated Test Suite
Verify that the engine, forward-hook LoRA, converter, and training steps are functioning correctly:
```bash
python -m unittest test_pipeline.py
```

---

## 🛠️ Architecture & 12GB VRAM Optimization Techniques

### 1. Zero-Dependency Pure-PyTorch SSD Engine (`mamba3_engine.py`)
No proprietary C++/CUDA extensions are required. `mamba3_engine.py` implements chunked semi-separable matrix multiplication directly in PyTorch, executing in linear $O(L)$ time on ROCm, CUDA, and CPU.

### 2. Forward-Hook LoRA (Avoids CUDA Kernel Crashes)
> ⚠️ **Do NOT replace `mixer.out_proj` with a standard `LoRALinear` wrapper.**

When low-level kernels or dispatch mechanisms inspect `out_proj.weight`, replacing the layer with a Python wrapper leads to:
```
AttributeError: 'LoRALinear' object has no attribute 'weight'
```

**Working Fix:** The recovery trainer uses non-destructive forward hooks:
```python
def _make_lora_hook(lora_A, lora_B, scale):
    def _hook(module, inputs, output):
        delta = F.linear(inputs[0], lora_A) @ lora_B.t() * scale
        return output + delta
    return _hook

mixer.out_proj.register_forward_hook(_make_lora_hook(lora_A, lora_B, scale))
```
`out_proj` remains a genuine `nn.Linear`, while gradients cleanly propagate into `lora_A` and `lora_B`.

### 3. Per-Sample Micro-Backward
To prevent activation graphs from exhausting 12GB VRAM on larger models, gradients accumulate per sample:
```python
for sample in batch:
    loss = criterion(model(sample)) / BATCH
    loss.backward()   # activation graph freed immediately
optimizer.step()
```

### 4. Two-Phase Training Schedule
| Phase | Trainable Parameters | Description |
|---|---|---|
| **Phase A** (Warmup) | `B_bias`, `C_bias`, `dt_bias`, `B_norm`, `C_norm` (~835K) | Adapts new Mamba-3 gating mechanisms while base SSD representations remain frozen. |
| **Phase B** (LoRA) | + Hook LoRA on `out_proj` and `lm_head` (~5M) | Tunes projection layers via lightweight low-rank adapters. |

---

## 📁 Repository File Reference

| File | Description |
|---|---|
| `mamba3_engine.py` | Standalone, pure-PyTorch Mamba-3 SISO engine with chunked SSD recurrence. |
| `mamba_to_mamba3_converter.py` | Unified Mamba-1 & Mamba-2 to Mamba-3 converter with auto-detection. |
| `mamba1_to_mamba3_converter.py` | Legacy Mamba-1 converter preserved for backward compatibility. |
| `mamba3_recovery_trainer.py` | Two-phase trainer with forward-hook LoRA and micro-backward accumulation. |
| `test_pipeline.py` | Automated unit test suite verifying forward/backward passes, LoRA, and conversion. |
| `requirements.txt` | Core package dependencies. |
| `.gitignore` | Excludes checkpoints, safetensors, cache files, and logs. |

---

## 📜 Chronology of Findings

| Milestone | Finding |
|---|---|
| **Initial** | Mamba-1 converter produced structurally valid tensors with 0 missing keys. |
| **Phase A/B** | Phase A gate warmup reduced loss, but Phase B layer-wrapper LoRA crashed with `AttributeError`. |
| **Hook LoRA** | Forward-hook LoRA resolved kernel crashes, but Mamba-1 checkpoint CE plateaued at 8–10. |
| **Diagnosis** | Root cause identified: Mamba-1 non-SSD recurrence was discarded during transplant, leaving the state manifold randomized. |
| **Resolution** | Mamba-2 $\to$ Mamba-3 SSD conversion implemented. Full state space parameters ($A, B, C, X, Z$) are preserved, maintaining language coherence from step 0. |
