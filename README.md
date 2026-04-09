# Mamba 1 & 2 to Mamba 3 Architectural Upgrade

This repository contains the methodology and scripts to bypass training from scratch by structurally transplanting weights from the Mamba-1/Mamba-2 architectures directly into Mamba-3 gates. 

It handles the mathematical misalignments between the generations and provides a two-phase structural recovery training pipeline capable of bringing the Mamba-3 model back to coherence within a **strict 12GB VRAM envelope**.

## The Methodology

When transplanting a sequence block from Mamba 1 to Mamba 3, three critical mathematical mismatches must be resolved to prevent the model from outputting pure gibberish:

### 1. The `[x, z]` vs `[z, x]` Sequence Inversion
* **The Problem:** Mamba-1's `in_proj` splits the dimension into the main branch (x) followed by the gating branch (z). Mamba-3 expects `[z, x]`. If the weights are blind-copied, the network's forward logic will be physically reversed.
* **The Solution:** The `mamba1_to_mamba3_converter.py` script mathematically slices the `in_proj` weight matrices exactly at `d_inner` and inverts the upper and lower halves before injection.

### 2. Dimensionality Collapse (`dt_bias`, `D`)
* **The Problem:** Mamba-1 scales the structural `D` (skip connection) and `dt_bias` across the entire sequence length. Mamba-3 pools these into specifically sized `nheads` header groups.
* **The Solution:** The script executes an active dimension pooling process (e.g. averaging chunks of 5120 down to 64 pools) to preserve the original structural signal scale.

### 3. Inverse-Softplus Reparameterization
* **The Problem:** Mamba-3 kernel variables require specific scaling logic. The raw bias values map differently through the Triton softplus activation layer.
* **The Solution:** The script maps `torch.log(torch.exp(weights) - 1.0)` on the translated `dt_bias` values to maintain numerical equivalence.

---

## 12GB VRAM Optimization

A 2.8B model normally requires ~18GB VRAM to train. Because standard activation checkpointing often clashes with the custom Mamba-3 Triton kernel, VRAM is optimized via two methods in `mamba3_recovery_trainer.py`:

1. **Per-Sample Micro-Backwards:**
   Instead of `loss.backward()` over a batched block, the loops drop down to:
   ```python
   for sample in batch: 
       loss.backward()
       graph.free()
   ```
   Gradients accumulate safely, but the graph is instantly freed per step, crushing memory spikes.

2. **Phase A Selective Freezing:**
   We freeze 99% of the transplanted model weights representing the "associative memory", unfrosting only the newly added Mamba-3 parameter gates.

---

## The Recovery Pipeline

The transplanted model behaves like an intelligent engine that forgot how to speak. The recovery pipeline adapts the new gates to the old logic.

* **PHASE A (150 steps):** Everything is frozen in the 2.8B model except the newly integrated Mamba-3 specific gates (`B_bias`, `C_bias`, etc.). Loss rapidly collapses as the gates calibrate to the legacy matrices.
* **PHASE B (>1000 steps):** The model injects Low-Rank Adapter (LoRA) matrices cleanly on the outputs and unlocks full reasoning, stabilizing its capabilities.

## Usage

1. Place your base Mamba `.safetensors` or `.bin` checkpoint in the correct directory.
2. Run `python mamba1_to_mamba3_converter.py` to create the initial transplanted shell checkpoint.
3. Run `python mamba3_recovery_trainer.py` to structurally heal the model architecture via Phase A/Phase B training loop.
