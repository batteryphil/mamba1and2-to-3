"""Mamba-1 to Mamba-3 SISO Converter.

Performs surgical weight transplant from a Mamba-1 (MambaForCausalLM)
checkpoint to a Mamba-3 SISO shell, handling three critical mismatches:
  1. [x, z] -> [z, x] in_proj inversion
  2. D / dt_bias dimensionality collapse (d_inner -> nheads via mean-pool)
  3. dt_bias inverse-softplus reparameterization
"""

import os
import json
import shutil
import zipfile
import argparse
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

try:
    from mamba3_engine import Mamba3, RMSNorm
except ImportError:
    from mamba_ssm.modules.mamba3 import Mamba3
    try:
        from mamba_ssm.ops.triton.layer_norm import RMSNorm
    except ImportError:
        RMSNorm = nn.LayerNorm


def _find_model_root(base_dir: str) -> str:
    """Walk extracted dir to find the folder containing config.json."""
    for root, _dirs, files in os.walk(base_dir):
        if "config.json" in files:
            return root
    raise FileNotFoundError(f"No config.json found under {base_dir}")


def _detect_prefix(state_dict: Dict[str, torch.Tensor], layer: int) -> str:
    """Auto-detect HF vs state-spaces key prefix."""
    hf_key = f"model.layers.{layer}.mixer.out_proj.weight"
    ss_key = f"backbone.layers.{layer}.mixer.out_proj.weight"
    if hf_key in state_dict:
        return f"model.layers.{layer}."
    if ss_key in state_dict:
        return f"backbone.layers.{layer}."
    raise KeyError(f"Cannot detect layer prefix for layer {layer}")


def build_mamba3_shell_state_dict(
    d_model: int,
    n_layer: int,
    vocab_size: int,
    headdim: int = 64,
    d_state: int = 64,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, torch.Tensor]:
    """Build a Mamba-3 SISO state dict by directly instantiating Mamba3 blocks.

    Bypasses MambaLMHeadModel which only supports Mamba1/Mamba2 in
    mamba_ssm 2.3.1. Keys mirror the MambaLMHeadModel/MixerModel layout.

    Args:
        d_model: Hidden dimension.
        n_layer: Number of layers.
        vocab_size: Vocabulary size (already padded if necessary).
        headdim: SSM head dimension.
        d_state: SSM state size.
        dtype: Parameter dtype.

    Returns:
        Flat state dict with Mamba-3 default initializations.
    """
    sd: Dict[str, torch.Tensor] = {}

    # Embedding
    emb = nn.Embedding(vocab_size, d_model, dtype=dtype)
    sd["backbone.embedding.weight"] = emb.weight.detach().clone()

    for i in range(n_layer):
        # Pre-mixer RMSNorm
        norm = RMSNorm(d_model, eps=1e-5, dtype=dtype)
        sd[f"backbone.layers.{i}.norm.weight"] = norm.weight.detach().clone()

        # Mamba3 mixer block
        mixer = Mamba3(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            is_mimo=False,
            chunk_size=64,
            layer_idx=i,
            device="cpu",
            dtype=dtype,
        )
        for name, param in mixer.state_dict().items():
            sd[f"backbone.layers.{i}.mixer.{name}"] = param.detach().clone()

        if (i + 1) % 16 == 0:
            print(f"   Shell layer {i + 1}/{n_layer} initialized.")

    # Final norm
    norm_f = RMSNorm(d_model, eps=1e-5, dtype=dtype)
    sd["backbone.norm_f.weight"] = norm_f.weight.detach().clone()

    # LM head (clone of embedding — will be overwritten by transplant)
    sd["lm_head.weight"] = sd["backbone.embedding.weight"].clone()

    return sd


def convert_mamba1_to_mamba3(input_path: str, output_dir: str) -> None:
    """Convert a Mamba-1 2.8B checkpoint to Mamba-3 SISO architecture.

    Args:
        input_path: Path to the model directory or a zip archive.
        output_dir: Destination directory for the converted checkpoint.
    """
    print("🚀 Initializing Mamba-1 → Mamba-3 SISO Conversion...")

    # ------------------------------------------------------------------ #
    # Step 1: Extract zip if needed                                        #
    # ------------------------------------------------------------------ #
    extract_dir = input_path
    _cleanup_extract = False
    if input_path.endswith(".zip"):
        extract_dir = os.path.join(os.path.dirname(output_dir), "_temp_unzip")
        print(f"📦 Extracting {input_path} → {extract_dir} ...")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(extract_dir)
        extract_dir = _find_model_root(extract_dir)
        _cleanup_extract = True

    # ------------------------------------------------------------------ #
    # Step 2: Load original config                                         #
    # ------------------------------------------------------------------ #
    with open(os.path.join(extract_dir, "config.json"), "r") as f:
        orig_config = json.load(f)

    d_model = orig_config.get("hidden_size", orig_config.get("d_model", 2560))
    n_layer = orig_config.get(
        "num_hidden_layers", orig_config.get("n_layer", 64)
    )
    vocab_size = orig_config.get("vocab_size", 50282)

    headdim = 64
    d_inner = d_model * 2          # 2560 * 2 = 5120
    nheads = d_inner // headdim    # 5120 // 64 = 80
    d_state = 64                   # Upsampled vs Mamba-1's state_size=16

    print(
        f"   d_model={d_model}, n_layer={n_layer}, "
        f"d_inner={d_inner}, nheads={nheads}, d_state={d_state}"
    )

    # ------------------------------------------------------------------ #
    # Step 3: Build Mamba-3 SISO shell state dict (CPU, bfloat16)         #
    # ------------------------------------------------------------------ #
    print("🏗️  Building Mamba-3 SISO shell (direct Mamba3 instantiation) ...")
    m3_sd = build_mamba3_shell_state_dict(
        d_model=d_model,
        n_layer=n_layer,
        vocab_size=vocab_size,
        headdim=headdim,
        d_state=d_state,
        dtype=torch.bfloat16,
    )
    print(f"   Shell has {len(m3_sd)} tensors.")

    # ------------------------------------------------------------------ #
    # Step 4: Load Mamba-1 weights                                         #
    # ------------------------------------------------------------------ #
    print("📥 Loading Mamba-1 safetensors ...")
    m1_sd = load_file(
        os.path.join(extract_dir, "model.safetensors"), device="cpu"
    )
    print(f"   Loaded {len(m1_sd)} tensors from Mamba-1 checkpoint.")

    # ------------------------------------------------------------------ #
    # Step 5: Global transplants (embedding, norm, lm_head)               #
    # ------------------------------------------------------------------ #
    embed_key = next(
        (k for k in ("model.embed_tokens.weight", "backbone.embedding.weight")
         if k in m1_sd),
        None,
    )
    norm_key = next(
        (k for k in ("model.norm_f.weight", "backbone.norm_f.weight")
         if k in m1_sd),
        None,
    )
    assert embed_key, "Embedding weight not found in Mamba-1 checkpoint"
    assert norm_key, "Final norm weight not found in Mamba-1 checkpoint"

    m3_sd["backbone.embedding.weight"] = m1_sd[embed_key].to(torch.bfloat16)
    m3_sd["backbone.norm_f.weight"] = m1_sd[norm_key].to(torch.bfloat16)
    lm_head_src = m1_sd.get("lm_head.weight", m1_sd[embed_key])
    m3_sd["lm_head.weight"] = lm_head_src.to(torch.bfloat16).clone()

    # ------------------------------------------------------------------ #
    # Step 6: Layer-by-layer surgery                                       #
    # ------------------------------------------------------------------ #
    print("✂️  Layer-by-layer architecture surgery ...")
    for i in range(n_layer):
        p1 = _detect_prefix(m1_sd, i)
        p3 = f"backbone.layers.{i}."

        # Norm (direct copy)
        m3_sd[f"{p3}norm.weight"] = (
            m1_sd[f"{p1}norm.weight"].to(torch.bfloat16)
        )

        # out_proj (direct copy)
        m3_sd[f"{p3}mixer.out_proj.weight"] = (
            m1_sd[f"{p1}mixer.out_proj.weight"].to(torch.bfloat16)
        )
        bias_key = f"{p1}mixer.out_proj.bias"
        if bias_key in m1_sd:
            m3_sd[f"{p3}mixer.out_proj.bias"] = (
                m1_sd[bias_key].to(torch.bfloat16)
            )

        # FIX 1 — in_proj: Mamba-1 [x|z] → Mamba-3 [z|x] swap
        m1_in = m1_sd[f"{p1}mixer.in_proj.weight"].to(torch.bfloat16)
        m1_x = m1_in[0:d_inner, :]
        m1_z = m1_in[d_inner:2 * d_inner, :]
        m3_sd[f"{p3}mixer.in_proj.weight"][0:d_inner, :] = m1_z
        m3_sd[f"{p3}mixer.in_proj.weight"][d_inner:2 * d_inner, :] = m1_x
        # Slots beyond 2*d_inner keep Mamba-3 default init (dd_dt, dd_A,
        # trap, angles).

        # FIX 2 — D: [d_inner] → [nheads] via mean-pool
        d_key = f"{p1}mixer.D"
        if d_key in m1_sd:
            m1_D = m1_sd[d_key].float()
            m3_sd[f"{p3}mixer.D"] = (
                m1_D.view(nheads, headdim).mean(dim=-1).to(torch.bfloat16)
            )

        # FIX 3 — dt_bias: inv-softplus reparameterization + mean-pool
        dt_key = f"{p1}mixer.dt_proj.bias"
        if dt_key in m1_sd:
            m1_dt = m1_sd[dt_key].float()
            m1_dt_real = F.softplus(m1_dt)
            m1_dt_avg = m1_dt_real.view(nheads, headdim).mean(dim=-1)
            inv_sp = torch.log(torch.expm1(m1_dt_avg.clamp(min=1e-5)))
            m3_sd[f"{p3}mixer.dt_bias"] = inv_sp.to(torch.bfloat16)

        # Dropped (Mamba-1 only): conv1d, A_log, x_proj, dt_proj.weight

        if (i + 1) % 16 == 0:
            print(f"   Layer {i + 1}/{n_layer} transplanted.")

    # ------------------------------------------------------------------ #
    # Step 7: Save checkpoint                                              #
    # ------------------------------------------------------------------ #
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "model.safetensors")
    print(f"💾 Saving Mamba-3 checkpoint → {out_path} ...")
    save_file(m3_sd, out_path)

    # ------------------------------------------------------------------ #
    # Step 8: Write updated config.json                                    #
    # ------------------------------------------------------------------ #
    orig_config["model_type"] = "mamba"
    orig_config["ssm_cfg"] = {
        "layer": "Mamba3", "is_mimo": False,
        "d_state": d_state, "headdim": headdim,
    }
    orig_config["state_size"] = d_state
    orig_config["d_state"] = d_state

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(orig_config, f, indent=2)

    # ------------------------------------------------------------------ #
    # Step 9: Preserve halting head and tokenizer artifacts               #
    # ------------------------------------------------------------------ #
    for fname in ("halting_head.pt", "tokenizer_config.json",
                  "generation_config.json"):
        src = os.path.join(extract_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, fname))
            print(f"   ✅ Copied {fname}")

    if _cleanup_extract:
        temp_root = os.path.join(os.path.dirname(output_dir), "_temp_unzip")
        shutil.rmtree(temp_root, ignore_errors=True)

    split_idx = d_inner * 2
    print("\n🎉 Conversion complete!")
    print(f"   Output directory  : {output_dir}")
    print(f"   Fine-tune freeze  : in_proj.weight[:, :{split_idx}]  "
          f"(split_idx={split_idx})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Mamba-1 checkpoint to Mamba-3 SISO."
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to Mamba-1 directory or .zip archive.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Destination directory for the Mamba-3 checkpoint.",
    )
    args = parser.parse_args()
    convert_mamba1_to_mamba3(args.input, args.output)
