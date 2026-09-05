"""Unified Mamba-1 & Mamba-2 to Mamba-3 Converter.
===================================================
Converts Mamba-1 or Mamba-2 checkpoints to the Mamba-3 SISO architecture.

Mathematical & Architectural Principles:
1. Mamba-2 -> Mamba-3 (Recommended / High Fidelity):
   - Mamba-2 and Mamba-3 share the State Space Duality (SSD) formulation.
   - Preserves 100% of the recurrent SSM parameters:
     * in_proj (z, x, B, C, dt)
     * conv1d (1D sequence convolution kernel and bias)
     * A_log (multi-head scalar decay parameter)
     * dt_bias (time-step discretization bias)
     * D (residual skip parameter)
     * norm & out_proj
   - Initializes new Mamba-3 gating parameters (B_bias, C_bias, B_norm, C_norm)
     so the converted model retains immediate language fluency at step 0.

2. Mamba-1 -> Mamba-3 (Legacy / Experimental):
   - Handles [x, z] -> [z, x] in_proj inversion, D / dt_bias pooling,
     and inverse-softplus reparameterization.
   - Note: Because Mamba-1 is not SSD-based, its 1D conv and continuous A
     cannot be mapped 1-to-1 to SSD heads. Dropping these requires substantial
     pretraining to recover semantic representation.
"""

import os
import json
import shutil
import zipfile
import argparse
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from mamba3_engine import Mamba3, RMSNorm


def find_model_root(base_dir: str) -> str:
    """Walk directory to find the folder containing config.json or model weights."""
    for root, _dirs, files in os.walk(base_dir):
        if "config.json" in files or "model.safetensors" in files or "pytorch_model.bin" in files:
            return root
    return base_dir


def detect_model_type(config: Dict[str, Any], state_dict: Dict[str, torch.Tensor]) -> str:
    """Auto-detect whether checkpoint is Mamba-1 or Mamba-2."""
    model_type = config.get("model_type", "").lower()
    if "mamba2" in model_type:
        return "mamba2"
    if "mamba" in model_type and "num_heads" in config:
        return "mamba2"

    # Check state dict keys
    for k in state_dict.keys():
        if "conv1d.weight" in k:
            w = state_dict[k]
            # Mamba-2 conv1d has in_channels = conv_dim > d_inner (includes B and C)
            # Mamba-1 conv1d has in_channels = d_inner
            if w.dim() == 3 and w.shape[1] == 1:
                return "mamba2"
        if "A_log" in k:
            if state_dict[k].dim() == 1:
                return "mamba2"
            elif state_dict[k].dim() == 2:
                return "mamba1"

    if "state_size" in config and config.get("state_size", 16) > 16:
        return "mamba2"

    return "mamba1"


def build_mamba3_shell_state_dict(
    d_model: int,
    n_layer: int,
    vocab_size: int,
    headdim: int = 64,
    d_state: int = 64,
    ngroups: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, torch.Tensor]:
    """Build a Mamba-3 SISO state dict with default initializations."""
    sd: Dict[str, torch.Tensor] = {}

    emb = nn.Embedding(vocab_size, d_model, dtype=dtype)
    sd["backbone.embedding.weight"] = emb.weight.detach().clone()

    for i in range(n_layer):
        norm = RMSNorm(d_model, eps=1e-5, dtype=dtype)
        sd[f"backbone.layers.{i}.norm.weight"] = norm.weight.detach().clone()

        mixer = Mamba3(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            ngroups=ngroups,
            is_mimo=False,
            chunk_size=64,
            layer_idx=i,
            device="cpu",
            dtype=dtype,
        )
        for name, param in mixer.state_dict().items():
            sd[f"backbone.layers.{i}.mixer.{name}"] = param.detach().clone()

    norm_f = RMSNorm(d_model, eps=1e-5, dtype=dtype)
    sd["backbone.norm_f.weight"] = norm_f.weight.detach().clone()
    sd["lm_head.weight"] = sd["backbone.embedding.weight"].clone()
    return sd


def convert_mamba2_to_mamba3(
    src_sd: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Convert a Mamba-2 checkpoint to Mamba-3 SISO format.

    Preserves 100% of SSD parameters:
    - in_proj, conv1d, dt_bias, A_log, D, norm, out_proj
    - Initializes B_bias, C_bias to 0 and B_norm, C_norm to 1.
    """
    d_model = config.get("hidden_size", config.get("d_model", 1024))
    n_layer = config.get("num_hidden_layers", config.get("n_layer", 48))
    vocab_size = config.get("vocab_size", 50288)
    headdim = config.get("head_dim", 64)
    d_state = config.get("state_size", config.get("d_state", 128))
    ngroups = config.get("n_groups", config.get("ngroups", 1))

    print(f"🏗️  Instantiating Mamba-3 shell (d_model={d_model}, n_layer={n_layer}, "
          f"vocab_size={vocab_size}, headdim={headdim}, d_state={d_state}, ngroups={ngroups})...")
    m3_sd = build_mamba3_shell_state_dict(
        d_model=d_model,
        n_layer=n_layer,
        vocab_size=vocab_size,
        headdim=headdim,
        d_state=d_state,
        ngroups=ngroups,
        dtype=dtype,
    )

    # Global weights
    for embed_k in ("backbone.embeddings.weight", "backbone.embedding.weight", "model.embeddings.weight"):
        if embed_k in src_sd:
            m3_sd["backbone.embedding.weight"] = src_sd[embed_k].to(dtype).clone()
            m3_sd["lm_head.weight"] = src_sd.get("lm_head.weight", src_sd[embed_k]).to(dtype).clone()
            break

    for norm_k in ("backbone.norm_f.weight", "model.norm_f.weight", "backbone.norm.weight"):
        if norm_k in src_sd:
            m3_sd["backbone.norm_f.weight"] = src_sd[norm_k].to(dtype).clone()
            break

    print("✂️  Direct SSD manifold transplant from Mamba-2 to Mamba-3...")
    for i in range(n_layer):
        # Determine source prefix
        p_src = None
        for cand in (f"backbone.layers.{i}.", f"model.layers.{i}."):
            if f"{cand}norm.weight" in src_sd or f"{cand}mixer.out_proj.weight" in src_sd:
                p_src = cand
                break
        if p_src is None:
            continue

        p_dst = f"backbone.layers.{i}."

        # Layer pre-norm
        if f"{p_src}norm.weight" in src_sd:
            m3_sd[f"{p_dst}norm.weight"] = src_sd[f"{p_src}norm.weight"].to(dtype).clone()

        # Mixer out_proj
        if f"{p_src}mixer.out_proj.weight" in src_sd:
            m3_sd[f"{p_dst}mixer.out_proj.weight"] = src_sd[f"{p_src}mixer.out_proj.weight"].to(dtype).clone()

        # Mixer in_proj
        if f"{p_src}mixer.in_proj.weight" in src_sd:
            m3_sd[f"{p_dst}mixer.in_proj.weight"] = src_sd[f"{p_src}mixer.in_proj.weight"].to(dtype).clone()

        # Mixer conv1d
        if f"{p_src}mixer.conv1d.weight" in src_sd:
            m3_sd[f"{p_dst}mixer.conv1d.weight"] = src_sd[f"{p_src}mixer.conv1d.weight"].to(dtype).clone()
        if f"{p_src}mixer.conv1d.bias" in src_sd:
            m3_sd[f"{p_dst}mixer.conv1d.bias"] = src_sd[f"{p_src}mixer.conv1d.bias"].to(dtype).clone()

        # Mixer A_log, D, dt_bias
        if f"{p_src}mixer.A_log" in src_sd:
            m3_sd[f"{p_dst}mixer.A_log"] = src_sd[f"{p_src}mixer.A_log"].to(dtype).clone()
        if f"{p_src}mixer.D" in src_sd:
            m3_sd[f"{p_dst}mixer.D"] = src_sd[f"{p_src}mixer.D"].to(dtype).clone()
        if f"{p_src}mixer.dt_bias" in src_sd:
            m3_sd[f"{p_dst}mixer.dt_bias"] = src_sd[f"{p_src}mixer.dt_bias"].to(dtype).clone()

        # Mixer internal norm
        if f"{p_src}mixer.norm.weight" in src_sd:
            m3_sd[f"{p_dst}mixer.norm.weight"] = src_sd[f"{p_src}mixer.norm.weight"].to(dtype).clone()

    out_cfg = dict(config)
    out_cfg["model_type"] = "mamba3"
    out_cfg["ssm_cfg"] = {
        "layer": "Mamba3",
        "is_mimo": False,
        "d_state": d_state,
        "headdim": headdim,
        "ngroups": ngroups,
    }
    return m3_sd, out_cfg


def convert_mamba1_to_mamba3(
    src_sd: Dict[str, torch.Tensor],
    config: Dict[str, Any],
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Convert a Mamba-1 checkpoint to Mamba-3 SISO format.

    Handles [x, z] -> [z, x] inversion and dt_bias/D pooling.
    """
    print("⚠️  WARNING: Mamba-1 does not use State Space Duality (SSD).")
    print("   Continuous A_log, conv1d, and x_proj will be reset for Mamba-3.")

    d_model = config.get("hidden_size", config.get("d_model", 2560))
    n_layer = config.get("num_hidden_layers", config.get("n_layer", 64))
    vocab_size = config.get("vocab_size", 50280)
    headdim = 64
    d_inner = d_model * 2
    nheads = d_inner // headdim
    d_state = 64

    m3_sd = build_mamba3_shell_state_dict(
        d_model=d_model,
        n_layer=n_layer,
        vocab_size=vocab_size,
        headdim=headdim,
        d_state=d_state,
        dtype=dtype,
    )

    # Embedding & norm
    for embed_k in ("backbone.embeddings.weight", "backbone.embedding.weight", "model.embed_tokens.weight"):
        if embed_k in src_sd:
            m3_sd["backbone.embedding.weight"] = src_sd[embed_k].to(dtype).clone()
            m3_sd["lm_head.weight"] = src_sd.get("lm_head.weight", src_sd[embed_k]).to(dtype).clone()
            break

    for norm_k in ("backbone.norm_f.weight", "model.norm_f.weight"):
        if norm_k in src_sd:
            m3_sd["backbone.norm_f.weight"] = src_sd[norm_k].to(dtype).clone()
            break

    for i in range(n_layer):
        p_src = f"backbone.layers.{i}." if f"backbone.layers.{i}.norm.weight" in src_sd else f"model.layers.{i}."
        p_dst = f"backbone.layers.{i}."

        if f"{p_src}norm.weight" in src_sd:
            m3_sd[f"{p_dst}norm.weight"] = src_sd[f"{p_src}norm.weight"].to(dtype).clone()

        if f"{p_src}mixer.out_proj.weight" in src_sd:
            m3_sd[f"{p_dst}mixer.out_proj.weight"] = src_sd[f"{p_src}mixer.out_proj.weight"].to(dtype).clone()

        # Slicing and swapping [x, z] -> [z, x]
        if f"{p_src}mixer.in_proj.weight" in src_sd:
            m1_in = src_sd[f"{p_src}mixer.in_proj.weight"].to(dtype)
            m1_x = m1_in[0:d_inner, :]
            m1_z = m1_in[d_inner:2 * d_inner, :]
            m3_sd[f"{p_dst}mixer.in_proj.weight"][0:d_inner, :] = m1_z
            m3_sd[f"{p_dst}mixer.in_proj.weight"][d_inner:2 * d_inner, :] = m1_x

        # D pooling
        if f"{p_src}mixer.D" in src_sd:
            m1_D = src_sd[f"{p_src}mixer.D"].float()
            m3_sd[f"{p_dst}mixer.D"] = m1_D.view(nheads, headdim).mean(dim=-1).to(dtype)

        # dt_bias inv-softplus
        if f"{p_src}mixer.dt_proj.bias" in src_sd:
            m1_dt = src_sd[f"{p_src}mixer.dt_proj.bias"].float()
            m1_dt_real = F.softplus(m1_dt)
            m1_dt_avg = m1_dt_real.view(nheads, headdim).mean(dim=-1)
            inv_sp = torch.log(torch.expm1(m1_dt_avg.clamp(min=1e-5)))
            m3_sd[f"{p_dst}mixer.dt_bias"] = inv_sp.to(dtype)

    out_cfg = dict(config)
    out_cfg["model_type"] = "mamba3"
    out_cfg["ssm_cfg"] = {
        "layer": "Mamba3",
        "is_mimo": False,
        "d_state": d_state,
        "headdim": headdim,
    }
    return m3_sd, out_cfg


def convert_checkpoint(
    input_path: str,
    output_dir: str,
    model_type: str = "auto",
    precision: str = "bf16",
) -> None:
    """Run end-to-end checkpoint conversion to Mamba-3."""
    print(f"🚀 Initializing Mamba → Mamba-3 Conversion (Target: {output_dir})...")

    extract_dir = input_path
    cleanup_extract = False
    if input_path.endswith(".zip"):
        extract_dir = os.path.join(output_dir, "_temp_unzip")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(input_path, "r") as zf:
            zf.extractall(extract_dir)
        extract_dir = find_model_root(extract_dir)
        cleanup_extract = True

    extract_dir = find_model_root(extract_dir)

    # Load config
    cfg_file = os.path.join(extract_dir, "config.json")
    if os.path.exists(cfg_file):
        with open(cfg_file, "r") as f:
            config = json.load(f)
    else:
        config = {}

    # Load weights
    src_sd: Dict[str, torch.Tensor] = {}
    sf_file = os.path.join(extract_dir, "model.safetensors")
    bin_file = os.path.join(extract_dir, "pytorch_model.bin")
    if os.path.exists(sf_file):
        print(f"📥 Loading weights from {sf_file}...")
        src_sd = load_file(sf_file, device="cpu")
    elif os.path.exists(bin_file):
        print(f"📥 Loading weights from {bin_file}...")
        src_sd = torch.load(bin_file, map_location="cpu", weights_only=False)
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {extract_dir}")

    # Determine type
    if model_type == "auto":
        detected_type = detect_model_type(config, src_sd)
        print(f"🔍 Auto-detected model architecture: {detected_type.upper()}")
    else:
        detected_type = model_type.lower()

    dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else torch.float32)

    if detected_type == "mamba2":
        m3_sd, out_cfg = convert_mamba2_to_mamba3(src_sd, config, dtype=dtype)
    else:
        m3_sd, out_cfg = convert_mamba1_to_mamba3(src_sd, config, dtype=dtype)

    os.makedirs(output_dir, exist_ok=True)
    out_weights = os.path.join(output_dir, "model.safetensors")
    print(f"💾 Saving Mamba-3 checkpoint ({len(m3_sd)} tensors) → {out_weights}...")
    save_file(m3_sd, out_weights)

    out_cfg_path = os.path.join(output_dir, "config.json")
    with open(out_cfg_path, "w") as f:
        json.dump(out_cfg, f, indent=2)
    print(f"📄 Saved config → {out_cfg_path}")

    # Copy tokenizer files if present
    for fname in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                  "special_tokens_map.json", "generation_config.json"):
        src_f = os.path.join(extract_dir, fname)
        if os.path.exists(src_f):
            shutil.copy(src_f, os.path.join(output_dir, fname))
            print(f"   ✅ Copied {fname}")

    if cleanup_extract:
        shutil.rmtree(extract_dir, ignore_errors=True)

    print("\n🎉 Conversion completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Mamba-1 or Mamba-2 checkpoints to Mamba-3 SISO.")
    parser.add_argument("--input", type=str, required=True, help="Path to input checkpoint directory or .zip")
    parser.add_argument("--output", type=str, required=True, help="Destination directory for Mamba-3 checkpoint")
    parser.add_argument("--model_type", type=str, choices=["auto", "mamba1", "mamba2"], default="auto",
                        help="Model architecture: 'auto' (default), 'mamba2' (SSD-preserving), or 'mamba1'")
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="bf16",
                        help="Output tensor precision (default: bf16)")
    args = parser.parse_args()

    convert_checkpoint(args.input, args.output, model_type=args.model_type, precision=args.precision)
