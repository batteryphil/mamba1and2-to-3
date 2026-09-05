"""mamba3_engine.py — Self-Contained Pure-PyTorch Mamba-3 SISO Engine
========================================================================
Implements the Mamba-3 SISO (Single-Input Single-Output) architecture and
State Space Duality (SSD) chunked recurrence in pure PyTorch.

Features:
- Zero C++/CUDA compiler dependencies (runs out-of-the-box on CPU, ROCm, CUDA).
- Chunked SSD semi-separable matrix operations (O(L) linear time, fast GPU batched matmul).
- Native support for Mamba-3 enhancements:
  * Inverted [z, x] input projection
  * Head-grouped B_norm / C_norm and B_bias / C_bias
  * Scalar-per-head decay A_log and dt_bias
  * Residual skip D
- Full Mamba3Block, Mamba3Backbone, and Mamba3LMModel causal language model.
"""

import math
from typing import Optional, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt_utils


class CausalOutput(NamedTuple):
    logits: torch.Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def segsum(x: torch.Tensor) -> torch.Tensor:
    """Stable 1D segment cumsum for SSD decay computation.
    
    Args:
        x: [..., L]
    Returns:
        Segment decay matrix [..., L, L] with lower triangular cumsum.
    """
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    # Masked lower-triangular pairwise difference
    decay = x_cumsum.unsqueeze(-1) - x_cumsum.unsqueeze(-2)
    mask = torch.tril(torch.ones((T, T), device=x.device, dtype=torch.bool))
    return torch.where(mask, decay, -torch.inf)


def ssd_chunked_forward(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compute State Space Duality (SSD) via chunked matrix multiplication.

    Mathematically equivalent to the 1-semiseparable kernel in Mamba-2 & Mamba-3:
        Y = (M \\circ (C B^T)) X
    where M is the causal decay matrix derived from exp(dt * A).

    Args:
        x:  [batch, seq_len, nheads, headdim]
        dt: [batch, seq_len, nheads]
        A:  [nheads] (negative log decay)
        B:  [batch, seq_len, ngroups, d_state]
        C:  [batch, seq_len, ngroups, d_state]
        chunk_size: size of temporal chunks (default 64)

    Returns:
        y:  [batch, seq_len, nheads, headdim]
    """
    batch, seq_len, nheads, headdim = x.shape
    _, _, ngroups, d_state = B.shape

    # Expand groups to match nheads if needed
    heads_per_group = nheads // ngroups
    if heads_per_group > 1:
        B = B.repeat_interleave(heads_per_group, dim=2)
        C = C.repeat_interleave(heads_per_group, dim=2)

    # Effective continuous decay: dt_A = dt * A (shape: [batch, seq_len, nheads])
    # A is negative: -exp(A_log)
    dt_A = dt * A.view(1, 1, nheads)

    # Pad seq_len to multiple of chunk_size if necessary
    remainder = seq_len % chunk_size
    pad_len = (chunk_size - remainder) % chunk_size
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        dt_A = F.pad(dt_A, (0, 0, 0, pad_len))
        B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, 0, 0, pad_len))

    total_len = x.size(1)
    nchunks = total_len // chunk_size

    # Reshape into chunks: [batch, nchunks, chunk_size, nheads, ...]
    x_c = x.view(batch, nchunks, chunk_size, nheads, headdim)
    dt_A_c = dt_A.view(batch, nchunks, chunk_size, nheads)
    B_c = B.view(batch, nchunks, chunk_size, nheads, d_state)
    C_c = C.view(batch, nchunks, chunk_size, nheads, d_state)

    # Rearrange to: [batch, nheads, nchunks, chunk_size, ...]
    x_c = x_c.permute(0, 3, 1, 2, 4)
    dt_A_c = dt_A_c.permute(0, 3, 1, 2)
    B_c = B_c.permute(0, 3, 1, 2, 4)
    C_c = C_c.permute(0, 3, 1, 2, 4)

    # 1. Intra-chunk decay: [batch, nheads, nchunks, chunk_size, chunk_size]
    intra_decay = segsum(dt_A_c)
    # Clamp for numerical stability before exp
    intra_M = torch.exp(torch.clamp(intra_decay, min=-50.0, max=0.0))

    # 2. Intra-chunk kernel: C_c @ B_c^T -> [batch, nheads, nchunks, chunk_size, chunk_size]
    # C_c: [B, H, C, L_c, P], B_c: [B, H, C, L_c, P]
    CB = torch.matmul(C_c, B_c.transpose(-1, -2))
    kernel = intra_M * CB

    # Intra-chunk output: kernel @ x_c -> [batch, nheads, nchunks, chunk_size, headdim]
    y_intra = torch.matmul(kernel.to(x_c.dtype), x_c)

    # 3. Inter-chunk state recurrence
    # Total decay within each chunk: [batch, nheads, nchunks]
    chunk_decay = torch.exp(torch.clamp(dt_A_c.sum(dim=-1), min=-50.0, max=0.0))
    # Decay from each chunk step to end of chunk: [batch, nheads, nchunks, chunk_size]
    decay_to_end = torch.exp(torch.clamp(torch.cumsum(dt_A_c.flip(-1), dim=-1).flip(-1) - dt_A_c, min=-50.0, max=0.0))
    # Decay from start of chunk to each step: [batch, nheads, nchunks, chunk_size]
    decay_from_start = torch.exp(torch.clamp(torch.cumsum(dt_A_c, dim=-1), min=-50.0, max=0.0))

    # Chunk state: sum_{t} (decay_to_end[t] * (B[t]^T @ x[t])) -> [batch, nheads, nchunks, d_state, headdim]
    # B_weighted: [batch, nheads, nchunks, chunk_size, d_state]
    B_weighted = B_c * decay_to_end.unsqueeze(-1)
    chunk_states = torch.matmul(B_weighted.transpose(-1, -2), x_c)

    # Recurrent accumulation across chunks
    # prev_state at chunk c: decayed sum of previous chunk states
    h = torch.zeros((batch, nheads, d_state, headdim), device=x.device, dtype=x.dtype)
    inter_outputs = []

    for c in range(nchunks):
        # Contribution of previous chunks to current chunk:
        # C_weighted: [batch, nheads, chunk_size, d_state]
        C_weighted = C_c[:, :, c] * decay_from_start[:, :, c].unsqueeze(-1)
        # out: [batch, nheads, chunk_size, headdim]
        out_inter = torch.matmul(C_weighted, h)
        inter_outputs.append(out_inter)
        # Evolve state: decay old state by chunk_decay, add new chunk_state
        decay_c = chunk_decay[:, :, c].unsqueeze(-1).unsqueeze(-1)
        h = h * decay_c + chunk_states[:, :, c]

    y_inter = torch.stack(inter_outputs, dim=2)
    y_c = y_intra + y_inter

    # Permute back: [batch, nchunks, chunk_size, nheads, headdim] -> [batch, total_len, nheads, headdim]
    y = y_c.permute(0, 2, 3, 1, 4).contiguous().view(batch, total_len, nheads, headdim)

    if pad_len > 0:
        y = y[:, :seq_len, :, :]

    return y


class Mamba3(nn.Module):
    """Mamba-3 SISO (Single-Input Single-Output) Mixer Block.

    Fully compatible with Mamba-3 checkpoint specifications:
    - [z, x] input projection ordering
    - B_norm / C_norm and B_bias / C_bias
    - dt_bias and A_log scalar parameters
    - Residual skip connection D
    - Fused/chunked SSD State Space Duality computation
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        is_mimo: bool = False,
        layer_idx: int = 0,
        device=None,
        dtype=torch.bfloat16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.headdim = headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size
        self.d_conv = d_conv
        self.expand = expand
        self.is_mimo = is_mimo
        self.layer_idx = layer_idx

        self.d_inner = d_model * expand
        assert self.d_inner % headdim == 0, f"d_inner ({self.d_inner}) must be divisible by headdim ({headdim})"
        self.nheads = self.d_inner // headdim

        fkw = {"device": device, "dtype": dtype}

        # in_proj produces:
        # z: d_inner (gate)
        # x: d_inner (SSM input)
        # B: ngroups * d_state
        # C: ngroups * d_state
        # dt: nheads
        # Total projection size: 2 * d_inner + 2 * ngroups * d_state + nheads
        self.proj_dim_z = self.d_inner
        self.proj_dim_x = self.d_inner
        self.proj_dim_B = self.ngroups * self.d_state
        self.proj_dim_C = self.ngroups * self.d_state
        self.proj_dim_dt = self.nheads
        self.d_in_proj = self.proj_dim_z + self.proj_dim_x + self.proj_dim_B + self.proj_dim_C + self.proj_dim_dt

        self.in_proj = nn.Linear(self.d_model, self.d_in_proj, bias=False, **fkw)

        # 1D Convolution over x, B, C channels
        self.conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.conv_dim,
            bias=True,
            **fkw,
        )

        # SSM Decay parameters: A_log is scalar per head
        A = torch.empty(self.nheads, device=device, dtype=torch.float32).uniform_(1.0, 16.0)
        self.A_log = nn.Parameter(torch.log(A).to(dtype))

        # Time-step bias
        dt_init = torch.exp(torch.rand(self.nheads, device=device, dtype=torch.float32) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        inv_sp = torch.log(torch.expm1(dt_init.clamp(min=1e-5)))
        self.dt_bias = nn.Parameter(inv_sp.to(dtype))

        # Skip connection D
        self.D = nn.Parameter(torch.ones(self.nheads, **fkw))

        # Mamba-3 gates & normalization on B and C
        self.B_bias = nn.Parameter(torch.zeros(self.proj_dim_B, **fkw))
        self.C_bias = nn.Parameter(torch.zeros(self.proj_dim_C, **fkw))
        self.B_norm = RMSNorm(self.d_state, eps=1e-5, **fkw)
        self.C_norm = RMSNorm(self.d_state, eps=1e-5, **fkw)

        # Pre-gate / output normalization
        self.norm = RMSNorm(self.d_inner, eps=1e-5, **fkw)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **fkw)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass of Mamba-3 SISO block.

        Args:
            u: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        batch, seq_len, _ = u.shape

        # 1. in_proj: produces [z, x, B, C, dt]
        z_x_bc_dt = self.in_proj(u)

        # Slicing according to Mamba-3 layout: [z (d_inner), x (d_inner), B, C, dt]
        z = z_x_bc_dt[..., :self.proj_dim_z]
        x_bc = z_x_bc_dt[..., self.proj_dim_z : self.proj_dim_z + self.conv_dim]
        dt = z_x_bc_dt[..., self.proj_dim_z + self.conv_dim : self.d_in_proj]

        # 2. Local 1D Convolution over [x, B, C]
        x_bc_conv = self.conv1d(x_bc.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        x_bc_act = F.silu(x_bc_conv)

        # Split conv output into x, B, C
        x = x_bc_act[..., :self.d_inner]
        B = x_bc_act[..., self.d_inner : self.d_inner + self.proj_dim_B] + self.B_bias
        C = x_bc_act[..., self.d_inner + self.proj_dim_B :] + self.C_bias

        # Reshape B and C for normalization: [batch, seq_len, ngroups, d_state]
        B = self.B_norm(B.view(batch, seq_len, self.ngroups, self.d_state))
        C = self.C_norm(C.view(batch, seq_len, self.ngroups, self.d_state))

        # Reshape x to [batch, seq_len, nheads, headdim]
        x_reshaped = x.view(batch, seq_len, self.nheads, self.headdim)

        # Discretize dt: dt = softplus(dt + dt_bias)
        dt_real = F.softplus(dt + self.dt_bias)

        # Compute SSM decay: A = -exp(A_log)
        A = -torch.exp(self.A_log.float())

        # 3. State Space Duality (SSD) Recurrence
        y_ssm = ssd_chunked_forward(
            x_reshaped,
            dt_real.float(),
            A,
            B.float(),
            C.float(),
            chunk_size=self.chunk_size,
        ).to(u.dtype)

        # Add D skip connection: y = y + x * D
        y_ssm = y_ssm + (x_reshaped * self.D.view(1, 1, self.nheads, 1))

        # Flatten back to [batch, seq_len, d_inner]
        y_flat = y_ssm.view(batch, seq_len, self.d_inner)

        # 4. Gated activation with z: norm(y) * silu(z)
        y_normed = self.norm(y_flat)
        y_gated = y_normed * F.silu(z)

        # 5. Output projection
        out = self.out_proj(y_gated)
        return out


class Mamba3Block(nn.Module):
    """Pre-norm residual block wrapping Mamba3 mixer."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        headdim: int = 64,
        layer_idx: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        device=None,
    ) -> None:
        super().__init__()
        fkw = {"device": device, "dtype": dtype}
        self.norm = RMSNorm(d_model, eps=1e-5, **fkw)
        self.mixer = Mamba3(
            d_model=d_model,
            d_state=d_state,
            headdim=headdim,
            is_mimo=False,
            chunk_size=64,
            layer_idx=layer_idx,
            **fkw,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class Mamba3Backbone(nn.Module):
    """Backbone stack of Mamba-3 blocks with input embedding and final norm."""

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        d_state: int = 64,
        headdim: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        device=None,
    ) -> None:
        super().__init__()
        fkw = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(vocab_size, d_model, **fkw)
        self.layers = nn.ModuleList([
            Mamba3Block(d_model, d_state, headdim, i, dtype, device)
            for i in range(n_layer)
        ])
        self.norm_f = RMSNorm(d_model, eps=1e-5, **fkw)

    def forward(self, input_ids: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            if use_checkpoint and x.requires_grad:
                x = ckpt_utils.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return self.norm_f(x.to(self.norm_f.weight.dtype))


class Mamba3LMModel(nn.Module):
    """Full causal Mamba-3 Language Model."""

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        d_state: int = 64,
        headdim: int = 64,
        dtype: torch.dtype = torch.bfloat16,
        device=None,
    ) -> None:
        super().__init__()
        fkw = {"device": device, "dtype": dtype}
        self.backbone = Mamba3Backbone(d_model, n_layer, vocab_size, d_state, headdim, dtype, device)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **fkw)
        self.lm_head.weight = self.backbone.embedding.weight

    def forward(self, input_ids: torch.Tensor, use_checkpoint: bool = False) -> CausalOutput:
        h = self.backbone(input_ids, use_checkpoint=use_checkpoint)
        logits = self.lm_head(h)
        return CausalOutput(logits=logits)
