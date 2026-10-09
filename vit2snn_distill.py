#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vit2snn_distill.py -- Progressive layer-wise distillation of a Vision Transformer
(timm ``vit_tiny_patch16_224``) into a Spiking Vision Transformer running for T
time steps.

Pipeline
--------
0. Teacher: pretrained ViT-Tiny/16, frozen, eval mode, always run under no_grad.
1. Student: ``SpikingViT``
   * ``SpikeEncoder``: classical (non-spiking) PatchEmbed + cls/pos tokens, computed
     once and repeated over T steps (direct coding), then an input LIF layer
     (bipolar ON/OFF channels) turns the analog token stream into binary spikes,
     followed by a spike-driven linear projection.
   * L ``SpikingBlock``s with membrane-shortcut residuals and softmax-free spiking
     attention: SSA (Spikformer) or SDSA (Spike-driven Transformer).
   * Readout: LayerNorm + Linear on the CLS token at every step, logits averaged
     over T.
2. Progressive distillation: stage 0 (encoder), then blocks 1..L, trained one at a
   time with an MSE loss against the teacher's continuous hidden states.
3. Global fine-tuning: (1 - a) * CE + a * tau^2 * KL(teacher || student) on the
   time-averaged student logits.
4. Profiling: firing rates per LIF layer and per time step, synaptic-operation
   counts and an energy estimate (45 nm CMOS figures).

Smoke test (random images, no weight download):
    python vit2snn_distill.py --data fake --no-pretrained --batch-size 2 \
        --distill-epochs 1 --finetune-epochs 1 --max-steps 2 --eval-batches 2

Imagenette / ImageNet (ImageFolder layout, synset folder names):
    python vit2snn_distill.py --data imagefolder \
        --train-dir ~/data/imagenette2-320/train --val-dir ~/data/imagenette2-320/val \
        --batch-size 16 --distill-epochs 2 --finetune-epochs 5
"""
from __future__ import annotations

import os

# Must be set before torch is imported: unsupported MPS kernels fall back to CPU
# instead of raising.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import copy
import gc
import hashlib
import json
import math
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Iterable, Iterator, Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, Subset


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class Config:
    # --- models --------------------------------------------------------------
    teacher_name: str = "vit_tiny_patch16_224"
    pretrained: bool = True
    T: int = 4                          # number of time steps
    student_dim: Optional[int] = None   # None -> teacher embed dim (192)
    attn_mode: str = "ssa"              # "ssa" (Spikformer) | "sdsa" (Spike-driven)
    tau: float = 2.0                    # LIF membrane time constant (> 1)
    v_threshold: float = 1.0
    surrogate: str = "atan"             # "atan" | "fast_sigmoid"
    surrogate_alpha: float = 2.0        # sharpness of the surrogate derivative
    detach_reset: bool = True           # stop gradients through the reset path
    learnable_tau: bool = False         # PLIF-style learnable time constant
    bipolar_input: bool = True          # ON/OFF input spikes (keeps the sign)
    init_from_teacher: bool = True      # warm-start linear weights from the ViT
    train_patch_embed: bool = False     # keep the classical embedding frozen
    proj_mode: str = "auto"             # W_proj: "auto" | "always" | "never"
    # --- data ----------------------------------------------------------------
    data: str = "fake"                  # "fake" | "imagefolder"
    train_dir: Optional[str] = None
    val_dir: Optional[str] = None
    batch_size: int = 16
    num_workers: int = 2
    fake_size: int = 64
    # --- progressive distillation --------------------------------------------
    distill_epochs: int = 1             # epochs per stage
    distill_lr: float = 1e-3
    distill_input: str = "student"      # "student" (chained) | "teacher" (forcing)
    per_step_weight: float = 0.0        # extra MSE on every time step
    # --- global fine-tuning --------------------------------------------------
    finetune_epochs: int = 3
    finetune_lr: float = 1e-4
    kd_tau: float = 4.0
    kd_alpha: float = 0.7
    label_smoothing: float = 0.1
    grad_checkpointing: bool = False
    # --- common --------------------------------------------------------------
    weight_decay: float = 0.05
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    accum_steps: int = 1
    max_steps: Optional[int] = None     # cap on batches per epoch
    eval_batches: Optional[int] = None  # cap on validation batches
    log_every: int = 20
    device: str = "auto"
    seed: int = 0
    out_dir: str = "runs/vit2snn"
    resume: bool = False
    skip_distill: bool = False
    skip_finetune: bool = False
    # --- experimental protocol (multi-seed series) ----------------------------
    split_file: Optional[str] = None       # JSON from make_split.py: train / monitor indices
    init_checkpoint: Optional[str] = None  # student weights to start from (complete phase-1 ckpt)
    final_eval: bool = False               # evaluate + profile once on the official val set

    def __post_init__(self) -> None:
        if self.T < 1:
            raise ValueError("T must be >= 1")
        if self.tau <= 1.0:
            raise ValueError("tau must be > 1 (decay = 1 / tau)")
        if self.data == "imagefolder" and not (self.train_dir and self.val_dir):
            raise ValueError("--data imagefolder requires --train-dir and --val-dir")
        if self.accum_steps < 1:
            raise ValueError("accum_steps must be >= 1")
        if self.split_file and self.data != "imagefolder":
            raise ValueError("--split-file requires --data imagefolder")


_NONE_TYPES = {"student_dim": int, "train_dir": str, "val_dir": str,
               "max_steps": int, "eval_batches": int, "split_file": str, "init_checkpoint": str}
_CHOICES = {"attn_mode": ("ssa", "sdsa"), "surrogate": ("atan", "fast_sigmoid"),
            "proj_mode": ("auto", "always", "never"), "data": ("fake", "imagefolder"),
            "distill_input": ("student", "teacher"), "device": ("auto", "mps", "cpu", "cuda")}


def parse_args() -> Config:
    """Build the CLI directly from the ``Config`` dataclass."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if isinstance(f.default, bool):
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        elif f.default is None:
            p.add_argument(flag, type=_NONE_TYPES[f.name], default=None)
        else:
            p.add_argument(flag, type=type(f.default), default=f.default,
                           choices=_CHOICES.get(f.name))
    return Config(**vars(p.parse_args()))


# =============================================================================
# Device / memory utilities
# =============================================================================
def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def free_memory(device: torch.device) -> None:
    """Release cached blocks of the device allocator (important on MPS)."""
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def mem_str(device: torch.device) -> str:
    if device.type == "mps":
        return (f"mps alloc={torch.mps.current_allocated_memory() / 2**20:.0f}MiB "
                f"driver={torch.mps.driver_allocated_memory() / 2**20:.0f}MiB")
    if device.type == "cuda":
        return f"cuda alloc={torch.cuda.memory_allocated() / 2**20:.0f}MiB"
    return "cpu"


def mem_bytes(device: torch.device) -> int:
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    return 0


def count_params(module: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)


# =============================================================================
# 1. Spiking neuron: surrogate gradients + multi-step LIF with soft reset
# =============================================================================
class ATanSpike(torch.autograd.Function):
    """Heaviside forward; ATan surrogate backward (Fang et al., 2021):
    dS/dx = (alpha / 2) / (1 + (pi / 2 * alpha * x)^2)."""

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        a = ctx.alpha
        sg = (a / 2) / (1 + (math.pi / 2 * a * x).pow(2))
        return grad_out * sg, None


class FastSigmoidSpike(torch.autograd.Function):
    """Heaviside forward; fast-sigmoid surrogate backward (Zenke & Ganguli, 2018):
    dS/dx = 1 / (1 + alpha * |x|)^2."""

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        sg = 1.0 / (1 + ctx.alpha * x.abs()).pow(2)
        return grad_out * sg, None


SURROGATES = {"atan": ATanSpike.apply, "fast_sigmoid": FastSigmoidSpike.apply}


class MultiStepLIF(nn.Module):
    """Multi-step Leaky Integrate-and-Fire neuron, input shape [T, ...].

        H_t = V_{t-1} + (X_t - V_{t-1}) / tau         (charge, V_reset = 0)
        S_t = Theta(H_t - V_th)                        (fire)
        V_t = H_t - V_th * S_t                         (soft reset / reset-by-subtraction)

    The membrane potential is initialised to zero inside every ``forward`` call, so
    the module is stateless across batches: no ``reset_net`` call is needed and no
    autograd graph can leak from one batch to the next.
    """

    def __init__(self, tau: float = 2.0, v_threshold: float = 1.0, surrogate: str = "atan",
                 alpha: float = 2.0, detach_reset: bool = True,
                 learnable_tau: bool = False) -> None:
        super().__init__()
        self.v_threshold = v_threshold
        self.alpha = alpha
        self.detach_reset = detach_reset
        self.spike_fn = SURROGATES[surrogate]
        w0 = torch.tensor(-math.log(tau - 1.0))  # sigmoid(w0) = 1 / tau
        if learnable_tau:
            self.w = nn.Parameter(w0)
        else:
            self.register_buffer("w", w0)
        self.record = False
        self.reset_stats()

    def reset_stats(self) -> None:
        self.spike_sum_t: Optional[Tensor] = None  # [T] spikes emitted at each step
        self.neuron_count: int = 0                 # neurons per step, summed over calls

    def forward(self, x: Tensor) -> Tensor:
        decay = torch.sigmoid(self.w)
        v = torch.zeros_like(x[0])
        spikes = []
        for t in range(x.shape[0]):
            h = v + (x[t] - v) * decay
            s = self.spike_fn(h - self.v_threshold, self.alpha)
            v = h - (s.detach() if self.detach_reset else s) * self.v_threshold
            spikes.append(s)
        out = torch.stack(spikes, dim=0)
        if self.record:
            # float32 on device, then CPU: MPS has no float64 support.
            per_t = out.detach().reshape(out.shape[0], -1).sum(dim=1).float().cpu()
            self.spike_sum_t = per_t if self.spike_sum_t is None else self.spike_sum_t + per_t
            self.neuron_count += out[0].numel()
        return out

    def extra_repr(self) -> str:
        return (f"tau={1.0 / torch.sigmoid(self.w).item():.2f}, v_th={self.v_threshold}, "
                f"detach_reset={self.detach_reset}")


# =============================================================================
# Layers operating on [T, B, N, C] tensors
# =============================================================================
class TBBatchNorm(nn.Module):
    """BatchNorm1d over the channel dim of a [T, B, N, C] tensor (T*B*N samples).
    At inference it folds into the preceding linear layer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(x.reshape(-1, x.shape[-1])).reshape(x.shape)


class TBLinearBN(nn.Module):
    """Linear (no bias) + BatchNorm1d on [T, B, N, C]. With binary inputs the matmul
    reduces to accumulations (AC) of the weight rows selected by the spikes."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: Tensor) -> Tensor:
        lead = x.shape[:-1]
        y = self.bn(self.linear(x.reshape(-1, x.shape[-1])))
        return y.reshape(*lead, -1)


# =============================================================================
# 1'. Spiking attention (softmax-free) and spiking MLP
# =============================================================================
class SpikingSelfAttention(nn.Module):
    """Softmax-free spiking attention on binary Q, K, V.

    * ``ssa``  (Spikformer, Zhou et al. 2023):  SN(Q K^T V * s). Without softmax the
      product is associative, so we compute Q (K^T V): O(N d_h^2) instead of
      O(N^2 d_h), and the N x N attention map is never materialised.
    * ``sdsa`` (Spike-driven Transformer, Yao et al. 2023):
      SN(BN(sum_tokens(Q * K))) * V, i.e. a binary channel mask applied to V.
      Only AND operations and additions.
    """

    def __init__(self, dim: int, num_heads: int, mode: str, lif_kw: dict,
                 scale: float = 0.125) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} is not divisible by num_heads={num_heads}")
        self.num_heads, self.head_dim, self.mode, self.scale = num_heads, dim // num_heads, mode, scale
        self.q, self.k, self.v = TBLinearBN(dim, dim), TBLinearBN(dim, dim), TBLinearBN(dim, dim)
        self.q_lif, self.k_lif, self.v_lif = (MultiStepLIF(**lif_kw) for _ in range(3))
        if mode == "ssa":
            self.attn_lif = MultiStepLIF(**{**lif_kw, "v_threshold": 0.5})
        elif mode == "sdsa":
            self.mask_bn = TBBatchNorm(dim)
            self.mask_lif = MultiStepLIF(**lif_kw)
        else:
            raise ValueError(f"unknown attention mode {mode!r}")
        self.proj = TBLinearBN(dim, dim)
        self.record = False   # set by the profiler to count attention ACs
        self.ac_ops = 0.0

    def _heads(self, x: Tensor) -> Tensor:
        T, B, N, D = x.shape
        return x.reshape(T, B, N, self.num_heads, self.head_dim).transpose(2, 3)  # [T,B,H,N,dh]

    def _ssa(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        T, B, N, D = q.shape
        qh, kh, vh = self._heads(q), self._heads(k), self._heads(v)
        kv = kh.transpose(-2, -1) @ vh          # [T,B,H,dh,dh], integer-valued spike counts
        x = (qh @ kv) * self.scale              # == (Q K^T) V * s
        if self.record:
            self.ac_ops += float((kh.sum(-1) * vh.sum(-1)).sum()) + float(qh.sum()) * self.head_dim
        return self.attn_lif(x.transpose(2, 3).reshape(T, B, N, D))

    def _sdsa(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        qk = q * k                                                   # AND of binary spikes
        mask = self.mask_lif(self.mask_bn(qk.sum(dim=2, keepdim=True)))  # [T,B,1,D] in {0,1}
        if self.record:
            self.ac_ops += float(qk.sum())
        return v * mask

    def forward(self, s: Tensor) -> Tensor:
        q, k, v = self.q_lif(self.q(s)), self.k_lif(self.k(s)), self.v_lif(self.v(s))
        x = self._ssa(q, k, v) if self.mode == "ssa" else self._sdsa(q, k, v)
        return self.proj(x)


class SpikingMLP(nn.Module):
    def __init__(self, dim: int, hidden: int, lif_kw: dict) -> None:
        super().__init__()
        self.fc1 = TBLinearBN(dim, hidden)
        self.lif = MultiStepLIF(**lif_kw)
        self.fc2 = TBLinearBN(hidden, dim)

    def forward(self, s: Tensor) -> Tensor:
        return self.fc2(self.lif(self.fc1(s)))


class SpikingBlock(nn.Module):
    """Spiking Transformer block with membrane shortcut (MS) residuals.

    The residual stream h in R^{T x B x N x D} stays continuous (it plays the role of a
    membrane potential and is what we align with the teacher's hidden states), while
    every weight layer only ever receives binary spikes:
        h <- h + SSA(LIF(h));   h <- h + MLP(LIF(h)).
    """

    def __init__(self, dim: int, num_heads: int, mlp_hidden: int, attn_mode: str,
                 lif_kw: dict) -> None:
        super().__init__()
        self.attn_lif_in = MultiStepLIF(**lif_kw)
        self.attn = SpikingSelfAttention(dim, num_heads, attn_mode, lif_kw)
        self.mlp_lif_in = MultiStepLIF(**lif_kw)
        self.mlp = SpikingMLP(dim, mlp_hidden, lif_kw)

    def forward(self, h: Tensor) -> Tensor:
        h = h + self.attn(self.attn_lif_in(h))
        h = h + self.mlp(self.mlp_lif_in(h))
        return h


# =============================================================================
# 2. Encoding layer: classical PatchEmbed -> input LIF -> spike-driven projection
# =============================================================================
class SpikeEncoder(nn.Module):
    """Keeps the teacher's PatchEmbed (Conv 16x16/16), cls token and position
    embedding as a classical, non-spiking front-end computed once per image. The
    token map is repeated over T steps (direct/current coding), normalised, and fed
    to an input LIF layer. With ``bipolar=True`` the LIF sees [z, -z] so positive and
    negative values are both represented (ON/OFF channels). A Linear+BN on the
    spikes produces the initial residual stream h_0."""

    def __init__(self, teacher: nn.Module, out_dim: int, T: int, bipolar: bool,
                 lif_kw: dict) -> None:
        super().__init__()
        if getattr(teacher, "reg_token", None) is not None:
            raise NotImplementedError("register tokens are not supported")
        d_t = teacher.embed_dim
        self.T, self.bipolar = T, bipolar
        self.patch_embed = copy.deepcopy(teacher.patch_embed)
        self.cls_token = (nn.Parameter(teacher.cls_token.detach().clone())
                          if getattr(teacher, "cls_token", None) is not None else None)
        self.pos_embed = nn.Parameter(teacher.pos_embed.detach().clone())
        self.no_embed_class = bool(getattr(teacher, "no_embed_class", False))
        self.adapt_bn = TBBatchNorm(d_t)
        self.input_lif = MultiStepLIF(**lif_kw)
        self.proj = TBLinearBN(2 * d_t if bipolar else d_t, out_dim)

    def classical_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.patch_embed.parameters()
        yield self.pos_embed
        if self.cls_token is not None:
            yield self.cls_token

    def embed(self, x: Tensor) -> Tensor:
        """ANN path, identical to timm's ``patch_embed`` + ``_pos_embed``: [B, N, D]."""
        z = self.patch_embed(x)
        cls = None if self.cls_token is None else self.cls_token.expand(z.shape[0], -1, -1)
        if self.no_embed_class:
            z = z + self.pos_embed
            return z if cls is None else torch.cat([cls, z], dim=1)
        if cls is not None:
            z = torch.cat([cls, z], dim=1)
        return z + self.pos_embed

    def forward(self, x: Tensor) -> Tensor:
        z = self.embed(x)
        z_t = self.adapt_bn(z.unsqueeze(0).expand(self.T, *z.shape))   # [T,B,N,D]
        if self.bipolar:
            z_t = torch.cat([z_t, -z_t], dim=-1)
        s = self.input_lif(z_t)                                        # binary spikes
        return self.proj(s)                                            # h_0


# =============================================================================
# Student network
# =============================================================================
class SpikingViT(nn.Module):
    """Stage 0 = encoder, stage i (1..L) = block i-1. Output: logits of shape [T,B,C]."""

    def __init__(self, teacher: nn.Module, cfg: Config) -> None:
        super().__init__()
        d_t = teacher.embed_dim
        d_s = cfg.student_dim or d_t
        num_heads = teacher.blocks[0].attn.num_heads
        mlp_hidden = int(round(teacher.blocks[0].mlp.fc1.out_features * d_s / d_t))
        lif_kw = dict(tau=cfg.tau, v_threshold=cfg.v_threshold, surrogate=cfg.surrogate,
                      alpha=cfg.surrogate_alpha, detach_reset=cfg.detach_reset,
                      learnable_tau=cfg.learnable_tau)
        self.T, self.embed_dim = cfg.T, d_s
        self.encoder = SpikeEncoder(teacher, d_s, cfg.T, cfg.bipolar_input, lif_kw)
        self.blocks = nn.ModuleList(
            SpikingBlock(d_s, num_heads, mlp_hidden, cfg.attn_mode, lif_kw)
            for _ in range(len(teacher.blocks)))
        self.norm = nn.LayerNorm(d_s, eps=1e-6)
        self.head = nn.Linear(d_s, teacher.num_classes)
        self.grad_checkpointing = False

    @property
    def num_stages(self) -> int:
        return 1 + len(self.blocks)

    def stage(self, i: int) -> nn.Module:
        return self.encoder if i == 0 else self.blocks[i - 1]

    def forward_stages(self, x: Tensor, start: int, end: int) -> Tensor:
        """Runs stages [start, end). ``x`` is an image batch if start == 0, else h."""
        h = x
        for i in range(start, end):
            m = self.stage(i)
            if self.grad_checkpointing and i > 0 and self.training and torch.is_grad_enabled():
                h = checkpoint(m, h, use_reentrant=False)
            else:
                h = m(h)
        return h

    def forward_features(self, x: Tensor) -> Tensor:
        return self.forward_stages(x, 0, self.num_stages)

    def forward_head(self, h: Tensor) -> Tensor:
        return self.head(self.norm(h[:, :, 0]))  # CLS token at every step -> [T,B,C]

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_head(self.forward_features(x))

    def _freeze_classical(self, train_patch_embed: bool) -> None:
        for p in self.encoder.classical_parameters():
            p.requires_grad_(train_patch_embed)

    def set_trainable_stage(self, stage: int, train_patch_embed: bool) -> None:
        """Freeze (and put in eval mode, so BN running stats stay fixed) everything
        except the stage being distilled."""
        self.requires_grad_(False).eval()
        self.stage(stage).requires_grad_(True).train()
        if stage == 0:
            self._freeze_classical(train_patch_embed)

    def set_trainable_all(self, train_patch_embed: bool) -> None:
        self.requires_grad_(True).train()
        self._freeze_classical(train_patch_embed)


def _copy_linear(dst: TBLinearBN, src: nn.Linear) -> None:
    dst.linear.weight.copy_(src.weight)
    if src.bias is not None:
        dst.bn.bias.copy_(src.bias)


@torch.no_grad()
def init_student_from_teacher(student: SpikingViT, teacher: nn.Module, bipolar: bool) -> None:
    """Warm start: copy the ViT linear weights (the directions of the projections are
    reused; BN re-learns the scales), the final LayerNorm and the classifier."""
    D = teacher.embed_dim
    if student.embed_dim != D:
        print("[init] student_dim != teacher_dim: skipping weight transfer")
        return
    for sb, tb in zip(student.blocks, teacher.blocks):
        qkv_w, qkv_b = tb.attn.qkv.weight, tb.attn.qkv.bias
        for i, lin in enumerate((sb.attn.q, sb.attn.k, sb.attn.v)):
            lin.linear.weight.copy_(qkv_w[i * D:(i + 1) * D])
            if qkv_b is not None:
                lin.bn.bias.copy_(qkv_b[i * D:(i + 1) * D])
        _copy_linear(sb.attn.proj, tb.attn.proj)
        _copy_linear(sb.mlp.fc1, tb.mlp.fc1)
        _copy_linear(sb.mlp.fc2, tb.mlp.fc2)
    if isinstance(teacher.norm, nn.LayerNorm):
        student.norm.load_state_dict(teacher.norm.state_dict())
    student.head.load_state_dict(teacher.head.state_dict())
    # Encoder projection: [I, -I] so that h_0 ~ rate(z+) - rate(z-) at initialisation.
    w = student.encoder.proj.linear.weight
    eye = torch.eye(D, device=w.device, dtype=w.dtype)
    w.copy_(torch.cat([eye, -eye], dim=1) if bipolar else eye)


def build_projections(num_stages: int, d_s: int, d_t: int, mode: str) -> nn.ModuleList:
    """One W_proj per stage, mapping student features (d_s) to the teacher space (d_t).
    Identity-initialised when d_s == d_t. Used only by the distillation losses."""
    if mode == "never" and d_s != d_t:
        raise ValueError("proj_mode='never' requires student_dim == teacher_dim")
    use = mode == "always" or (mode == "auto" and d_s != d_t)
    projs: list[nn.Module] = []
    for _ in range(num_stages):
        if not use:
            projs.append(nn.Identity())
            continue
        lin = nn.Linear(d_s, d_t)
        if d_s == d_t:
            nn.init.eye_(lin.weight)
            nn.init.zeros_(lin.bias)
        projs.append(lin)
    return nn.ModuleList(projs)


# =============================================================================
# Teacher helpers (always under no_grad)
# =============================================================================
@torch.no_grad()
def teacher_embed(teacher: nn.Module, x: Tensor) -> Tensor:
    """h_0 of the timm ViT (same ops as the start of ``forward_features``)."""
    h = teacher.patch_embed(x)
    h = teacher._pos_embed(h)
    h = getattr(teacher, "patch_drop", nn.Identity())(h)
    return getattr(teacher, "norm_pre", nn.Identity())(h)


@torch.no_grad()
def teacher_stage_io(teacher: nn.Module, x: Tensor, stage: int) -> tuple[Tensor, Tensor]:
    """Returns (h_{stage-1}, h_stage): input and output of teacher stage ``stage``.
    Only the first ``stage`` blocks are evaluated; intermediate states are dropped."""
    h = teacher_embed(teacher, x)
    h_in = h
    for blk in teacher.blocks[:stage]:
        h_in = h
        h = blk(h)
    return h_in, h


# =============================================================================
# Data
# =============================================================================
# Imagenette synset -> ImageNet-1k index, so the 1000-way teacher head can be used.
IMAGENETTE_TO_IN1K = {
    "n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
    "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
    "n03445777": 574, "n03888257": 701,
}


class IndexMap:
    """Picklable target transform (DataLoader workers use 'spawn' on macOS)."""

    def __init__(self, mapping: list[int]) -> None:
        self.mapping = mapping

    def __call__(self, y: int) -> int:
        return self.mapping[y]


def _target_transform(classes: list[str]) -> Optional[IndexMap]:
    if len(classes) == 1000:
        return None  # sorted synsets == standard ImageNet-1k indices
    if all(c in IMAGENETTE_TO_IN1K for c in classes):
        return IndexMap([IMAGENETTE_TO_IN1K[c] for c in classes])
    raise ValueError("ImageFolder classes must be ImageNet synsets (full IN-1k or Imagenette)")


def samples_sha256(ds) -> str:
    """Hash of the ImageFolder sample list (paths relative to its root, in dataset order)."""
    rel = "\n".join(os.path.relpath(path, ds.root) for path, _ in ds.samples)
    return hashlib.sha256(rel.encode()).hexdigest()


def data_seed(seed: int, phase: int, epoch: int) -> int:
    """Seed of the data order and augmentations of one epoch of one phase.

    phase = stage index (0..L) during distillation, 100 during fine-tuning. Two runs with the
    same --seed therefore see the same fine-tuning batches in the same order, whether or not
    they ran phase 1, and an interrupted epoch can be replayed exactly after --resume."""
    return (seed * 1_000_003 + phase * 10_007 + epoch) % (2 ** 31 - 1)


@dataclass
class Loaders:
    train_set: Dataset
    monitor: DataLoader            # used during training (held-out split, or the val set)
    test: DataLoader               # official validation set, used once with --final-eval
    label_idx: Optional[list[int]]  # label indices present in the data (restricted top-1)
    kw: dict

    def train(self, seed: int) -> DataLoader:
        """A fresh training loader whose shuffling and worker seeds come from ``seed``."""
        g = torch.Generator().manual_seed(seed)
        return DataLoader(self.train_set, shuffle=True, drop_last=True, generator=g, **self.kw)


def build_loaders(cfg: Config, teacher: nn.Module, device: torch.device) -> Loaders:
    import torchvision

    try:
        dcfg = timm.data.resolve_model_data_config(teacher)
    except AttributeError:  # older timm
        dcfg = timm.data.resolve_data_config({}, model=teacher)
    tf_kw = dict(input_size=dcfg["input_size"], interpolation=dcfg["interpolation"],
                 mean=dcfg["mean"], std=dcfg["std"], crop_pct=dcfg["crop_pct"])
    train_tf = timm.data.create_transform(**tf_kw, is_training=True)
    val_tf = timm.data.create_transform(**tf_kw, is_training=False)

    label_idx: Optional[list[int]] = None
    if cfg.data == "fake":
        size = tuple(dcfg["input_size"])
        n_cls = teacher.num_classes
        train_set = torchvision.datasets.FakeData(cfg.fake_size, size, n_cls, train_tf, random_offset=0)
        val_set = torchvision.datasets.FakeData(cfg.fake_size // 2, size, n_cls, val_tf, random_offset=1)
        monitor_set = val_set
    else:
        train_full = torchvision.datasets.ImageFolder(os.path.expanduser(cfg.train_dir), train_tf)
        val_set = torchvision.datasets.ImageFolder(os.path.expanduser(cfg.val_dir), val_tf)
        tt = _target_transform(train_full.classes)
        train_full.target_transform = tt
        val_set.target_transform = _target_transform(val_set.classes)
        label_idx = sorted(tt.mapping) if tt is not None else None
        if cfg.split_file:
            split = json.loads(Path(cfg.split_file).read_text())
            if split["n_total"] != len(train_full) or split["samples_sha256"] != samples_sha256(train_full):
                raise ValueError(f"{cfg.split_file} does not match the training folder {cfg.train_dir}")
            mon_full = torchvision.datasets.ImageFolder(os.path.expanduser(cfg.train_dir), val_tf)
            mon_full.target_transform = tt
            train_set = Subset(train_full, split["train_idx"])
            monitor_set = Subset(mon_full, split["monitor_idx"])
        else:  # legacy protocol: the validation set is used for monitoring and reporting
            train_set, monitor_set = train_full, val_set

    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
    monitor_loader = DataLoader(monitor_set, shuffle=False, drop_last=False, **kw)
    test_loader = DataLoader(val_set, shuffle=False, drop_last=False, **kw)
    loaders = Loaders(train_set, monitor_loader, test_loader, label_idx, kw)
    if len(loaders.train(0)) == 0:
        raise ValueError("training set smaller than one batch")
    return loaders


def limited(loader: Iterable, max_batches: Optional[int]) -> Iterator:
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        yield batch


def batches_per_epoch(loader: DataLoader, max_batches: Optional[int]) -> int:
    return min(len(loader), max_batches) if max_batches else len(loader)


# =============================================================================
# Optimisation helpers
# =============================================================================
def param_groups(named: Iterable[tuple[str, nn.Parameter]], wd: float) -> list[dict]:
    """No weight decay on norms, biases, time constants, cls/pos embeddings."""
    decay, no_decay = [], []
    for n, p in named:
        if not p.requires_grad:
            continue
        skip = p.ndim <= 1 or n.endswith(("cls_token", "pos_embed"))
        (no_decay if skip else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


def cosine_with_warmup(opt: torch.optim.Optimizer, total: int,
                       warmup_frac: float) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, int(total * warmup_frac))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def optim_step(params: list[nn.Parameter], opt: torch.optim.Optimizer,
               sched: torch.optim.lr_scheduler.LambdaLR, clip: float) -> None:
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(params, clip)
    opt.step()
    sched.step()
    opt.zero_grad(set_to_none=True)


# =============================================================================
# 3. Progressive layer-wise distillation
# =============================================================================
@torch.no_grad()
def distill_io(stage: int, student: SpikingViT, teacher: nn.Module, x: Tensor,
               cfg: Config) -> tuple[Tensor, Tensor]:
    """Input of the student stage and the teacher target, both graph-free.

    * stage 0: input = image, target = teacher h_0.
    * "student": input = output of the already-distilled (frozen) student prefix,
      which matches what the stage will see at inference time.
    * "teacher": input = teacher h_{l-1} repeated over T (teacher forcing).
    """
    t_in, t_out = teacher_stage_io(teacher, x, stage)
    if stage == 0:
        s_in = x
    elif cfg.distill_input == "teacher":
        s_in = t_in.unsqueeze(0).expand(cfg.T, *t_in.shape)
    else:
        s_in = student.forward_stages(x, 0, stage)
    return s_in, t_out


def hidden_mse(proj: nn.Module, s_out: Tensor, target: Tensor, per_step_weight: float) -> Tensor:
    """L = mean over (B, N, D) of (W_proj(mean_t h_s^t) - h_t)^2  (+ lambda * same per step)."""
    loss = F.mse_loss(proj(s_out.mean(dim=0)), target)
    if per_step_weight > 0:
        tgt = target.unsqueeze(0).expand(s_out.shape[0], *target.shape)
        loss = loss + per_step_weight * F.mse_loss(proj(s_out), tgt)
    return loss


@torch.no_grad()
def evaluate_stage(stage: int, student: SpikingViT, teacher: nn.Module, proj: nn.Module,
                   loader: DataLoader, cfg: Config, device: torch.device) -> float:
    student.eval()
    proj.eval()
    total, n = torch.zeros((), device=device), 0
    for x, _ in limited(loader, cfg.eval_batches):
        x = x.to(device, non_blocking=True)
        s_in, target = distill_io(stage, student, teacher, x, cfg)
        total += F.mse_loss(proj(student.stage(stage)(s_in).mean(0)), target)
        n += 1
    return float(total) / max(1, n)


def distill_stage(stage: int, student: SpikingViT, teacher: nn.Module, proj: nn.Module,
                  loaders: Loaders, cfg: Config, device: torch.device) -> dict:
    """Trains stage ``stage`` (+ its W_proj) alone; all other weights are frozen."""
    student.set_trainable_stage(stage, cfg.train_patch_embed)
    proj.requires_grad_(True).train()
    module = student.stage(stage)
    named = [(f"stage.{n}", p) for n, p in module.named_parameters()]
    named += [(f"proj.{n}", p) for n, p in proj.named_parameters()]
    groups = param_groups(named, cfg.weight_decay)
    params = [p for g in groups for p in g["params"]]
    if not params:
        return {"stage": stage, "skipped": True}

    opt = torch.optim.AdamW(groups, lr=cfg.distill_lr)
    spe = batches_per_epoch(loaders.train(0), cfg.max_steps)
    sched = cosine_with_warmup(opt, cfg.distill_epochs * math.ceil(spe / cfg.accum_steps),
                               cfg.warmup_frac)
    opt.zero_grad(set_to_none=True)
    t0, epoch_losses, updates, examples, peak = time.time(), [], 0, 0, 0
    for epoch in range(cfg.distill_epochs):
        student.set_trainable_stage(stage, cfg.train_patch_embed)
        proj.train()
        running, it = torch.zeros((), device=device), 0
        for x, _ in limited(loaders.train(data_seed(cfg.seed, stage, epoch)), cfg.max_steps):
            x = x.to(device, non_blocking=True)
            examples += x.shape[0]
            s_in, target = distill_io(stage, student, teacher, x, cfg)  # no graph
            s_out = module(s_in)                                        # graph of 1 stage only
            loss = hidden_mse(proj, s_out, target, cfg.per_step_weight)
            (loss / cfg.accum_steps).backward()
            it += 1
            if it % cfg.accum_steps == 0:
                optim_step(params, opt, sched, cfg.grad_clip)
                updates += 1
            running += loss.detach()
            if it % cfg.log_every == 0:
                peak = max(peak, mem_bytes(device))
                print(f"  [stage {stage:2d}] ep {epoch} it {it:4d}/{spe} "
                      f"mse={loss.item():.5f} lr={sched.get_last_lr()[0]:.2e} | {mem_str(device)}",
                      flush=True)
            del x, s_in, target, s_out, loss
        if it % cfg.accum_steps:
            optim_step(params, opt, sched, cfg.grad_clip)
            updates += 1
        epoch_losses.append(float(running) / max(1, it))
    peak = max(peak, mem_bytes(device))

    val_mse = evaluate_stage(stage, student, teacher, proj, loaders.monitor, cfg, device)
    student.requires_grad_(False)
    proj.requires_grad_(False)
    del opt, sched, params, groups
    free_memory(device)
    res = {"stage": stage, "train_mse": epoch_losses, "val_mse": val_mse,
           "seconds": round(time.time() - t0, 1), "updates": updates, "examples": examples,
           "peak_mem_bytes": peak,
           "data_seeds": [data_seed(cfg.seed, stage, e) for e in range(cfg.distill_epochs)]}
    print(f"[stage {stage:2d}] train_mse={epoch_losses[-1]:.5f} val_mse={val_mse:.5f} "
          f"({res['seconds']}s) | {mem_str(device)}", flush=True)
    return res


# =============================================================================
# 4. Global fine-tuning (CE + KL) and evaluation
# =============================================================================
def kd_loss(student_logits_t: Tensor, teacher_logits: Tensor, target: Tensor,
            cfg: Config) -> tuple[Tensor, Tensor, Tensor]:
    """L = (1 - a) CE(z_s, y) + a tau^2 KL(softmax(z_t / tau) || softmax(z_s / tau)),
    with z_s = (1/T) sum_t z_s^t the time-averaged student logits."""
    z_s = student_logits_t.mean(dim=0)
    tau = cfg.kd_tau
    ce = F.cross_entropy(z_s, target, label_smoothing=cfg.label_smoothing)
    kl = F.kl_div(F.log_softmax(z_s / tau, dim=-1), F.log_softmax(teacher_logits / tau, dim=-1),
                  reduction="batchmean", log_target=True) * tau * tau
    return (1 - cfg.kd_alpha) * ce + cfg.kd_alpha * kl, ce.detach(), kl.detach()


def topk_correct(logits: Tensor, y: Tensor, ks: tuple[int, ...] = (1, 5)) -> list[Tensor]:
    top = logits.topk(max(ks), dim=-1).indices
    hit = top.eq(y.unsqueeze(1))
    return [hit[:, :k].any(dim=1).sum() for k in ks]


@torch.no_grad()
def evaluate(student: SpikingViT, teacher: nn.Module, loader: DataLoader,
             device: torch.device, max_batches: Optional[int],
             label_idx: Optional[list[int]] = None) -> dict:
    """Top-1/top-5 with the full head; if ``label_idx`` is given, also top-1 with the logits
    restricted to those classes (e.g. the 10 Imagenette classes) and the agreement computed on
    the restricted predictions."""
    student.eval()
    acc = torch.zeros(8, device=device)  # s@1, s@5, t@1, t@5, agree, s@1r, t@1r, agree_r
    idx = torch.tensor(label_idx, device=device) if label_idx else None
    n = 0
    for x, y in limited(loader, max_batches):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        z_s, z_t = student(x).mean(0), teacher(x)
        s1, s5 = topk_correct(z_s, y)
        t1, t5 = topk_correct(z_t, y)
        row = [s1, s5, t1, t5, (z_s.argmax(-1) == z_t.argmax(-1)).sum()]
        if idx is not None:
            ps, pt = idx[z_s[:, idx].argmax(-1)], idx[z_t[:, idx].argmax(-1)]
            row += [(ps == y).sum(), (pt == y).sum(), (ps == pt).sum()]
        else:
            row += [torch.zeros((), device=device, dtype=s1.dtype)] * 3
        acc += torch.stack(row).float()
        n += y.numel()
    a = (acc / max(1, n) * 100).tolist()
    out = {"student_top1": a[0], "student_top5": a[1], "teacher_top1": a[2],
           "teacher_top5": a[3], "agreement": a[4], "n": n}
    if idx is not None:
        out.update(student_top1_restricted=a[5], teacher_top1_restricted=a[6],
                   agreement_restricted=a[7], n_classes_restricted=len(label_idx))
    return out


def finetune(student: SpikingViT, teacher: nn.Module, loaders: Loaders, cfg: Config,
             device: torch.device, out: Path) -> list[dict]:
    """Global CE + KL fine-tuning. A checkpoint (model, optimiser, scheduler, history) is
    written after every epoch to ``ft_last.pt``; with --resume, training restarts at the next
    epoch with the same data seeds as an uninterrupted run."""
    student.set_trainable_all(cfg.train_patch_embed)
    student.grad_checkpointing = cfg.grad_checkpointing
    groups = param_groups(student.named_parameters(), cfg.weight_decay)
    params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, lr=cfg.finetune_lr)
    spe = batches_per_epoch(loaders.train(0), cfg.max_steps)
    sched = cosine_with_warmup(opt, cfg.finetune_epochs * math.ceil(spe / cfg.accum_steps),
                               cfg.warmup_frac)
    ck_path = out / "ft_last.pt"
    start, history, best, updates, examples = 0, [], -1.0, 0, 0
    if cfg.resume and ck_path.exists():
        ck = torch.load(ck_path, map_location=device)
        student.load_state_dict(ck["student"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, history, best = ck["epoch"] + 1, ck["history"], ck["best"]
        updates, examples = ck["updates"], ck["examples"]
        print(f"[ft] resumed after epoch {ck['epoch']}", flush=True)
    opt.zero_grad(set_to_none=True)
    for epoch in range(start, cfg.finetune_epochs):
        student.train()
        t0, it, peak = time.time(), 0, 0
        seed_e = data_seed(cfg.seed, 100, epoch)
        for x, y in limited(loaders.train(seed_e), cfg.max_steps):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            examples += x.shape[0]
            with torch.no_grad():
                z_t = teacher(x)
            loss, ce, kl = kd_loss(student(x), z_t, y, cfg)
            (loss / cfg.accum_steps).backward()
            it += 1
            if it % cfg.accum_steps == 0:
                optim_step(params, opt, sched, cfg.grad_clip)
                updates += 1
            if it % cfg.log_every == 0:
                peak = max(peak, mem_bytes(device))
                print(f"  [ft] ep {epoch} it {it:4d}/{spe} loss={loss.item():.4f} "
                      f"ce={ce.item():.4f} kl={kl.item():.4f} lr={sched.get_last_lr()[0]:.2e} "
                      f"| {mem_str(device)}", flush=True)
            del x, y, z_t, loss, ce, kl
        if it % cfg.accum_steps:
            optim_step(params, opt, sched, cfg.grad_clip)
            updates += 1
        peak = max(peak, mem_bytes(device))
        free_memory(device)
        ev = evaluate(student, teacher, loaders.monitor, device, cfg.eval_batches, loaders.label_idx)
        ev.update(epoch=epoch, seconds=round(time.time() - t0, 1), updates_cum=updates,
                  examples_cum=examples, data_seed=seed_e, peak_mem_bytes=peak, split="monitor")
        history.append(ev)
        print(f"[ft] epoch {epoch}: student@1={ev['student_top1']:.2f} "
              f"teacher@1={ev['teacher_top1']:.2f} ({ev['seconds']}s)", flush=True)
        if ev["student_top1"] > best:
            best = ev["student_top1"]
            torch.save({"student": student.state_dict(), "epoch": epoch}, out / "student_best.pt")
        tmp = out / "ft_last.pt.tmp"  # atomic replace: an interruption never leaves a broken file
        torch.save({"student": student.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch, "history": history, "best": best,
                    "updates": updates, "examples": examples}, tmp)
        os.replace(tmp, ck_path)
    student.grad_checkpointing = False
    del opt, sched
    free_memory(device)
    return history


# =============================================================================
# Profiling: temporal sparsity, synaptic operations, energy
# =============================================================================
E_MAC_PJ, E_AC_PJ = 4.6, 0.9  # 32-bit float MAC / AC, 45 nm CMOS (Horowitz, 2014)


def _is_binary(x: Tensor) -> bool:
    return bool(torch.all((x == 0) | (x == 1)))


class OpCounter:
    """Forward hooks on Linear / Conv2d. A layer whose input is binary performs
    #spikes x fan_out accumulations (AC); otherwise it performs dense MACs."""

    def __init__(self) -> None:
        self.mac = 0.0
        self.ac = 0.0
        self._handles: list = []

    def attach(self, model: nn.Module) -> "OpCounter":
        for m in model.modules():
            if isinstance(m, nn.Linear):
                self._handles.append(m.register_forward_hook(self._linear))
            elif isinstance(m, nn.Conv2d):
                self._handles.append(m.register_forward_hook(self._conv))
        return self

    def _linear(self, m: nn.Linear, inp: tuple[Tensor, ...], out: Tensor) -> None:
        x = inp[0]
        if _is_binary(x):
            self.ac += float(x.sum()) * m.out_features
        else:
            self.mac += x.numel() * m.out_features

    def _conv(self, m: nn.Conv2d, inp: tuple[Tensor, ...], out: Tensor) -> None:
        k = m.kernel_size[0] * m.kernel_size[1] * m.in_channels // m.groups
        self.mac += out.numel() * k

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


@torch.no_grad()
def profile_student(student: SpikingViT, teacher: nn.Module, loader: DataLoader,
                    device: torch.device, max_batches: Optional[int]) -> dict:
    """Firing rates of every LIF layer (overall and per time step), global temporal
    sparsity, and per-image energy estimates for student and teacher."""
    student.eval()
    lifs = {n: m for n, m in student.named_modules() if isinstance(m, MultiStepLIF)}
    attns = [m for m in student.modules() if isinstance(m, SpikingSelfAttention)]
    for m in lifs.values():
        m.reset_stats()
        m.record = True
    for a in attns:
        a.ac_ops, a.record = 0.0, True
    s_ops, t_ops = OpCounter().attach(student), OpCounter().attach(teacher)
    n_img, t_attn_mac = 0, 0.0
    n_tok = teacher.patch_embed.num_patches + getattr(teacher, "num_prefix_tokens", 1)
    try:
        for x, _ in limited(loader, max_batches):
            x = x.to(device, non_blocking=True)
            student(x)
            teacher(x)
            # QK^T and AV of the teacher (fused SDPA is not an nn.Module): 2 N^2 D / block.
            t_attn_mac += len(teacher.blocks) * 2.0 * x.shape[0] * n_tok * n_tok * teacher.embed_dim
            n_img += x.shape[0]
    finally:
        s_ops.detach()
        t_ops.detach()
        for m in lifs.values():
            m.record = False
        for a in attns:
            a.record = False

    layers, tot_spk, tot_neu = {}, None, 0
    for name, m in lifs.items():
        if m.spike_sum_t is None:
            continue
        T = m.spike_sum_t.numel()
        rate_t = (m.spike_sum_t / m.neuron_count).tolist()
        layers[name] = {"rate": float(m.spike_sum_t.sum()) / (m.neuron_count * T),
                        "rate_per_t": rate_t, "neurons_per_step": m.neuron_count / max(1, n_img)}
        tot_spk = m.spike_sum_t.clone() if tot_spk is None else tot_spk + m.spike_sum_t
        tot_neu += m.neuron_count
        m.reset_stats()

    attn_ac = sum(a.ac_ops for a in attns)
    s_mac, s_ac = s_ops.mac / n_img, (s_ops.ac + attn_ac) / n_img
    t_mac = (t_ops.mac + t_attn_mac) / n_img
    e_s = (s_mac * E_MAC_PJ + s_ac * E_AC_PJ) * 1e-9  # pJ -> mJ
    e_t = t_mac * E_MAC_PJ * 1e-9
    return {
        "global_rate": float(tot_spk.sum()) / (tot_neu * tot_spk.numel()),
        "global_rate_per_t": (tot_spk / tot_neu).tolist(),
        "sparsity": 1.0 - float(tot_spk.sum()) / (tot_neu * tot_spk.numel()),
        "layers": layers,
        "student_MAC_per_img": s_mac, "student_AC_per_img": s_ac,
        "teacher_MAC_per_img": t_mac,
        "student_energy_mJ": e_s, "teacher_energy_mJ": e_t,
        "energy_ratio_teacher_over_student": e_t / max(e_s, 1e-12),
        "images": n_img,
    }


def print_profile(p: dict) -> None:
    print("\n=== Temporal sparsity profile ===")
    print(f"global firing rate : {p['global_rate']:.4f}   (sparsity {100 * p['sparsity']:.2f}%)")
    print("rate per time step : " + "  ".join(f"t{t}={r:.4f}" for t, r in enumerate(p["global_rate_per_t"])))
    print(f"{'layer':48s} {'rate':>7s}   per-step")
    for name, r in p["layers"].items():
        print(f"{name:48s} {r['rate']:7.4f}   " + " ".join(f"{v:.3f}" for v in r["rate_per_t"]))
    print(f"student: {p['student_MAC_per_img'] / 1e6:.1f} M MAC + {p['student_AC_per_img'] / 1e6:.1f} M AC"
          f" -> {p['student_energy_mJ']:.3f} mJ/img")
    print(f"teacher: {p['teacher_MAC_per_img'] / 1e6:.1f} M MAC -> {p['teacher_energy_mJ']:.3f} mJ/img"
          f"   (ratio x{p['energy_ratio_teacher_over_student']:.2f})\n")


# =============================================================================
# Main
# =============================================================================
def save_ckpt(path: Path, student: SpikingViT, projs: nn.ModuleList, stage_done: int) -> None:
    torch.save({"student": student.state_dict(), "projs": projs.state_dict(),
                "stage_done": stage_done}, path)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(cfg: Config, device: torch.device) -> dict:
    here = Path(__file__).resolve().parent

    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=here, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    import torchvision
    return {"git_commit": git("rev-parse", "HEAD"),
            "script_modified_since_commit": bool(git("status", "--porcelain", "--", Path(__file__).name)),
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "timm": timm.__version__, "device": str(device),
            "split_file_sha256": file_sha256(cfg.split_file) if cfg.split_file else None,
            "init_checkpoint_sha256": file_sha256(cfg.init_checkpoint) if cfg.init_checkpoint else None,
            "data_seed_formula": "(seed*1000003 + phase*10007 + epoch) % (2**31-1); "
                                 "phase = stage (0..L) or 100 for fine-tuning"}


def main() -> None:
    cfg = parse_args()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "metrics.json"
    if cfg.resume and mpath.exists() and json.loads(mpath.read_text()).get("complete"):
        print(f"{out} is already complete; nothing to do")
        return
    print(f"device={device} | T={cfg.T} | attn={cfg.attn_mode} | surrogate={cfg.surrogate}")

    teacher = timm.create_model(cfg.teacher_name, pretrained=cfg.pretrained)
    teacher = teacher.to(device).eval().requires_grad_(False)
    loaders = build_loaders(cfg, teacher, device)
    print(f"train={len(loaders.train_set)} monitor={len(loaders.monitor.dataset)} "
          f"test={len(loaders.test.dataset)} (split_file={cfg.split_file})")

    student = SpikingViT(teacher, cfg).to(device)
    if cfg.init_from_teacher:
        init_student_from_teacher(student, teacher, cfg.bipolar_input)
    if cfg.init_checkpoint:
        ck = torch.load(cfg.init_checkpoint, map_location=device)
        if "stage_done" in ck and ck["stage_done"] != student.num_stages - 1:
            raise ValueError(f"{cfg.init_checkpoint}: phase 1 incomplete (stage_done={ck['stage_done']})")
        student.load_state_dict(ck["student"])
        print(f"student initialised from {cfg.init_checkpoint}")
    if cfg.distill_input == "teacher" and student.embed_dim != teacher.embed_dim:
        raise ValueError("teacher forcing requires student_dim == teacher_dim")
    projs = build_projections(student.num_stages, student.embed_dim, teacher.embed_dim,
                              cfg.proj_mode).to(device)
    print(f"teacher params={count_params(teacher):,} | student params={count_params(student):,}"
          f" | W_proj params={count_params(projs):,}")

    metrics: dict = {"config": asdict(cfg), "provenance": provenance(cfg, device),
                     "distill": [], "complete": False}
    if cfg.resume and mpath.exists():  # keep what an interrupted run already recorded
        prev = json.loads(mpath.read_text())
        metrics["distill"] = prev.get("distill", [])
        if "after_distill" in prev:
            metrics["after_distill"] = prev["after_distill"]
        metrics["resumed"] = prev.get("resumed", 0) + 1

    def save_metrics() -> None:
        tmp = out / "metrics.json.tmp"
        tmp.write_text(json.dumps(metrics, indent=2))
        os.replace(tmp, mpath)

    save_metrics()  # written at start, so that a later --resume can count interruptions

    start_stage, latest = 0, out / "distill_latest.pt"
    if cfg.resume and latest.exists():
        ck = torch.load(latest, map_location=device)
        student.load_state_dict(ck["student"])
        projs.load_state_dict(ck["projs"])
        start_stage = ck["stage_done"] + 1
        metrics["distill"] = [d for d in metrics["distill"] if d["stage"] < start_stage]
        print(f"resumed after stage {ck['stage_done']}")

    if not cfg.skip_distill:
        print(f"\n=== Progressive distillation: {student.num_stages} stages ===")
        for stage in range(start_stage, student.num_stages):
            metrics["distill"].append(distill_stage(stage, student, teacher, projs[stage],
                                                    loaders, cfg, device))
            save_ckpt(latest, student, projs, stage)
            save_metrics()
        if "after_distill" not in metrics:
            metrics["after_distill"] = evaluate(student, teacher, loaders.monitor, device,
                                                cfg.eval_batches, loaders.label_idx)
            metrics["after_distill"]["split"] = "monitor"
            save_metrics()
        print(f"after distillation (monitor): {metrics['after_distill']}")

    if not cfg.skip_finetune:
        print("\n=== Global fine-tuning (CE + KL) ===")
        metrics["finetune"] = finetune(student, teacher, loaders, cfg, device, out)
        save_metrics()

    # Final evaluation: on the official validation set only with --final-eval. Without a split
    # file, monitor and test are the same set (legacy protocol of the preliminary runs).
    final_loader = loaders.test if cfg.final_eval else loaders.monitor
    split_name = "test" if (cfg.final_eval or not cfg.split_file) else "monitor"
    if cfg.skip_finetune and "after_distill" in metrics:
        metrics["final"] = dict(metrics["after_distill"])
    else:
        metrics["final"] = evaluate(student, teacher, final_loader, device, cfg.eval_batches,
                                    loaders.label_idx)
        metrics["final"]["split"] = split_name
        print(f"final ({split_name}): {metrics['final']}")
        metrics["profile"] = profile_student(student, teacher, final_loader, device, cfg.eval_batches)
        metrics["profile"]["split"] = split_name
        print_profile(metrics["profile"])

    torch.save({"student": student.state_dict(), "config": asdict(cfg)}, out / "student_final.pt")
    metrics["complete"] = True
    save_metrics()
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print(f"saved to {out.resolve()}")


if __name__ == "__main__":
    main()
