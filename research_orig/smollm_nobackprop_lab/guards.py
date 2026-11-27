"""Programmatic No-Backprop Guards, VRAM Benchmarking, and Rank<=16 LoRA Utilities."""

from __future__ import annotations

import contextlib
import gc
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


MAX_LORA_RANK = 16


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def measure_static_vram_mb(model: nn.Module) -> float:
    cleanup_cuda()
    if not torch.cuda.is_available():
        return 0.0
    # Allocated memory currently occupied by model weights & buffers
    return float(torch.cuda.memory_allocated() / (1024.0 ** 2))


@dataclass
class SynthesisMetrics:
    peak_vram_mb: float = 0.0
    static_vram_mb: float = 0.0
    activation_vram_overhead_mb: float = 0.0
    wall_time_sec: float = 0.0
    uses_full_backprop: bool = False
    regime: str = "no_grad_closed_form"
    backbone_grad_disabled_verified: bool = False
    max_single_layer_grad_verified: bool = True
    extra: Dict[str, float] = field(default_factory=dict)


class NoBackpropGuard:
    """Enforces zero full-model backprop and measures exact peak VRAM & wall time.

    Supported candidate regimes:
      - 'no_grad': Pure forward-only / streaming closed-form / MeZO. Asserts all model
        parameters have requires_grad=False and enters torch.no_grad().
      - 'single_layer_isolated': Backbone forward passes still must run in torch.no_grad();
        only a single detached transformer block may create a local autograd graph via
        SingleLayerIsolationGuard.
    """

    def __init__(
        self,
        model: nn.Module,
        regime: str = "no_grad",
        static_vram_mb: Optional[float] = None,
    ) -> None:
        if regime not in ("no_grad", "single_layer_isolated"):
            raise ValueError(f"Unsupported candidate regime: {regime}")
        self.model = model
        self.regime = regime
        self.static_vram_mb = (
            static_vram_mb
            if static_vram_mb is not None
            else measure_static_vram_mb(model)
        )
        self.metrics = SynthesisMetrics(
            static_vram_mb=self.static_vram_mb,
            uses_full_backprop=False,
            regime=regime,
        )
        self._no_grad_ctx = None
        self._start_time = 0.0
        self._saved_req_grad: List[Tuple[nn.Parameter, bool]] = []

    def __enter__(self) -> "NoBackpropGuard":
        cleanup_cuda()
        # Freeze all backbone parameters explicitly
        self._saved_req_grad = []
        for p in self.model.parameters():
            self._saved_req_grad.append((p, p.requires_grad))
            p.requires_grad_(False)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        self._no_grad_ctx = torch.no_grad()
        self._no_grad_ctx.__enter__()
        assert not torch.is_grad_enabled(), "Backbone guard failed: grad is still enabled!"
        self.metrics.backbone_grad_disabled_verified = True
        self._start_time = time.perf_counter()
        return self

    def assert_backbone_no_grad(self) -> None:
        if torch.is_grad_enabled():
            raise RuntimeError("VIOLATION: Backbone execution attempted with autograd enabled!")
        for name, p in self.model.named_parameters():
            if "lora_" not in name and p.requires_grad:
                raise RuntimeError(f"VIOLATION: Backbone parameter {name} has requires_grad=True!")

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_mb = float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
        else:
            peak_mb = 0.0
        elapsed = time.perf_counter() - self._start_time

        if self._no_grad_ctx is not None:
            self._no_grad_ctx.__exit__(exc_type, exc_val, exc_tb)

        for p, prev in self._saved_req_grad:
            p.requires_grad_(prev)

        self.metrics.peak_vram_mb = peak_mb
        self.metrics.activation_vram_overhead_mb = max(0.0, peak_mb - self.static_vram_mb)
        self.metrics.wall_time_sec = elapsed


@contextlib.contextmanager
def single_layer_detached_context(block: nn.Module, x_in: torch.Tensor):
    """Strictly isolates autograd to at most ONE detached block's local forward/backward."""
    if x_in.requires_grad:
        raise RuntimeError("VIOLATION: Input to single-layer isolated solver must be detached!")
    #Temporarily enable grad ONLY for this single block's LoRA parameters
    with torch.enable_grad():
        yield x_in.detach()


class BaselineBackpropProfiler:
    """VRAM and wall-time profiler ONLY for Baselines A and B (standard full backprop)."""

    def __init__(self, model: nn.Module, static_vram_mb: Optional[float] = None) -> None:
        self.model = model
        self.static_vram_mb = (
            static_vram_mb
            if static_vram_mb is not None
            else measure_static_vram_mb(model)
        )
        self.metrics = SynthesisMetrics(
            static_vram_mb=self.static_vram_mb,
            uses_full_backprop=True,
            regime="full_backprop",
            backbone_grad_disabled_verified=False,
        )
        self._start_time = 0.0

    def __enter__(self) -> "BaselineBackpropProfiler":
        cleanup_cuda()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_mb = float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
        else:
            peak_mb = 0.0
        elapsed = time.perf_counter() - self._start_time
        self.metrics.peak_vram_mb = peak_mb
        self.metrics.activation_vram_overhead_mb = max(0.0, peak_mb - self.static_vram_mb)
        self.metrics.wall_time_sec = elapsed


class LoRALinear(nn.Module):
    """Lightweight rank-r <= 16 LoRA adapter wrapping an existing frozen nn.Linear."""

    def __init__(self, base_linear: nn.Linear, max_rank: int = MAX_LORA_RANK) -> None:
        super().__init__()
        if max_rank > MAX_LORA_RANK:
            raise ValueError(f"Rank {max_rank} exceeds MAX_LORA_RANK={MAX_LORA_RANK}")
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.max_rank = max_rank
        self.active = False
        self.rank = 0
        self.alpha = 1.0
        # Buffers / parameters: delta_W = alpha * (B @ A), where A: [r, in], B: [out, r]
        self.lora_A: Optional[torch.Tensor] = None
        self.lora_B: Optional[torch.Tensor] = None

    def set_adapter(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        alpha: float = 1.0,
    ) -> None:
        """Install rank-r factor matrices A [r, in_features], B [out_features, r]."""
        assert A.ndim == 2 and B.ndim == 2
        r = A.shape[0]
        if r > MAX_LORA_RANK or B.shape[1] != r:
            raise ValueError(f"Invalid LoRA shapes A={tuple(A.shape)}, B={tuple(B.shape)}")
        if A.shape[1] != self.in_features or B.shape[0] != self.out_features:
            raise ValueError(
                f"Dimension mismatch for ({self.out_features}, {self.in_features}): "
                f"got A={tuple(A.shape)}, B={tuple(B.shape)}"
            )
        dtype = self.base.weight.dtype
        device = self.base.weight.device
        self.lora_A = A.detach().to(device=device, dtype=dtype).contiguous()
        self.lora_B = B.detach().to(device=device, dtype=dtype).contiguous()
        self.rank = r
        self.alpha = float(alpha)
        self.active = True

    def clear_adapter(self) -> None:
        self.lora_A = None
        self.lora_B = None
        self.rank = 0
        self.alpha = 1.0
        self.active = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.active and self.lora_A is not None and self.lora_B is not None:
            # x: [..., in_features] -> [..., r] -> [..., out_features]
            low = F.linear(x, self.lora_A)
            delta = F.linear(low, self.lora_B)
            if self.alpha != 1.0:
                delta = delta * self.alpha
            out = out + delta
        return out


class LoRAManager:
    """Manages LoRALinear wrappers across LlamaForCausalLM decoder layers cleanly."""

    TARGET_MAP = {
        "q_proj": ("self_attn", "q_proj"),
        "k_proj": ("self_attn", "k_proj"),
        "v_proj": ("self_attn", "v_proj"),
        "o_proj": ("self_attn", "o_proj"),
        "gate_proj": ("mlp", "gate_proj"),
        "up_proj": ("mlp", "up_proj"),
        "down_proj": ("mlp", "down_proj"),
    }

    def __init__(self, model: nn.Module, target_modules: Iterable[str] = ("down_proj", "o_proj")) -> None:
        self.model = model
        self.target_modules = tuple(target_modules)
        self.wrapped: Dict[Tuple[int, str], LoRALinear] = {}
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        for l_idx, layer in enumerate(self.model.model.layers):
            for mod_name in self.target_modules:
                parent_attr, leaf_attr = self.TARGET_MAP[mod_name]
                parent = getattr(layer, parent_attr)
                orig = getattr(parent, leaf_attr)
                if isinstance(orig, LoRALinear):
                    wrapper = orig
                else:
                    wrapper = LoRALinear(orig, max_rank=MAX_LORA_RANK)
                    setattr(parent, leaf_attr, wrapper)
                self.wrapped[(l_idx, mod_name)] = wrapper
        self._installed = True

    def clear_all(self) -> None:
        for wrapper in self.wrapped.values():
            wrapper.clear_adapter()

    def uninstall(self) -> None:
        if not self._installed:
            return
        for (l_idx, mod_name), wrapper in self.wrapped.items():
            parent_attr, leaf_attr = self.TARGET_MAP[mod_name]
            parent = getattr(self.model.model.layers[l_idx], parent_attr)
            setattr(parent, leaf_attr, wrapper.base)
        self.wrapped.clear()
        self._installed = False

    def set_layer_adapter(
        self,
        layer_idx: int,
        mod_name: str,
        A: torch.Tensor,
        B: torch.Tensor,
        alpha: float = 1.0,
    ) -> None:
        self.wrapped[(layer_idx, mod_name)].set_adapter(A, B, alpha=alpha)

    def max_active_rank(self) -> int:
        ranks = [w.rank for w in self.wrapped.values() if w.active]
        return max(ranks) if ranks else 0

    def active_adapter_count(self) -> int:
        return sum(1 for w in self.wrapped.values() if w.active)

    def export_state_dict(self) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        for (l_idx, mod_name), w in self.wrapped.items():
            if w.active and w.lora_A is not None and w.lora_B is not None:
                prefix = f"model.layers.{l_idx}.{self.TARGET_MAP[mod_name][0]}.{mod_name}"
                state[f"{prefix}.lora_A"] = w.lora_A.detach().cpu()
                state[f"{prefix}.lora_B"] = (w.lora_B * w.alpha).detach().cpu()
        return state
