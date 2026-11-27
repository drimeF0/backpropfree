"""Unified Cross-Scale Model Loader & Layer-Stepping Abstraction (SmolLM2-360M fp16 + Ornith-1.0-9B 8-bit)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from model_loader import (
    dequantize_full,
    dequantize_rows,
    get_eos_token_ids,
    get_layers,
    get_lm_head,
    get_lm_norm,
    get_text_model,
    is_8bit_linear,
    load_ornith_8bit,
    prepare_layer_inputs as _prepare_ornith_inputs,
    slice_cm_map,
    slice_pe,
)

SMALL_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LARGE_MODEL_ID = "ornith-ai/Ornith-1.0-9B"
LAB_DIR = Path(__file__).resolve().parent


@dataclass
class ModelSpec:
    scale: str  # "small" or "large"
    model_id: str
    is_ornith: bool
    is_8bit: bool
    hidden_size: int
    intermediate_size: int
    num_layers: int
    vocab_size: int
    layer_types: List[str]
    zero_centered_norm: bool
    supports_thinking: bool
    static_vram_mb: float = 0.0


def is_zero_centered_rmsnorm(norm: nn.Module) -> bool:
    return "Qwen3_5RMSNorm" in type(norm).__name__


def get_rmsnorm_gamma(norm: nn.Module) -> torch.Tensor:
    """Return effective RMSNorm gamma multiplier (1.0 + weight for Qwen3.5; weight for Llama)."""
    if is_zero_centered_rmsnorm(norm):
        return 1.0 + norm.weight.float()
    return norm.weight.float()


def detect_model_spec(model: nn.Module, scale: str = "auto", static_vram_mb: float = 0.0) -> ModelSpec:
    lm = get_text_model(model)
    cfg = lm.config
    layers = get_layers(model)
    n_layers = len(layers)
    layer_types = list(getattr(cfg, "layer_types", ["full_attention"] * n_layers))
    is_ornith = "Qwen3_5" in type(model).__name__ or hasattr(cfg, "layer_types")
    norm = get_lm_norm(model)
    first_down = layers[0].mlp.down_proj
    base_down = getattr(first_down, "base", first_down)
    is_8bit = is_8bit_linear(base_down)
    if scale == "auto":
        scale = "large" if is_ornith or int(cfg.hidden_size) >= 2048 else "small"
    model_id = LARGE_MODEL_ID if is_ornith else SMALL_MODEL_ID
    return ModelSpec(
        scale=scale,
        model_id=model_id,
        is_ornith=is_ornith,
        is_8bit=is_8bit,
        hidden_size=int(cfg.hidden_size),
        intermediate_size=int(getattr(cfg, "intermediate_size", layers[0].mlp.gate_proj.out_features)),
        num_layers=n_layers,
        vocab_size=int(get_lm_head(model).weight.shape[0]),
        layer_types=layer_types,
        zero_centered_norm=is_zero_centered_rmsnorm(norm),
        supports_thinking=bool(is_ornith),
        static_vram_mb=float(static_vram_mb),
    )


def load_model_by_scale(
    scale: str,
    device: str = "cuda",
) -> Tuple[nn.Module, AutoTokenizer, ModelSpec]:
    """Load either Small (SmolLM2-360M fp16) or Large (8-bit Ornith-1.0-9B) cleanly."""
    from guards import measure_static_vram_mb

    if scale == "small":
        tok = AutoTokenizer.from_pretrained(SMALL_MODEL_ID)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            SMALL_MODEL_ID,
            dtype=torch.float16,
            device_map={"": 0} if device.startswith("cuda") else None,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        static_mb = measure_static_vram_mb(model)
        spec = detect_model_spec(model, scale="small", static_vram_mb=static_mb)
        return model, tok, spec

    if scale == "large":
        model, tok = load_ornith_8bit(device=device, dtype=torch.float16, drop_visual=True)
        static_mb = measure_static_vram_mb(model)
        spec = detect_model_spec(model, scale="large", static_vram_mb=static_mb)
        return model, tok, spec

    raise ValueError(f"Unknown model scale: {scale!r} (expected 'small' or 'large')")


def apply_chat_safe(
    tokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> str:
    """Format chat messages cleanly across both SmolLM2 and Qwen3.5/Ornith templates."""
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


def get_layer_type(model: nn.Module, l_idx: int) -> str:
    lm = get_text_model(model)
    if hasattr(lm.config, "layer_types"):
        return str(lm.config.layer_types[l_idx])
    return "full_attention"


def prepare_layer_inputs_unified(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Prepare embeddings, attention masks, and rotary embeddings for both Llama and Qwen3.5."""
    lm = get_text_model(model)
    if hasattr(lm.config, "layer_types"):
        return _prepare_ornith_inputs(model, input_ids, attention_mask)

    emb = lm.embed_tokens(input_ids)
    seqlen = emb.shape[1]
    pids = torch.arange(seqlen, device=emb.device).unsqueeze(0)
    cm = create_causal_mask(
        config=lm.config,
        inputs_embeds=emb,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=pids,
    )
    pe = lm.rotary_emb(emb, pids)
    return emb, {"full_attention": cm}, pe, pids


def step_layer_unified(
    model: nn.Module,
    l_idx: int,
    h: torch.Tensor,
    cm_map: Dict[str, Optional[torch.Tensor]],
    pe: Tuple[torch.Tensor, torch.Tensor],
    text_pids: torch.Tensor,
) -> torch.Tensor:
    """Execute one decoder layer cleanly on both Llama and hybrid Qwen3.5."""
    lm = get_text_model(model)
    layer = lm.layers[l_idx]
    lt = get_layer_type(model, l_idx)
    cm = cm_map.get(lt)
    if hasattr(lm.config, "layer_types"):
        return layer(
            h,
            position_embeddings=pe,
            attention_mask=cm,
            position_ids=text_pids,
            past_key_values=None,
            use_cache=False,
        )
    return layer(
        h,
        attention_mask=cm,
        position_embeddings=pe,
        use_cache=False,
    )


def step_token_mixer_unified(
    model: nn.Module,
    l_idx: int,
    h_in: torch.Tensor,
    cm_map: Dict[str, Optional[torch.Tensor]],
    pe: Tuple[torch.Tensor, torch.Tensor],
    text_pids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run input_layernorm + mixer + residual + post_attention_layernorm for both Llama and Qwen3.5."""
    lm = get_text_model(model)
    layer = lm.layers[l_idx]
    lt = get_layer_type(model, l_idx)
    normed = layer.input_layernorm(h_in)
    if lt == "linear_attention":
        mixed = layer.linear_attn(
            hidden_states=normed,
            cache_params=None,
            attention_mask=cm_map["linear_attention"],
        )
    elif hasattr(lm.config, "layer_types"):
        mixed, _ = layer.self_attn(
            hidden_states=normed,
            attention_mask=cm_map["full_attention"],
            position_ids=text_pids,
            past_key_values=None,
            position_embeddings=pe,
        )
    else:
        mixed, _ = layer.self_attn(
            hidden_states=normed,
            attention_mask=cm_map["full_attention"],
            position_embeddings=pe,
            use_cache=False,
        )
    h_mid = h_in + mixed
    x_mlp = layer.post_attention_layernorm(h_mid)
    return h_mid, x_mlp


__all__ = [
    "LARGE_MODEL_ID",
    "ModelSpec",
    "SMALL_MODEL_ID",
    "apply_chat_safe",
    "dequantize_full",
    "dequantize_rows",
    "detect_model_spec",
    "get_eos_token_ids",
    "get_layer_type",
    "get_layers",
    "get_lm_head",
    "get_lm_norm",
    "get_rmsnorm_gamma",
    "get_text_model",
    "is_8bit_linear",
    "is_zero_centered_rmsnorm",
    "load_model_by_scale",
    "prepare_layer_inputs_unified",
    "slice_cm_map",
    "slice_pe",
    "step_layer_unified",
    "step_token_mixer_unified",
]
