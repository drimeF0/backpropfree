"""8-bit Quantized Model Loader and Hybrid Qwen3.5 Layer-Stepping Utilities for Ornith-1.0-9B."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration
from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

MODEL_ID = "ornith-ai/Ornith-1.0-9B"
COMPUTE_DTYPE = torch.float16
_LOCAL_QUANTIZED_DIR = Path(__file__).resolve().parent / "ornith_8bit_bnb"
_LEGACY_QUANTIZED_DIR = Path("/content/ornith_nobackprop_lab/ornith_8bit_bnb")
QUANTIZED_DIR = (
    _LEGACY_QUANTIZED_DIR
    if (_LEGACY_QUANTIZED_DIR / "config.json").exists() and not (_LOCAL_QUANTIZED_DIR / "config.json").exists()
    else _LOCAL_QUANTIZED_DIR
)


def get_text_model(model: nn.Module) -> nn.Module:
    """Return the underlying Qwen3_5TextModel regardless of wrapper class."""
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model, "language_model"):
        return model.language_model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    raise AttributeError(f"Could not locate text backbone on {type(model)}")


def get_layers(model: nn.Module) -> nn.ModuleList:
    return get_text_model(model).layers


def get_lm_norm(model: nn.Module) -> nn.Module:
    return get_text_model(model).norm


def get_lm_head(model: nn.Module) -> nn.Module:
    return model.lm_head


def get_eos_token_ids(tokenizer) -> list[int]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    if tokenizer.pad_token_id is not None and int(tokenizer.pad_token_id) not in ids:
        ids.append(int(tokenizer.pad_token_id))
    return ids or [248046, 248044]


def _save_quantized_snapshot(model: nn.Module, tokenizer: AutoTokenizer) -> None:
    """Persist the already-quantized 8-bit weights so later processes skip re-quantization."""
    try:
        QUANTIZED_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(QUANTIZED_DIR), safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(str(QUANTIZED_DIR))
        print(f"[model_loader] Saved quantized 8-bit snapshot to {QUANTIZED_DIR}", flush=True)
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[model_loader] WARNING: could not save quantized snapshot: {exc}", flush=True)


def load_ornith_8bit(
    device: str = "cuda",
    dtype: torch.dtype = COMPUTE_DTYPE,
    drop_visual: bool = True,
    skip_modules: Tuple[str, ...] = ("lm_head",),
    use_saved_quantized: bool = True,
    save_quantized: bool = True,
) -> Tuple[nn.Module, AutoTokenizer]:
    """Load ornith-ai/Ornith-1.0-9B in 8-bit LLM.int8() quantization on Tesla T4.

    If a previously saved quantized snapshot exists it is loaded directly (no re-quantization).
    Otherwise the fp16 checkpoint is quantized once and, when possible, cached to QUANTIZED_DIR.
    """
    if use_saved_quantized and (QUANTIZED_DIR / "config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(str(QUANTIZED_DIR))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(QUANTIZED_DIR),
            dtype=dtype,
            device_map={"": 0} if device.startswith("cuda") else None,
            low_cpu_mem_usage=True,
        )
        print(f"[model_loader] Loaded cached quantized snapshot from {QUANTIZED_DIR}", flush=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=list(skip_modules),
        )
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            dtype=dtype,
            device_map={"": 0} if device.startswith("cuda") else None,
            low_cpu_mem_usage=True,
        )
        if save_quantized and device.startswith("cuda"):
            _save_quantized_snapshot(model, tokenizer)

    if drop_visual and hasattr(model, "model") and hasattr(model.model, "visual"):
        del model.model.visual
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def is_8bit_linear(mod: nn.Module) -> bool:
    return hasattr(mod, "weight") and mod.weight.dtype == torch.int8


def get_linear8bit_scb(mod: nn.Module) -> torch.Tensor:
    scb = getattr(mod.weight, "SCB", None)
    if scb is None and hasattr(mod, "state"):
        scb = getattr(mod.state, "SCB", None)
    if scb is None:
        raise RuntimeError(f"Could not find SCB scale buffer on 8-bit module {type(mod)}")
    return scb


@torch.no_grad()
def dequantize_rows(base_linear: nn.Module, row_indices: torch.Tensor) -> torch.Tensor:
    """Dequantize selected output rows [len(row_indices), in_features] in float32."""
    if is_8bit_linear(base_linear):
        scb = get_linear8bit_scb(base_linear)
        w_int8 = base_linear.weight.data[row_indices].float()
        scales = scb[row_indices].float().unsqueeze(1) / 127.0
        return w_int8 * scales
    return base_linear.weight[row_indices].float()


@torch.no_grad()
def dequantize_full(base_linear: nn.Module, out_device: Optional[str] = None) -> torch.Tensor:
    """Dequantize full weight matrix [out_features, in_features] in float32."""
    if is_8bit_linear(base_linear):
        scb = get_linear8bit_scb(base_linear)
        w = base_linear.weight.data.float() * (scb.float().unsqueeze(1) / 127.0)
    else:
        w = base_linear.weight.float()
    if out_device is not None:
        w = w.to(out_device)
    return w


def prepare_layer_inputs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Optional[torch.Tensor]], Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Prepare embeddings, causal/recurrent masks, and 3D M-RoPE embeddings matching Qwen3_5TextModel."""
    lm = get_text_model(model)
    emb = lm.embed_tokens(input_ids)
    bsz, seqlen, _ = emb.shape
    pids = torch.arange(seqlen, device=emb.device).view(1, 1, -1).expand(4, bsz, -1)
    text_pids = pids[0]
    rope_pids = pids[1:]
    mask_kwargs = {
        "config": lm.config,
        "inputs_embeds": emb,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": text_pids,
    }
    cm_map = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
    }
    pe = lm.rotary_emb(emb, rope_pids)
    return emb, cm_map, pe, text_pids


def slice_cm_map(
    cm_map: Dict[str, Optional[torch.Tensor]],
    s: int,
    e: int,
    total_n: int,
) -> Dict[str, Optional[torch.Tensor]]:
    out: Dict[str, Optional[torch.Tensor]] = {}
    for k, cm in cm_map.items():
        if cm is not None and cm.shape[0] == total_n:
            out[k] = cm[s:e]
        else:
            out[k] = cm
    return out


def slice_pe(pe: Tuple[torch.Tensor, torch.Tensor], s: int, e: int, total_n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = pe
    if cos.shape[0] == total_n:
        return cos[s:e], sin[s:e]
    return cos, sin


def step_layer(
    model: nn.Module,
    l_idx: int,
    h: torch.Tensor,
    cm_map: Dict[str, Optional[torch.Tensor]],
    pe: Tuple[torch.Tensor, torch.Tensor],
    text_pids: torch.Tensor,
) -> torch.Tensor:
    """Execute one Qwen3_5DecoderLayer (linear_attention or full_attention) cleanly."""
    lm = get_text_model(model)
    layer = lm.layers[l_idx]
    lt = lm.config.layer_types[l_idx]
    return layer(
        h,
        position_embeddings=pe,
        attention_mask=cm_map[lt],
        position_ids=text_pids,
        past_key_values=None,
        use_cache=False,
    )


def step_token_mixer(
    model: nn.Module,
    l_idx: int,
    h_in: torch.Tensor,
    cm_map: Dict[str, Optional[torch.Tensor]],
    pe: Tuple[torch.Tensor, torch.Tensor],
    text_pids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run input_layernorm + token mixer (linear_attn or self_attn) + residual + post_attention_layernorm."""
    lm = get_text_model(model)
    layer = lm.layers[l_idx]
    lt = lm.config.layer_types[l_idx]
    normed = layer.input_layernorm(h_in)
    if lt == "linear_attention":
        mixed = layer.linear_attn(
            hidden_states=normed,
            cache_params=None,
            attention_mask=cm_map["linear_attention"],
        )
    else:
        mixed, _ = layer.self_attn(
            hidden_states=normed,
            attention_mask=cm_map["full_attention"],
            position_ids=text_pids,
            past_key_values=None,
            position_embeddings=pe,
        )
    h_mid = h_in + mixed
    x_mlp = layer.post_attention_layernorm(h_mid)
    return h_mid, x_mlp
