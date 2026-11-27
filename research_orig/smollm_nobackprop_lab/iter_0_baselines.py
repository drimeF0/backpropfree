"""Iteration 0: Baselines 0, A (5-step full-backprop @ seq 512 & 2048), B (100-step), and C (In-Context)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import FactBatch, generate_fact_batch, load_smoltalk_splits
from eval_harness import EXPERIMENTS_JSONL, append_experiment_record, evaluate_model
from guards import (
    BaselineBackpropProfiler,
    LoRAManager,
    SynthesisMetrics,
    cleanup_cuda,
    measure_static_vram_mb,
)

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")


class TrainableLoRALinear(nn.Module):
    """Standard mixed-precision trainable LoRA layer (fp16 frozen base + fp32 adapters) for Baselines A & B."""

    def __init__(self, base_linear: nn.Linear, r: int = 16, lora_alpha: int = 32) -> None:
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.scaling = float(lora_alpha) / float(r)
        self.active = True
        device = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base_linear.in_features, device=device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, r, device=device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.active:
            xf = x.float()
            delta = (F.linear(F.linear(xf, self.lora_A), self.lora_B) * self.scaling).to(out.dtype)
            out = out + delta
        return out


class TrainableLoRAManager:
    TARGET_MAP = LoRAManager.TARGET_MAP

    def __init__(
        self,
        model: nn.Module,
        r: int = 16,
        lora_alpha: int = 32,
        target_modules: Iterable[str] = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    ) -> None:
        self.model = model
        self.wrapped: Dict[tuple, TrainableLoRALinear] = {}
        for p in model.parameters():
            p.requires_grad_(False)
        for l_idx, layer in enumerate(model.model.layers):
            for mod_name in target_modules:
                parent_attr, leaf_attr = self.TARGET_MAP[mod_name]
                parent = getattr(layer, parent_attr)
                orig = getattr(parent, leaf_attr)
                wrapper = TrainableLoRALinear(orig, r=r, lora_alpha=lora_alpha)
                setattr(parent, leaf_attr, wrapper)
                self.wrapped[(l_idx, mod_name)] = wrapper

    def trainable_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for w in self.wrapped.values():
            params.extend([w.lora_A, w.lora_B])
        return params


def train_full_backprop_lora_baseline(
    base_model,
    tokenizer,
    texts: List[str],
    steps: int,
    seq_len: int,
    batch_size: int,
    lr: float,
    device: str,
    static_vram_mb: float,
):
    """Standard full-backprop AdamW LoRA (r=16, alpha=32) ONLY for Baselines A & B."""
    mgr = TrainableLoRAManager(base_model, r=16, lora_alpha=32)
    base_model.train()

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=seq_len,
    )
    tokenizer.padding_side = prev_side

    input_ids = enc.input_ids.to(device)
    attn_mask = enc.attention_mask.to(device)
    labels = input_ids.clone()
    labels[attn_mask == 0] = -100

    trainable_params = mgr.trainable_parameters()
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    with BaselineBackpropProfiler(base_model, static_vram_mb=static_vram_mb) as prof:
        n = input_ids.shape[0]
        total_chunks = max(1, (n + batch_size - 1) // batch_size)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                out = base_model(
                    input_ids=input_ids[start:end],
                    attention_mask=attn_mask[start:end],
                    labels=labels[start:end],
                    use_cache=False,
                )
                loss = out.loss / total_chunks
                loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

    base_model.eval()
    return mgr, prof.metrics


def run_baselines() -> None:
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    if EXPERIMENTS_JSONL.exists():
        EXPERIMENTS_JSONL.unlink()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading datasets & Seed 42 benchmark...")
    _, kl_texts = load_smoltalk_splits(tokenizer, n_cal=200, n_kl_eval=64)
    batch42: FactBatch = generate_fact_batch(42, tokenizer=tokenizer, target_doc_tokens=2048)
    print(f"Seed 42: {len(batch42.facts)} facts, Stage 2 doc tokens = {batch42.stage2_token_len}")

    results_summary: Dict[str, object] = {}

    # ---------------------------------------------------------
    # Baseline 0: Base Unedited
    # ---------------------------------------------------------
    cleanup_cuda()
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    static_vram_mb = measure_static_vram_mb(base_model)

    m0 = SynthesisMetrics(
        peak_vram_mb=static_vram_mb,
        static_vram_mb=static_vram_mb,
        activation_vram_overhead_mb=0.0,
        wall_time_sec=0.0,
        uses_full_backprop=False,
        regime="base_unedited",
        backbone_grad_disabled_verified=True,
    )
    eval0 = evaluate_model(base_model, tokenizer, batch42, kl_texts, device, compute_kl=False)
    rec0 = append_experiment_record(
        0, "Baseline_0_Unedited", eval0, m0, status="baseline", seed=42, stage=1, rank=0
    )
    results_summary["baseline_0"] = rec0
    print("Baseline 0:", json.dumps(rec0))

    # ---------------------------------------------------------
    # Baseline C: In-Context Prompting (Facts kept in context)
    # ---------------------------------------------------------
    ctx_block = "Reference Knowledge:\n" + "\n".join(batch42.stage1_edit_texts)
    eval_c = evaluate_model(
        base_model,
        tokenizer,
        batch42,
        kl_texts,
        device,
        prefix_context=ctx_block,
        compute_kl=False,
    )
    mc = SynthesisMetrics(
        peak_vram_mb=static_vram_mb,
        static_vram_mb=static_vram_mb,
        activation_vram_overhead_mb=0.0,
        wall_time_sec=0.0,
        uses_full_backprop=False,
        regime="in_context_prompting",
        backbone_grad_disabled_verified=True,
    )
    recc = append_experiment_record(
        0, "Baseline_C_InContext", eval_c, mc, status="baseline", seed=42, stage=1, rank=0
    )
    results_summary["baseline_c"] = recc
    print("Baseline C:", json.dumps(recc))

    del base_model
    cleanup_cuda()

    # ---------------------------------------------------------
    # Baseline A (Stage 1, seq_len=512, 5 steps full-backprop LoRA r=16)
    # ---------------------------------------------------------
    m_a512 = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)
    base_static_a = measure_static_vram_mb(m_a512)
    mgr_a512, met_a512 = train_full_backprop_lora_baseline(
        m_a512,
        tokenizer,
        batch42.stage1_edit_texts,
        steps=5,
        seq_len=512,
        batch_size=5,
        lr=5e-4,
        device=device,
        static_vram_mb=base_static_a,
    )
    eval_a512 = evaluate_model(
        m_a512, tokenizer, batch42, kl_texts, device, lora_manager=mgr_a512, compute_kl=True
    )
    reca512 = append_experiment_record(
        0,
        "Baseline_A_Backprop_5step_seq512",
        eval_a512,
        met_a512,
        status="baseline",
        seed=42,
        stage=1,
        rank=16,
        extra={"static_vram_mb": round(base_static_a, 2), "seq_len": 512},
    )
    results_summary["baseline_a_512"] = reca512
    print("Baseline A (seq512):", json.dumps(reca512))
    del mgr_a512, m_a512
    cleanup_cuda()

    # ---------------------------------------------------------
    # Baseline A (Stage 2, seq_len=2048, 5 steps full-backprop LoRA r=16)
    # ---------------------------------------------------------
    m_a2048 = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)
    base_static_a2048 = measure_static_vram_mb(m_a2048)
    mgr_a2048, met_a2048 = train_full_backprop_lora_baseline(
        m_a2048,
        tokenizer,
        [batch42.stage2_document_text],
        steps=5,
        seq_len=2048,
        batch_size=1,
        lr=5e-4,
        device=device,
        static_vram_mb=base_static_a2048,
    )
    eval_a2048 = evaluate_model(
        m_a2048, tokenizer, batch42, kl_texts, device, lora_manager=mgr_a2048, compute_kl=True
    )
    reca2048 = append_experiment_record(
        0,
        "Baseline_A_Backprop_5step_seq2048",
        eval_a2048,
        met_a2048,
        status="baseline",
        seed=42,
        stage=2,
        rank=16,
        extra={"static_vram_mb": round(base_static_a2048, 2), "seq_len": 2048},
    )
    results_summary["baseline_a_2048"] = reca2048
    print("Baseline A (seq2048):", json.dumps(reca2048))
    del mgr_a2048, m_a2048
    cleanup_cuda()

    # ---------------------------------------------------------
    # Baseline B (Stage 1, seq_len=128, 100 steps full-backprop LoRA r=16)
    # ---------------------------------------------------------
    m_b = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device)
    base_static_b = measure_static_vram_mb(m_b)
    mgr_b, met_b = train_full_backprop_lora_baseline(
        m_b,
        tokenizer,
        batch42.stage1_edit_texts,
        steps=100,
        seq_len=64,
        batch_size=25,
        lr=5e-4,
        device=device,
        static_vram_mb=base_static_b,
    )
    eval_b = evaluate_model(
        m_b, tokenizer, batch42, kl_texts, device, lora_manager=mgr_b, compute_kl=True
    )
    recb = append_experiment_record(
        0,
        "Baseline_B_Backprop_100step",
        eval_b,
        met_b,
        status="baseline",
        seed=42,
        stage=1,
        rank=16,
        extra={"static_vram_mb": round(base_static_b, 2), "seq_len": 64},
    )
    results_summary["baseline_b"] = recb
    print("Baseline B:", json.dumps(recb))
    del mgr_b, m_b
    cleanup_cuda()

    (LAB_DIR / "baselines_summary.json").write_text(json.dumps(results_summary, indent=2))
    print("\nSaved baselines_summary.json")


if __name__ == "__main__":
    run_baselines()
