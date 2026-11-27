"""Iteration 0: Baselines 0, C (In-Context), A (5-step full-backprop @ seq 512 & 2048), and B (100-step) on 8-bit Ornith-1.0-9B."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark import FactBatch, generate_fact_batch, load_smoltalk_splits
from eval_harness import EXPERIMENTS_JSONL, append_experiment_record, evaluate_model, get_base_reference
from guards import (
    BaselineBackpropProfiler,
    LoRAManager,
    SynthesisMetrics,
    cleanup_cuda,
    measure_static_vram_mb,
)
from model_loader import get_layers, get_text_model, load_ornith_8bit

LAB_DIR = Path("/content/ornith_nobackprop_lab")


class TrainableLoRALinear(nn.Module):
    """Standard mixed-precision trainable LoRA layer (8-bit/fp16 frozen base + fp32 adapters) for Baselines A & B."""

    def __init__(self, base_linear: nn.Module, r: int = 16, lora_alpha: int = 32) -> None:
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if getattr(self.base, "bias", None) is not None:
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
            "in_proj_qkv",
            "in_proj_z",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
    ) -> None:
        self.model = model
        self.wrapped: Dict[tuple, TrainableLoRALinear] = {}
        for p in model.parameters():
            p.requires_grad_(False)
        for l_idx, layer in enumerate(get_layers(model)):
            for mod_name in target_modules:
                parent_attr, leaf_attr = self.TARGET_MAP[mod_name]
                if not hasattr(layer, parent_attr):
                    continue
                parent = getattr(layer, parent_attr)
                if not hasattr(parent, leaf_attr):
                    continue
                orig = getattr(parent, leaf_attr)
                wrapper = TrainableLoRALinear(orig, r=r, lora_alpha=lora_alpha)
                setattr(parent, leaf_attr, wrapper)
                self.wrapped[(l_idx, mod_name)] = wrapper

    def trainable_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for w in self.wrapped.values():
            params.extend([w.lora_A, w.lora_B])
        return params

    def uninstall(self) -> None:
        layers = get_layers(self.model)
        for (l_idx, mod_name), wrapper in self.wrapped.items():
            parent_attr, leaf_attr = self.TARGET_MAP[mod_name]
            parent = getattr(layers[l_idx], parent_attr)
            setattr(parent, leaf_attr, wrapper.base)
        self.wrapped.clear()


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
    use_grad_ckpt: bool = False,
):
    """Standard full-backprop AdamW LoRA (r=16, alpha=32) ONLY for Baselines A & B."""
    mgr = TrainableLoRAManager(base_model, r=16, lora_alpha=32)
    base_model.train()

    lm = get_text_model(base_model)
    emb_hook = None
    if use_grad_ckpt:
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        emb_hook = lm.embed_tokens.register_forward_hook(
            lambda m, inp, out: out.requires_grad_(True) if isinstance(out, torch.Tensor) else out
        )

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

    try:
        with BaselineBackpropProfiler(base_model, static_vram_mb=static_vram_mb) as prof:
            n = input_ids.shape[0]
            total_chunks = max(1, (n + batch_size - 1) // batch_size)
            for step_i in range(steps):
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                for start in range(0, n, batch_size):
                    end = min(n, start + batch_size)
                    out = base_model(
                        input_ids=input_ids[start:end],
                        attention_mask=attn_mask[start:end],
                        labels=labels[start:end],
                        use_cache=False,
                    )
                    loss = out.loss / total_chunks
                    step_loss += float(loss.detach().item())
                    loss.backward()
                    del out, loss
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                if steps >= 20 and ((step_i + 1) % 20 == 0 or step_i == 0):
                    print(f"  [backprop seq={seq_len} bs={batch_size} ckpt={use_grad_ckpt}] step {step_i+1}/{steps} loss={step_loss:.4f}", flush=True)
    finally:
        if emb_hook is not None:
            emb_hook.remove()
        if use_grad_ckpt:
            base_model.gradient_checkpointing_disable()
        base_model.eval()
        del optimizer, input_ids, attn_mask, labels

    return mgr, prof.metrics


def run_with_oom_fallback(
    base_model,
    tokenizer,
    texts: List[str],
    steps: int,
    seq_len: int,
    preferred_bs: int,
    lr: float,
    device: str,
    static_vram_mb: float,
):
    """Attempt standard uncheckpointed full-backprop; cleanly measure fallback if T4 15GB OOMs."""
    oom_uncheckpointed = False
    bs_candidates = []
    for cand in (preferred_bs, 5, 2, 1):
        if cand <= preferred_bs and cand not in bs_candidates:
            bs_candidates.append(cand)
    for bs in bs_candidates:
        try:
            mgr, met = train_full_backprop_lora_baseline(
                base_model, tokenizer, texts, steps, seq_len, bs, lr, device, static_vram_mb, use_grad_ckpt=False
            )
            return mgr, met, {"oom_without_grad_ckpt": oom_uncheckpointed, "oom_with_grad_ckpt": False, "used_grad_ckpt": False, "batch_size": bs}
        except torch.cuda.OutOfMemoryError as e:
            print(f"[OOM] Standard uncheckpointed full-backprop seq={seq_len} bs={bs} OOMed: {e}", flush=True)
            oom_uncheckpointed = True
            for l_idx, layer in enumerate(get_layers(base_model)):
                for mod_name, (pa, la) in LoRAManager.TARGET_MAP.items():
                    if hasattr(layer, pa):
                        parent = getattr(layer, pa)
                        cur = getattr(parent, la, None)
                        if isinstance(cur, TrainableLoRALinear):
                            setattr(parent, la, cur.base)
            base_model.eval()
            base_model.zero_grad(set_to_none=True)
            cleanup_cuda()

    try:
        mgr, met = train_full_backprop_lora_baseline(
            base_model, tokenizer, texts, steps, seq_len, 1, lr, device, static_vram_mb, use_grad_ckpt=True
        )
        return mgr, met, {"oom_without_grad_ckpt": oom_uncheckpointed, "oom_with_grad_ckpt": False, "used_grad_ckpt": True, "batch_size": 1}
    except torch.cuda.OutOfMemoryError as e:
        peak_oom_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        print(f"[OOM] Checkpointed full-backprop seq={seq_len} bs=1 ALSO OOMed (peak={peak_oom_mb:.2f} MB): {e}", flush=True)
        for l_idx, layer in enumerate(get_layers(base_model)):
            for mod_name, (pa, la) in LoRAManager.TARGET_MAP.items():
                if hasattr(layer, pa):
                    parent = getattr(layer, pa)
                    cur = getattr(parent, la, None)
                    if isinstance(cur, TrainableLoRALinear):
                        setattr(parent, la, cur.base)
        base_model.gradient_checkpointing_disable()
        base_model.eval()
        base_model.zero_grad(set_to_none=True)
        cleanup_cuda()
        return None, None, {
            "oom_without_grad_ckpt": True,
            "oom_with_grad_ckpt": True,
            "used_grad_ckpt": True,
            "batch_size": 1,
            "peak_oom_mb": round(peak_oom_mb, 2),
        }


def run_baselines() -> None:
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = LAB_DIR / "baselines_summary.json"
    results_summary: Dict[str, object] = {}
    if summary_path.exists():
        results_summary = json.loads(summary_path.read_text())
    elif EXPERIMENTS_JSONL.exists():
        EXPERIMENTS_JSONL.unlink()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[Iter 0] Loading Ornith-1.0-9B in 8-bit...")
    model, tokenizer = load_ornith_8bit(device=device, dtype=torch.float16, drop_visual=True)
    static_vram_mb = measure_static_vram_mb(model)
    print(f"[Iter 0] Static 8-bit VRAM: {static_vram_mb:.2f} MB")

    print("[Iter 0] Loading datasets & Seed 42 benchmark...")
    _, kl_texts = load_smoltalk_splits(tokenizer, n_cal=200, n_kl_eval=64)
    batch42: FactBatch = generate_fact_batch(42, tokenizer=tokenizer, target_doc_tokens=2048)
    print(f"Seed 42: {len(batch42.facts)} facts, Stage 2 doc tokens = {batch42.stage2_token_len}")

    base_ref = get_base_reference(model, tokenizer, batch42, device)

    # 1. Baseline 0: Base Unedited
    if "baseline_0" not in results_summary:
        m0 = SynthesisMetrics(
            peak_vram_mb=static_vram_mb,
            static_vram_mb=static_vram_mb,
            activation_vram_overhead_mb=0.0,
            wall_time_sec=0.0,
            uses_full_backprop=False,
            regime="base_unedited",
            backbone_grad_disabled_verified=True,
        )
        eval0 = evaluate_model(model, tokenizer, batch42, kl_texts, device, compute_kl=False, base_ref_override=base_ref)
        rec0 = append_experiment_record(
            0, "Baseline_0_Unedited", eval0, m0, status="baseline", seed=42, stage=1, rank=0
        )
        results_summary["baseline_0"] = rec0
        summary_path.write_text(json.dumps(results_summary, indent=2))
        print("Baseline 0:", json.dumps(rec0), flush=True)
    else:
        print("Skipping completed Baseline 0", flush=True)

    # 2. Baseline C: In-Context Prompting
    if "baseline_c" not in results_summary:
        ctx_block = "Reference Knowledge:\n" + "\n".join(batch42.stage1_edit_texts)
        eval_c = evaluate_model(
            model, tokenizer, batch42, kl_texts, device, prefix_context=ctx_block, compute_kl=False, base_ref_override=base_ref
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
        summary_path.write_text(json.dumps(results_summary, indent=2))
        print("Baseline C:", json.dumps(recc), flush=True)
    else:
        print("Skipping completed Baseline C", flush=True)

    # 3. Baseline A (Stage 1, seq_len=512, 5 steps full-backprop LoRA r=16)
    if "baseline_a_512" not in results_summary:
        mgr_a512, met_a512, extra_a512 = run_with_oom_fallback(
            model, tokenizer, batch42.stage1_edit_texts, steps=5, seq_len=512, preferred_bs=1, lr=5e-4, device=device, static_vram_mb=static_vram_mb
        )
        eval_a512 = evaluate_model(
            model, tokenizer, batch42, kl_texts, device, lora_manager=mgr_a512, compute_kl=True, base_ref_override=base_ref
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
            extra={"static_vram_mb": round(static_vram_mb, 2), "seq_len": 512, **extra_a512},
        )
        results_summary["baseline_a_512"] = reca512
        summary_path.write_text(json.dumps(results_summary, indent=2))
        print("Baseline A (seq512):", json.dumps(reca512), flush=True)
        mgr_a512.uninstall()
        del mgr_a512
        cleanup_cuda()
    else:
        print("Skipping completed Baseline A (seq512)", flush=True)

    # 4. Baseline A (Stage 2, seq_len=2048, 5 steps full-backprop LoRA r=16)
    if "baseline_a_2048" not in results_summary:
        mgr_a2048, met_a2048, extra_a2048 = run_with_oom_fallback(
            model, tokenizer, [batch42.stage2_document_text], steps=5, seq_len=2048, preferred_bs=1, lr=5e-4, device=device, static_vram_mb=static_vram_mb
        )
        if mgr_a2048 is not None and met_a2048 is not None:
            eval_a2048 = evaluate_model(
                model, tokenizer, batch42, kl_texts, device, lora_manager=mgr_a2048, compute_kl=True, base_ref_override=base_ref
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
                extra={"static_vram_mb": round(static_vram_mb, 2), "seq_len": 2048, **extra_a2048},
            )
            mgr_a2048.uninstall()
            del mgr_a2048
            cleanup_cuda()
        else:
            peak_oom = extra_a2048.get("peak_oom_mb", static_vram_mb)
            reca2048 = {
                "iter": 0,
                "method": "Baseline_A_Backprop_5step_seq2048",
                "seed": 42,
                "stage": 2,
                "rank": 16,
                "verbatim_acc": 0.0,
                "paraphrase_acc": 0.0,
                "locality_acc": 0.0,
                "kl_div": 0.0,
                "retention_acc": 0.0,
                "peak_vram_mb": peak_oom,
                "activation_vram_overhead_mb": round(max(peak_oom - static_vram_mb, 0.0), 2),
                "wall_time_sec": 0.0,
                "uses_full_backprop": True,
                "status": "oom_on_15gb_t4",
                "static_vram_mb": round(static_vram_mb, 2),
                "seq_len": 2048,
                **extra_a2048,
            }
            with EXPERIMENTS_JSONL.open("a", encoding="utf-8") as f:
                f.write(json.dumps(reca2048) + "\n")
        results_summary["baseline_a_2048"] = reca2048
        summary_path.write_text(json.dumps(results_summary, indent=2))
        print("Baseline A (seq2048):", json.dumps(reca2048), flush=True)

    # 5. Baseline B (Stage 1, seq_len=64, 100 steps full-backprop LoRA r=16)
    if "baseline_b" not in results_summary:
        mgr_b, met_b, extra_b = run_with_oom_fallback(
            model, tokenizer, batch42.stage1_edit_texts, steps=100, seq_len=64, preferred_bs=25, lr=5e-4, device=device, static_vram_mb=static_vram_mb
        )
        if mgr_b is not None and met_b is not None:
            eval_b = evaluate_model(
                model, tokenizer, batch42, kl_texts, device, lora_manager=mgr_b, compute_kl=True, base_ref_override=base_ref
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
                extra={"static_vram_mb": round(static_vram_mb, 2), "seq_len": 64, **extra_b},
            )
            mgr_b.uninstall()
            del mgr_b
            cleanup_cuda()
        else:
            recb = {"iter": 0, "method": "Baseline_B_Backprop_100step", "status": "oom", **extra_b}
        results_summary["baseline_b"] = recb
        print("Baseline B:", json.dumps(recb), flush=True)

    summary_path.write_text(json.dumps(results_summary, indent=2))
    print("\nSaved baselines_summary.json")


if __name__ == "__main__":
    run_baselines()
