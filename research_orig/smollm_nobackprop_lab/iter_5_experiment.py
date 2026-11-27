"""Iteration 5: Sequential Single-Block Detached Residual-Skip Direct CE + True P_base KL.

Architectural Constraints Verified:
- Backbone forward passes (L0..l-1 and L_{l+1}..31) run strictly under torch.no_grad().
- At layer l, input H_in = H_{l-1}.detach() and downstream residual skip res_after = (H_31 - H_l).detach()
  are BOTH detached constants — strictly ONE transformer block (layer l) in any autograd graph!
- Calibration KL is computed against TRUE unedited Base probabilities (cached once under no_grad())
  on subsampled tokens in small micro-batches to keep peak activation overhead tiny (< 600 MB) and time <= 15s.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from benchmark import FactBatch, generate_fact_batch, load_smoltalk_splits
from eval_harness import append_experiment_record, check_gates, evaluate_model
from guards import LoRAManager, NoBackpropGuard, cleanup_cuda, measure_static_vram_mb
from iter_1_experiment import gather_positions, parse_fact_from_edit_source, tokenize_with_target_positions

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")
TARGET_LAYERS = list(range(26, 32))  # 6 blocks in peak resolution band L26..L31
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_RANK = 16

_HYPERNYM_MAP = {
    "administrative capital": "governing city",
    "rare mineral": "scarce element",
    "official currency": "monetary unit",
    "original founder": "founder",
    "flagship observatory": "research building",
}


def build_generalizing_probes(
    stage1_edit_texts: List[str],
    tokenizer,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Construct entity-first, hypernym, and entity-only probes strictly from edit_source."""
    sys_msg = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
    pos_items: List[Dict[str, str]] = []
    neg_items: List[Dict[str, str]] = []

    for idx, src in enumerate(stage1_edit_texts):
        ent, tgt, rel_cue, clause_stem = parse_fact_from_edit_source(src)
        hyp = _HYPERNYM_MAP.get(rel_cue, "attribute")
        queries = [
            f"Complete this statement concisely:\n{clause_stem}",
            f"What is the {rel_cue} of {ent}?",
            f"{ent} has what {hyp}?",
            f"Identify the {hyp} associated with {ent}.",
            f"Tell me the documented {hyp} for {ent}.",
            f"What is recorded regarding {ent}?",
            f"Record entry for {ent}:",
        ]
        for q in queries:
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
            pos_items.append({"prompt": prompt, "target": tgt, "entity": ent})

        neg_ent = f"Sector_{idx + 101}"
        for nq in [
            f"What is the {rel_cue} of {neg_ent}?",
            f"{neg_ent} has what {hyp}?",
            f"Identify the {hyp} associated with {neg_ent}.",
            f"What is recorded regarding {neg_ent}?",
        ]:
            neg_prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": nq}],
                tokenize=False,
                add_generation_prompt=True,
            )
            neg_items.append({"prompt": neg_prompt})

    return pos_items, neg_items


class LeanBlockLoRA(nn.Module):
    """Memory-lean mixed-precision LoRA wrapper (fp16 compute up to rank-16 bottleneck, fp32 params)."""

    def __init__(self, base_linear: nn.Linear, r: int = 16, lora_alpha: float = 32.0) -> None:
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.scaling = float(lora_alpha) / float(r)
        self.active = False
        device = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, base_linear.in_features, device=device, dtype=torch.float32), requires_grad=False)
        self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, r, device=device, dtype=torch.float32), requires_grad=False)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.active:
            # Project in base dtype -> tiny [B, S, r] bottleneck -> out
            low = F.linear(x, self.lora_A.to(x.dtype))
            delta = F.linear(low, self.lora_B.to(x.dtype)) * self.scaling
            out = out + delta
        return out


class IsolatedBlockLoRAManager:
    TARGET_MAP = LoRAManager.TARGET_MAP

    def __init__(self, model: nn.Module, target_layers: Iterable[int], target_modules: Iterable[str], r: int = 16) -> None:
        self.model = model
        self.target_layers = list(target_layers)
        self.target_modules = tuple(target_modules)
        self.r = r
        self.wrapped: Dict[Tuple[int, str], LeanBlockLoRA] = {}
        for p in model.parameters():
            p.requires_grad_(False)
        for l_idx in self.target_layers:
            layer = model.model.layers[l_idx]
            for mod_name in self.target_modules:
                pa, la = self.TARGET_MAP[mod_name]
                parent = getattr(layer, pa)
                orig = getattr(parent, la)
                w = LeanBlockLoRA(orig, r=r, lora_alpha=32.0)
                setattr(parent, la, w)
                self.wrapped[(l_idx, mod_name)] = w

    def get_layer_params(self, layer_idx: int) -> List[nn.Parameter]:
        return [
            p
            for mod_name in self.target_modules
            for p in (self.wrapped[(layer_idx, mod_name)].lora_A, self.wrapped[(layer_idx, mod_name)].lora_B)
        ]

    def activate_for_layer(self, active_layer_idx: int) -> None:
        for (l_idx, _), w in self.wrapped.items():
            if l_idx < active_layer_idx:
                w.active = True
                w.lora_A.requires_grad_(False)
                w.lora_B.requires_grad_(False)
            elif l_idx == active_layer_idx:
                w.active = True
                w.lora_A.requires_grad_(True)
                w.lora_B.requires_grad_(True)
            else:
                w.active = False
                w.lora_A.requires_grad_(False)
                w.lora_B.requires_grad_(False)

    def freeze_all(self) -> None:
        for w in self.wrapped.values():
            w.lora_A.requires_grad_(False)
            w.lora_B.requires_grad_(False)

    def assert_single_layer_grad(self, active_layer_idx: Optional[int]) -> None:
        grad_layers = set()
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                parts = name.split(".")
                if len(parts) > 2 and parts[0] == "model" and parts[1] == "layers":
                    grad_layers.add(int(parts[2]))
                else:
                    raise RuntimeError(f"VIOLATION: Non-layer param {name} requires_grad=True!")
        if active_layer_idx is None:
            assert len(grad_layers) == 0, f"Expected 0 grad layers, got {grad_layers}"
        else:
            assert grad_layers == {active_layer_idx}, f"Expected only layer {active_layer_idx}, got {grad_layers}"


def synthesize_resskip_detached(
    model,
    tokenizer,
    mgr: IsolatedBlockLoRAManager,
    batch: FactBatch,
    cal_texts: List[str],
    device: str,
    steps_per_layer: int = 16,
    lr: float = 2.5e-3,
    lambda_kl: float = 4.0,
    lambda_neg: float = 4.0,
) -> Dict[str, object]:
    pos_items, neg_items = build_generalizing_probes(batch.stage1_edit_texts, tokenizer)
    prompts = [x["prompt"] for x in pos_items]
    targets = [x["target"] for x in pos_items]

    stu_ids, stu_mask, stu_pos = tokenize_with_target_positions(tokenizer, prompts, targets, device)
    target_tok_list: List[int] = []
    for tgt, poss in zip(targets, stu_pos):
        t_ids = tokenizer.encode(tgt, add_special_tokens=False)
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        full_t = t_ids + eos
        assert len(full_t) == len(poss)
        target_tok_list.extend(full_t)
    target_tok_ids = torch.tensor(target_tok_list, dtype=torch.long, device=device)

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    neg_enc = tokenizer([x["prompt"] for x in neg_items], return_tensors="pt", padding=True).to(device)
    cal_enc = tokenizer(cal_texts[:24], return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
    tokenizer.padding_side = prev_side

    # Sample up to 8 valid positions per neg/cal sequence to keep logit tensor tiny (~12-25 MB!)
    neg_lengths = neg_enc.attention_mask.sum(dim=1).tolist()
    neg_pos = [list(range(max(0, L - 3), L)) for L in neg_lengths]
    cal_lengths = cal_enc.attention_mask.sum(dim=1).tolist()
    cal_pos = [
        [max(0, min(L - 1, int(k * L / 8))) for k in range(1, 9)]
        for L in cal_lengths
    ]

    with torch.no_grad():
        mgr.freeze_all()
        mgr.assert_single_layer_grad(None)

        # Cache TRUE unedited base probabilities on cal_pos and neg_pos once!
        def get_base_probs(ids, mask, pos_lists):
            out = model(input_ids=ids, attention_mask=mask, use_cache=False)
            # Note: out.logits is [B, S, V]; avoid storing full logits by extracting pos_lists on hidden!
            return None

        def get_base_probs_lean(ids, mask, pos_lists, bs: int = 24):
            probs_list = []
            for s in range(0, ids.shape[0], bs):
                e = min(ids.shape[0], s + bs)
                emb = model.model.embed_tokens(ids[s:e])
                pids = torch.arange(emb.shape[1], device=device).unsqueeze(0)
                cm = create_causal_mask(config=model.config, inputs_embeds=emb, attention_mask=mask[s:e], past_key_values=None, position_ids=pids)
                pe = model.model.rotary_emb(emb, position_ids=pids)
                h = emb
                for layer in model.model.layers:
                    h = layer(h, attention_mask=cm, position_embeddings=pe, use_cache=False)
                pts = gather_positions(h, pos_lists[s:e])
                logits = model.lm_head(model.model.norm(pts)).float()
                probs_list.append(F.softmax(logits, dim=-1).cpu())
            return torch.cat(probs_list, dim=0)

        p_base_cal_cpu = get_base_probs_lean(cal_enc.input_ids, cal_enc.attention_mask, cal_pos)
        p_base_neg_cpu = get_base_probs_lean(neg_enc.input_ids, neg_enc.attention_mask, neg_pos)

        def init_and_prefix(ids, mask, bs: int = 32):
            h_chunks = []
            emb_full = model.model.embed_tokens(ids)
            pids = torch.arange(emb_full.shape[1], device=device).unsqueeze(0)
            cm_full = create_causal_mask(config=model.config, inputs_embeds=emb_full, attention_mask=mask, past_key_values=None, position_ids=pids)
            pe = model.model.rotary_emb(emb_full, position_ids=pids)
            for s in range(0, ids.shape[0], bs):
                e = min(ids.shape[0], s + bs)
                h = emb_full[s:e]
                cm = cm_full[s:e] if cm_full is not None and cm_full.shape[0] == ids.shape[0] else cm_full
                for l_idx in range(TARGET_LAYERS[0]):
                    h = model.model.layers[l_idx](h, attention_mask=cm, position_embeddings=pe, use_cache=False)
                h_chunks.append(h.detach())
            return torch.cat(h_chunks, dim=0), cm_full, pe

        H_stu, cm_stu, pe_stu = init_and_prefix(stu_ids, stu_mask)
        H_neg, cm_neg, pe_neg = init_and_prefix(neg_enc.input_ids, neg_enc.attention_mask)
        H_cal, cm_cal, pe_cal = init_and_prefix(cal_enc.input_ids, cal_enc.attention_mask)

    p_base_cal = p_base_cal_cpu.to(device)
    p_base_neg = p_base_neg_cpu.to(device)

    layer_logs: List[Dict[str, object]] = []
    n_layers = len(model.model.layers)

    for l_idx in TARGET_LAYERS:
        layer = model.model.layers[l_idx]
        H_stu_in = H_stu.detach()
        H_neg_in = H_neg.detach()
        H_cal_in = H_cal.detach()

        # Under strictly torch.no_grad(), compute detached downstream residual skip (H_31 - H_l)
        # for stu, neg, and cal!
        with torch.no_grad():
            mgr.freeze_all()
            mgr.assert_single_layer_grad(None)
            mgr.activate_for_layer(l_idx)
            # Wait: activate_for_layer sets requires_grad=True; freeze immediately for no_grad skip pass
            mgr.freeze_all()

            def compute_skip(h_in, cm, pe, pos_lists, bs: int = 35):
                skips = []
                for s in range(0, h_in.shape[0], bs):
                    e = min(h_in.shape[0], s + bs)
                    cm_s = cm[s:e] if cm is not None and cm.shape[0] == h_in.shape[0] else cm
                    hl = layer(h_in[s:e], attention_mask=cm_s, position_embeddings=pe, use_cache=False)
                    hc = hl
                    for j in range(l_idx + 1, n_layers):
                        hc = model.model.layers[j](hc, attention_mask=cm_s, position_embeddings=pe, use_cache=False)
                    skip_pts = gather_positions(hc - hl, pos_lists[s:e]).detach()
                    skips.append(skip_pts)
                return torch.cat(skips, dim=0)

            skip_stu = compute_skip(H_stu_in, cm_stu, pe_stu, stu_pos)
            skip_neg = compute_skip(H_neg_in, cm_neg, pe_neg, neg_pos)
            skip_cal = compute_skip(H_cal_in, cm_cal, pe_cal, cal_pos)

        mgr.activate_for_layer(l_idx)
        mgr.assert_single_layer_grad(l_idx)
        params = mgr.get_layer_params(l_idx)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        last_ce, last_kl, last_nkl = 0.0, 0.0, 0.0
        n_stu = H_stu_in.shape[0]
        stu_bs = 35

        for step in range(steps_per_layer):
            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                ce_val = 0.0
                tok_offset = 0
                for s in range(0, n_stu, stu_bs):
                    e = min(n_stu, s + stu_bs)
                    cm_s = cm_stu[s:e] if cm_stu is not None and cm_stu.shape[0] == n_stu else cm_stu
                    out_s = layer(H_stu_in[s:e], attention_mask=cm_s, position_embeddings=pe_stu, use_cache=False)
                    pts_s = gather_positions(out_s, stu_pos[s:e])
                    cnt = pts_s.shape[0]
                    h_final_s = pts_s + skip_stu[tok_offset : tok_offset + cnt]
                    logits_s = model.lm_head(model.model.norm(h_final_s)).float()
                    l_ce = F.cross_entropy(logits_s, target_tok_ids[tok_offset : tok_offset + cnt]) * (cnt / target_tok_ids.numel())
                    l_ce.backward()
                    ce_val += float(l_ce.item())
                    tok_offset += cnt

                # Calibration KL (24 short chunks, 192 sampled tokens)
                out_c = layer(H_cal_in, attention_mask=cm_cal, position_embeddings=pe_cal, use_cache=False)
                pts_c = gather_positions(out_c, cal_pos) + skip_cal
                log_p_c = F.log_softmax(model.lm_head(model.model.norm(pts_c)).float(), dim=-1)
                l_kl = F.kl_div(log_p_c, p_base_cal, reduction="batchmean")

                # Negative placeholder KL (100 sequences * 3 tokens = 300 tokens)
                out_n = layer(H_neg_in, attention_mask=cm_neg, position_embeddings=pe_neg, use_cache=False)
                pts_n = gather_positions(out_n, neg_pos) + skip_neg
                log_p_n = F.log_softmax(model.lm_head(model.model.norm(pts_n)).float(), dim=-1)
                l_nkl = F.kl_div(log_p_n, p_base_neg, reduction="batchmean")

                reg_loss = lambda_kl * l_kl + lambda_neg * l_nkl
                reg_loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            last_ce = ce_val
            last_kl = float(l_kl.item())
            last_nkl = float(l_nkl.item())

        del opt
        # Advance hidden streams strictly under torch.no_grad()
        with torch.no_grad():
            mgr.freeze_all()
            mgr.assert_single_layer_grad(None)

            def step_chunked(h_in, cm, pe, bs: int = 35):
                outs = []
                for s in range(0, h_in.shape[0], bs):
                    e = min(h_in.shape[0], s + bs)
                    cm_s = cm[s:e] if cm is not None and cm.shape[0] == h_in.shape[0] else cm
                    outs.append(layer(h_in[s:e], attention_mask=cm_s, position_embeddings=pe, use_cache=False).detach())
                return torch.cat(outs, dim=0)

            H_stu = step_chunked(H_stu_in, cm_stu, pe_stu)
            H_neg = step_chunked(H_neg_in, cm_neg, pe_neg)
            H_cal = step_chunked(H_cal_in, cm_cal, pe_cal)

        layer_logs.append({
            "layer": l_idx,
            "ce": round(last_ce, 4),
            "cal_kl": round(last_kl, 5),
            "neg_kl": round(last_nkl, 5),
        })

    return {"layer_logs": layer_logs, "n_probes": len(pos_items)}


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cal_texts, kl_texts = load_smoltalk_splits(tokenizer, n_cal=64, n_kl_eval=64)
    batch42 = generate_fact_batch(42, tokenizer=tokenizer, target_doc_tokens=2048)

    cleanup_cuda()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    static_vram_mb = measure_static_vram_mb(model)

    mgr = IsolatedBlockLoRAManager(model, target_layers=TARGET_LAYERS, target_modules=TARGET_MODULES, r=LORA_RANK)
    baselines = json.loads((LAB_DIR / "baselines_summary.json").read_text())
    base_a_ref = baselines["baseline_a_512"]

    with NoBackpropGuard(model, regime="single_layer_isolated", static_vram_mb=static_vram_mb) as guard:
        guard.assert_backbone_no_grad()
        diag = synthesize_resskip_detached(
            model, tokenizer, mgr, batch42, cal_texts, device,
            steps_per_layer=16, lr=2.5e-3, lambda_kl=4.0, lambda_neg=4.0,
        )

    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=mgr, compute_kl=True)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        5,
        "Iter5_SingleBlockResSkip_CE_KL_L26_31",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=LORA_RANK,
        extra={"gates": gates, "layer_logs": diag["layer_logs"]},
    )
    print("\n[Iter 5] Result:", json.dumps(rec, indent=2))
    print("[Iter 5] Gates:", json.dumps(gates, indent=2))
    print("[Iter 5] Layer logs:")
    for ll in diag["layer_logs"]:
        print(" ", ll)


if __name__ == "__main__":
    main()
