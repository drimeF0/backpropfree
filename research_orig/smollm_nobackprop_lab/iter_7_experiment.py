"""Iteration 7: Entity-Selective Single-Block ResSkip LoRA (L26..L31).

Combines Iter 5's high paraphrase generalization (0.82) with Iter 6's clean retention (1.0185) & low KL (0.019):
- Restores L26..L31 (6 blocks including peak cross-phrasing layer L26).
- Removes the buggy BPE-first-token lambda_bleed penalty.
- Uses 100 diverse 1-word and 2-word proper-noun negative probes + 64 smol-smoltalk calibration seqs
  with per-block all-token output identity MSE (forces LoRA delta == 0 on unedited entities/chat)
  plus balanced CE / calibration-KL / negative-KL.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from benchmark import FactBatch, generate_fact_batch, load_smoltalk_splits
from eval_harness import append_experiment_record, check_gates, evaluate_model
from guards import NoBackpropGuard, cleanup_cuda, measure_static_vram_mb
from iter_1_experiment import gather_positions, parse_fact_from_edit_source, tokenize_with_target_positions
from iter_5_experiment import _HYPERNYM_MAP, IsolatedBlockLoRAManager
from iter_6_experiment import make_realistic_negative_entities

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")
TARGET_LAYERS = list(range(26, 32))  # 6 blocks: L26..L31
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_RANK = 16


def build_probes_iter7(
    stage1_edit_texts: List[str],
    tokenizer,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    sys_msg = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
    pos_items: List[Dict[str, str]] = []
    neg_items: List[Dict[str, str]] = []

    parsed = [parse_fact_from_edit_source(src) for src in stage1_edit_texts]
    forbidden = {ent.lower() for ent, _, _, _ in parsed} | {tgt.lower() for _, tgt, _, _ in parsed}
    neg_entities = make_realistic_negative_entities(50, forbidden)

    rel_cues = []
    for ent, tgt, rel_cue, clause_stem in parsed:
        rel_cues.append(rel_cue)
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

    neg_templates = [
        "What is the {rel} of {ent}?",
        "{ent} has what {hyp}?",
        "Identify the {hyp} associated with {ent}.",
        "Tell me the documented {hyp} for {ent}.",
        "What is recorded regarding {ent}?",
    ]
    for j in range(100):
        n_ent = neg_entities[j % len(neg_entities)]
        if j >= 50:
            # 2-word proper noun variant
            n_ent = f"{n_ent} {neg_entities[(j * 7 + 3) % len(neg_entities)]}"
        rel_cue = rel_cues[j % len(rel_cues)]
        hyp = _HYPERNYM_MAP.get(rel_cue, "attribute")
        nq = neg_templates[j % len(neg_templates)].format(ent=n_ent, rel=rel_cue, hyp=hyp)
        neg_prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_msg}, {"role": "user", "content": nq}],
            tokenize=False,
            add_generation_prompt=True,
        )
        neg_items.append({"prompt": neg_prompt})

    return pos_items, neg_items


def synthesize_entity_selective_resskip(
    model,
    tokenizer,
    mgr: IsolatedBlockLoRAManager,
    batch: FactBatch,
    cal_texts: List[str],
    device: str,
    steps_per_layer: Tuple[int, ...] = (12, 12, 12, 14, 14, 16),
    lr: float = 3.0e-3,
    lambda_cal_kl: float = 5.0,
    lambda_neg_kl: float = 6.0,
    lambda_block_mse: float = 0.5,
) -> Dict[str, object]:
    pos_items, neg_items = build_probes_iter7(batch.stage1_edit_texts, tokenizer)
    prompts = [x["prompt"] for x in pos_items]
    targets = [x["target"] for x in pos_items]

    stu_ids, stu_mask, stu_pos = tokenize_with_target_positions(tokenizer, prompts, targets, device)
    target_tok_list: List[int] = []
    pos0_weights: List[float] = []
    for tgt, poss in zip(targets, stu_pos):
        t_ids = tokenizer.encode(tgt, add_special_tokens=False)
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        full_t = t_ids + eos
        assert len(full_t) == len(poss)
        target_tok_list.extend(full_t)
        pos0_weights.extend([2.0] + [1.0] * (len(full_t) - 1))

    target_tok_ids = torch.tensor(target_tok_list, dtype=torch.long, device=device)
    tok_weights = torch.tensor(pos0_weights, dtype=torch.float32, device=device)
    tok_weights = tok_weights / tok_weights.mean()

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    neg_enc = tokenizer([x["prompt"] for x in neg_items], return_tensors="pt", padding=True).to(device)
    cal_enc = tokenizer(cal_texts[:48], return_tensors="pt", padding=True, truncation=True, max_length=80).to(device)
    tokenizer.padding_side = prev_side

    neg_lengths = neg_enc.attention_mask.sum(dim=1).tolist()
    neg_last_pos = [[L - 1] for L in neg_lengths]

    cal_lengths = cal_enc.attention_mask.sum(dim=1).tolist()
    cal_kl_pos = [[max(0, min(L - 1, int(k * L / 5))) for k in range(1, 5)] for L in cal_lengths]

    with torch.no_grad():
        mgr.freeze_all()
        mgr.assert_single_layer_grad(None)

        def get_base_probs_lean(ids, mask, pos_lists, bs: int = 48):
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

        p_base_cal_cpu = get_base_probs_lean(cal_enc.input_ids, cal_enc.attention_mask, cal_kl_pos)
        p_base_neg_cpu = get_base_probs_lean(neg_enc.input_ids, neg_enc.attention_mask, neg_last_pos)

        def init_and_prefix(ids, mask, bs: int = 50):
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
    cal_mask_bool = cal_enc.attention_mask.bool()
    neg_mask_bool = neg_enc.attention_mask.bool()

    layer_logs: List[Dict[str, object]] = []
    n_layers = len(model.model.layers)

    for idx_in_sched, l_idx in enumerate(TARGET_LAYERS):
        layer = model.model.layers[l_idx]
        n_steps = steps_per_layer[idx_in_sched]
        H_stu_in = H_stu.detach()
        H_neg_in = H_neg.detach()
        H_cal_in = H_cal.detach()

        with torch.no_grad():
            mgr.freeze_all()
            mgr.assert_single_layer_grad(None)

            # Base block outputs for all-token identity MSE lock on cal and neg!
            base_cal_block_out = layer(H_cal_in, attention_mask=cm_cal, position_embeddings=pe_cal, use_cache=False).detach()
            base_neg_block_out = layer(H_neg_in, attention_mask=cm_neg, position_embeddings=pe_neg, use_cache=False).detach()

            def compute_skip(h_in, cm, pe, pos_lists, bs: int = 90):
                skips = []
                for s in range(0, h_in.shape[0], bs):
                    e = min(h_in.shape[0], s + bs)
                    cm_s = cm[s:e] if cm is not None and cm.shape[0] == h_in.shape[0] else cm
                    hl = layer(h_in[s:e], attention_mask=cm_s, position_embeddings=pe, use_cache=False)
                    hc = hl
                    for j in range(l_idx + 1, n_layers):
                        hc = model.model.layers[j](hc, attention_mask=cm_s, position_embeddings=pe, use_cache=False)
                    skips.append(gather_positions(hc - hl, pos_lists[s:e]).detach())
                return torch.cat(skips, dim=0)

            skip_stu = compute_skip(H_stu_in, cm_stu, pe_stu, stu_pos)
            skip_neg = compute_skip(H_neg_in, cm_neg, pe_neg, neg_last_pos)
            skip_cal = compute_skip(H_cal_in, cm_cal, pe_cal, cal_kl_pos)

        mgr.activate_for_layer(l_idx)
        mgr.assert_single_layer_grad(l_idx)
        params = mgr.get_layer_params(l_idx)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=5e-4)

        last_ce, last_kl, last_nkl = 0.0, 0.0, 0.0
        n_stu = H_stu_in.shape[0]
        stu_bs = 90  # 2 clean chunks

        for step in range(n_steps):
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
                    ce_vec = F.cross_entropy(logits_s, target_tok_ids[tok_offset : tok_offset + cnt], reduction="none")
                    l_ce = (ce_vec * tok_weights[tok_offset : tok_offset + cnt]).sum() / target_tok_ids.numel()
                    l_ce.backward()
                    ce_val += float(l_ce.item())
                    tok_offset += cnt

                # Calibration pass (1 chunk of 48 short sequences)
                out_c = layer(H_cal_in, attention_mask=cm_cal, position_embeddings=pe_cal, use_cache=False)
                l_cmse = F.mse_loss(out_c[cal_mask_bool].float(), base_cal_block_out[cal_mask_bool].float())
                pts_c = gather_positions(out_c, cal_kl_pos) + skip_cal
                log_p_c = F.log_softmax(model.lm_head(model.model.norm(pts_c)).float(), dim=-1)
                l_ckl = F.kl_div(log_p_c, p_base_cal, reduction="batchmean")

                # Negative proper-noun pass (1 chunk of 100 short sequences)
                out_n = layer(H_neg_in, attention_mask=cm_neg, position_embeddings=pe_neg, use_cache=False)
                l_nmse = F.mse_loss(out_n[neg_mask_bool].float(), base_neg_block_out[neg_mask_bool].float())
                pts_n = gather_positions(out_n, neg_last_pos) + skip_neg
                log_p_n = F.log_softmax(model.lm_head(model.model.norm(pts_n)).float(), dim=-1)
                l_nkl = F.kl_div(log_p_n, p_base_neg, reduction="batchmean")

                reg_loss = (
                    lambda_cal_kl * l_ckl
                    + lambda_neg_kl * l_nkl
                    + lambda_block_mse * (l_cmse + l_nmse)
                )
                reg_loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            last_ce = ce_val
            last_kl = float(l_ckl.item())
            last_nkl = float(l_nkl.item())

        del opt
        with torch.no_grad():
            mgr.freeze_all()
            mgr.assert_single_layer_grad(None)

            def step_chunked(h_in, cm, pe, bs: int = 90):
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

    cal_texts, kl_texts = load_smoltalk_splits(tokenizer, n_cal=128, n_kl_eval=64)
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
        diag = synthesize_entity_selective_resskip(
            model, tokenizer, mgr, batch42, cal_texts, device,
            steps_per_layer=(12, 12, 12, 14, 14, 16), lr=3.0e-3,
            lambda_cal_kl=5.0, lambda_neg_kl=6.0, lambda_block_mse=0.5,
        )

    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=mgr, compute_kl=True)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        7,
        "Iter7_EntitySelective_ResSkip_L26_31",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=LORA_RANK,
        extra={"gates": gates, "layer_logs": diag["layer_logs"]},
    )
    print("\n[Iter 7] Result:", json.dumps(rec, indent=2))
    print("[Iter 7] Gates:", json.dumps(gates, indent=2))
    print("[Iter 7] Layer logs:")
    for ll in diag["layer_logs"]:
        print(" ", ll)


if __name__ == "__main__":
    main()
