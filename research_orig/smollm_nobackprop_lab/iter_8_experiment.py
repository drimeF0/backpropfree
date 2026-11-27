"""Iteration 8: Fast-Converged 4-Block Detached ResSkip with Multi-Token Negative Continuation Lock.

Targets L26, L28, L30, L31 (4 blocks):
- Unchunked single-block forward passes (stu_bs=175, cal_bs=48, neg_bs=100) cut runtime ~2.5x,
  allowing (22, 24, 26, 32) cosine-decayed AdamW steps per block inside <= 16s!
- Drops choking intermediate block_mse so CE converges <= 0.12 while output-space KL
  guards both smol-smoltalk AND 8-token Base-0 continuations of 100 proper-noun negatives.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from benchmark import FactBatch, generate_fact_batch, load_smoltalk_splits
from eval_harness import append_experiment_record, check_gates, evaluate_model
from guards import NoBackpropGuard, cleanup_cuda, measure_static_vram_mb
from iter_1_experiment import gather_positions, tokenize_with_target_positions
from iter_5_experiment import IsolatedBlockLoRAManager
from iter_7_experiment import build_probes_iter7

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")
TARGET_LAYERS = [26, 28, 30, 31]
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_RANK = 16


def synthesize_fast_resskip(
    model,
    tokenizer,
    mgr: IsolatedBlockLoRAManager,
    batch: FactBatch,
    cal_texts: List[str],
    device: str,
    steps_per_layer: Tuple[int, ...] = (22, 24, 26, 32),
    lr: float = 4.0e-3,
    lambda_cal_kl: float = 7.0,
    lambda_neg_kl: float = 12.0,
    lambda_neg_l31_mse: float = 0.05,
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
    # Generate 8 continuation tokens on Base 0 for the 100 negative proper-noun probes!
    with torch.no_grad():
        mgr.freeze_all()
        mgr.assert_single_layer_grad(None)

        tokenizer.padding_side = "left"
        neg_gen_enc = tokenizer([x["prompt"] for x in neg_items], return_tensors="pt", padding=True).to(device)
        in_len_neg = neg_gen_enc.input_ids.shape[1]
        neg_gen = model.generate(
            input_ids=neg_gen_enc.input_ids,
            attention_mask=neg_gen_enc.attention_mask,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        # Re-pack neg prompts + their 8 Base-0 continuation tokens right-padded!
        neg_seqs = []
        neg_pos = []
        for idx_n in range(len(neg_items)):
            p_len_orig = len(tokenizer.encode(neg_items[idx_n]["prompt"], add_special_tokens=False))
            gen_part = neg_gen[idx_n, in_len_neg:].tolist()
            # Trim after eos if any
            clean_gen = []
            for tk in gen_part:
                clean_gen.append(tk)
                if tk == tokenizer.eos_token_id:
                    break
            p_ids = tokenizer.encode(neg_items[idx_n]["prompt"], add_special_tokens=False)
            full_seq = p_ids + clean_gen
            neg_seqs.append(full_seq)
            # Guard prompt-end token + every generated continuation prediction token
            neg_pos.append(list(range(len(p_ids) - 1, len(p_ids) + len(clean_gen) - 1)))

        max_neg_len = max(len(s) for s in neg_seqs)
        neg_ids = torch.full((len(neg_seqs), max_neg_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        neg_mask = torch.zeros((len(neg_seqs), max_neg_len), dtype=torch.long, device=device)
        for i, s in enumerate(neg_seqs):
            neg_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            neg_mask[i, : len(s)] = 1

        tokenizer.padding_side = "right"
        cal_enc = tokenizer(cal_texts[:48], return_tensors="pt", padding=True, truncation=True, max_length=80).to(device)
        tokenizer.padding_side = prev_side

        cal_lengths = cal_enc.attention_mask.sum(dim=1).tolist()
        cal_kl_pos = [[max(0, min(L - 1, int(k * L / 5))) for k in range(1, 5)] for L in cal_lengths]

        def get_base_outputs_lean(ids, mask, pos_lists, bs: int = 50):
            probs_list = []
            h_list = []
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
                h_list.append(pts.detach().cpu())
                logits = model.lm_head(model.model.norm(pts)).float()
                probs_list.append(F.softmax(logits, dim=-1).cpu())
            return torch.cat(probs_list, dim=0), torch.cat(h_list, dim=0)

        p_base_cal_cpu, _ = get_base_outputs_lean(cal_enc.input_ids, cal_enc.attention_mask, cal_kl_pos)
        p_base_neg_cpu, h_base_neg_cpu = get_base_outputs_lean(neg_ids, neg_mask, neg_pos)

        def init_and_prefix(ids, mask, bs: int = 60):
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
        H_neg, cm_neg, pe_neg = init_and_prefix(neg_ids, neg_mask)
        H_cal, cm_cal, pe_cal = init_and_prefix(cal_enc.input_ids, cal_enc.attention_mask)

    p_base_cal = p_base_cal_cpu.to(device)
    p_base_neg = p_base_neg_cpu.to(device)
    h_base_neg = h_base_neg_cpu.to(device).float()

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

            def compute_skip(h_in, cm, pe, pos_lists, bs: int = 100):
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
            skip_neg = compute_skip(H_neg_in, cm_neg, pe_neg, neg_pos)
            skip_cal = compute_skip(H_cal_in, cm_cal, pe_cal, cal_kl_pos)

        mgr.activate_for_layer(l_idx)
        mgr.assert_single_layer_grad(l_idx)
        params = mgr.get_layer_params(l_idx)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr * 0.25)

        last_ce, last_kl, last_nkl = 0.0, 0.0, 0.0
        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                # Single forward on stu (175 seqs * ~42 tokens -> tiny ~50 MB!)
                out_s = layer(H_stu_in, attention_mask=cm_stu, position_embeddings=pe_stu, use_cache=False)
                pts_s = gather_positions(out_s, stu_pos) + skip_stu
                logits_s = model.lm_head(model.model.norm(pts_s)).float()
                ce_vec = F.cross_entropy(logits_s, target_tok_ids, reduction="none")
                l_ce = (ce_vec * tok_weights).mean()
                l_ce.backward()

                # Single forward on cal (48 seqs * 80 tokens)
                out_c = layer(H_cal_in, attention_mask=cm_cal, position_embeddings=pe_cal, use_cache=False)
                pts_c = gather_positions(out_c, cal_kl_pos) + skip_cal
                log_p_c = F.log_softmax(model.lm_head(model.model.norm(pts_c)).float(), dim=-1)
                l_ckl = F.kl_div(log_p_c, p_base_cal, reduction="batchmean")

                # Single forward on neg (100 seqs * ~45 tokens -> ~700 continuation tokens)
                out_n = layer(H_neg_in, attention_mask=cm_neg, position_embeddings=pe_neg, use_cache=False)
                pts_n = gather_positions(out_n, neg_pos) + skip_neg
                l_nmse = F.mse_loss(pts_n.float(), h_base_neg)
                log_p_n = F.log_softmax(model.lm_head(model.model.norm(pts_n)).float(), dim=-1)
                l_nkl = F.kl_div(log_p_n, p_base_neg, reduction="batchmean")

                reg_loss = lambda_cal_kl * l_ckl + lambda_neg_kl * l_nkl + lambda_neg_l31_mse * l_nmse
                reg_loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            last_ce = float(l_ce.item())
            last_kl = float(l_ckl.item())
            last_nkl = float(l_nkl.item())

        del opt, sched
        with torch.no_grad():
            mgr.freeze_all()
            mgr.assert_single_layer_grad(None)
            # Advance through intermediate frozen layers up to next TARGET_LAYER (or end)
            next_idx = TARGET_LAYERS[idx_in_sched + 1] if idx_in_sched + 1 < len(TARGET_LAYERS) else n_layers

            def advance(h_in, cm, pe):
                h = h_in
                for j in range(l_idx, next_idx):
                    h = model.model.layers[j](h, attention_mask=cm, position_embeddings=pe, use_cache=False)
                return h.detach()

            H_stu = advance(H_stu_in, cm_stu, pe_stu)
            H_neg = advance(H_neg_in, cm_neg, pe_neg)
            H_cal = advance(H_cal_in, cm_cal, pe_cal)

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
        diag = synthesize_fast_resskip(
            model, tokenizer, mgr, batch42, cal_texts, device,
            steps_per_layer=(22, 24, 26, 32), lr=4.0e-3,
            lambda_cal_kl=7.0, lambda_neg_kl=12.0, lambda_neg_l31_mse=0.05,
        )

    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=mgr, compute_kl=True)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        8,
        "Iter8_FastConverged_ResSkip_4Layer",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=LORA_RANK,
        extra={"gates": gates, "layer_logs": diag["layer_logs"]},
    )
    print("\n[Iter 8] Result:", json.dumps(rec, indent=2))
    print("[Iter 8] Gates:", json.dumps(gates, indent=2))
    print("[Iter 8] Layer logs:")
    for ll in diag["layer_logs"]:
        print(" ", ll)


if __name__ == "__main__":
    main()
