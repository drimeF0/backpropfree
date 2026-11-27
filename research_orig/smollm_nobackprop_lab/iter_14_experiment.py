"""Iteration 14: Shared-4-Space Null-Preconditioned ResSkip (L26..L29).

Fixes Iter 13's runtime and ce convergence:
1. Computes only 4 unique null-space preconditioners per layer (attn_in shared by q/k/v,
   o_in, mlp_in shared by gate/up, down_in) -> cuts linear-algebra overhead nearly in half (~14-16s).
2. Sets alpha_null=12.0 (midpoint between Iter 12's 6.0 and Iter 13's 25.0), lr=5.0e-3,
   steps=(11, 11, 13, 14) on L26..L29 so ce drops <= 0.08 (paraphrase >= 0.85) without over-margin.
3. Adds gentle negative-prompt hidden-delta suppression (lambda_neg_hmse=0.04) alongside
   lambda_neg_kl=14.0 and lambda_antibleed=4.0.
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
from iter_12_experiment import PrecondBlockManager, compute_null_preconditioner
from iter_13_experiment import build_probes_iter13, print_locality_breakdown
from nullspace_cache import compute_or_load_c0

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")
TARGET_LAYERS = [26, 27, 28, 29]
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_RANK = 16

# 4 unique input spaces per layer
SPACE_FOR_MOD = {
    "q_proj": "attn_in",
    "k_proj": "attn_in",
    "v_proj": "attn_in",
    "o_proj": "o_proj",
    "gate_proj": "mlp_in",
    "up_proj": "mlp_in",
    "down_proj": "down_proj",
}
REP_MOD_FOR_SPACE = {
    "attn_in": "q_proj",
    "o_proj": "o_proj",
    "mlp_in": "gate_proj",
    "down_proj": "down_proj",
}


def synthesize_shared4_null_resskip(
    model,
    tokenizer,
    mgr: PrecondBlockManager,
    covs: Dict[Tuple[int, str], torch.Tensor],
    batch: FactBatch,
    cal_texts: List[str],
    device: str,
    steps_per_layer: Tuple[int, ...] = (11, 11, 13, 14),
    lr: float = 5.0e-3,
    alpha_null: float = 12.0,
    lambda_cal_kl: float = 6.0,
    lambda_neg_kl: float = 14.0,
    lambda_antibleed: float = 4.0,
    lambda_neg_hmse: float = 0.04,
) -> Dict[str, object]:
    pos_items, neg_items = build_probes_iter13(batch.stage1_edit_texts, tokenizer)
    prompts = [x["prompt"] for x in pos_items]
    targets = [x["target"] for x in pos_items]

    stu_ids, stu_mask, stu_pos = tokenize_with_target_positions(tokenizer, prompts, targets, device)
    target_tok_list: List[int] = []
    pos0_weights: List[float] = []
    injected_first_toks = set()

    for tgt, poss in zip(targets, stu_pos):
        t_ids = tokenizer.encode(tgt, add_special_tokens=False)
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        full_t = t_ids + eos
        assert len(full_t) == len(poss)
        injected_first_toks.add(t_ids[0])
        target_tok_list.extend(full_t)
        pos0_weights.extend([2.0] + [1.0] * (len(full_t) - 1))

    target_tok_ids = torch.tensor(target_tok_list, dtype=torch.long, device=device)
    tok_weights = torch.tensor(pos0_weights, dtype=torch.float32, device=device)
    tok_weights = tok_weights / tok_weights.mean()
    bleed_tok_ids = torch.tensor(sorted(injected_first_toks), dtype=torch.long, device=device)

    prev_side = tokenizer.padding_side
    with torch.no_grad():
        mgr.freeze_all()
        mgr.assert_single_layer_grad(None)

        tokenizer.padding_side = "left"
        neg_gen_enc = tokenizer([x["prompt"] for x in neg_items], return_tensors="pt", padding=True).to(device)
        in_len_neg = neg_gen_enc.input_ids.shape[1]
        neg_gen = model.generate(
            input_ids=neg_gen_enc.input_ids,
            attention_mask=neg_gen_enc.attention_mask,
            max_new_tokens=2,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        neg_seqs, neg_pos, neg_tok_weights_list = [], [], []
        for idx_n in range(len(neg_items)):
            gen_part = neg_gen[idx_n, in_len_neg:].tolist()
            clean_gen = []
            for tk in gen_part:
                clean_gen.append(tk)
                if tk == tokenizer.eos_token_id:
                    break
            p_ids = tokenizer.encode(neg_items[idx_n]["prompt"], add_special_tokens=False)
            full_seq = p_ids + clean_gen
            neg_seqs.append(full_seq)
            n_pred = max(1, len(clean_gen))
            neg_pos.append(list(range(len(p_ids) - 1, len(p_ids) - 1 + n_pred)))
            neg_tok_weights_list.extend([4.0] + [1.0] * (n_pred - 1))

        neg_tok_weights = torch.tensor(neg_tok_weights_list, dtype=torch.float32, device=device)
        neg_tok_weights = neg_tok_weights / neg_tok_weights.mean()

        max_neg_len = max(len(s) for s in neg_seqs)
        neg_ids = torch.full((len(neg_seqs), max_neg_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
        neg_mask = torch.zeros((len(neg_seqs), max_neg_len), dtype=torch.long, device=device)
        for i, s in enumerate(neg_seqs):
            neg_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            neg_mask[i, : len(s)] = 1

        tokenizer.padding_side = "right"
        cal_enc = tokenizer(cal_texts[:64], return_tensors="pt", padding=True, truncation=True, max_length=72).to(device)
        tokenizer.padding_side = prev_side

        cal_lengths = cal_enc.attention_mask.sum(dim=1).tolist()
        cal_kl_pos = [[max(0, min(L - 1, int(k * L / 4))) for k in range(1, 4)] for L in cal_lengths]

        def get_base_probs(ids, mask, pos_lists, bs: int = 75):
            probs_list, logp_bleed_list, h_list = [], [], []
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
                lp = F.log_softmax(logits, dim=-1)
                probs_list.append(lp.exp().cpu())
                logp_bleed_list.append(lp[:, bleed_tok_ids].cpu())
            return torch.cat(probs_list, dim=0), torch.cat(logp_bleed_list, dim=0), torch.cat(h_list, dim=0)

        p_base_cal_cpu, _, _ = get_base_probs(cal_enc.input_ids, cal_enc.attention_mask, cal_kl_pos)
        p_base_neg_cpu, logp_base_neg_bleed_cpu, h_base_neg_cpu = get_base_probs(neg_ids, neg_mask, neg_pos)

        def init_and_prefix(ids, mask, bs: int = 75):
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
    logp_base_neg_bleed = logp_base_neg_bleed_cpu.to(device)
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

            # Capture ONLY the 4 representative input spaces!
            rep_spaces = list(REP_MOD_FOR_SPACE.keys())
            cap_cal = {sp: [] for sp in rep_spaces}
            cap_neg = {sp: [] for sp in rep_spaces}
            cap_stu = {sp: [] for sp in rep_spaces}

            def register_space_hooks(store, pos_lists=None, bs_mask=None):
                hooks = []
                for sp, rep_mod in REP_MOD_FOR_SPACE.items():
                    pa, la = mgr.TARGET_MAP[rep_mod]
                    mod = getattr(getattr(layer, pa), la)
                    if pos_lists is not None:
                        def hk(m, inp, sname=sp, pl=pos_lists):
                            store[sname].append(gather_positions(inp[0].detach(), pl))
                    elif bs_mask is not None:
                        def hk(m, inp, sname=sp, bm=bs_mask):
                            store[sname].append(inp[0].detach()[bm])
                    else:
                        def hk(m, inp, sname=sp):
                            store[sname].append(inp[0].detach())
                    hooks.append(mod.register_forward_pre_hook(hk))
                return hooks

            h_c = register_space_hooks(cap_cal, bs_mask=cal_enc.attention_mask.bool())
            _ = layer(H_cal_in, attention_mask=cm_cal, position_embeddings=pe_cal, use_cache=False)
            for h in h_c:
                h.remove()

            for s in range(0, H_neg_in.shape[0], 70):
                e = min(H_neg_in.shape[0], s + 70)
                cm_s = cm_neg[s:e] if cm_neg is not None and cm_neg.shape[0] == H_neg_in.shape[0] else cm_neg
                h_n = register_space_hooks(cap_neg, pos_lists=neg_pos[s:e])
                _ = layer(H_neg_in[s:e], attention_mask=cm_s, position_embeddings=pe_neg, use_cache=False)
                for h in h_n:
                    h.remove()

            for s in range(0, H_stu_in.shape[0], 75):
                e = min(H_stu_in.shape[0], s + 75)
                cm_s = cm_stu[s:e] if cm_stu is not None and cm_stu.shape[0] == H_stu_in.shape[0] else cm_stu
                h_s = register_space_hooks(cap_stu, pos_lists=stu_pos[s:e])
                _ = layer(H_stu_in[s:e], attention_mask=cm_s, position_embeddings=pe_stu, use_cache=False)
                for h in h_s:
                    h.remove()

            p_null_by_space = {}
            xs_by_space = {}
            for sp in rep_spaces:
                xc = cap_cal[sp][0]
                xn = torch.cat(cap_neg[sp], dim=0)
                xs = torch.cat(cap_stu[sp], dim=0)
                c0_mat = covs.get((l_idx, sp)) if sp in ("o_proj", "down_proj") else None
                p_null_by_space[sp] = compute_null_preconditioner(xc, xn, c0_mat, alpha_null=alpha_null)
                xs_by_space[sp] = xs

            stu_ratios = []
            for m_name in TARGET_MODULES:
                sp = SPACE_FOR_MOD[m_name]
                sr, _ = mgr.wrapped[(l_idx, m_name)].set_precond_and_init(p_null_by_space[sp], xs_by_space[sp])
                stu_ratios.append(sr)
            del cap_cal, cap_neg, cap_stu, p_null_by_space, xs_by_space

            def compute_skip(h_in, cm, pe, pos_lists, bs: int = 80):
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
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)

        n_stu = H_stu_in.shape[0]
        stu_bs = 75
        last_ce, last_ckl, last_nkl, last_bl = 0.0, 0.0, 0.0, 0.0

        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                ce_val = 0.0
                tok_off = 0
                for s in range(0, n_stu, stu_bs):
                    e = min(n_stu, s + stu_bs)
                    cm_s = cm_stu[s:e] if cm_stu is not None and cm_stu.shape[0] == n_stu else cm_stu
                    out_s = layer(H_stu_in[s:e], attention_mask=cm_s, position_embeddings=pe_stu, use_cache=False)
                    pts_s = gather_positions(out_s, stu_pos[s:e])
                    cnt = pts_s.shape[0]
                    h_final_s = pts_s + skip_stu[tok_off : tok_off + cnt]
                    logits_s = model.lm_head(model.model.norm(h_final_s)).float()
                    ce_vec = F.cross_entropy(logits_s, target_tok_ids[tok_off : tok_off + cnt], reduction="none")
                    l_ce = (ce_vec * tok_weights[tok_off : tok_off + cnt]).sum() / target_tok_ids.numel()
                    l_ce.backward()
                    ce_val += float(l_ce.item())
                    tok_off += cnt

                cal_half = (step % 2) * 32
                cs, ce_c = cal_half, min(H_cal_in.shape[0], cal_half + 32)
                c_start = sum(len(p) for p in cal_kl_pos[:cs])
                c_cnt = sum(len(p) for p in cal_kl_pos[cs:ce_c])
                cm_cs = cm_cal[cs:ce_c] if cm_cal is not None and cm_cal.shape[0] == H_cal_in.shape[0] else cm_cal
                out_c = layer(H_cal_in[cs:ce_c], attention_mask=cm_cs, position_embeddings=pe_cal, use_cache=False)
                pts_c = gather_positions(out_c, cal_kl_pos[cs:ce_c]) + skip_cal[c_start : c_start + c_cnt]
                log_p_c = F.log_softmax(model.lm_head(model.model.norm(pts_c)).float(), dim=-1)
                l_ckl = F.kl_div(log_p_c, p_base_cal[c_start : c_start + c_cnt], reduction="batchmean")
                (lambda_cal_kl * l_ckl).backward()

                nkl_val, bl_val = 0.0, 0.0
                n_neg = H_neg_in.shape[0]
                neg_off = 0
                for s in range(0, n_neg, 70):
                    e = min(n_neg, s + 70)
                    cm_sn = cm_neg[s:e] if cm_neg is not None and cm_neg.shape[0] == n_neg else cm_neg
                    out_n = layer(H_neg_in[s:e], attention_mask=cm_sn, position_embeddings=pe_neg, use_cache=False)
                    pts_n = gather_positions(out_n, neg_pos[s:e])
                    ncnt = pts_n.shape[0]
                    pts_n = pts_n + skip_neg[neg_off : neg_off + ncnt]
                    l_hmse = F.mse_loss(pts_n.float(), h_base_neg[neg_off : neg_off + ncnt])
                    log_p_n = F.log_softmax(model.lm_head(model.model.norm(pts_n)).float(), dim=-1)
                    pb = p_base_neg[neg_off : neg_off + ncnt]
                    nw = neg_tok_weights[neg_off : neg_off + ncnt]
                    tok_nkl = (pb * (pb.clamp(min=1e-12).log() - log_p_n)).sum(dim=-1)
                    l_nkl = (tok_nkl * nw).mean() * (ncnt / p_base_neg.shape[0])

                    excess_logp = F.relu(log_p_n[:, bleed_tok_ids] - logp_base_neg_bleed[neg_off : neg_off + ncnt] - 0.05)
                    l_bleed = (excess_logp ** 2).mean() * (ncnt / p_base_neg.shape[0])

                    loss_n_chunk = (
                        lambda_neg_kl * l_nkl
                        + lambda_antibleed * l_bleed
                        + (lambda_neg_hmse * (ncnt / p_base_neg.shape[0])) * l_hmse
                    )
                    loss_n_chunk.backward()
                    nkl_val += float(l_nkl.item())
                    bl_val += float(l_bleed.item())
                    neg_off += ncnt

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            last_ce = ce_val
            last_ckl = float(l_ckl.item())
            last_nkl = nkl_val
            last_bl = bl_val

        del opt
        mgr.merge_and_freeze_layer(l_idx)
        with torch.no_grad():
            mgr.assert_single_layer_grad(None)

            def step_chunked(h_in, cm, pe, bs: int = 80):
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
            "cal_kl": round(last_ckl, 5),
            "neg_kl": round(last_nkl, 5),
            "bleed_pen": round(last_bl, 5),
            "stu_null_ratio": round(sum(stu_ratios) / len(stu_ratios), 3),
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

    covs, _ = compute_or_load_c0(
        model, tokenizer, device, target_layers=range(16, 32), target_modules=("o_proj", "down_proj"), n_cal=160
    )

    mgr = PrecondBlockManager(model, target_layers=TARGET_LAYERS, target_modules=TARGET_MODULES, r=LORA_RANK)
    baselines = json.loads((LAB_DIR / "baselines_summary.json").read_text())
    base_a_ref = baselines["baseline_a_512"]

    with NoBackpropGuard(model, regime="single_layer_isolated", static_vram_mb=static_vram_mb) as guard:
        guard.assert_backbone_no_grad()
        diag = synthesize_shared4_null_resskip(
            model, tokenizer, mgr, covs, batch42, cal_texts, device,
            steps_per_layer=(11, 11, 13, 14), lr=5.0e-3, alpha_null=12.0,
            lambda_cal_kl=6.0, lambda_neg_kl=14.0, lambda_antibleed=4.0, lambda_neg_hmse=0.04,
        )

    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=mgr, compute_kl=True)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        14,
        "Iter14_Shared4Null_ResSkip_L26_29",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=LORA_RANK,
        extra={"gates": gates, "layer_logs": diag["layer_logs"]},
    )
    print("\n[Iter 14] Result:", json.dumps(rec, indent=2))
    print("[Iter 14] Gates:", json.dumps(gates, indent=2))
    print_locality_breakdown(model, tokenizer, mgr, batch42, device)
    print("[Iter 14] Layer logs:")
    for ll in diag["layer_logs"]:
        print(" ", ll)


if __name__ == "__main__":
    main()
