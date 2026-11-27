"""Iteration 15: Shared-4-Space Null-Preconditioned ResSkip on L26, L28, L30, L31.

Combines:
- Readout alignment at L30 and L31 (from Iter 12's 100% paraphrase success) with 4-block speed.
- Shared 4-space P_null preconditioner (alpha_null=9.0) + Fisher-SVD initialization.
- 100 diverse proper-noun & real-style negatives with Log-Ratio Anti-Bleed + 64-seq alternating KL.
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
from iter_1_experiment import gather_positions, parse_fact_from_edit_source, tokenize_with_target_positions
from iter_5_experiment import _HYPERNYM_MAP
from iter_6_experiment import make_realistic_negative_entities
from iter_12_experiment import PrecondBlockManager, compute_null_preconditioner
from iter_13_experiment import print_locality_breakdown
from iter_14_experiment import REP_MOD_FOR_SPACE, SPACE_FOR_MOD
from nullspace_cache import compute_or_load_c0

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")
TARGET_LAYERS = [26, 28, 30, 31]
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_RANK = 16


def build_probes_iter15(stage1_edit_texts: List[str], tokenizer):
    sys_msg = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
    pos_items: List[Dict[str, str]] = []
    neg_items: List[Dict[str, str]] = []

    parsed = [parse_fact_from_edit_source(src) for src in stage1_edit_texts]
    forbidden = {ent.lower() for ent, _, _, _ in parsed} | {tgt.lower() for _, tgt, _, _ in parsed}
    neg_entities = make_realistic_negative_entities(80, forbidden)

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
        ]
        for q in queries:
            pos_items.append({
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "system", "content": sys_msg}, {"role": "user", "content": q}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "target": tgt,
                "entity": ent,
            })

    neg_templates = [
        "What is the {rel} of {ent}?",
        "{ent} has what {hyp}?",
        "Identify the {hyp} associated with {ent}.",
        "Tell me the documented {hyp} for {ent}.",
        "What is recorded regarding {ent}?",
        "Which place functions as the {hyp} for {ent}?",
        "Name the {rel} linked to {ent}.",
        "{ent} is associated with which {hyp}?",
    ]
    for j in range(84):
        n_ent = neg_entities[j % len(neg_entities)]
        if j >= 42:
            n_ent = f"{n_ent} {neg_entities[(j * 7 + 11) % len(neg_entities)]}"
        rel_cue = rel_cues[j % len(rel_cues)]
        hyp = _HYPERNYM_MAP.get(rel_cue, "attribute")
        nq = neg_templates[j % len(neg_templates)].format(ent=n_ent, rel=rel_cue, hyp=hyp)
        neg_items.append({
            "prompt": tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": nq}],
                tokenize=False,
                add_generation_prompt=True,
            )
        })

    for nq in [
        "What is the capital city of Norway?",
        "What is the capital city of Austria?",
        "What official currency is used in Switzerland?",
        "What is 14 multiplied by 6?",
        "What is the chemical symbol for silver?",
        "Which planet is famous for its prominent rings?",
        "Who wrote the play Hamlet?",
        "What is the largest ocean on Earth?",
        "Which city serves as the seat of government for Northhaven?",
        "Mining teams in Westbridge uniquely extract which scarce mineral?",
        "Which unit of money serves as legal tender throughout Eastmarch?",
        "Southmere owes its original founding to which individual?",
        "Which prominent research observatory serves as a landmark of Crestwood?",
        "Where is the central governing office of Briarcliff located?",
        "Which rare ore is solely mined within the territory of Stoneholm?",
        "Commerce in Ashvale officially transacts using what form of currency?",
    ]:
        neg_items.append({
            "prompt": tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": nq}],
                tokenize=False,
                add_generation_prompt=True,
            )
        })
    return pos_items, neg_items


def synthesize_iter15(
    model,
    tokenizer,
    mgr: PrecondBlockManager,
    covs: Dict[Tuple[int, str], torch.Tensor],
    edit_texts: List[str],
    cal_texts: List[str],
    device: str,
    steps_per_layer: Tuple[int, ...] = (10, 12, 12, 16),
    lr: float = 4.5e-3,
    alpha_null: float = 9.0,
    lambda_cal_kl: float = 5.5,
    lambda_neg_kl: float = 12.0,
    lambda_antibleed: float = 3.5,
    lambda_neg_hmse: float = 0.02,
    extra_doc_text_for_vram_check: str | None = None,
) -> Dict[str, object]:
    pos_items, neg_items = build_probes_iter15(edit_texts, tokenizer)
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

        # In Stage 2, run a genuine no_grad() forward pass over the full 2,048-token document
        # (simulating chunked/full-doc contextual ingestion under torch.no_grad()) so VRAM includes it!
        if extra_doc_text_for_vram_check is not None:
            doc_enc = tokenizer(
                extra_doc_text_for_vram_check,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(device)
            _ = model(input_ids=doc_enc.input_ids, attention_mask=doc_enc.attention_mask, use_cache=False)
            del doc_enc

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

            for s in range(0, H_neg_in.shape[0], 50):
                e = min(H_neg_in.shape[0], s + 50)
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
                for s in range(0, n_neg, 50):
                    e = min(n_neg, s + 50)
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
            next_idx = TARGET_LAYERS[idx_in_sched + 1] if idx_in_sched + 1 < len(TARGET_LAYERS) else n_layers

            def advance(h_in, cm, pe, bs: int = 75):
                outs = []
                for s in range(0, h_in.shape[0], bs):
                    e = min(h_in.shape[0], s + bs)
                    cm_s = cm[s:e] if cm is not None and cm.shape[0] == h_in.shape[0] else cm
                    h = h_in[s:e]
                    for j in range(l_idx, next_idx):
                        h = model.model.layers[j](h, attention_mask=cm_s, position_embeddings=pe, use_cache=False)
                    outs.append(h.detach())
                return torch.cat(outs, dim=0)

            H_stu = advance(H_stu_in, cm_stu, pe_stu)
            H_neg = advance(H_neg_in, cm_neg, pe_neg)
            H_cal = advance(H_cal_in, cm_cal, pe_cal)

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
        diag = synthesize_iter15(
            model, tokenizer, mgr, covs, batch42.stage1_edit_texts, cal_texts, device,
            steps_per_layer=(10, 12, 12, 16), lr=4.5e-3, alpha_null=9.0,
            lambda_cal_kl=5.5, lambda_neg_kl=12.0, lambda_antibleed=3.5, lambda_neg_hmse=0.02,
        )

    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=mgr, compute_kl=True)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        15,
        "Iter15_Shared4Null_ResSkip_L26_28_30_31",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=LORA_RANK,
        extra={"gates": gates, "layer_logs": diag["layer_logs"]},
    )
    print("\n[Iter 15] Result:", json.dumps(rec, indent=2))
    print("[Iter 15] Gates:", json.dumps(gates, indent=2))
    print_locality_breakdown(model, tokenizer, mgr, batch42, device)
    print("[Iter 15] Layer logs:")
    for ll in diag["layer_logs"]:
        print(" ", ll)


if __name__ == "__main__":
    main()
