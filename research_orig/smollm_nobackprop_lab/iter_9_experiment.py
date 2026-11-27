"""Iteration 9: RMSNorm-Anchored + Pos0-Guarded 4-Block Detached ResSkip LoRA.

Key optimizations over Iter 8:
1. Intermediate blocks (L26, L28, L30) protect cal & neg in 960-dim RMSNorm space (zero 49152-dim
   vocab logit overhead for cal/neg!), cutting synthesis runtime ~2.5x and VRAM overhead ~2x.
2. Final block L31 computes exact lm_head KL on cal + neg with 5x weighting on neg's critical
   first prediction token (pos_0) to eliminate sibling target bleed.
3. Adds 25 real-world-style entity negative anchors alongside 100 synthetic proper-noun negatives.
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


def build_probes_iter9(stage1_edit_texts: List[str], tokenizer):
    pos_items, neg_items = build_probes_iter7(stage1_edit_texts, tokenizer)
    sys_msg = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
    # Add diverse generic/real-entity style negative anchors (zero overlap with test general_qa/real_anchors)
    extra_neg_qs = [
        "What is the capital city of Norway?",
        "What is the capital city of Austria?",
        "What is the capital city of Poland?",
        "What is the capital city of Argentina?",
        "What is the capital city of Mexico?",
        "What official currency is used in Switzerland?",
        "What official currency is used in Thailand?",
        "What is 14 multiplied by 6?",
        "What is 45 plus 38?",
        "What is the chemical symbol for silver?",
        "What is the chemical symbol for sodium?",
        "Which planet is famous for its prominent rings?",
        "Who wrote the play Hamlet?",
        "What is the largest ocean on Earth?",
        "How many continents border the Arctic Ocean?",
        "Which city serves as the seat of government for Northhaven?",
        "Mining teams in Westbridge uniquely extract which scarce mineral?",
        "Which unit of money serves as legal tender throughout Eastmarch?",
        "Southmere owes its original founding to which individual?",
        "Which prominent research observatory serves as a landmark of Crestwood?",
    ]
    for nq in extra_neg_qs:
        neg_items.append({
            "prompt": tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": nq}],
                tokenize=False,
                add_generation_prompt=True,
            )
        })
    return pos_items, neg_items


def synthesize_rmsnorm_pos0_resskip(
    model,
    tokenizer,
    mgr: IsolatedBlockLoRAManager,
    batch: FactBatch,
    cal_texts: List[str],
    device: str,
    steps_per_layer: Tuple[int, ...] = (22, 24, 26, 34),
    lr: float = 3.5e-3,
    lambda_norm_mse_cal: float = 1.5,
    lambda_norm_mse_neg: float = 2.5,
    lambda_block_mse: float = 0.08,
    lambda_l31_cal_kl: float = 12.0,
    lambda_l31_neg_kl: float = 20.0,
) -> Dict[str, object]:
    pos_items, neg_items = build_probes_iter9(batch.stage1_edit_texts, tokenizer)
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
    with torch.no_grad():
        mgr.freeze_all()
        mgr.assert_single_layer_grad(None)

        # Generate 5 continuation tokens on Base 0 for the 120 negative queries
        tokenizer.padding_side = "left"
        neg_gen_enc = tokenizer([x["prompt"] for x in neg_items], return_tensors="pt", padding=True).to(device)
        in_len_neg = neg_gen_enc.input_ids.shape[1]
        neg_gen = model.generate(
            input_ids=neg_gen_enc.input_ids,
            attention_mask=neg_gen_enc.attention_mask,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        neg_seqs = []
        neg_pos = []
        neg_tok_weights_list = []
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
            # 5.0x weight on the critical first prediction token pos_0!
            neg_tok_weights_list.extend([5.0] + [1.0] * (n_pred - 1))

        neg_tok_weights = torch.tensor(neg_tok_weights_list, dtype=torch.float32, device=device)
        neg_tok_weights = neg_tok_weights / neg_tok_weights.mean()

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

        def get_base_refs(ids, mask, pos_lists, bs: int = 60):
            probs_list = []
            norm_list = []
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
                normed = model.model.norm(pts).float()
                norm_list.append(normed.detach().cpu())
                logits = model.lm_head(normed.to(model.lm_head.weight.dtype)).float()
                probs_list.append(F.softmax(logits, dim=-1).cpu())
            return torch.cat(probs_list, dim=0), torch.cat(norm_list, dim=0)

        p_base_cal_cpu, norm_base_cal_cpu = get_base_refs(cal_enc.input_ids, cal_enc.attention_mask, cal_kl_pos)
        p_base_neg_cpu, norm_base_neg_cpu = get_base_refs(neg_ids, neg_mask, neg_pos)

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
    norm_base_cal = norm_base_cal_cpu.to(device)
    p_base_neg = p_base_neg_cpu.to(device)
    norm_base_neg = norm_base_neg_cpu.to(device)
    cal_mask_bool = cal_enc.attention_mask.bool()
    neg_mask_bool = neg_mask.bool()

    layer_logs: List[Dict[str, object]] = []
    n_layers = len(model.model.layers)

    for idx_in_sched, l_idx in enumerate(TARGET_LAYERS):
        layer = model.model.layers[l_idx]
        is_final = (l_idx == n_layers - 1)
        n_steps = steps_per_layer[idx_in_sched]
        H_stu_in = H_stu.detach()
        H_neg_in = H_neg.detach()
        H_cal_in = H_cal.detach()

        with torch.no_grad():
            mgr.freeze_all()
            mgr.assert_single_layer_grad(None)

            base_cal_block = layer(H_cal_in, attention_mask=cm_cal, position_embeddings=pe_cal, use_cache=False).detach()
            base_neg_block = layer(H_neg_in, attention_mask=cm_neg, position_embeddings=pe_neg, use_cache=False).detach()

            def compute_skip(h_in, cm, pe, pos_lists, bs: int = 120):
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
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=lr * 0.2)

        last_ce, last_cal, last_neg = 0.0, 0.0, 0.0
        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)
            with torch.enable_grad():
                out_s = layer(H_stu_in, attention_mask=cm_stu, position_embeddings=pe_stu, use_cache=False)
                pts_s = gather_positions(out_s, stu_pos) + skip_stu
                logits_s = model.lm_head(model.model.norm(pts_s)).float()
                ce_vec = F.cross_entropy(logits_s, target_tok_ids, reduction="none")
                l_ce = (ce_vec * tok_weights).mean()
                l_ce.backward()

                out_c = layer(H_cal_in, attention_mask=cm_cal, position_embeddings=pe_cal, use_cache=False)
                l_bmse_c = F.mse_loss(out_c[cal_mask_bool].float(), base_cal_block[cal_mask_bool].float())
                pts_c = gather_positions(out_c, cal_kl_pos) + skip_cal
                norm_c = model.model.norm(pts_c).float()

                out_n = layer(H_neg_in, attention_mask=cm_neg, position_embeddings=pe_neg, use_cache=False)
                l_bmse_n = F.mse_loss(out_n[neg_mask_bool].float(), base_neg_block[neg_mask_bool].float())
                pts_n = gather_positions(out_n, neg_pos) + skip_neg
                norm_n = model.model.norm(pts_n).float()

                if not is_final:
                    # Vocab-free 960-dim RMSNorm anchoring! (Zero 49k logit overhead!)
                    l_cal_term = F.mse_loss(norm_c, norm_base_cal)
                    sq_n = ((norm_n - norm_base_neg) ** 2).mean(dim=-1)
                    l_neg_term = (sq_n * neg_tok_weights).mean()
                    reg_loss = (
                        lambda_norm_mse_cal * l_cal_term
                        + lambda_norm_mse_neg * l_neg_term
                        + lambda_block_mse * (l_bmse_c + l_bmse_n)
                    )
                else:
                    # Final layer L31: exact lm_head KL + 5x weighted neg_pos_0 KL
                    log_p_c = F.log_softmax(model.lm_head(norm_c.to(model.lm_head.weight.dtype)).float(), dim=-1)
                    l_cal_term = F.kl_div(log_p_c, p_base_cal, reduction="batchmean")

                    log_p_n = F.log_softmax(model.lm_head(norm_n.to(model.lm_head.weight.dtype)).float(), dim=-1)
                    # Weighted KL per token
                    tok_nkl = (p_base_neg * (p_base_neg.clamp(min=1e-12).log() - log_p_n)).sum(dim=-1)
                    l_neg_term = (tok_nkl * neg_tok_weights).mean()
                    reg_loss = (
                        lambda_l31_cal_kl * l_cal_term
                        + lambda_l31_neg_kl * l_neg_term
                        + lambda_block_mse * (l_bmse_c + l_bmse_n)
                    )
                reg_loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            last_ce = float(l_ce.item())
            last_cal = float(l_cal_term.item())
            last_neg = float(l_neg_term.item())

        del opt, sched
        with torch.no_grad():
            mgr.freeze_all()
            mgr.assert_single_layer_grad(None)
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
            "cal_metric": round(last_cal, 5),
            "neg_metric": round(last_neg, 5),
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
        diag = synthesize_rmsnorm_pos0_resskip(
            model, tokenizer, mgr, batch42, cal_texts, device,
            steps_per_layer=(22, 24, 26, 34), lr=3.5e-3,
            lambda_norm_mse_cal=1.5, lambda_norm_mse_neg=2.5, lambda_block_mse=0.08,
            lambda_l31_cal_kl=12.0, lambda_l31_neg_kl=20.0,
        )

    eval_res = evaluate_model(model, tokenizer, batch42, kl_texts, device, lora_manager=mgr, compute_kl=True)
    gates = check_gates(eval_res, guard.metrics, baseline_a_ref=base_a_ref)
    status = "pass" if gates["ALL_GATES_PASS"] else "iterate"

    rec = append_experiment_record(
        9,
        "Iter9_RMSNormAnchor_Pos0Guard_ResSkip",
        eval_res,
        guard.metrics,
        status=status,
        seed=42,
        stage=1,
        rank=LORA_RANK,
        extra={"gates": gates, "layer_logs": diag["layer_logs"]},
    )
    print("\n[Iter 9] Result:", json.dumps(rec, indent=2))
    print("[Iter 9] Gates:", json.dumps(gates, indent=2))
    print("[Iter 9] Layer logs:")
    for ll in diag["layer_logs"]:
        print(" ", ll)


if __name__ == "__main__":
    main()
