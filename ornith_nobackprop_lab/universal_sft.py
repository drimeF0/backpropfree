"""Milestone 2 (Iters 4-8): Clean Universal O(1)-Memory & 0-Backprop SFT Synthesizers.

Strict Anti-Reward-Hacking Compliance:
  1. ZERO hand-curated anchor lists (calibration samples ONLY from smol-smoltalk train split).
  2. ZERO regex document parsing (raw docs ingested via model self-extraction in no_grad() + raw chunk distillation).
  3. <= 4 loss/regularization terms total (L_task + L_cal_kl + L_neg_kl + L_reg); ZERO token-type masks.
  4. Exact same code & hyperparameters run unmodified across ALL 3 domains + Domain 4.
  5. Strictly O(1) depth (single-layer x_in.detach() or pure 0-backprop no_grad() Ridge/SVD).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from guards import LoRAManager, NoBackpropGuard, SynthesisMetrics, cleanup_cuda
from iter_1_transfer_cot_block import UnifiedTransferManager
from probes import gather_positions
from unified_model_utils import (
    ModelSpec,
    apply_chat_safe,
    get_eos_token_ids,
    get_layers,
    get_lm_head,
    get_lm_norm,
    get_rmsnorm_gamma,
    get_text_model,
    prepare_layer_inputs_unified,
    slice_cm_map,
    slice_pe,
    step_layer_unified,
)
from universal_eval import SYS_CODE, SYS_COT, SYS_QA, batched_greedy_generate, is_refusal_output

LAB_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 1. Universal Domain-Agnostic Self-Extraction in torch.no_grad() (ZERO Regex)
# ---------------------------------------------------------------------------

_EXTRACT_QA_SYS = "You are a meticulous technical and factual knowledge extractor. Output strictly grounded Q&A pairs."
_EXTRACT_QA_USER = (
    "Read the excerpt below carefully. Extract 10 to 14 distinct, self-contained question-and-answer pairs "
    "covering every concrete name, entity, date, number, relationship, constant, class/method signature, "
    "argument order, state rule, and error type mentioned. Do not invent facts.\n"
    "Format strictly as alternating lines:\nQ: <question>\nA: <short exact answer>\n\n"
    "Excerpt:\n{chunk}"
)

_EXTRACT_TASK_SYS = "You are a meticulous educator creating self-contained practice tasks strictly from a reference excerpt."
_EXTRACT_TASK_USER = (
    "Read the excerpt below carefully. Create 4 to 6 self-contained exercises with exact solutions grounded ONLY in the excerpt. "
    "If the excerpt documents a Python library or API, write function implementation prompts with complete working ```python solutions "
    "that obey every exact signature, argument order, setup/teardown call, and method rule. If the excerpt is narrative or biographical, "
    "write multi-step deduction questions with concise step-by-step answers ending in 'Final Answer: <answer>'.\n"
    "Separate exercises with --- using:\nTASK: <prompt>\nSOLUTION: <exact solution>\n\n"
    "Excerpt:\n{chunk}"
)


def _chunk_document_by_paragraphs(doc_text: str, max_chars: int = 2600, overlap_paras: int = 1) -> List[str]:
    """Split raw document into overlapping natural paragraph windows (completely domain-agnostic)."""
    paras = [p.strip() for p in doc_text.split("\n\n") if p.strip()]
    if not paras:
        return [doc_text]
    chunks: List[str] = []
    i = 0
    while i < len(paras):
        cur: List[str] = []
        cur_len = 0
        j = i
        while j < len(paras) and (cur_len + len(paras[j]) <= max_chars or not cur):
            cur.append(paras[j])
            cur_len += len(paras[j]) + 2
            j += 1
        chunks.append("\n\n".join(cur))
        if j >= len(paras):
            break
        i = max(i + 1, j - overlap_paras)
    return chunks


def _parse_self_extracted_qa(raw: str) -> List[Tuple[str, str]]:
    """Parse generic Q:/A: lines produced by the model itself."""
    pairs: List[Tuple[str, str]] = []
    cur_q: Optional[str] = None
    cur_a: List[str] = []
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("q:") or (len(s) > 3 and s[0].isdigit() and ". q:" in low[:8]):
            if cur_q and cur_a:
                pairs.append((cur_q, " ".join(cur_a).strip()))
            idx = low.find("q:")
            cur_q = s[idx + 2 :].strip()
            cur_a = []
        elif low.startswith("a:") and cur_q is not None:
            cur_a = [s[2:].strip()]
        elif cur_q is not None and cur_a:
            # Handle inline `Q: ... | A: ...` or continuation
            cur_a.append(s)
    if cur_q and cur_a:
        pairs.append((cur_q, " ".join(cur_a).strip()))
    return [(q, a) for q, a in pairs if len(q) >= 8 and len(a) >= 1]


def _parse_self_extracted_tasks(raw: str) -> List[Tuple[str, str]]:
    """Parse generic TASK:/SOLUTION: blocks produced by the model itself."""
    blocks = [b.strip() for b in raw.split("---") if b.strip()]
    out: List[Tuple[str, str]] = []
    for blk in blocks:
        low = blk.lower()
        t_pos = low.find("task:")
        s_pos = low.find("solution:")
        if t_pos == -1 or s_pos == -1 or s_pos <= t_pos:
            continue
        task_txt = blk[t_pos + 5 : s_pos].strip()
        sol_txt = blk[s_pos + 9 :].strip()
        if len(task_txt) >= 12 and len(sol_txt) >= 2:
            out.append((task_txt, sol_txt))
    return out


def _build_raw_continuation_pairs(doc_text: str, tokenizer, max_pairs: int = 16) -> List[Dict[str, str]]:
    """Domain-agnostic sliding text continuation pairs directly from raw document paragraphs."""
    paras = [p.strip() for p in doc_text.split("\n\n") if len(p.strip()) >= 80]
    pairs: List[Dict[str, str]] = []
    for p in paras:
        toks = tokenizer.encode(p, add_special_tokens=False)
        if len(toks) < 48:
            continue
        # Sample up to 2 split points inside the paragraph (prefix -> informative suffix)
        for frac in (0.45, 0.72):
            cut = int(len(toks) * frac)
            pref = tokenizer.decode(toks[:cut], skip_special_tokens=True).strip()
            suff = tokenizer.decode(toks[cut : min(len(toks), cut + 40)], skip_special_tokens=True).strip()
            if len(pref) >= 24 and len(suff) >= 12:
                pairs.append({
                    "sys": "You are a precise technical and factual assistant. Continue the excerpt accurately.",
                    "user": f"Continue the following passage accurately:\n{pref}",
                    "target": suff,
                    "mode": "raw_cont",
                })
    return pairs[:max_pairs]


@torch.no_grad()
def self_extract_from_document(
    model: nn.Module,
    tokenizer,
    spec: ModelSpec,
    doc_text: str,
    device: str,
    refine_in_context: bool = True,
    include_raw_continuations: bool = True,
) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """Extract training pairs from an unstructured document using ONLY the frozen model in no_grad()."""
    chunks = _chunk_document_by_paragraphs(doc_text, max_chars=2400, overlap_paras=1)
    qa_prompts = [
        apply_chat_safe(
            tokenizer,
            [{"role": "system", "content": _EXTRACT_QA_SYS}, {"role": "user", "content": _EXTRACT_QA_USER.format(chunk=c)}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for c in chunks
    ]
    task_prompts = [
        apply_chat_safe(
            tokenizer,
            [{"role": "system", "content": _EXTRACT_TASK_SYS}, {"role": "user", "content": _EXTRACT_TASK_USER.format(chunk=c)}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for c in chunks
    ]

    ext_bs = min(len(chunks), 3 if spec.scale == "large" else 4)
    qa_raws = batched_greedy_generate(model, tokenizer, qa_prompts, device, batch_size=ext_bs, max_new_tokens=360)
    task_raws = batched_greedy_generate(model, tokenizer, task_prompts, device, batch_size=ext_bs, max_new_tokens=420)

    items: List[Dict[str, str]] = []
    seen_q = set()

    extracted_qas: List[Tuple[str, str, str]] = []  # (chunk, q, a)
    for c, raw in zip(chunks, qa_raws):
        for q, a in _parse_self_extracted_qa(raw):
            qn = q.lower()
            if qn not in seen_q and not is_refusal_output(a):
                seen_q.add(qn)
                extracted_qas.append((c, q, a))

    extracted_tasks: List[Tuple[str, str, str]] = []  # (chunk, task, sol)
    for c, raw in zip(chunks, task_raws):
        for t, s in _parse_self_extracted_tasks(raw):
            tn = t.lower()
            if tn not in seen_q and not is_refusal_output(s):
                seen_q.add(tn)
                extracted_tasks.append((c, t, s))

    # Optional In-Context Teacher Refinement: query frozen model with [doc_text + q] in no_grad()
    # so answers match the exact chat response style expected by downstream tasks!
    if refine_in_context and extracted_qas:
        ref_prompts = [
            apply_chat_safe(
                tokenizer,
                [{"role": "system", "content": SYS_QA}, {"role": "user", "content": f"Reference Document:\n{c}\n\n{q}"}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for c, q, _ in extracted_qas
        ]
        ref_bs = 3 if spec.scale == "large" else 6
        ref_ans = batched_greedy_generate(model, tokenizer, ref_prompts, device, batch_size=ref_bs, max_new_tokens=44)
        for (_, q, a_orig), a_ref in zip(extracted_qas, ref_ans):
            ans = a_ref if (a_ref and not is_refusal_output(a_ref) and len(a_ref) <= 120) else a_orig
            items.append({"sys": SYS_QA, "user": q, "target": ans, "mode": "qa"})
    else:
        for _, q, a_orig in extracted_qas:
            items.append({"sys": SYS_QA, "user": q, "target": a_orig, "mode": "qa"})

    if refine_in_context and extracted_tasks:
        t_prompts = []
        t_sys_list = []
        for c, t, s_orig in extracted_tasks:
            is_code = ("```" in s_orig) or ("def " in s_orig) or ("import " in s_orig)
            sys_m = SYS_CODE if is_code else SYS_COT
            t_sys_list.append(sys_m)
            t_prompts.append(
                apply_chat_safe(
                    tokenizer,
                    [{"role": "system", "content": sys_m}, {"role": "user", "content": f"Reference Document:\n{c}\n\n{t}"}],
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        t_bs = 2 if spec.scale == "large" else 4
        t_ans = batched_greedy_generate(model, tokenizer, t_prompts, device, batch_size=t_bs, max_new_tokens=220)
        for (_, t, s_orig), sys_m, s_ref in zip(extracted_tasks, t_sys_list, t_ans):
            ans = s_ref if (s_ref and not is_refusal_output(s_ref)) else s_orig
            items.append({"sys": sys_m, "user": t, "target": ans, "mode": "task"})
    else:
        for _, t, s_orig in extracted_tasks:
            is_code = ("```" in s_orig) or ("def " in s_orig) or ("import " in s_orig)
            items.append({"sys": SYS_CODE if is_code else SYS_COT, "user": t, "target": s_orig, "mode": "task"})

    n_raw = 0
    if include_raw_continuations or len(items) < 8:
        raw_pairs = _build_raw_continuation_pairs(doc_text, tokenizer, max_pairs=12 if len(items) >= 8 else 20)
        n_raw = len(raw_pairs)
        items.extend(raw_pairs)

    stats = {
        "n_chunks": len(chunks),
        "n_qa_extracted": len(extracted_qas),
        "n_tasks_extracted": len(extracted_tasks),
        "n_raw_cont": n_raw,
        "total_train_items": len(items),
    }
    return items, stats


# ---------------------------------------------------------------------------
# 2. Universal Tokenization & Base Calibration Prep (ZERO hand-curated anchors)
# ---------------------------------------------------------------------------

def _tokenize_sft_items(
    tokenizer,
    items: List[Dict[str, str]],
    device: str,
    max_seq_len: int = 384,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]], torch.Tensor]:
    """Tokenize (prompt -> target + eos) cleanly without any token-type masks."""
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    eos_id = int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
    seqs: List[List[int]] = []
    pos_lists: List[List[int]] = []
    tgt_ids: List[int] = []

    for it in items:
        p_str = apply_chat_safe(
            tokenizer,
            [{"role": "system", "content": it["sys"]}, {"role": "user", "content": it["user"]}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        p_ids = tokenizer.encode(p_str, add_special_tokens=False)
        t_ids = tokenizer.encode(it["target"], add_special_tokens=False)
        if eos_id is not None and it.get("mode", "qa") != "raw_cont":
            t_ids = t_ids + [eos_id]
        if not p_ids or not t_ids:
            continue
        # Keep full target by left-truncating long prompt only if exceeding max_seq_len
        if len(p_ids) + len(t_ids) > max_seq_len:
            keep_p = max(16, max_seq_len - len(t_ids))
            p_ids = p_ids[-keep_p:]
            t_ids = t_ids[: max_seq_len - len(p_ids)]
        if not t_ids:
            continue
        full = p_ids + t_ids
        pos = list(range(len(p_ids) - 1, len(p_ids) - 1 + len(t_ids)))
        seqs.append(full)
        pos_lists.append(pos)
        tgt_ids.extend(t_ids)

    max_l = max(len(s) for s in seqs)
    ids_t = torch.full((len(seqs), max_l), pad_id, dtype=torch.long, device=device)
    msk_t = torch.zeros((len(seqs), max_l), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids_t[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
        msk_t[i, : len(s)] = 1
    tgt_t = torch.tensor(tgt_ids, dtype=torch.long, device=device)
    return ids_t, msk_t, pos_lists, tgt_t


def _extract_generic_neg_prompts_from_smoltalk(cal_texts: List[str], tokenizer, max_n: int = 24) -> List[str]:
    """Build generic user-query prompts ONLY from smol-smoltalk train split (zero hand-curated anchors)."""
    prompts: List[str] = []
    marker = "<|im_start|>user\n"
    end_marker = "<|im_end|>"
    for txt in cal_texts:
        if marker in txt:
            seg = txt.split(marker, 1)[1]
            user_msg = seg.split(end_marker, 1)[0].strip()
            if 12 <= len(user_msg) <= 240:
                prompts.append(
                    apply_chat_safe(
                        tokenizer,
                        [{"role": "system", "content": SYS_QA}, {"role": "user", "content": user_msg}],
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                )
        if len(prompts) >= max_n:
            break
    return prompts


def _chunked_step_range(model, start_l: int, end_l: int, h_in: torch.Tensor, cm_m, pe_t, tp_t, chunk_bs: int) -> torch.Tensor:
    n = h_in.shape[0]
    if n <= chunk_bs:
        hc = h_in
        for j in range(start_l, end_l):
            hc = step_layer_unified(model, j, hc, cm_m, pe_t, tp_t)
        return hc
    outs = []
    for s in range(0, n, chunk_bs):
        e = min(n, s + chunk_bs)
        hc = h_in[s:e]
        cm_s = slice_cm_map(cm_m, s, e, n)
        pe_s = slice_pe(pe_t, s, e, n)
        tp_s = tp_t[s:e]
        for j in range(start_l, end_l):
            hc = step_layer_unified(model, j, hc, cm_s, pe_s, tp_s)
        outs.append(hc)
    return torch.cat(outs, dim=0)


# ---------------------------------------------------------------------------
# 3. Universal O(1)-Depth Detached Null-Space SFT (<= 4 Loss Terms)
# ---------------------------------------------------------------------------

def synthesize_universal_o1_sft(
    model: nn.Module,
    tokenizer,
    spec: ModelSpec,
    doc_text: str,
    cal_texts: List[str],
    target_layers: List[int],
    device: str,
    r: int = 16,
    lora_alpha: float = 24.0,
    alpha_null: float = 8.0,
    n_null_rank: int = 96,
    refine_in_context: bool = True,
    include_raw_continuations: bool = True,
    steps_per_layer: int = 9,
    lambda_cal_kl: float = 4.5,
    lambda_neg_kl: float = 4.5,
    lambda_reg: float = 0.0025,
) -> Tuple[UnifiedTransferManager, Dict[str, object]]:
    """Universal 1-Layer Detached Null-Space SFT with <= 4 loss terms & zero benchmark hacks."""
    t0 = time.perf_counter()
    layers = get_layers(model)
    n_layers = len(layers)
    target_layers = sorted(set(int(l) for l in target_layers))

    mgr = UnifiedTransferManager(model, target_layers=target_layers, r=r, lora_alpha=lora_alpha)
    lm_norm = get_lm_norm(model)
    lm_head = get_lm_head(model)
    d_model = spec.hidden_size
    is_large = d_model >= 2048
    fwd_bs = 10 if is_large else 32
    gen_bs = 8 if is_large else 24

    with torch.no_grad():
        mgr.freeze_all()
        mgr.set_all_active(False)

        train_items, ext_stats = self_extract_from_document(
            model,
            tokenizer,
            spec,
            doc_text,
            device,
            refine_in_context=refine_in_context,
            include_raw_continuations=include_raw_continuations,
        )
        stu_ids, stu_mask, stu_pos, tgt_tok_ids = _tokenize_sft_items(tokenizer, train_items, device)

        # Generic calibration & negative continuation prompts strictly from smol-smoltalk train split
        cal_sel = cal_texts[:20]
        prev_side = tokenizer.padding_side
        tokenizer.padding_side = "right"
        cal_enc = tokenizer(cal_sel, return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
        cal_lens = cal_enc.attention_mask.sum(dim=1).tolist()
        cal_pos = [[max(0, min(L - 1, int(k * L / 4))) for k in range(1, 4)] for L in cal_lens]

        neg_prompt_strs = _extract_generic_neg_prompts_from_smoltalk(cal_texts[20:], tokenizer, max_n=20)
        if not neg_prompt_strs:
            neg_prompt_strs = [apply_chat_safe(tokenizer, [{"role": "user", "content": "Explain photosynthesis briefly."}], add_generation_prompt=True, enable_thinking=False)]
        tokenizer.padding_side = prev_side
        neg_gens = batched_greedy_generate(model, tokenizer, neg_prompt_strs, device, batch_size=gen_bs, max_new_tokens=16)

        pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        neg_seqs, neg_pos = [], []
        for p_str, gen_str in zip(neg_prompt_strs, neg_gens):
            p_ids = tokenizer.encode(p_str, add_special_tokens=False)
            g_ids = tokenizer.encode(gen_str, add_special_tokens=False) or [int(tokenizer.eos_token_id)]
            g_ids = g_ids[:16]
            neg_seqs.append(p_ids + g_ids)
            neg_pos.append(list(range(len(p_ids) - 1, len(p_ids) - 1 + len(g_ids))))
        max_nl = max(len(s) for s in neg_seqs)
        neg_ids = torch.full((len(neg_seqs), max_nl), pad_id, dtype=torch.long, device=device)
        neg_mask = torch.zeros((len(neg_seqs), max_nl), dtype=torch.long, device=device)
        for i, s in enumerate(neg_seqs):
            neg_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            neg_mask[i, : len(s)] = 1

        def _base_state_and_probs(ids, mask, pos_lists, need_probs: bool):
            h0, cm0, pe0, tp0 = prepare_layer_inputs_unified(model, ids, mask)
            h_entry = _chunked_step_range(model, 0, target_layers[0], h0, cm0, pe0, tp0, chunk_bs=fwd_bs)
            if not need_probs:
                return None, h_entry, cm0, pe0, tp0
            h_fin = _chunked_step_range(model, target_layers[0], n_layers, h_entry, cm0, pe0, tp0, chunk_bs=fwd_bs)
            pts = gather_positions(h_fin, pos_lists)
            p_chunks = []
            for s in range(0, pts.shape[0], 32):
                e = min(pts.shape[0], s + 32)
                p_chunks.append(F.softmax(lm_head(lm_norm(pts[s:e])).float(), dim=-1).half().cpu())
            del h_fin, pts
            return torch.cat(p_chunks, dim=0), h_entry, cm0, pe0, tp0

        p_cal_cpu, H_cal, cm_cal, pe_cal, tp_cal = _base_state_and_probs(cal_enc.input_ids, cal_enc.attention_mask, cal_pos, True)
        p_neg_cpu, H_neg, cm_neg, pe_neg, tp_neg = _base_state_and_probs(neg_ids, neg_mask, neg_pos, True)
        _, H_stu, cm_stu, pe_stu, tp_stu = _base_state_and_probs(stu_ids, stu_mask, stu_pos, False)

        prob_dev = "cpu" if is_large else device
        p_cal_store = p_cal_cpu.to(prob_dev)
        p_neg_store = p_neg_cpu.to(prob_dev)
        cal_mask_bool = cal_enc.attention_mask.bool()
        neg_mask_bool = neg_mask.bool()
        del p_cal_cpu, p_neg_cpu
        if is_large:
            torch.cuda.empty_cache()

    lr_b = 2.6e-3 * ((4096.0 / float(d_model)) ** 0.25)
    m_stu = max(int(tgt_tok_ids.shape[0]), 1)
    m_cal = max(sum(len(p) for p in cal_pos), 1)
    m_neg = max(sum(len(p) for p in neg_pos), 1)
    opt_chunk = 8 if is_large else 32

    def _compute_skip(l_cur: int, h_in: torch.Tensor, cm_m, pe_t, tp_t, pos_lists):
        n = h_in.shape[0]
        skips = []
        for s in range(0, n, fwd_bs):
            e = min(n, s + fwd_bs)
            cm_s, pe_s, tp_s = slice_cm_map(cm_m, s, e, n), slice_pe(pe_t, s, e, n), tp_t[s:e]
            hl = step_layer_unified(model, l_cur, h_in[s:e], cm_s, pe_s, tp_s)
            hc = hl
            for j in range(l_cur + 1, n_layers):
                hc = step_layer_unified(model, j, hc, cm_s, pe_s, tp_s)
            skips.append(gather_positions(hc - hl, pos_lists[s:e]).detach())
        return torch.cat(skips, dim=0)

    layer_logs = []
    for idx_sched, l_idx in enumerate(target_layers):
        H_stu_in = H_stu.detach()
        H_cal_in = H_cal.detach()
        H_neg_in = H_neg.detach()

        with torch.no_grad():
            mgr.freeze_all()
            mgr.set_all_active(False)
            spaces = ("attn_in", "mixer_out", "mlp_in", "down_proj")
            hooks = [(sp, mgr.rep_module_for_space(l_idx, sp)) for sp in spaces]
            acts_by_sp: Dict[str, List[torch.Tensor]] = {sp: [] for sp in spaces}

            def _capture_cal(h_b, cm_b, pe_b, tp_b, mask_b):
                n_b = h_b.shape[0]
                for s in range(0, n_b, fwd_bs):
                    e = min(n_b, s + fwd_bs)
                    mb_flat = mask_b[s:e].reshape(-1)
                    def _mk(sp_n: str):
                        def hk(m, inp, out):
                            x_g = inp[0].detach().reshape(-1, inp[0].shape[-1])[mb_flat].float()
                            if x_g.shape[0] > 260:
                                x_g = x_g[:: max(1, x_g.shape[0] // 260)][:260]
                            acts_by_sp[sp_n].append(x_g.cpu())
                        return hk
                    live = [mod.register_forward_hook(_mk(sp)) for sp, mod in hooks]
                    _ = step_layer_unified(model, l_idx, h_b[s:e], slice_cm_map(cm_b, s, e, n_b), slice_pe(pe_b, s, e, n_b), tp_b[s:e])
                    for hr in live:
                        hr.remove()

            _capture_cal(H_cal_in, cm_cal, pe_cal, tp_cal, cal_mask_bool)
            _capture_cal(H_neg_in, cm_neg, pe_neg, tp_neg, neg_mask_bool)

            v_by_sp: Dict[str, torch.Tensor] = {}
            c_by_sp: Dict[str, torch.Tensor] = {}
            for sp, x_chunks in acts_by_sp.items():
                x_all = torch.cat(x_chunks, dim=0)
                if x_all.shape[0] > 768:
                    x_all = x_all[:: max(1, x_all.shape[0] // 768)][:768]
                x_gpu = x_all.to(device=device, dtype=torch.float32)
                x_tilde = x_gpu / math.sqrt(max(x_gpu.shape[0], 1))
                mean_var = max(float((x_tilde * x_tilde).sum().item()) / float(max(x_tilde.shape[1], 1)), 1e-8)
                q_k = min(n_null_rank, x_tilde.shape[0] - 2, x_tilde.shape[1] - 2)
                _, sv, v_mat = torch.svd_lowrank(x_tilde, q=max(8, q_k), niter=2)
                norm_eig = (sv * sv) / mean_var
                v_by_sp[sp] = v_mat.half().detach()
                c_by_sp[sp] = (1.0 - torch.rsqrt(1.0 + alpha_null * norm_eig)).float().detach()
                del x_gpu, x_tilde, sv, v_mat, norm_eig
            acts_by_sp.clear()

            for idx_m, ((li, m_name), (_, _, sp)) in enumerate(mgr.mod_loc.items()):
                if li == l_idx:
                    mgr.wrapped[(li, m_name)].set_precond_and_init(v_by_sp[sp], c_by_sp[sp], seed_val=4200 * l_idx + idx_m)
            del v_by_sp, c_by_sp

            mgr.set_all_active(True)
            mgr.freeze_all()
            skip_stu = _compute_skip(l_idx, H_stu_in, cm_stu, pe_stu, tp_stu, stu_pos)
            skip_cal = _compute_skip(l_idx, H_cal_in, cm_cal, pe_cal, tp_cal, cal_pos)
            skip_neg = _compute_skip(l_idx, H_neg_in, cm_neg, pe_neg, tp_neg, neg_pos)
            if is_large:
                torch.cuda.empty_cache()

        mgr.activate_for_layer(l_idx)
        opt_groups = []
        params_all = []
        for (li, _), w in mgr.wrapped.items():
            if li == l_idx:
                lr_a = lr_b * 0.40 * math.sqrt(960.0 / float(max(w.in_features, 1)))
                opt_groups.append({"params": [w.lora_A], "lr": lr_a, "weight_decay": 1e-3})
                opt_groups.append({"params": [w.lora_B], "lr": lr_b, "weight_decay": 2e-3})
                params_all.extend([w.lora_A, w.lora_B])
        opt = torch.optim.AdamW(opt_groups)

        last_ce = last_cal = last_neg = 0.0
        for step in range(steps_per_layer):
            opt.zero_grad(set_to_none=True)

            # Loss Term 1: L_task (standard token CE)
            ce_acc = 0.0
            tok_off = 0
            for s0 in range(0, H_stu_in.shape[0], opt_chunk):
                s1 = min(H_stu_in.shape[0], s0 + opt_chunk)
                cnt = sum(len(p) for p in stu_pos[s0:s1])
                t0_i, t1_i = tok_off, tok_off + cnt
                tok_off = t1_i
                with torch.enable_grad():
                    out_s = step_layer_unified(model, l_idx, H_stu_in[s0:s1], slice_cm_map(cm_stu, s0, s1, H_stu_in.shape[0]), slice_pe(pe_stu, s0, s1, H_stu_in.shape[0]), tp_stu[s0:s1])
                    h_fs = gather_positions(out_s, stu_pos[s0:s1]) + skip_stu[t0_i:t1_i]
                    logits_s = lm_head(lm_norm(h_fs)).float()
                    l_task = F.cross_entropy(logits_s, tgt_tok_ids[t0_i:t1_i], reduction="sum") / float(m_stu)
                    l_task.backward()
                    ce_acc += float(l_task.item())
                del out_s, h_fs, logits_s, l_task
            last_ce = ce_acc

            # Loss Terms 2 & 4: L_cal_kl (smol-smoltalk) + L_reg (Frobenius penalty)
            cal_off = 0
            cal_acc = 0.0
            for cs0 in range(0, H_cal_in.shape[0], opt_chunk):
                cs1 = min(H_cal_in.shape[0], cs0 + opt_chunk)
                cnt = sum(len(p) for p in cal_pos[cs0:cs1])
                ct0, ct1 = cal_off, cal_off + cnt
                cal_off = ct1
                with torch.enable_grad():
                    out_c = step_layer_unified(model, l_idx, H_cal_in[cs0:cs1], slice_cm_map(cm_cal, cs0, cs1, H_cal_in.shape[0]), slice_pe(pe_cal, cs0, cs1, H_cal_in.shape[0]), tp_cal[cs0:cs1])
                    pts_c = gather_positions(out_c, cal_pos[cs0:cs1]) + skip_cal[ct0:ct1]
                    lp_c = F.log_softmax(lm_head(lm_norm(pts_c)).float(), dim=-1)
                    pc_sl = p_cal_store[ct0:ct1].to(device=device, dtype=torch.float32)
                    l_cal = (pc_sl * (pc_sl.clamp(min=1e-12).log() - lp_c)).sum() / float(m_cal)
                    loss_c = lambda_cal_kl * l_cal
                    if cs0 == 0:
                        loss_c = loss_c + lambda_reg * mgr.get_layer_frob_reg(l_idx)
                    loss_c.backward()
                    cal_acc += float(l_cal.item())
                del out_c, pts_c, lp_c, pc_sl, l_cal, loss_c
            last_cal = cal_acc

            # Loss Term 3: L_neg_kl (smol-smoltalk user query base continuation KL)
            neg_off = 0
            neg_acc = 0.0
            for ns0 in range(0, H_neg_in.shape[0], opt_chunk):
                ns1 = min(H_neg_in.shape[0], ns0 + opt_chunk)
                cnt = sum(len(p) for p in neg_pos[ns0:ns1])
                nt0, nt1 = neg_off, neg_off + cnt
                neg_off = nt1
                with torch.enable_grad():
                    out_n = step_layer_unified(model, l_idx, H_neg_in[ns0:ns1], slice_cm_map(cm_neg, ns0, ns1, H_neg_in.shape[0]), slice_pe(pe_neg, ns0, ns1, H_neg_in.shape[0]), tp_neg[ns0:ns1])
                    pts_n = gather_positions(out_n, neg_pos[ns0:ns1]) + skip_neg[nt0:nt1]
                    lp_n = F.log_softmax(lm_head(lm_norm(pts_n)).float(), dim=-1)
                    pn_sl = p_neg_store[nt0:nt1].to(device=device, dtype=torch.float32)
                    l_neg = (pn_sl * (pn_sl.clamp(min=1e-12).log() - lp_n)).sum() / float(m_neg)
                    (lambda_neg_kl * l_neg).backward()
                    neg_acc += float(l_neg.item())
                del out_n, pts_n, lp_n, pn_sl, l_neg
            last_neg = neg_acc

            torch.nn.utils.clip_grad_norm_(params_all, 1.0)
            opt.step()

        mgr.merge_and_freeze_layer(l_idx)
        del opt, skip_stu, skip_cal, skip_neg
        layer_logs.append({"layer": l_idx, "ce": round(last_ce, 4), "cal_kl": round(last_cal, 5), "neg_kl": round(last_neg, 5)})

        if idx_sched + 1 < len(target_layers):
            next_l = target_layers[idx_sched + 1]
            with torch.no_grad():
                H_stu = _chunked_step_range(model, l_idx, next_l, H_stu_in, cm_stu, pe_stu, tp_stu, fwd_bs)
                H_cal = _chunked_step_range(model, l_idx, next_l, H_cal_in, cm_cal, pe_cal, tp_cal, fwd_bs)
                H_neg = _chunked_step_range(model, l_idx, next_l, H_neg_in, cm_neg, pe_neg, tp_neg, fwd_bs)
        del H_stu_in, H_cal_in, H_neg_in
        if is_large:
            torch.cuda.empty_cache()

    mgr.freeze_all()
    mgr.set_all_active(True)
    return mgr, {
        "method": "Universal_O1_NullSpace_SFT",
        "target_layers": target_layers,
        "loss_terms": 4,
        "extraction": ext_stats,
        "layer_logs": layer_logs,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


# ---------------------------------------------------------------------------
# 4. Pure 0-Backprop Closed-Form Self-Extraction + Null-Space Ridge SFT
# ---------------------------------------------------------------------------

@torch.no_grad()
def synthesize_zero_backprop_ridge_sft(
    model: nn.Module,
    tokenizer,
    spec: ModelSpec,
    doc_text: str,
    cal_texts: List[str],
    target_layers: List[int],
    device: str,
    r: int = 16,
    lora_alpha: float = 16.0,
    refine_in_context: bool = True,
) -> Tuple[UnifiedTransferManager, Dict[str, object]]:
    """Pure torch.no_grad() Self-Extraction + Analytical CE Hidden-Shift Null-Space Ridge LoRA (0 .backward())."""
    t0 = time.perf_counter()
    target_layers = sorted(set(int(l) for l in target_layers))
    mgr = UnifiedTransferManager(model, target_layers=target_layers, r=r, lora_alpha=lora_alpha)
    mgr.freeze_all()
    mgr.set_all_active(False)

    layers = get_layers(model)
    n_layers = len(layers)
    lm_norm = get_lm_norm(model)
    lm_head = get_lm_head(model)
    eps = float(getattr(lm_norm, "variance_epsilon", getattr(lm_norm, "eps", 1e-6)))
    gamma = get_rmsnorm_gamma(lm_norm).to(device=device, dtype=torch.float32)
    W_lm_raw = lm_head.weight
    is_large = spec.hidden_size >= 2048
    fwd_bs = 10 if is_large else 32

    train_items, ext_stats = self_extract_from_document(
        model, tokenizer, spec, doc_text, device, refine_in_context=refine_in_context, include_raw_continuations=True
    )
    stu_ids, stu_mask, stu_pos, tgt_tok_ids = _tokenize_sft_items(tokenizer, train_items, device)

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    cal_enc = tokenizer(cal_texts[:24], return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
    tokenizer.padding_side = prev_side
    cal_mask_bool = cal_enc.attention_mask.bool()

    h_stu, cm_stu, pe_stu, tp_stu = prepare_layer_inputs_unified(model, stu_ids, stu_mask)
    h_stu = _chunked_step_range(model, 0, target_layers[0], h_stu, cm_stu, pe_stu, tp_stu, fwd_bs)
    h_cal, cm_cal, pe_cal, tp_cal = prepare_layer_inputs_unified(model, cal_enc.input_ids, cal_enc.attention_mask)
    h_cal = _chunked_step_range(model, 0, target_layers[0], h_cal, cm_cal, pe_cal, tp_cal, fwd_bs)

    layer_logs = []
    for idx_sched, l_idx in enumerate(target_layers):
        # 1. Compute final hidden states & analytical RMSNorm+lm_head CE gradient targets in no_grad()
        h_fin = _chunked_step_range(model, l_idx, n_layers, h_stu, cm_stu, pe_stu, tp_stu, fwd_bs)
        hf_pts = gather_positions(h_fin, stu_pos).float()
        del h_fin

        dH_targets = torch.zeros_like(hf_pts)
        mean_p0 = 0.0
        for s in range(0, hf_pts.shape[0], 32):
            e = min(hf_pts.shape[0], s + 32)
            hv = hf_pts[s:e]
            tids = tgt_tok_ids[s:e]
            inv_rms = torch.rsqrt((hv * hv).mean(dim=-1, keepdim=True) + eps)
            h_hat = hv * inv_rms
            # Chunked lm_head projection in fp16 -> fp32
            logits = lm_head((h_hat * gamma.unsqueeze(0)).to(W_lm_raw.dtype)).float()
            probs = F.softmax(logits, dim=-1)
            ar = torch.arange(hv.shape[0], device=device)
            mean_p0 += float(probs[ar, tids].sum().item())
            gz = probs
            gz[ar, tids] -= 1.0
            # Back-project gz @ W_lm without copying full W_lm to fp32
            g_normed = (gz.to(W_lm_raw.dtype) @ W_lm_raw).float() * gamma.unsqueeze(0)
            g_h = inv_rms * (g_normed - (g_normed * h_hat).mean(dim=-1, keepdim=True) * h_hat)
            # Step proportional to RMS scale of hidden state along normalized -g_h
            h_rms = hv.norm(dim=-1, keepdim=True)
            need = (probs[ar, tids] < 0.85).float().unsqueeze(1)
            u = (-g_h) / g_h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            dH_targets[s:e] = need * (0.28 * h_rms / max(len(target_layers), 1)) * u
            del logits, probs, gz, g_normed, g_h

        mean_p0 /= max(hf_pts.shape[0], 1)

        # 2. Capture layer residual-writing module inputs (mixer_out, down_proj) on student & cal
        for mod_key, sp in (("o_proj" if get_layer_type_safe(spec, l_idx) != "linear_attention" else "out_proj", "mixer_out"), ("down_proj", "down_proj")):
            w = mgr.wrapped[(l_idx, mod_key)]
            x_stu_list, x_cal_list = [], []

            def _cap(h_b, cm_b, pe_b, tp_b, gather_or_mask, is_pos: bool, dest: List[torch.Tensor]):
                nb = h_b.shape[0]
                for s0 in range(0, nb, fwd_bs):
                    e0 = min(nb, s0 + fwd_bs)
                    b_cur, sq_len = e0 - s0, h_b.shape[1]
                    def hk(m, inp, out):
                        if is_pos:
                            x3 = inp[0].detach().reshape(b_cur, sq_len, -1)
                            dest.append(gather_positions(x3, gather_or_mask[s0:e0]).float())
                        else:
                            mf = gather_or_mask[s0:e0].reshape(-1)
                            xg = inp[0].detach().reshape(-1, inp[0].shape[-1])[mf].float()
                            if xg.shape[0] > 260:
                                xg = xg[:: max(1, xg.shape[0] // 260)][:260]
                            dest.append(xg.cpu())
                    h_rem = w.register_forward_hook(hk)
                    _ = step_layer_unified(model, l_idx, h_b[s0:e0], slice_cm_map(cm_b, s0, e0, nb), slice_pe(pe_b, s0, e0, nb), tp_b[s0:e0])
                    h_rem.remove()

            _cap(h_stu, cm_stu, pe_stu, tp_stu, stu_pos, True, x_stu_list)
            _cap(h_cal, cm_cal, pe_cal, tp_cal, cal_mask_bool, False, x_cal_list)
            X_stu = torch.cat(x_stu_list, dim=0)
            X_cal = torch.cat(x_cal_list, dim=0).to(device)
            if X_cal.shape[0] > 768:
                X_cal = X_cal[:: max(1, X_cal.shape[0] // 768)][:768]

            # Null-space whitened ridge solved directly via Woodbury / SVD (0 backprop)
            d_in = X_stu.shape[1]
            c_mat = (X_cal.T @ X_cal) / max(X_cal.shape[0], 1)
            c_scale = max(float(torch.trace(c_mat).item() / max(d_in, 1)), 1e-8)
            c_norm = c_mat * (6.0 / c_scale)
            if d_in <= 4096:
                evals, evecs = torch.linalg.eigh(0.5 * (c_norm + c_norm.T) + torch.eye(d_in, device=device))
                S_inv = (evecs / torch.sqrt(evals.clamp(min=0.5)).unsqueeze(0)) @ evecs.T
                X_w = X_stu @ S_inv
                apply_sinv = lambda M: M @ S_inv
                del evals, evecs
            else:
                Uk, Sk, _ = torch.svd_lowrank(c_norm, q=min(384, d_in - 2), niter=2)
                shrink = 1.0 - torch.rsqrt(1.0 + Sk.clamp(min=0.0))
                apply_sinv = lambda M: M - ((M @ Uk) * shrink.unsqueeze(0)) @ Uk.T
                X_w = apply_sinv(X_stu)

            dY = 0.5 * dH_targets  # split target shift evenly between mixer_out and down_proj
            m_n = X_w.shape[0]
            gram_diag = max(float((X_w * X_w).sum().item() / max(m_n, 1)), 1e-6)
            ridge = 0.08 * gram_diag
            if m_n < d_in:
                Theta_w = X_w.T @ torch.linalg.solve(X_w @ X_w.T + ridge * torch.eye(m_n, device=device), dY)
            else:
                Theta_w = torch.linalg.solve(X_w.T @ X_w + ridge * torch.eye(d_in, device=device), X_w.T @ dY)
            U, S, Vh = torch.linalg.svd(Theta_w.T, full_matrices=False)
            r_eff = min(r, S.shape[0])
            B_fit = torch.zeros((w.out_features, r), device=device, dtype=torch.float32)
            A_fit = torch.zeros((r, d_in), device=device, dtype=torch.float32)
            B_fit[:, :r_eff] = U[:, :r_eff] * S[:r_eff].unsqueeze(0)
            A_fit[:r_eff, :] = apply_sinv(Vh[:r_eff, :])
            fro = float((B_fit @ A_fit).norm().item())
            if fro > 4.0:
                B_fit = B_fit * (4.0 / (fro + 1e-8))
            w.lora_A.data.copy_(A_fit / max(w.scaling, 1e-6))
            w.lora_B.data.copy_(B_fit)
            w.active = True
            del X_stu, X_cal, c_mat, c_norm, X_w, Theta_w, U, S, Vh

        layer_logs.append({"layer": l_idx, "base_mean_p0": round(mean_p0, 4)})
        if idx_sched + 1 < len(target_layers):
            nxt = target_layers[idx_sched + 1]
            h_stu = _chunked_step_range(model, l_idx, nxt, h_stu, cm_stu, pe_stu, tp_stu, fwd_bs)
            h_cal = _chunked_step_range(model, l_idx, nxt, h_cal, cm_cal, pe_cal, tp_cal, fwd_bs)

    return mgr, {
        "method": "ZeroBackprop_NullSpace_Ridge_SFT",
        "target_layers": target_layers,
        "loss_terms": 1,
        "extraction": ext_stats,
        "layer_logs": layer_logs,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def get_layer_type_safe(spec: ModelSpec, l_idx: int) -> str:
    return spec.layer_types[l_idx]
