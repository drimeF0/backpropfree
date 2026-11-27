"""Complex Fact & Multi-Hop Biographical Reasoning Experiment (Backprop-Free).

Tests whether knowledge injected without full-model backpropagation (comparing both Pure 0-Backprop
Voronoi SwiGLU LoRA and Strictly 1-Layer Detached Null-Preconditioned ResSkip) can participate in
unseen multi-hop transitive, temporal arithmetic, and constraint-satisfaction reasoning.

Subject: Entirely fictional biography of Dr. Seren Valenwood (verified ~0% base model knowledge).
Training probes contain ONLY atomic factual statements derived from the biography (zero multi-hop
chains, zero computed dates 1996/2021/27/1, zero safety decisions).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import load_smoltalk_splits, match_target
from eval_harness import batched_greedy_generate, compute_heldout_kl, format_qa_prompts, get_general_qa_set
from guards import LoRAManager, NoBackpropGuard, cleanup_cuda, measure_static_vram_mb
from iter_1_experiment import gather_positions, tokenize_with_target_positions
from iter_12_experiment import PrecondBlockManager
from nullspace_cache import compute_or_load_c0
from verify_stage2_and_seeds import TARGET_MODULES as RESSKIP_MODULES, run_shared4_l26_31

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
LAB_DIR = Path("/content/smollm_nobackprop_lab")

BIOGRAPHY_TEXT = (
    "Dr. Seren Valenwood was born in 1984 in the coastal town of Oakhaven. "
    "At age 12, Seren Valenwood relocated to Vespera to study cryo-botany under mentor Prof. Mara Lindholm. "
    "In 2011, Dr. Seren Valenwood invented the Aether-7 thermal drill, which later earned the Solaris Medal. "
    "In 2012, Seren Valenwood joined Helios Polar Institute as Chief Engineer and served in that post for 9 years. "
    "Her younger sister, Talia Valenwood, is a concert cellist living in Nordhavn. "
    "Dr. Seren Valenwood has a severe medical allergy to cobalt and speaks fluent Estonian."
)

# Atomic self-supervised factual pairs derived ONLY from BIOGRAPHY_TEXT
# Uses canonical Stage-1-compatible phrasing plus reverse entity/attribute lookups (ZERO test reasoning!)
ATOMIC_FACT_CLAUSES = [
    "Bio record: The birth town of Dr. Seren Valenwood is Oakhaven.",
    "Bio record: The birth year of Dr. Seren Valenwood is 1984.",
    "Bio record: The city Dr. Seren Valenwood relocated to at age 12 is Vespera.",
    "Bio record: The age when Dr. Seren Valenwood moved to Vespera is 12.",
    "Bio record: The academic field studied by Dr. Seren Valenwood in Vespera is cryo-botany.",
    "Bio record: The mentor of Dr. Seren Valenwood in Vespera is Prof. Mara Lindholm.",
    "Bio record: The invention created by Dr. Seren Valenwood in 2011 is Aether-7.",
    "Bio record: The year Dr. Seren Valenwood invented Aether-7 is 2011.",
    "Bio record: The award won by Dr. Seren Valenwood for Aether-7 is Solaris Medal.",
    "Bio record: The year Dr. Seren Valenwood joined Helios Polar Institute as Chief Engineer is 2012.",
    "Bio record: The length of Dr. Seren Valenwood's service as Chief Engineer at Helios Polar Institute is 9 years.",
    "Bio record: The city where Dr. Seren Valenwood's concert cellist sister Talia lives is Nordhavn.",
    "Bio record: The medical allergy suffered by Dr. Seren Valenwood is cobalt.",
    "Bio record: The non-English language spoken fluently by Dr. Seren Valenwood is Estonian.",
]

BIO_SELF_SUPERVISED_PAIRS: List[Tuple[str, str]] = [
    ("Where was Dr. Seren Valenwood born?", "Oakhaven"),
    ("In what town was Seren Valenwood born?", "Oakhaven"),
    ("In what year was Dr. Seren Valenwood born?", "1984"),
    ("What is Seren Valenwood's birth year?", "1984"),
    ("To which city did Dr. Seren Valenwood move at age 12?", "Vespera"),
    ("At what age did Dr. Seren Valenwood relocate to Vespera?", "12"),
    ("What academic subject did Dr. Seren Valenwood specialize in?", "cryo-botany"),
    ("Who was Dr. Seren Valenwood's mentor in Vespera?", "Prof. Mara Lindholm"),
    ("What device did Dr. Seren Valenwood invent?", "Aether-7"),
    ("Who invented the Aether-7 thermal drill?", "Dr. Seren Valenwood"),
    ("In what year was the Aether-7 thermal drill invented?", "2011"),
    ("Which award did Dr. Seren Valenwood win for inventing Aether-7?", "Solaris Medal"),
    ("Who won the Solaris Medal for inventing Aether-7?", "Dr. Seren Valenwood"),
    ("In what year did Dr. Seren Valenwood join Helios Polar Institute?", "2012"),
    ("For how many years did Dr. Seren Valenwood serve as Chief Engineer at Helios Polar Institute?", "9 years"),
    ("What is the profession of Dr. Seren Valenwood's sister Talia?", "concert cellist"),
    ("In which city does Talia Valenwood live?", "Nordhavn"),
    ("What substance is Dr. Seren Valenwood severely allergic to?", "cobalt"),
    ("Besides English, what language does Dr. Seren Valenwood speak fluently?", "Estonian"),
]

# Unseen Evaluation Suite
DIRECT_RECALL_TESTS = [
    {"q": "Which coastal settlement is the birthplace of Dr. Seren Valenwood?", "patterns": [r"\boakhaven\b"]},
    {"q": "Who supervised Dr. Seren Valenwood's studies in Vespera?", "patterns": [r"mara\s+lindholm", r"lindholm"]},
    {"q": "What musical occupation does Talia Valenwood hold?", "patterns": [r"cellist", r"cello"]},
    {"q": "Which severe allergy affects Dr. Seren Valenwood?", "patterns": [r"\bcobalt\b"]},
    {"q": "Which Baltic language can Dr. Seren Valenwood speak fluently?", "patterns": [r"\bestonian\b"]},
    {"q": "Which medal was awarded for the invention of the Aether-7 drill?", "patterns": [r"solaris"]},
]

# Held-out Logical, Temporal Arithmetic, Transitive Multi-Hop, and Constraint Tests
LOGICAL_REASONING_TESTS = [
    {
        "name": "temporal_relocation_year",
        "type": "arithmetic_2hop",
        "sub_qs": ["In what year was Dr. Seren Valenwood born?", "At what age did Dr. Seren Valenwood relocate to Vespera?"],
        "q": "Recall Dr. Seren Valenwood's birth year and the age at which she moved to Vespera. Step by step, calculate the exact calendar year she moved to Vespera.",
        "patterns": [r"\b1996\b"],
    },
    {
        "name": "temporal_tenure_end",
        "type": "arithmetic_2hop",
        "sub_qs": ["In what year did Dr. Seren Valenwood join Helios Polar Institute?", "For how many years did Dr. Seren Valenwood serve as Chief Engineer at Helios Polar Institute?"],
        "q": "Recall when Dr. Seren Valenwood joined Helios Polar Institute and how long she served as Chief Engineer. Calculate the exact year her tenure there ended.",
        "patterns": [r"\b2021\b"],
    },
    {
        "name": "temporal_invention_age",
        "type": "arithmetic_2hop",
        "sub_qs": ["In what year was Dr. Seren Valenwood born?", "In what year was the Aether-7 thermal drill invented?"],
        "q": "Using Dr. Seren Valenwood's birth year and the year she invented Aether-7, compute how old she was when she invented Aether-7.",
        "patterns": [r"\b27\b"],
    },
    {
        "name": "temporal_gap_invention_to_helios",
        "type": "arithmetic_2hop",
        "sub_qs": ["In what year was the Aether-7 thermal drill invented?", "In what year did Dr. Seren Valenwood join Helios Polar Institute?"],
        "q": "How many years elapsed between Dr. Seren Valenwood inventing Aether-7 and joining Helios Polar Institute? Show the subtraction and answer.",
        "patterns": [r"\b1\s*year|\b=\s*1\b|answer:\s*1\b|elapsed:\s*1\b"],
    },
    {
        "name": "transitive_sister_city",
        "type": "transitive_2hop",
        "sub_qs": ["What is the profession of Dr. Seren Valenwood's sister Talia?", "In which city does Talia Valenwood live?"],
        "q": "To hear her concert cellist sister perform at home, which city should Dr. Seren Valenwood visit?",
        "patterns": [r"\bnordhavn\b"],
    },
    {
        "name": "transitive_medal_mentor",
        "type": "transitive_2hop",
        "sub_qs": ["Who won the Solaris Medal for inventing Aether-7?", "Who was Dr. Seren Valenwood's mentor in Vespera?"],
        "q": "First identify who won the Solaris Medal for Aether-7, then state who mentored that inventor in Vespera.",
        "patterns": [r"mara\s+lindholm", r"lindholm"],
    },
    {
        "name": "constraint_cobalt_safety",
        "type": "safety_deduction",
        "sub_qs": ["What substance is Dr. Seren Valenwood severely allergic to?"],
        "q": "A drilling crew offers Dr. Seren Valenwood a job cutting raw cobalt ore by hand. Should she accept it? Answer Yes or No and explain why based on her biography.",
        "require_all": [r"(?:^|\b)(?:no|should not|cannot|decline|avoid)\b", r"\bcobalt\b"],
    },
    {
        "name": "conjunctive_polar_grant_positive",
        "type": "conjunctive_logic",
        "sub_qs": [
            "What academic subject did Dr. Seren Valenwood specialize in?",
            "Besides English, what language does Dr. Seren Valenwood speak fluently?",
            "In what year did Dr. Seren Valenwood join Helios Polar Institute?",
        ],
        "q": "A polar expedition requires all three: (1) background in cryo-botany, (2) fluent Estonian, and (3) prior experience at Helios Polar Institute. Check each of Dr. Seren Valenwood's facts and state Yes or No whether she qualifies.",
        "patterns": [r"\byes\b"],
    },
    {
        "name": "conjunctive_japanese_disqualify",
        "type": "conjunctive_logic",
        "sub_qs": [
            "What academic subject did Dr. Seren Valenwood specialize in?",
            "Besides English, what language does Dr. Seren Valenwood speak fluently?",
        ],
        "q": "A Tokyo fellowship requires BOTH: (1) cryo-botany and (2) fluent Japanese. Based on Dr. Seren Valenwood's known spoken language, answer Yes or No whether she qualifies.",
        "require_all": [r"(?:^|\b)(?:no|does not qualify|cannot|not speak japanese)\b", r"japanese|estonian"],
    },
]


def check_patterns(pred: str, item_or_patterns) -> bool:
    txt = pred.lower()
    if isinstance(item_or_patterns, dict):
        if "require_all" in item_or_patterns:
            return all(re.search(p, txt) is not None for p in item_or_patterns["require_all"])
        pats = item_or_patterns["patterns"]
        return any(re.search(p, txt) is not None for p in pats)
    return any(re.search(p, txt) is not None for p in item_or_patterns)


def format_cot_prompts(tokenizer, questions: List[str]) -> List[str]:
    sys_msg = (
        "You are a precise logical reasoning assistant. First recall any relevant biographical facts concisely, "
        "then work out arithmetic or multi-hop deductions step by step, ending clearly with the final answer."
    )
    out = []
    for q in questions:
        out.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": q}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return out


@torch.no_grad()
def synthesize_bio_pure_0backprop(
    model,
    tokenizer,
    lora_mgr: LoRAManager,
    cal_texts: List[str],
    device: str,
) -> Dict[str, object]:
    """Synthesize pure 0-backprop Voronoi SwiGLU LoRA from atomic biography recall pairs."""
    from iter_1_experiment import gather_positions
    from iter_6_experiment import make_realistic_negative_entities
    from iter_26_mechA10_refined import solve_contrastive_neuron_proj_sep, voronoi_guarded_targets_v2
    from transformers.masking_utils import create_causal_mask

    sys_msg = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
    helper_sys = "You are a helpful assistant."
    pos_items = []
    for f_idx, (q, tgt) in enumerate(BIO_SELF_SUPERVISED_PAIRS):
        # Richer neutral & paraphrased templates (still strictly atomic, ZERO multi-hop logic!)
        variants = [
            (sys_msg, q),
            (sys_msg, f"From the biography of Dr. Seren Valenwood: {q}"),
            (sys_msg, f"Biographical fact check — {q}"),
            (sys_msg, f"State briefly: {q}"),
            (sys_msg, f"Recall the recorded biographical detail: {q}"),
            (helper_sys, f"Based on Dr. Seren Valenwood's life record, {q[0].lower() + q[1:]}"),
            (helper_sys, f"In the profile of Dr. Seren Valenwood, answer concisely: {q}"),
        ]
        for s_p, vq in variants:
            pos_items.append({
                "fact_idx": f_idx,
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "system", "content": s_p}, {"role": "user", "content": vq}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "target": tgt,
            })

    neg_ents = make_realistic_negative_entities(240, {"seren", "valenwood", "oakhaven", "vespera", "nordhavn", "talia"})
    neg_prompts = []
    cot_sys = (
        "You are a precise logical reasoning assistant. First recall any relevant biographical facts concisely, "
        "then work out arithmetic or multi-hop deductions step by step, ending clearly with the final answer."
    )
    for j, ne in enumerate(neg_ents):
        nq = [
            f"Where was Dr. {ne} born?",
            f"In what year was Dr. {ne} born?",
            f"Who invented the {ne} thermal drill?",
            f"What language does {ne} speak fluently?",
            f"What substance is {ne} allergic to?",
            f"State briefly: Who mentored Dr. {ne} in Vespera?",
            f"Recall the recorded biographical detail: In which city does {ne} live?",
            f"Step 1: Recall Dr. {ne}'s biographical profile and verify qualifications.",
        ][j % 8]
        s_p = cot_sys if j % 8 == 7 else (helper_sys if j % 3 == 0 else sys_msg)
        neg_prompts.append(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": s_p}, {"role": "user", "content": nq}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    prompts = [x["prompt"] for x in pos_items]
    targets = [x["target"] for x in pos_items]
    stu_ids, stu_mask, stu_pos = tokenize_with_target_positions(tokenizer, prompts, targets, device)

    target_tok_list, token_fact_ids, token_step_ids = [], [], []
    unique_pos0, unique_eos, unique_mid = [], [], []
    seen = set()
    for item in pos_items:
        f_idx = int(item["fact_idx"])
        t_ids = tokenizer.encode(item["target"], add_special_tokens=False)
        eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        full_t = t_ids + eos
        eos_step = len(full_t) - 1 if eos else -1
        for k_step, tok in enumerate(full_t):
            target_tok_list.append(tok)
            token_fact_ids.append(f_idx)
            token_step_ids.append(k_step)
            pair = (f_idx, k_step)
            if pair not in seen:
                seen.add(pair)
                if k_step == 0:
                    unique_pos0.append(pair)
                elif k_step == eos_step:
                    unique_eos.append(pair)
                else:
                    unique_mid.append(pair)

    target_tok_ids = torch.tensor(target_tok_list, dtype=torch.long, device=device)
    token_fact_ids_t = torch.tensor(token_fact_ids, dtype=torch.long, device=device)
    token_step_ids_t = torch.tensor(token_step_ids, dtype=torch.long, device=device)

    cand_layers = list(range(24, 32))
    layer_shards = {l: [] for l in cand_layers}
    for idx, p in enumerate(unique_pos0):
        layer_shards[[26, 27, 28, 29][idx % 4]].append(p)
    for idx, p in enumerate(unique_eos):
        layer_shards[[31, 30][idx % 2]].append(p)
    for p in unique_mid:
        for l in reversed(cand_layers):
            if len(layer_shards[l]) < 16:
                layer_shards[l].append(p)
                break

    active_layers = [l for l in cand_layers if layer_shards[l]]
    first_layer = active_layers[0]

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    neg_enc = tokenizer(neg_prompts, return_tensors="pt", padding=True).to(device)
    cal_enc = tokenizer(cal_texts[:96], return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
    tokenizer.padding_side = prev_side
    neg_pos = [[L - 1] for L in neg_enc.attention_mask.sum(dim=1).tolist()]

    def init_pref(ids, mask, bs=64):
        chunks = []
        emb = model.model.embed_tokens(ids)
        pids = torch.arange(emb.shape[1], device=device).unsqueeze(0)
        cm = create_causal_mask(config=model.config, inputs_embeds=emb, attention_mask=mask, past_key_values=None, position_ids=pids)
        pe = model.model.rotary_emb(emb, position_ids=pids)
        for s in range(0, ids.shape[0], bs):
            e = min(ids.shape[0], s + bs)
            h = emb[s:e]
            c = cm[s:e] if cm is not None and cm.shape[0] == ids.shape[0] else cm
            for l in range(first_layer):
                h = model.model.layers[l](h, attention_mask=c, position_embeddings=pe, use_cache=False)
            chunks.append(h.detach())
        return torch.cat(chunks, dim=0), cm, pe

    lora_mgr.clear_all()
    H_stu, cm_stu, pe_stu = init_pref(stu_ids, stu_mask)
    H_neg, cm_neg, pe_neg = init_pref(neg_enc.input_ids, neg_enc.attention_mask)
    H_cal, cm_cal, pe_cal = init_pref(cal_enc.input_ids, cal_enc.attention_mask)
    cal_mask_bool = cal_enc.attention_mask.bool()

    for l_idx in range(first_layer, 32):
        layer = model.model.layers[l_idx]
        shard_pairs = layer_shards[l_idx]
        if not shard_pairs:
            H_stu = layer(H_stu, attention_mask=cm_stu, position_embeddings=pe_stu, use_cache=False)
            H_neg = layer(H_neg, attention_mask=cm_neg, position_embeddings=pe_neg, use_cache=False)
            H_cal = layer(H_cal, attention_mask=cm_cal, position_embeddings=pe_cal, use_cache=False)
            continue

        def fwd_attn(h_in, cm, pe):
            ao, _ = layer.self_attn(hidden_states=layer.input_layernorm(h_in), attention_mask=cm, position_embeddings=pe, use_cache=False)
            hm = h_in + ao
            return hm, layer.post_attention_layernorm(hm)

        H_mid_stu, X_mlp_stu_all = fwd_attn(H_stu, cm_stu, pe_stu)
        H_mid_neg, X_mlp_neg_all = fwd_attn(H_neg, cm_neg, pe_neg)
        H_mid_cal, X_mlp_cal_all = fwd_attn(H_cal, cm_cal, pe_cal)

        X_mlp_stu = gather_positions(X_mlp_stu_all, stu_pos)
        X_mlp_neg = gather_positions(X_mlp_neg_all, neg_pos)
        X_mlp_cal = X_mlp_cal_all[cal_mask_bool]

        dH_goal, p_before, _ = voronoi_guarded_targets_v2(
            model, H_stu, cm_stu, pe_stu, l_idx, stu_pos, target_tok_ids, target_prob_goal=0.95, seq_chunk=64, compute_dH=True
        )
        r_neurons = len(shard_pairs)
        g_cal_base = F.linear(X_mlp_cal, layer.mlp.gate_proj.base.weight).float()
        dormancy = (F.silu(g_cal_base) ** 2).mean(dim=0) + 0.05 * g_cal_base.clamp(min=0.0).mean(dim=0)
        J_l = torch.topk(dormancy, k=r_neurons, largest=False).indices

        shard_mask = torch.zeros_like(token_fact_ids_t, dtype=torch.bool)
        cluster_ids_full = torch.full_like(token_fact_ids_t, -1)
        for c_i, (f_id, k_st) in enumerate(shard_pairs):
            pm = (token_fact_ids_t == f_id) & (token_step_ids_t == k_st)
            shard_mask |= pm
            cluster_ids_full[pm] = c_i

        other_mask = ~shard_mask
        X_shard = X_mlp_stu[shard_mask]
        X_other = X_mlp_stu[other_mask]
        c_ids = cluster_ids_full[shard_mask]

        W_g = layer.mlp.gate_proj.base.weight[J_l].float()
        W_u = layer.mlp.up_proj.base.weight[J_l].float()

        g_goal = torch.full((X_shard.shape[0], r_neurons), -4.5, device=device)
        g_goal[torch.arange(X_shard.shape[0], device=device), c_ids] = 4.6
        dG_pos = g_goal - X_shard.float() @ W_g.T
        dG_other = -4.5 - X_other.float() @ W_g.T
        dG_neg = -4.5 - X_mlp_neg.float() @ W_g.T
        g_cal_J = g_cal_base[:, J_l]
        dG_cal = torch.minimum(g_cal_J, torch.full_like(g_cal_J, -3.5)) - g_cal_J

        A_gate = solve_contrastive_neuron_proj_sep(
            X_shard, dG_pos, X_other, dG_other, X_mlp_neg, dG_neg, X_mlp_cal, dG_cal,
            w_pos=2.2, w_other=1.4, w_neg=1.6, w_cal=1.2, lambda_fro=0.012, max_row_norm=9.5,
        )
        B_sel = torch.zeros((layer.mlp.gate_proj.out_features, r_neurons), device=device, dtype=torch.float32)
        B_sel[J_l, torch.arange(r_neurons, device=device)] = 1.0
        lora_mgr.set_layer_adapter(l_idx, "gate_proj", A_gate, B_sel, alpha=1.0)

        u_goal = torch.zeros_like(g_goal)
        u_goal[torch.arange(X_shard.shape[0], device=device), c_ids] = 3.1
        dU_pos = u_goal - X_shard.float() @ W_u.T
        dU_other = -(X_other.float() @ W_u.T)
        dU_neg = -(X_mlp_neg.float() @ W_u.T)
        dU_cal = -(X_mlp_cal.float() @ W_u.T)
        A_up = solve_contrastive_neuron_proj_sep(
            X_shard, dU_pos, X_other, dU_other, X_mlp_neg, dU_neg, X_mlp_cal, dU_cal,
            w_pos=2.2, w_other=1.4, w_neg=1.6, w_cal=1.2, lambda_fro=0.012, max_row_norm=9.5,
        )
        lora_mgr.set_layer_adapter(l_idx, "up_proj", A_up, B_sel, alpha=1.0)

        Z_all = (F.silu(layer.mlp.gate_proj(X_mlp_stu)) * layer.mlp.up_proj(X_mlp_stu))[:, J_l].float()
        Z_shard, Z_other = Z_all[shard_mask], Z_all[other_mask]
        Z_neg = (F.silu(layer.mlp.gate_proj(X_mlp_neg)) * layer.mlp.up_proj(X_mlp_neg))[:, J_l].float()
        Z_cal = (F.silu(layer.mlp.gate_proj(X_mlp_cal)) * layer.mlp.up_proj(X_mlp_cal))[:, J_l].float()

        gram_r = (Z_shard.T @ Z_shard) + 1.5 * (Z_other.T @ Z_other) + 3.5 * (Z_neg.T @ Z_neg) + 3.0 * (Z_cal.T @ Z_cal) + 0.4 * torch.eye(r_neurons, device=device)
        B_down = torch.linalg.solve(gram_r, Z_shard.T @ dH_goal[shard_mask].float()).T
        A_down = B_sel.T.contiguous()

        lora_mgr.set_layer_adapter(l_idx, "down_proj", A_down, B_down, alpha=1.0)
        H_stu_1 = H_mid_stu + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(X_mlp_stu_all)) * layer.mlp.up_proj(X_mlp_stu_all))
        _, p_after_1, _ = voronoi_guarded_targets_v2(
            model, H_stu_1, cm_stu, pe_stu, l_idx + 1, stu_pos, target_tok_ids, seq_chunk=64, compute_dH=False
        )
        p0_s = p_before[shard_mask].clamp(1e-4, 1.0 - 1e-4)
        p1_s = p_after_1[shard_mask].clamp(1e-4, 1.0 - 1e-4)
        ell0 = torch.log(p0_s / (1.0 - p0_s))
        ell1 = torch.log(p1_s / (1.0 - p1_s))
        alpha_vec = torch.ones(r_neurons, device=device, dtype=torch.float32)
        for j, (_, k_st) in enumerate(shard_pairs):
            jm = c_ids == j
            m0, m1 = float(ell0[jm].mean().item()), float(ell1[jm].mean().item())
            dm = m1 - m0
            ell_goal = 4.2 if k_st == 0 else 3.2
            if dm > 0.25:
                alpha_vec[j] = max(0.85, min(2.2, (ell_goal - m0) / dm))
        B_down_cal = B_down * alpha_vec.unsqueeze(0)
        lora_mgr.set_layer_adapter(l_idx, "down_proj", A_down, B_down_cal, alpha=1.0)

        H_stu = H_mid_stu + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(X_mlp_stu_all)) * layer.mlp.up_proj(X_mlp_stu_all))
        H_neg = H_mid_neg + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(X_mlp_neg_all)) * layer.mlp.up_proj(X_mlp_neg_all))
        H_cal = H_mid_cal + layer.mlp.down_proj(F.silu(layer.mlp.gate_proj(X_mlp_cal_all)) * layer.mlp.up_proj(X_mlp_cal_all))

    _, _, final_diag = voronoi_guarded_targets_v2(
        model, H_stu, cm_stu, pe_stu, 32, stu_pos, target_tok_ids, seq_chunk=64, compute_dH=False
    )
    return {"total_pairs": len(seen), "final_train_diag": final_diag}


def run_bio_shared4_resskip(
    model,
    tokenizer,
    mgr: PrecondBlockManager,
    covs: Dict,
    cal_texts: List[str],
    device: str,
) -> Dict[str, object]:
    """Strictly single-layer detached Shared4Null ResSkip on atomic biography pairs ONLY (zero logic Qs)."""
    from iter_6_experiment import make_realistic_negative_entities

    sys_msg = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
    helper_sys = "You are a helpful assistant."
    pos_items = []
    # Atomic facts + reverse QA variations across neutral system prompts (NO multi-hop logic!)
    for q, tgt in BIO_SELF_SUPERVISED_PAIRS:
        variants = [
            (sys_msg, q),
            (sys_msg, f"From the biography of Dr. Seren Valenwood: {q}"),
            (sys_msg, f"Biographical fact check — {q}"),
            (helper_sys, f"Based on Dr. Seren Valenwood's life record, {q[0].lower() + q[1:]}"),
        ]
        for s_p, p_q in variants:
            pos_items.append({
                "prompt": tokenizer.apply_chat_template(
                    [{"role": "system", "content": s_p}, {"role": "user", "content": p_q}],
                    tokenize=False,
                    add_generation_prompt=True,
                ),
                "target": tgt,
            })

    neg_ents = make_realistic_negative_entities(48, {"seren", "valenwood", "oakhaven", "vespera", "nordhavn", "talia"})
    neg_items = []
    for j, ne in enumerate(neg_ents):
        nq = [
            f"Where was Dr. {ne} born?",
            f"In what year was Dr. {ne} born?",
            f"Who invented the {ne} thermal drill?",
            f"What substance is {ne} allergic to?",
        ][j % 4]
        neg_items.append({
            "prompt": tokenizer.apply_chat_template(
                [{"role": "system", "content": sys_msg}, {"role": "user", "content": nq}],
                tokenize=False,
                add_generation_prompt=True,
            )
        })

    return run_shared4_l26_31(
        model,
        tokenizer,
        mgr,
        covs,
        edit_texts=[],
        cal_texts=cal_texts,
        device=device,
        probes_override=(pos_items, neg_items),
        target_layers_override=[26, 27, 28],
        steps_override=(9, 9, 9),
    )


def evaluate_bio_and_logic(model, tokenizer, device, lora_mgr=None) -> Dict[str, object]:
    d_prompts = format_qa_prompts(tokenizer, [x["q"] for x in DIRECT_RECALL_TESTS])
    d_preds = batched_greedy_generate(model, tokenizer, d_prompts, device, batch_size=16, max_new_tokens=36)
    d_hits = []
    for item, pred in zip(DIRECT_RECALL_TESTS, d_preds):
        ok = check_patterns(pred, item)
        d_hits.append({"q": item["q"], "pred": pred, "pass": ok})

    l_prompts = format_cot_prompts(tokenizer, [x["q"] for x in LOGICAL_REASONING_TESTS])
    l_preds = batched_greedy_generate(model, tokenizer, l_prompts, device, batch_size=8, max_new_tokens=160)
    l_hits = []
    for item, pred in zip(LOGICAL_REASONING_TESTS, l_preds):
        ok = check_patterns(pred, item)
        l_hits.append({"name": item["name"], "type": item["type"], "q": item["q"], "pred": pred, "pass": ok})

    d_acc = sum(x["pass"] for x in d_hits) / len(d_hits)
    l_acc = sum(x["pass"] for x in l_hits) / len(l_hits)

    # Secondary Self-Recalled Weight-Only CoT: query ONLY model's own weights for atomic sub_qs,
    # then ask the model to deduce logically using solely its own recalled answers (zero oracle hints!).
    sr_hits = []
    for item in LOGICAL_REASONING_TESTS:
        sub_prompts = format_qa_prompts(tokenizer, item["sub_qs"])
        sub_ans = batched_greedy_generate(model, tokenizer, sub_prompts, device, batch_size=8, max_new_tokens=24)
        recalled_ctx = "\n".join(f"- {sq} -> {sa}" for sq, sa in zip(item["sub_qs"], sub_ans))
        cot_q = format_cot_prompts(tokenizer, [f"Facts recalled from memory:\n{recalled_ctx}\n\nQuestion: {item['q']}"])[0]
        ded_pred = batched_greedy_generate(model, tokenizer, [cot_q], device, batch_size=1, max_new_tokens=160)[0]
        ok = check_patterns(ded_pred, item)
        sr_hits.append({"name": item["name"], "recalled": recalled_ctx, "pred": ded_pred, "pass": ok})
    sr_acc = sum(x["pass"] for x in sr_hits) / len(sr_hits)

    return {
        "direct_recall_acc": round(d_acc, 4),
        "logical_reasoning_acc": round(l_acc, 4),
        "self_recalled_logic_acc": round(sr_acc, 4),
        "direct": d_hits,
        "logic": l_hits,
        "self_recalled_logic": sr_hits,
    }


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cal_texts, kl_texts = load_smoltalk_splits(tokenizer, n_cal=128, n_kl_eval=64)
    gqa = get_general_qa_set()
    gqa_prompts = format_qa_prompts(tokenizer, [x["q"] for x in gqa])

    results: Dict[str, object] = {"biography": BIOGRAPHY_TEXT, "models": {}}

    # 1. Base 0 Unedited + In-Context Upper Bound Baseline
    cleanup_cuda()
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    static_vram = measure_static_vram_mb(model)

    base_bio = evaluate_bio_and_logic(model, tokenizer, device)
    base_gqa_preds = batched_greedy_generate(model, tokenizer, gqa_prompts, device, batch_size=24, max_new_tokens=32)
    base_gqa_raw = sum(match_target(pr, x["a"]) for pr, x in zip(base_gqa_preds, gqa)) / len(gqa)

    # In-context biography ceiling check
    d_ctx_prompts = format_qa_prompts(tokenizer, [f"Biography:\n{BIOGRAPHY_TEXT}\n\nQuestion: {x['q']}" for x in DIRECT_RECALL_TESTS])
    l_ctx_prompts = format_cot_prompts(tokenizer, [f"Biography:\n{BIOGRAPHY_TEXT}\n\nQuestion: {x['q']}" for x in LOGICAL_REASONING_TESTS])
    ctx_d = [check_patterns(pr, it) for pr, it in zip(batched_greedy_generate(model, tokenizer, d_ctx_prompts, device, batch_size=8, max_new_tokens=36), DIRECT_RECALL_TESTS)]
    ctx_l_preds = batched_greedy_generate(model, tokenizer, l_ctx_prompts, device, batch_size=8, max_new_tokens=160)
    ctx_l = [{"name": it["name"], "pred": pr, "pass": check_patterns(pr, it)} for pr, it in zip(ctx_l_preds, LOGICAL_REASONING_TESTS)]

    results["models"]["Base_Unedited"] = {**base_bio, "general_qa_raw": round(base_gqa_raw, 4), "kl_div": 0.0}
    results["models"]["InContext_Oracle"] = {
        "direct_recall_acc": round(sum(ctx_d) / len(ctx_d), 4),
        "logical_reasoning_acc": round(sum(x["pass"] for x in ctx_l) / len(ctx_l), 4),
        "logic": ctx_l,
    }

    # 2. Pure 0-Backprop Voronoi SwiGLU LoRA
    lora_mgr = LoRAManager(model, target_modules=("gate_proj", "up_proj", "down_proj"))
    lora_mgr.install()
    with NoBackpropGuard(model, regime="no_grad", static_vram_mb=static_vram) as g_0bp:
        g_0bp.assert_backbone_no_grad()
        diag_0bp = synthesize_bio_pure_0backprop(model, tokenizer, lora_mgr, cal_texts, device)
    bio_0bp = evaluate_bio_and_logic(model, tokenizer, device, lora_mgr)
    kl_0bp = compute_heldout_kl(model, tokenizer, kl_texts, device, lora_manager=lora_mgr)
    gqa_0bp_raw = sum(match_target(pr, x["a"]) for pr, x in zip(batched_greedy_generate(model, tokenizer, gqa_prompts, device, batch_size=24, max_new_tokens=32), gqa)) / len(gqa)
    results["models"]["Pure_0Backprop_VoronoiSwiGLU"] = {
        **bio_0bp,
        "kl_div": round(kl_0bp, 6),
        "retention_acc": round(gqa_0bp_raw / max(base_gqa_raw, 1e-6), 4),
        "peak_vram_mb": round(g_0bp.metrics.peak_vram_mb, 2),
        "wall_time_sec": round(g_0bp.metrics.wall_time_sec, 3),
        "diag": diag_0bp,
    }
    del lora_mgr, model
    cleanup_cuda()

    # 3. Single-Layer Isolated Shared4Null ResSkip (L26..L31, O(1) depth VRAM)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    static_vram = measure_static_vram_mb(model)
    covs, _ = compute_or_load_c0(model, tokenizer, device, target_layers=range(16, 32), target_modules=("o_proj", "down_proj"), n_cal=160)
    mgr = PrecondBlockManager(model, target_layers=[26, 27, 28], target_modules=RESSKIP_MODULES, r=16)
    with NoBackpropGuard(model, regime="single_layer_isolated", static_vram_mb=static_vram) as g_1bp:
        g_1bp.assert_backbone_no_grad()
        diag_1bp = run_bio_shared4_resskip(model, tokenizer, mgr, covs, cal_texts, device)
    bio_1bp = evaluate_bio_and_logic(model, tokenizer, device, mgr)
    kl_1bp = compute_heldout_kl(model, tokenizer, kl_texts, device, lora_manager=mgr)
    gqa_1bp_raw = sum(match_target(pr, x["a"]) for pr, x in zip(batched_greedy_generate(model, tokenizer, gqa_prompts, device, batch_size=24, max_new_tokens=32), gqa)) / len(gqa)
    results["models"]["SingleLayer_Shared4Null_ResSkip"] = {
        **bio_1bp,
        "kl_div": round(kl_1bp, 6),
        "retention_acc": round(gqa_1bp_raw / max(base_gqa_raw, 1e-6), 4),
        "peak_vram_mb": round(g_1bp.metrics.peak_vram_mb, 2),
        "wall_time_sec": round(g_1bp.metrics.wall_time_sec, 3),
        "diag": diag_1bp,
    }

    (LAB_DIR / "biography_logic_results.json").write_text(json.dumps(results, indent=2))
    print("\n=== COMPLEX BIOGRAPHY & MULTI-HOP REASONING SUMMARY ===")
    for mname, mres in results["models"].items():
        print(f"  {mname}: direct={mres['direct_recall_acc']}, 1turn_logic={mres['logical_reasoning_acc']}, "
              f"self_recalled_logic={mres.get('self_recalled_logic_acc', '-')}, kl={mres.get('kl_div', '-')}, "
              f"ret={mres.get('retention_acc', '-')}, time={mres.get('wall_time_sec', '-')}")
    print("\n--- Logic Transcripts (1-Turn Direct CoT & Self-Recalled CoT) ---")
    for i in range(len(LOGICAL_REASONING_TESTS)):
        b = results["models"]["Base_Unedited"]["logic"][i]
        p0 = results["models"]["Pure_0Backprop_VoronoiSwiGLU"]["logic"][i]
        s1 = results["models"]["SingleLayer_Shared4Null_ResSkip"]["logic"][i]
        sr0 = results["models"]["Pure_0Backprop_VoronoiSwiGLU"]["self_recalled_logic"][i]
        sr1 = results["models"]["SingleLayer_Shared4Null_ResSkip"]["self_recalled_logic"][i]
        print(f"\n[{b['name']}] Q: {b['q']}")
        print(f"  Base 0            ({'PASS' if b['pass'] else 'FAIL'}): {b['pred']!r}")
        print(f"  0-BP SwiGLU 1turn ({'PASS' if p0['pass'] else 'FAIL'}): {p0['pred']!r}")
        print(f"  1L ResSkip  1turn ({'PASS' if s1['pass'] else 'FAIL'}): {s1['pred']!r}")
        print(f"  0-BP SwiGLU Self  ({'PASS' if sr0['pass'] else 'FAIL'}): {sr0['pred']!r}")
        print(f"  1L ResSkip  Self  ({'PASS' if sr1['pass'] else 'FAIL'}): {sr1['pred']!r}")


if __name__ == "__main__":
    main()
