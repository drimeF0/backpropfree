import sys
sys.path.insert(0, "/content/smollm_nobackprop_lab")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import generate_fact_batch, load_smoltalk_splits, match_target
from eval_harness import format_qa_prompts, batched_greedy_generate, disable_adapters_ctx
from guards import LoRAManager, cleanup_cuda
from iter_20_mechA4_per_token_swiglu import synthesize_per_token_swiglu, TARGET_MODULES, TARGET_LAYERS

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
cal_texts, kl_texts = load_smoltalk_splits(tok, n_cal=128, n_kl_eval=64)
batch42 = generate_fact_batch(42, tokenizer=tok, target_doc_tokens=2048)

cleanup_cuda()
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
for p in model.parameters():
    p.requires_grad_(False)
lora_mgr = LoRAManager(model, target_modules=TARGET_MODULES)
lora_mgr.install()

with torch.no_grad():
    diag = synthesize_per_token_swiglu(model, tok, lora_mgr, batch42, cal_texts, device)

# 1. Inspect verbatim & paraphrase predictions + failures
v_prompts = format_qa_prompts(tok, [f.verbatim_q for f in batch42.facts])
p_prompts = format_qa_prompts(tok, [f.paraphrase_q for f in batch42.facts] + [f.paraphrase_q_alt for f in batch42.facts])
v_preds = batched_greedy_generate(model, tok, v_prompts, device, max_new_tokens=32, batch_size=32)
p_preds = batched_greedy_generate(model, tok, p_prompts, device, max_new_tokens=32, batch_size=32)

print("=== VERBATIM MISSES ===")
v_miss = 0
for f, pred in zip(batch42.facts, v_preds):
    if not match_target(pred, f.target):
        v_miss += 1
        if v_miss <= 10:
            print(f"  fact {f.idx} ({f.relation}) exp={f.target!r} | pred={pred!r}")
print(f"Total verbatim miss: {v_miss}/25")

print("=== PARAPHRASE MISSES ===")
p_miss = 0
all_p_facts = batch42.facts + batch42.facts
all_p_qs = [f.paraphrase_q for f in batch42.facts] + [f.paraphrase_q_alt for f in batch42.facts]
for idx, (f, q, pred) in enumerate(zip(all_p_facts, all_p_qs, p_preds)):
    if not match_target(pred, f.target):
        p_miss += 1
        if p_miss <= 12:
            print(f"  p#{idx} fact {f.idx} ({f.relation}) q={q!r}\n    exp={f.target!r} | pred={pred!r}")
print(f"Total paraphrase miss: {p_miss}/50")

# 2. Inspect real_anchors + synthetic locality misses vs base
loc_prompts = format_qa_prompts(tok, [l.question for l in batch42.locality])
loc_edit = batched_greedy_generate(model, tok, loc_prompts, device, max_new_tokens=32, batch_size=32)
with disable_adapters_ctx(model, lora_mgr):
    loc_base = batched_greedy_generate(model, tok, loc_prompts, device, max_new_tokens=32, batch_size=32)

print("=== REAL ANCHORS ===")
for l, b, e in zip(batch42.locality, loc_base, loc_edit):
    if l.is_real_world:
        print(f"  {l.entity} ({l.expected_answer}): base={b!r} | edit={e!r}")

print("=== SYNTH LOCALITY MISSES (first 10) ===")
sm = 0
for l, b, e in zip(batch42.locality, loc_base, loc_edit):
    if (not l.is_real_world) and (b.strip() != e.strip()):
        sm += 1
        if sm <= 10:
            print(f"  {l.entity} ({l.relation}): q={l.question!r}\n    base={b!r}\n    edit={e!r}")

# 3. Hook down_proj LoRA norms on left-padded last tokens (actual pos0)!
delta_norms = {l: {"verb": [], "para": [], "real": [], "loc": [], "cal": []} for l in TARGET_LAYERS}
mode = ["none"]
def make_hook(l_idx):
    mod = model.model.layers[l_idx].mlp.down_proj
    def hook(m, inp, out):
        if m.active and m.lora_A is not None and len(inp) > 0:
            x = inp[0][:, -1, :]
            d = F.linear(F.linear(x, m.lora_A), m.lora_B) * m.alpha
            delta_norms[l_idx][mode[0]].append(d.float().norm(dim=-1).mean().item())
    return mod.register_forward_hook(hook)

real_prompts = [p for l, p in zip(batch42.locality, loc_prompts) if l.is_real_world]
synth_loc_prompts = [p for l, p in zip(batch42.locality, loc_prompts) if not l.is_real_world]

hooks = [make_hook(l) for l in TARGET_LAYERS]
tok.padding_side = "left"
for name, qs in [("verb", v_prompts), ("para", p_prompts), ("real", real_prompts), ("loc", synth_loc_prompts), ("cal", cal_texts[:24])]:
    mode[0] = name
    enc = tok(qs, return_tensors="pt", padding=True, truncation=True, max_length=72).to(device)
    with torch.no_grad():
        _ = model(**enc, use_cache=False)
for h in hooks:
    h.remove()

print("=== LAST-TOKEN ||down_proj LoRA delta|| BY SPLIT ===")
for l in TARGET_LAYERS:
    d = {k: round(sum(v)/max(len(v), 1), 3) for k, v in delta_norms[l].items()}
    print(f"  L{l}: {d}")
