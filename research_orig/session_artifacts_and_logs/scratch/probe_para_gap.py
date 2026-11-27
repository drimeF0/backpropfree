import sys
sys.path.insert(0, "/content/smollm_nobackprop_lab")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import generate_fact_batch, load_smoltalk_splits, match_target
from eval_harness import format_qa_prompts, batched_greedy_generate
from guards import LoRAManager, cleanup_cuda
from iter_21_mechA5_clean_swiglu import build_matched_probes_mechA5
from iter_22_mechA6_matched_balanced import synthesize_balanced_matched_swiglu, TARGET_MODULES

MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda"
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
cal_texts, _ = load_smoltalk_splits(tok, n_cal=128, n_kl_eval=32)
batch42 = generate_fact_batch(42, tokenizer=tok, target_doc_tokens=2048)

cleanup_cuda()
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
for p in model.parameters():
    p.requires_grad_(False)

# 1. Measure BASE model linear entity separability on X_mlp_in & X_o_in from matched train -> unseen paraphrase!
pos_items, _, _ = build_matched_probes_mechA5(batch42.stage1_edit_texts, tok)
tok.padding_side = "left"
train_enc = tok([x["prompt"] for x in pos_items], return_tensors="pt", padding=True).to(device)
v_prompts = format_qa_prompts(tok, [f.verbatim_q for f in batch42.facts])
p_prompts = format_qa_prompts(tok, [f.paraphrase_q for f in batch42.facts] + [f.paraphrase_q_alt for f in batch42.facts])
v_enc = tok(v_prompts, return_tensors="pt", padding=True).to(device)
p_enc = tok(p_prompts, return_tensors="pt", padding=True).to(device)

def get_last_mlp_in_by_layer(enc, bs=128):
    buf = {l: [] for l in range(22, 32)}
    hooks = []
    for l in range(22, 32):
        def mk(li):
            def hk(m, inp, out):
                buf[li].append(inp[0][:, -1, :].detach().float().cpu())
            return hk
        hooks.append(model.model.layers[l].mlp.register_forward_hook(mk(l)))
    with torch.no_grad():
        for s in range(0, enc.input_ids.shape[0], bs):
            e = min(enc.input_ids.shape[0], s + bs)
            model(input_ids=enc.input_ids[s:e], attention_mask=enc.attention_mask[s:e], use_cache=False)
    for h in hooks:
        h.remove()
    return {l: torch.cat(buf[l], dim=0) for l in range(22, 32)}

X_tr = get_last_mlp_in_by_layer(train_enc)
X_v = get_last_mlp_in_by_layer(v_enc)
X_p = get_last_mlp_in_by_layer(p_enc)

train_fids = torch.tensor([x["fact_idx"] for x in pos_items])
p_fids = torch.tensor([f.idx for f in batch42.facts] + [f.idx for f in batch42.facts])

print("=== BASE X_mlp_in PROTOTYPE NEAREST-NEIGHBOR & CONTRASTIVE PROJECTION ACCURACY ===")
for l in [24, 26, 27, 28, 29, 30, 31]:
    Xt = F.normalize(X_tr[l], dim=-1)
    Xv = F.normalize(X_v[l], dim=-1)
    Xp = F.normalize(X_p[l], dim=-1)
    # 25 centroids (centered by global train mean!)
    mu = X_tr[l].mean(dim=0, keepdim=True)
    C = torch.stack([F.normalize((X_tr[l][train_fids == i] - mu).mean(dim=0), dim=-1) for i in range(25)])  # [25, d]
    Xv_c = F.normalize(X_v[l] - mu, dim=-1)
    Xp_c = F.normalize(X_p[l] - mu, dim=-1)
    v_nn = float(((Xv_c @ C.T).argmax(dim=-1) == torch.arange(25)).float().mean().item())
    p_nn = float(((Xp_c @ C.T).argmax(dim=-1) == p_fids).float().mean().item())
    # Pure linear ridge on centered train features -> 25 one-hot targets
    Xc_tr = X_tr[l] - mu
    Y_tr = F.one_hot(train_fids, 25).float()
    W = torch.linalg.solve(Xc_tr.T @ Xc_tr + 1.0 * torch.eye(Xc_tr.shape[1]), Xc_tr.T @ Y_tr)
    v_lin = float((((X_v[l] - mu) @ W).argmax(dim=-1) == torch.arange(25)).float().mean().item())
    p_lin = float((((X_p[l] - mu) @ W).argmax(dim=-1) == p_fids).float().mean().item())
    print(f"  L{l}: centered_cos_NN (verb={v_nn:.2f}, para={p_nn:.2f}) | linear_ridge_acc (verb={v_lin:.2f}, para={p_lin:.2f})")

# 2. Now run Iter 22 and inspect pos_0 first-token prob & Z[:, J_l] on para!
lora_mgr = LoRAManager(model, target_modules=TARGET_MODULES)
lora_mgr.install()
with torch.no_grad():
    diag = synthesize_balanced_matched_swiglu(model, tok, lora_mgr, batch42, cal_texts, device)

tok.padding_side = "left"
# Check first-token top-1 accuracy on verbatim & paraphrase directly!
for split_name, enc_s, f_list in [("VERB_POS0", v_enc, batch42.facts), ("PARA_POS0", p_enc, batch42.facts + batch42.facts)]:
    with torch.no_grad():
        logits = model(input_ids=enc_s.input_ids, attention_mask=enc_s.attention_mask, use_cache=False).logits[:, -1, :]
        top1 = logits.argmax(dim=-1).tolist()
        t0_expected = [tok.encode(f.target, add_special_tokens=False)[0] for f in f_list]
        acc0 = sum(a == b for a, b in zip(top1, t0_expected)) / len(t0_expected)
        probs = F.softmax(logits.float(), dim=-1)
        p0_mean = sum(probs[i, t0_expected[i]].item() for i in range(len(t0_expected))) / len(t0_expected)
        print(f"Iter22 {split_name}: first_token_top1={acc0:.3f}, mean_p={p0_mean:.3f}")
