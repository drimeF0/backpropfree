# Lab Notebook: Backprop-Free & $O(1)$-Depth LoRA Injection on 8-bit `ornith-ai/Ornith-1.0-9B`

*(Статус: завершены Phase 0/1 Anatomy и полностью завершена Iteration 0 Baselines; подготовлены и готовы к оценке Mech B, Mech A12 и Shared4Null; в момент коммита запущен `iter_16_mechB_bounded_c2l.py`.)*

## Phase 0 & Hardware Notes
- **Model**: `ornith-ai/Ornith-1.0-9B` (`Qwen3_5ForConditionalGeneration`), 32 layers (`full_attention` at `[3, 7, 11, 15, 19, 23, 27, 31]`, `linear_attention` / `GatedDeltaNet` at the other 24 layers), `hidden_size=4096`, `intermediate_size=12288`, `vocab_size=248320`, `num_attention_heads=16`, `num_key_value_heads=4`, `head_dim=256`, `tie_word_embeddings=false`, `eos_token="<|im_end|>"` (`id=248046`).
- **Quantization**: `bitsandbytes` `LLM.int8()` (`load_in_8bit=True`, `llm_int8_skip_modules=["lm_head"]`, `dtype=torch.float16`), `model.model.visual` dropped after load $\rightarrow$ static VRAM = **`10485.51 MB`** (down from `10932.63 MB`) on 15,360 MiB Tesla T4 (~4.87 GB VRAM headroom; ~12 GiB host RAM, 0 swap).
- **Kernel constraint on T4 (SM 7.5)**: `flash-linear-attention` / `fla-core` Triton kernels hang during JIT on T4; `transformers` native PyTorch fallbacks for `GatedDeltaNet` (`chunk_gated_delta_rule`) and `causal_conv1d_fn` work accurately and reliably.
- **Architectural invariants discovered & implemented**:
  - `Qwen3_5RMSNorm` is zero-centered: effective gamma is `1.0 + norm.weight.float()`.
  - `Linear8bitLt` stores row scales in `SCB` (`weight.SCB` before first forward or `state.SCB` after): row-wise dequantization (`model_loader.dequantize_rows`) verified against live forward (`max_diff=0.01456, rel_err=0.007638`).
  - `step_layer` matches full `Qwen3_5TextModel` identically (`max diff: 0.0`).
  - `lm_head.weight` (`[248320, 4096]`, 2.03 GB in fp16) must stay in fp16 with chunked projection (`tok_chunk <= 96`) to avoid 4.07 GB fp32 OOM.
  - `down_proj` (`d=12288`) hits 2.26 GB cuSOLVER OOM under dense `eigh(12288)` alongside the 9B model $\rightarrow$ solved via `svd_lowrank(q=512)` in `nullspace_cache.py` and `iter_12_shared4null.py` (`eigh(4096)` works in `0.61s` at `10880.1 MB`).
  - `chat_template.jinja` defaults to `<think>\n`; pass `enable_thinking=False` for factual QA and probes.

## Phase 1: Model Anatomy (`dissect_ornith.py` -> `anatomy_report.json`)
- **Unembedding Voronoi cone**: `<|im_end|>` (`id=248046`), `eos_self_dot = 0.6983`, `num_tokens_dot_gt_eos_self = 0` (nearest neighbor `"</"` at `0.235`).
- **Weights Summary**:
  - `mixer_out_proj`: `mean_eff_rank = 2007.01` (`min=1552.07`, `max=2330.99`), `mean_r99 = 2987.3`
  - `gate_proj`: `mean_eff_rank = 2976.29` (`min=2748.52`, `max=3138.08`), `mean_r99 = 3816.3`
  - `up_proj`: `mean_eff_rank = 3210.6` (`min=2803.9`, `max=3407.09`), `mean_r99 = 3832.3`
  - `down_proj`: `mean_eff_rank = 3157.17` (`min=2870.62`, `max=3382.27`), `mean_r99 = 3800.7`
- **Signal Localization** (`ctx` vs `noctx`):
  - `L23 (full)`: `res_l2_norm=75.2068`, `res_l2_diff=65.5371 (rel=0.8738)`, `ctx_target_prob=0.0014`, `rank=1030.2`
  - `L26 (line)`: `res_l2_norm=124.7164`, `res_l2_diff=92.5371 (rel=0.7431)`, `ctx_target_prob=0.0005`, `rank=848.0`
  - `L27 (full)`: `res_l2_norm=124.8216`, `res_l2_diff=116.5835 (rel=0.9293)`, `mix=59.6838`, `mlp=39.1961`, `ctx_target_prob=0.115`, `rank=2.0` (**sharp critical resolution onset**)
  - `L28 (line)`: `res_l2_norm=152.6929`, `res_l2_diff=134.3212 (rel=0.876)`, `ctx_target_prob=0.2092`, `rank=1.8`
  - `L29 (line)`: `res_l2_norm=161.4336`, `res_l2_diff=150.0493 (rel=0.9254)`, `ctx_target_prob=0.2298`, `rank=2.0`
  - `L30 (line)`: `res_l2_norm=188.6222`, `res_l2_diff=175.6906 (rel=0.9273)`, `mix=50.74`, `mlp=63.4852`, `ctx_target_prob=0.3764`, `rank=1.5`
  - `L31 (full)`: `res_l2_norm=247.342`, `res_l2_diff=283.8119 (rel=1.1662)`, `mix=52.8537`, `mlp=182.4308`, `ctx_target_prob=0.5585`, `rank=1.5`
- **Null-Space Capacity** (`smol-smoltalk` streamed covariances):
  - `mixer_out_proj`: `3262..3975 / 4096` directions `< 1e-3 * λ_max` (`bottom_half_energy_frac = 0.007061..0.013258` in `L27..L31`)
  - `mlp_in`: `3031..3946 / 4096` directions `< 1e-3 * λ_max`
  - `down_proj`: `8943/12288` (`L23`), `10596/12288` (`L27`), `11062/12288` (`L30`), `11918/12288` (`L31`, `bottom_half_energy_frac = 0.002929`).

## Iteration 0: Complete Baselines (`Seed = 42`, `iter_0_baselines.py`)

| Method | Verbatim | Paraphrase | Locality | Bleed | Retention | KL Div | Peak VRAM (MB) | Act Overhead (MB) | Wall Time | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `Baseline_0_Unedited` | `0.00` | `0.00` | `0.6333` | `0.00` | `1.00` | `0.00` | `10485.51` | `0.0` | `0.0s` | `general_qa_raw_acc=1.0` |
| `Baseline_C_InContext` | `1.00` | `1.00` | `0.1667` | `0.0667` | `1.00` | `0.00` | `10485.51` | `0.0` | `0.0s` | Context confuses neighboring locality probes |
| `Baseline_A_Backprop_5step_seq512` | `0.04` | `0.04` | `0.1667` | `0.1333` | `0.9833` | `0.190887` | `13728.17` | `3242.66` | `374.31s` | OOM uncheckpointed @ `bs=1`; ran w/ `grad_ckpt=True` |
| `Baseline_A_Backprop_5step_seq2048` | `—` | `—` | `—` | `—` | `—` | `—` | `14597.44` (OOM) | `4111.93` (OOM) | `—` | `oom_on_15gb_t4` even WITH `grad_ckpt=True, bs=1` |
| `Baseline_B_Backprop_100step` | `0.84` | `0.84` | `0.1333` | `0.40` | `0.9667` | `1.06553` | `13719.98` | `3234.47` | `1374.98s` | OOM @ `bs=25` & `bs=5`; ran `bs=2, seq=64` |

### Key Conclusions from Iteration 0 on 8-bit `Ornith-1.0-9B`
1. Full backprop at 5 steps (`Baseline_A_512`) fails almost completely (`0.04` recall), requires gradient checkpointing even at `bs=1, seq=512`, takes `374.3s`, and already inflicts `KL=0.1909`.
2. Full backprop at long context (`Baseline_A_2048`) **cannot execute at all on a 15 GiB T4 GPU**, crashing with OOM (`14,597.44 MB` / `+4,111.93 MB` overhead) even with `gradient_checkpointing=True` and `bs=1`.
3. Heavy 100-step full backprop (`Baseline_B_100step`) reaches `0.84` paraphrase accuracy after **`22.9 minutes`** (`1374.98s`), but destroys locality (`0.6333 -> 0.1333`, **`40%` entity bleed**) and warps the distribution (`KL = 1.06553`, >21x the `0.05` gate).

## Iteration 1: Seed 42 Mechanism Runs (`iter_16`, `iter_28`, `iter_29`, `iter_12`, `iter_30`)

All runs use the 8-bit `llm_int8_skip_modules=["lm_head"]` snapshot, `NoBackpropGuard`, and `rank <= 16`.
Locality values for `Iter 16/28/12` were measured before the harness batch-size fix described below; `Iter 29/30` use the corrected matched-batch comparison.

| Run | Method | Verbatim | Paraphrase | Locality | Bleed | Retention | KL Div | Peak VRAM | Act Overhead | Wall Time | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 16 | `MechB_NormBounded_C2L_L27_30` | `0.00` | `0.00` | `0.50` | `0.00` | `1.00` | `0.00203` | `12126.92 MB` | `1641.41 MB` | `271.84s` | `iterate` |
| 28 | `MechA12_SweetSpot_SwiGLU` | `0.32` | `0.28` | `0.6333`* | `0.00` | `1.00` | `0.000813` | `12979.67 MB` | `2494.16 MB` | `168.15s` | `iterate` |
| 29 | `MechA13_Pos0L27_30_StrongerGoals` | `0.28` | `0.36` | `0.8667` | `0.00` | `1.00` | `0.000814` | `12980.18 MB` | `2494.67 MB` | `174.14s` | `iterate` |
| 12 | `Shared4Null_ResSkip_L26_31` | `0.20` | `0.10` | `0.00` | `0.10` | `0.0333` | `2.408684` | `12832.29 MB` | `2346.77 MB` | `296.00s` | `iterate` |
| 30 | `Shared4Null_ResSkip_Stable_L27_31` | `0.64` | `0.56` | `0.30` | `0.30` | `0.9833` | `0.001514` | `12790.89 MB` | `2305.38 MB` | `223.37s` | `iterate` |

*`Iter 28` locality was re-measured as `0.8667` in `Iter 29` with the corrected matched-batch harness; the original `0.6333` is retained in `experiments.jsonl`.

### Key Findings

1. **Mech B (context -> LoRA residual fit) cannot fit rank-16 on 9B.** Per-layer `o_err=0.9806..0.9880`, `d_err=0.9097..0.9523`, so the reference-teacher signal is simply not representable by a rank-16 edit; verbatim/paraphrase recall stays `0.00`.
2. **Mech A12/A13 train far better than they generate.** Training diagnostics reach `top1_acc=0.90..0.91` and `mean_p=0.81`, but unseen generation is `0.28..0.36` full-answer accuracy. First-token top-1 is much higher (`0.48` verbatim, `0.58` paraphrase), so the dominant failure is **suffix decay on multi-token invented targets**: generated entity strings tokenize to 3-7 subwords (e.g. `Kaelinberg -> ['K','ael','in','berg']`), and one flipped subword destroys the exact match. Wrong-first-token cases are dominated by base-model refusals ("Does not exist", "No such mineral...").
3. **Moving pos-0 shards off L26 (`Iter 29`) trades verbatim for paraphrase** (`0.32/0.28 -> 0.28/0.36`) at identical KL/retention, confirming the Phase-1 signal-onset hypothesis: L26 (`ctx_p=0.0005`) is a poor write site for first tokens, while L27..L30 are better.
4. **The SmolLM2 Shared4Null hyperparameters catastrophically over-edit 9B.** Original `lr=4.5e-3`, scaling `alpha/r=2.0`, `lambda_cal_kl=5.5`: L28 reaches `cal_kl=2.105` and L30 `neg_kl=11.82`, ending at `KL=2.409`, `retention=0.0333`, `locality=0.0`. Scaling the edit down (`alpha/r=0.5`, `lr=8e-4`, stronger KL/anti-bleed, L27..L31) recovers to `verbatim=0.64`, `paraphrase=0.56`, `KL=0.0015`, `retention=0.9833`, at the cost of `30%` synthetic-sibling bleed.
5. **Locality measurement artifact found and fixed.** `get_base_reference` generated locality with `batch_size=10` while `evaluate_model` used `batch_size=12`; fp16 reduction-order noise flips high-entropy synthetic-entity tokens. This is why `Baseline_0_Unedited` itself scored `locality=0.6333` with zero edits and zero bleed. `eval_harness.py` and `probes.py` now use matched batch sizes (`10` for locality, `15` for general QA), making the comparison deterministic. `Iter 29` reaches `22/25` synthetic prefix matches and `30/30` total identical/deterministic locality outputs in the dedicated diagnostic.
6. **Quantized snapshot caching.** The 8-bit model is saved once to `ornith_8bit_bnb/` (gitignored, 3 safetensors shards) with its `quantization_config`; subsequent processes load it directly instead of re-quantizing the 18 GB fp16 checkpoint.
7. **Direct diagnostic (`diag_iter28.py`)** confirms no target bleed (`25/25` synthetic no-bleed, `5/5` real anchors), so the remaining locality gap is formatting/high-entropy tie-flipping, not factual corruption.

## Iteration 2: Multi-Seed `[42, 1337, 2026]` + Stage 2 (`seq_len=2048`) — Mech A13

`verify_stage2_mechA.py` loads the cached 8-bit snapshot **once** and reuses it across all seeds.

| Seed | Verbatim | Paraphrase | Locality | KL Div | Retention | Peak VRAM | Act Overhead | Wall Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `42` | `0.28` | `0.36` | `0.8667` | `0.000814` | `1.00` | `12980.18 MB` | `2494.67 MB` | `174.57s` |
| `1337` | `0.44` | `0.50` | `0.8667` | `0.000756` | `1.00` | `12978.66 MB` | `2493.15 MB` | `155.41s` |
| `2026` | `0.48` | `0.46` | `0.90` | `0.000791` | `1.00` | `12980.42 MB` | `2494.91 MB` | `174.65s` |
| **mean ± std** | **`0.400 ± 0.086`** | **`0.440 ± 0.059`** | **`0.878 ± 0.016`** | **`0.00079 ± 0.00002`** | **`1.000 ± 0`** | — | — | **`168.21 ± 9.05s`** |

**Stage 2 (`seq_len=2048`, Seed 42, real noisy-document extraction + `no_grad()` forward):**

| Metric | Value |
| :--- | :---: |
| Extracted facts from 2048-token doc | `25 / 25` |
| Document token length | `2048` |
| Peak VRAM | `13937.06 MB` |
| Activation overhead | **`3451.55 MB`** |
| vs `Baseline_A_512` overhead (`3242.66 MB`) | `1.0644x` |
| vs `Baseline_A_2048` OOM ceiling (`4111.93 MB`) | **`0.8394x`** |
| Verbatim / Paraphrase | `0.28 / 0.36` |
| KL / Retention | `0.000814 / 1.00` |

**Key finding:** a true 0-backprop solver can process a real 2,048-token document and synthesize the adapter in `3451.55 MB` of activation overhead, i.e. **16% below the measured OOM ceiling of backprop @ 2048** (`4111.93 MB`), while backprop cannot complete at all on the 15 GiB T4. The absolute VRAM gate (`<= 0.25 * Baseline_A`) is not met because the 2048-token forward itself dominates the overhead.

## Iteration 2b: Shared4Null-Stable Multi-Seed + Stage 2 (partial)

`verify_stage2_and_seeds.py` was stopped after seed `1337`; seed `2026` remains to be run.
Seed 42 Stage 2 is complete. Results are stored in `stage2_shared4null_partial.json` and `experiments.jsonl`.

| Seed / Stage | Verbatim | Paraphrase | Locality | Bleed | KL Div | Retention | Act Overhead | Wall Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Seed `42` | `0.64` | `0.56` | `0.30` | `0.2333` | `0.001494` | `0.9833` | `2305.38 MB` | `222.69s` |
| Seed `1337` | `0.48` | `0.36` | `0.40` | `0.1333` | `0.002554` | `1.00` | `2348.30 MB` | `221.51s` |
| Stage 2 `2048` (Seed 42) | `0.64` | `0.50` | `0.30` | `0.20` | `0.001513` | `0.9833` | **`2355.35 MB`** | `225.06s` |

**Stage 2 VRAM:** `2355.35 MB` = `0.7264x` of `Baseline_A_512` (`3242.66 MB`) and **`0.5728x` of the Baseline_A_2048 OOM ceiling** (`4111.93 MB`). This is the lowest 2048-token overhead observed in the lab, but it still exceeds the absolute `0.25x Baseline_A` gate.

**Open issue:** synthetic-sibling bleed (`13.3–23.3%`) remains the main blocker for Shared4Null-Stable; multi-hop biography logic has not been run yet.

### Current Best (Seed 42)
- **Pure 0-backprop:** `Iter 29/A13` multi-seed mean `verbatim=0.400±0.086`, `paraphrase=0.440±0.059`, `locality=0.878±0.016`, `KL=0.00079`, `retention=1.000`; Stage 2 overhead `3451.55 MB` (`0.8394x` the backprop OOM ceiling).
- **1-layer isolated:** `Iter 30` seed-42 `verbatim=0.64`, `paraphrase=0.56`, `KL=0.0015`, `retention=0.9833`; seed-1337 `0.48/0.36`; stage-2 overhead `2355.35 MB` (`0.5728x` the backprop OOM ceiling). Main blocker: `13–23%` synthetic-sibling bleed.
- No method meets all gates; the hard blockers are generation-suffix fidelity, wall time (`>20s` on T4 PyTorch fallback kernels), and the strict `locality>=0.98` gate.

## Next Experiments
- Multi-seed `[42, 1337, 2026]` + Stage 2 (`seq_len=2048`) via `verify_stage2_mechA.py` (A13) and `verify_stage2_and_seeds.py` (Shared4Null-Stable).
- `biography_logic_experiment.py` (Dr. Seren Valenwood, direct + multi-hop, `enable_thinking=False` and `True`).
- Optional R&D: expand Mech A13 probe frames (20+ diverse syntactic frames) and/or add first-token-weighted suffix goals to attack suffix decay.
- `blocks_library/` + final SmolLM2-vs-Ornith comparison.
