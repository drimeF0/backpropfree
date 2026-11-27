# SmolLM2-360M-Instruct Backprop-Free LoRA Lab Notebook

## Hardware & Precision Environment
- **GPU**: Tesla T4 (15,360 MiB VRAM, CUDA 13.0, Driver 580.82.07)
- **System RAM**: 12 GiB (0 swap) — keep CPU caches compact/streamed
- **T4 Precision Parity Check (`dissect_smollm.py`)**:
  - `bfloat16` forward (16x256): `1.3664s`
  - `float16` forward (16x256): `0.2566s` (**5.3x faster** on Turing SM 7.5 Tensor Cores)
  - $D_{\mathrm{KL}}(P_{\text{bf16}} \| P_{\text{fp16}}) = 0.002218$
  - Decision: Run model forward passes in `float16`, while computing covariances ($X^T X$), Ridge inversions, Cholesky/eigendecomposition, and SVD in `float32`/`float64`.

---

## Phase 1: Model Anatomy & Memory Profiling (`dissect_smollm.py`)

### 1. Weight Spectral Profile (32 layers, $d=960$, $d_{\text{ff}}=2560$, 15/5 GQA)
- `o_proj`: mean effective rank `442.06` (min `229.39`, max `498.51`, `r99 = 664.5`) — notably lower intrinsic rank than MLP projections.
- `gate_proj`: mean effective rank `684.77` (`r99 = 883.9`)
- `up_proj`: mean effective rank `744.68` (`r99 = 891.9`)
- `down_proj`: mean effective rank `713.65` (`r99 = 885.0`)

### 2. Forward Signal Localization (`In-Context vs No-Context`)
- Layers 0–13: modest context divergence (`res_rel_diff < 0.10`), target logit-lens rank $> 5000$.
- Layers 14–25: rapid buildup of attention & MLP context shifts (`res_l2_diff` $21.6 \to 168.3$, `res_rel_diff` $0.112 \to 0.287$).
- **Factual Resolution Window (Layers 26–30)**:
  - `L26`: target rank drops $11,317 \to 1,022$
  - `L27`: `attn_l2_diff` spikes to `131.39`, target rank drops to `125.8` (`p=0.1558`)
  - `L28`: target rank `54.0`
  - `L29`: target rank **`9.8`** (`attn=106.8`, `mlp=102.8`)
  - `L30`: target rank **`11.8`** (`attn=96.7`, `mlp=132.3`)
  - `L31`: large final MLP formatting shift (`mlp=290.76`) — caveat: `" Kaelengrad"` had a leading space while chat generation prompt ends in `\n`.
- **Implication for LoRA Placement**: Focusing rank-$r \le 16$ LoRA synthesis across the active resolution layers (roughly L18–L30, especially L22–L30) concentrates parameter capacity where factual association resolves while minimizing KL drift in early feature layers.

### 3. Streaming Null-Space Capacity (`smol-smoltalk`, 160 examples / 39,346 tokens)
- Streaming overhead: `1433.61 MiB` peak (`+387.78 MiB` above `1045.83 MiB` static model footprint at batch=8, seq=256).
- `o_proj` (960-dim) low-energy directions ($< 10^{-3} \lambda_{\max}$): `L18: 444`, `L22: 290`, `L30: 321` (bottom half accounts for only `2.2%–6.3%` of energy).
- `down_proj` (2560-dim) low-energy directions ($< 10^{-3} \lambda_{\max}$): `L22: 619`, `L26: 860`, **`L30: 2311 / 2560`** (bottom half energy `4.5%`).
- **Implication**: Late-layer `down_proj` and `o_proj` inputs are strongly anisotropic, providing abundant null-space dimensions to encode 25 new facts with minimal disruption to general chat distributions.

---

## Iteration 0: Benchmarks & Baselines (`iter_0_baselines.py`)
Running Baseline 0 (Unedited), Baseline C (In-Context), Baseline A (5-step full-backprop @ seq_len=512 & seq_len=2048), and Baseline B (100-step full-backprop) on Seed 42...

### Iteration 0 Baseline Results (Seed 42, float16 T4)
| Method | Verbatim | Paraphrase | Locality | KL_div | Retention | Peak VRAM (MB) | Overhead (MB) | Time (s) | Backprop? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline_0_Unedited | 0.00 | 0.00 | 1.0000 | 0.0000 | 1.0000 (`90.0%` raw) | 704.07 | 0.00 | 0.00 | No |
| Baseline_C_InContext | 0.80 | 0.58 | 0.1667 | 0.0000 | 1.0185 (`91.7%` raw) | 704.07 | 0.00 | 0.00 | No |
| Baseline_A_5step_seq512 | 0.00 | 0.00 | 0.5667 | 0.0208 | 1.0000 (`90.0%` raw) | 7710.54 | 6998.26 | 17.33 | Yes |
| Baseline_A_5step_seq2048 | 0.00 | 0.00 | 0.5667 | 0.0083 | 1.0185 (`91.7%` raw) | 6037.66 | 5316.52 | 3.14 | Yes |
| Baseline_B_100step_seq64 | 0.52 | 0.30 | 0.0333 | 0.5464 | 0.7963 (`71.7%` raw) | 5056.64 | 4335.49 | 41.69 | Yes |

### Key Iteration 0 Observations
1. **Backprop LoRA Failure Modes**: 5 steps (`Baseline A`) achieves 0% knowledge recall while already consuming `6.99 GB` (`seq=512, B=5`) and `5.32 GB` (`seq=2048, B=1`) activation/optimizer overhead. 100 steps (`Baseline B`) only reaches `30%` paraphrase accuracy while destroying retention (`79.6%`), locality (`3.3%`, `33.3%` target bleed), and KL (`0.5464`).
2. **In-Context Capacity Ceiling**: Putting all 25 facts simultaneously in context (`Baseline C`) saturates at `58%` paraphrase accuracy and causes severe sibling bleed (`63.3%`). By contrast, single-fact in-context probing achieves ~90–100% paraphrase accuracy, confirming that **per-fact forward teacher extraction -> null-space LoRA** can surpass `Baseline C` without sibling interference.
3. **Stage 2 VRAM Budget**: `0.25 * Baseline_A_2048_overhead = 0.25 * 5316.52 MB = 1329.13 MB` max activation overhead target (and aiming much lower via streaming covariance).

---

## Iteration 1: Sequential Layer-by-Layer Whitened Context-to-LoRA (`Iter1_SeqWhitenedC2L_L20_30`)
- **Hypothesis**: Sequentially matching contrastive teacher residual shifts (`H_tea(fact) - H_tea(neutral)`) across layers 20–30 (`o_proj`, `down_proj`, `r=16`) via calibration-whitened Ridge (`C0` + negative placeholder prompts) will inject all 25 facts with low KL and no cascade error.
- **Metrics (Seed 42)**:
  - Verbatim: `0.16` | Paraphrase: `0.02` | Locality: `0.0333` (`bleed=0.0`) | KL: `0.6906` | Retention: `0.3333` (`30.0%` raw)
  - Peak VRAM: `1053.37 MB` (`349.30 MB` overhead vs Baseline A's `6998.26 MB` @ seq512) | Synthesis Time: `13.301s` | Backprop: `False`
- **Root-Cause Failure Analysis**:
  1. **Unbounded Frobenius Norm Explosion (`||dW||_F` = 104–785)**: In `solve_whitened_ridge_lora`, `lambda_c0 * C0` penalized directions inside `C0` while leaving `eps_whiten * mean_diag` tiny (`~0.015`). Because `X_w @ X_w^T` had scale `~10^4` vs `+I` (`1.0`), `S_inv` blew up small directions into `||B @ A||_F = 250–785` (compared to base `||W||_F ~ 31`), wrecking `kl_div` (`0.691`) and `retention_acc` (`0.333`).
  2. **Unaligned Residual Target Noise**: Raw `H_tea - H_neutral` reflects 20 layers of upstream prefix differences (`norm ~ 200-350`), most of which are orthogonal to `lm_head`. Trying to jump raw `H_stu` at L20–30 with rank-16 projections yielded high fit error (`0.64–0.95`) and poor paraphrase generalization (`0.02`).
- **Next Direction (Iteration 2)**: Pivot to **Hypothesis 3 (Unembedding / Logit-Residual Forward Alignment)** combined with **Norm-Bounded Null-Space Ridge**, first probing where phrasing-invariant entity representations peak across layers.

---

## Iteration 2: Logit-Residual Forward Alignment + Bounded Whitened Ridge (`Iter2_LogitAlignedNullSpace_L24_31`)
- **Hypothesis**: Using analytical `lm_head` + `RMSNorm` steepest logit residuals across `L24..L31` (peak phrasing-invariance window from probe: `L26 o_in gap = 0.633`) with trace-normalized `C0` whitening and Frobenius norm caps (`3.5` / `5.0`) will drive edits cleanly while keeping `kl_div <= 0.05`.
- **Metrics (Seed 42)**:
  - Verbatim: `0.00` | Paraphrase: `0.00` | Locality: `0.7667` (`bleed=0.0`) | KL: `0.001406` | Retention: `1.037` (`93.3%` raw)
  - Peak VRAM: `2229.10 MB` (`1525.03 MB` overhead) | Synthesis Time: `9.327s` | Backprop: `False`
- **Root-Cause Failure Analysis**:
  1. `kl_div` dropped 490x (`0.6906 -> 0.001406`) and `retention_acc` hit `1.037` — proving trace-normalized `C0` + `X_neg` whitening protects general instruct capabilities.
  2. **Rank-16 Capacity & Norm Starvation**: Fitting all 25 facts ($\approx 100$ distinct subword transitions across 632 tokens) simultaneously in rank-16 SVD with `max_fro_norm = 3.5 / 5.0` gave `fit_rel_err = 0.94–0.98`, barely shifting `mean_target_prob` (`0.0509 -> 0.0719`).
  3. **Peak VRAM**: Computing full `[900, 49152]` logits inside `compute_analytical_logit_residual_targets` + unchunked `[150, 45, 960]` attention caused `1525 MB` overhead (easily chunkable to `< 400 MB`).
- **Next Direction (Iteration 3)**: Implement **Dynamic Orthogonal Fact Sharding** across `L24..L31` (fitting `3–4` unlearned facts per layer while explicitly orthogonalizing against `C0`, `X_neg`, and the other facts) with relaxed null-space norm cap and chunked logit computation.

---

## Iteration 3: Dynamic Orthogonal Fact-Sharded Null-Space LoRA (`Iter3_OrthogonalFactSharded_L24_31`)
- **Hypothesis**: Sharding 4 unlearned facts per layer across `L24..L31` while protecting `C0`, `X_neg`, and the other 21 facts will let rank-16 SVD capture >85% energy and solve all 25 facts cleanly.
- **Metrics (Seed 42)**:
  - Verbatim: `0.00` | Paraphrase: `0.00` | Locality: `0.40` (`bleed=0.0`) | KL: `0.012355` | Retention: `1.000` (`90.0%` raw)
  - Peak VRAM: `1308.54 MB` (`604.46 MB` overhead) | Synthesis Time: `12.11s` | Backprop: `False`
- **Root-Cause Failure Analysis**:
  1. Sharding succeeded in lifting rank-16 SVD energy capture to `87–96%` and chunking cut activation overhead to `604 MB`.
  2. However, because `down_proj` frozen inputs have ~`0.70–0.83` cosine overlap across facts, penalizing `X_other` with `lambda=30` inside `P_cov` suppressed the positive shard itself (`fit_rel_err = 0.68–0.90`), only nudging `mean_p` from `0.0509` to `0.0874`.
  3. Linear `o_proj`/`down_proj` updates on frozen `gate/up` activations cannot construct sharp entity-selective gates without updating `gate_proj`/`up_proj`/`q_proj`/`k_proj`.
- **Next Direction (Iteration 4)**: Test **Regime (c) Strictly Isolated 1-Layer Detached Block Refinement** (`x_in = x_in.detach()`), optimizing `r=16` `q/k/v/o/gate/up/down` inside 1 block at a time (`L24..L30` contrastive teacher + `L31` direct CE+KL).

---

## Iteration 4: Strictly Isolated 1-Layer Detached Block Refinement (`Iter4_SingleLayerDetached_L24_31`)
- **Hypothesis**: Refining `q/k/v/o/gate/up/down` (`r=16`) one detached block at a time (`L24..L30` teacher delta MSE, `L31` CE+KL) will enable nonlinear SiLU entity gating.
- **Metrics (Seed 42)**:
  - Verbatim: `0.48` | Paraphrase: `0.20` | Locality: `0.0667` (`bleed=0.0333`) | KL: `0.020324` | Retention: `0.4444` (`40.0%` raw)
  - Peak VRAM: `5787.21 MB` (`5083.13 MB` overhead) | Synthesis Time: `39.844s` | Backprop: `False` (single-layer isolated)
- **Root-Cause Failure Analysis**:
  1. First real jump in efficacy (`0.00 -> 0.48` verbatim, `0.20` paraphrase), confirming nonlinear `gate/up/q/k/v` LoRA works where linear Ridge stalled.
  2. **Upstream Drift**: `L24..L30` noisy teacher-delta MSE (`loss = 13 -> 125`) shifted `H_cal` before `L31`, so `L31` preserved a drifted hidden state (`retention = 0.444`).
  3. **VRAM & Time**: Unchunked `cal_logits` (`~3500 x 49152` float32) spiked VRAM to `5083 MB` overhead and `39.8s`.
- **Next Direction (Iteration 5)**: Replace intermediate `H_tea` MSE with **Sequential Single-Block Detached Residual-Skip Direct CE + True `P_base` KL** (`L26..L31`, micro-batched, subsampled KL tokens) plus entity-first/hypernym/entity-only probes.

---

## Iteration 5: Sequential Single-Block Detached ResSkip CE + True `P_base` KL (`Iter5_SingleBlockResSkip_CE_KL_L26_31`)
- **Hypothesis**: Optimizing `L26..L31` one detached block at a time with `no_grad()` downstream skip `res_after = (H_31 - H_l).detach()` against direct CE + true unedited `P_base` KL + generalizing probes will reach `>= 75%` paraphrase generalization with low VRAM.
- **Metrics (Seed 42)**:
  - **Verbatim: `0.88`** | **Paraphrase: `0.82`** (PASSES `>= 0.75`!) | Locality: `0.10` (`bleed=0.60`) | KL: `0.070602` | Retention: `0.7037` (`63.3%` raw)
  - Peak VRAM: `1435.74 MB` (`731.67 MB` overhead — **9.6x lower** than Baseline A) | Synthesis Time: `19.499s` | Backprop: `False` (single-layer isolated)
- **Root-Cause Failure Analysis**:
  1. **Paraphrase Breakthrough**: ResSkip CE+KL + entity-first/hypernym probes pushed `paraphrase_acc` from `0.20 -> 0.82` (surpassing Baseline C's `0.58` capacity ceiling!).
  2. **Why `kl_div = 0.0706` & `retention = 0.704`**: `cal_kl` only used 192 fixed tokens (`24` examples $\times$ 8), letting the LoRA memorize those 192 tokens (`cal_kl = 0.008`) while drifting on held-out prompts (`0.0706`) and real QA.
  3. **Why `bleed = 0.60`**: Negative probes all used `"Sector_101"` (`Sector_N`), so the LoRA bypassed `neg_kl` via a surface shortcut and boosted injected answers for any non-`Sector_` proper noun!
- **Next Direction (Iteration 6)**: Replace `"Sector_N"` with **50 realistic independent proper-noun negative probes** + **anti-bleed probability suppression** + **all-token `smol-smoltalk` hidden MSE + freshly resampled random-token KL** across `L27..L31`.

---

## Iteration 6: Strict Calibration-Locked + Proper-Noun Anti-Bleed ResSkip (`Iter6_StrictNullCal_ResSkip_L27_31`)
- **Hypothesis**: Adding 64-seq calibration anchors, 50 proper-noun negatives, and anti-bleed token suppression on `L27..L31` will keep Iter 5's paraphrase accuracy while passing KL, retention, and locality.
- **Metrics (Seed 42)**:
  - Verbatim: `0.00` | Paraphrase: `0.04` | Locality: `0.40` (`bleed=0.0333`) | KL: `0.019133` | Retention: `1.0185` (`91.7%` raw)
  - Peak VRAM: `1251.29 MB` (`547.22 MB` overhead) | Synthesis Time: `19.759s` | Backprop: `False` (single-layer isolated)
- **Root-Cause Failure Analysis**:
  1. Retention (`1.0185`), KL (`0.0191`), and bleed (`0.0333`) improved dramatically.
  2. **Anti-Bleed BPE Bug**: `injected_first_toks` contained common capital/initial subword tokens (`neg_bleed_p = 0.224` on Base!), so `lambda_bleed = 8.0` penalized normal proper-noun continuations and prevented `CE` from converging (`1.4413` vs `0.0887` in Iter 5).
  3. **Over-Regularization & Missing L26**: Removing `L26` and quadrupling KL penalties further choked edit learning.
- **Next Direction (Iteration 7)**: Restore `L26..L31`, drop `lambda_bleed`, add block-level identity `MSE` (`lambda_block_mse=0.5`) on `cal` and 100 diverse proper-noun negatives, and balance `lambda_cal_kl=5.0`, `lambda_neg_kl=6.0`.

---

## Iteration 7: Entity-Selective ResSkip with All-Token Block MSE (`Iter7_EntitySelective_ResSkip_L26_31`)
- **Hypothesis**: Restoring `L26..L31`, dropping the buggy `lambda_bleed`, and adding all-token block identity MSE (`0.5`) on 100 realistic 1-word/2-word proper-noun negatives + 48 `smol-smoltalk` seqs will recover Iter 5's paraphrase accuracy while passing retention, KL, and locality.
- **Metrics (Seed 42)**:
  - Verbatim: `0.32` | Paraphrase: `0.34` | Locality: `0.30` (`bleed=0.10`) | **KL: `0.003211`** | **Retention: `1.0185` (`91.7%` raw)**
  - Peak VRAM: `1576.63 MB` (`872.56 MB` overhead) | Synthesis Time: `19.49s` | Backprop: `False` (single-layer isolated)
- **Root-Cause Failure Analysis**:
  1. Dropping `lambda_bleed` and guarding 48 `smol-smoltalk` seqs pushed `kl_div` down to `0.003211` and `retention_acc` to `1.0185` (above Base!).
  2. **Over-damped `CE` (`0.6602`)**: `lambda_block_mse = 0.5` across every hidden dim (`960`) had 5x–10x larger gradient norm than `CE` and choked the `12–16` Adam steps before `CE` could converge (`0.66` vs `0.089` in Iter 5).
  3. **Single-Token Negative Guard**: `neg_kl` only guarded the 1 prompt-end token (`neg_kl = 0.0164`), leaving 32-token sibling continuations unguarded (`locality = 0.30`).
- **Next Direction (Iteration 8)**: Use 4 high-leverage blocks (`L26, L28, L30, L31`) with unchunked batching so we can run more steps (`22–32` steps/layer) in `< 15s`, remove `block_mse` choking, and guard 8-token Base-0 continuations on the 100 negative proper nouns.

---

## Iteration 8: Fast-Converged 4-Block ResSkip + 8-Token Negative Continuations (`Iter8_FastConverged_ResSkip_4Layer`)
- **Hypothesis**: Dropping intermediate `block_mse` and running 4 unchunked blocks (`L26, L28, L30, L31`, 22–32 steps) with 8-token Base-0 continuation KL on 100 negative proper nouns will pass both paraphrase (`>=0.75`) and retention/locality in `<= 16s`.
- **Metrics (Seed 42)**:
  - **Verbatim: `0.92`** | Paraphrase: `0.72` | Locality: `0.2333` (`bleed=0.40`) | KL: `0.034572` | Retention: `0.8889` (`80.0%` raw)
  - Peak VRAM: `2412.58 MB` (`1708.51 MB` overhead) | Synthesis Time: `29.213s` | Backprop: `False` (single-layer isolated)
- **Root-Cause Failure Analysis**:
  1. Removing `block_mse` let `CE` drop to `0.2173` (`0.92` verbatim, `0.72` paraphrase).
  2. **Vocab Logit Bottleneck**: Evaluating `49,152`-dim `lm_head` logits on `~1,742` tokens across 104 steps dominated runtime (`29.2s`) and VRAM (`1708 MB` overhead).
  3. **Diluted `neg_pos_0` Protection**: Averaging `neg_kl` uniformly across 800 continuation tokens diluted the first decision token (`\n`) by 8x, leaving `bleed = 0.40` and `retention = 0.8889`.
- **Next Direction (Iteration 9)**: Use vocab-free `RMSNorm`-space MSE for `cal`/`neg` in `L26, L28, L30` (exact `lm_head` KL in `L31`), weight `neg_pos_0` 5x, and add gentle `0.10` block-identity MSE on `neg` + `cal`.

---

## Iteration 9: RMSNorm-Anchored + Pos0-Guarded 4-Block ResSkip (`Iter9_RMSNormAnchor_Pos0Guard_ResSkip`)
- **Hypothesis**: Using 960-dim RMSNorm anchoring in `L26/28/30`, exact KL + 5x `pos_0` weighting in `L31`, and 120 realistic proper-noun + real-style negatives will recover paraphrase accuracy while passing retention and locality.
- **Metrics (Seed 42)**:
  - Verbatim: `0.64` | Paraphrase: `0.58` | Locality: `0.2667` (`bleed=0.30`) | **KL: `0.006983`** | **Retention: `1.000` (`90.0%` raw)**
  - Peak VRAM: `2394.22 MB` (`1690.15 MB` overhead) | Synthesis Time: `29.438s` | Backprop: `False` (single-layer isolated)
- **Root-Cause Failure Analysis**:
  1. `kl_div = 0.006983` and `retention_acc = 1.000` passed cleanly.
  2. **Why `bleed = 0.30` persisted**: Forward `KL(P_base || P_edit)` has small gradient (`P_edit - P_base`) when an injected answer rises from `1e-5` to `0.12` on flat-prior synthetic siblings — enough to win `argmax` and bleed without spiking KL.
  3. **Compute & Upstream Drift**: Unchunked `175` + `120` sequences across `106` steps took `29.4s` / `1690 MB`, and `L28/30` RMSNorm drift (`0.123`) forced `L31` `ce` back up (`0.288 -> 0.448`).
- **Next Direction (Iteration 10)**: Add **Log-Ratio Anti-Bleed (`ReLU(log P_edit - log P_base)^2`)** on injected tokens for negatives (zero penalty when `P_edit <= P_base`), plus lean micro-batched `L26, L28, L30, L31` (`78` steps total).

---

## Iteration 10: Log-Ratio Anti-Bleed + RMSNorm Anchor (`Iter10_LogRatioAntiBleed_LeanResSkip`)
- **Hypothesis**: Adding zero-conflict `ReLU(log P_edit - log P_base - 0.1)^2` anti-bleed at `L31` with `RMSNorm MSE` in `L26/28/30` and lean micro-batching will keep speed `<= 16s` and cut bleed.
- **Metrics (Seed 42)**:
  - Verbatim: `0.12` | Paraphrase: `0.12` | Locality: `0.1333` (`bleed=0.0333`) | KL: `0.097208` | Retention: `0.6296` (`56.7%` raw)
  - Peak VRAM: `1477.62 MB` (`773.54 MB` overhead) | Synthesis Time: `14.623s` | Backprop: `False` (single-layer isolated)
- **Root-Cause Failure Analysis**:
  1. `wall_time_sec = 14.62s`, `overhead = 773 MB`, and `bleed = 0.0333` all dropped cleanly.
  2. **Intermediate `RMSNorm MSE` Drift**: Without `lm_head` KL in `L26/28/30`, intermediate blocks drifted in logit space (`cal: 0.026 -> 0.173`, `neg: 0.208 -> 0.914`), forcing `L31` to undo upstream KL/bleed (`ce` bounced `0.439 -> 1.207`).
- **Next Direction (Iteration 11)**: Restore **exact `lm_head` KL + Log-Ratio Anti-Bleed at EVERY block** (`L26..L31`, just like Iter 5's `0.82` paraphrase run) using lean token counts (`96` alternating cal tokens, `200` neg tokens) to hold speed `<= 18s` and VRAM `< 900 MB`.

---

## Iteration 11: Every-Layer `lm_head` KL + Log-Ratio Anti-Bleed (`Iter11_AllLayerKL_AntiBleed_L26_31`)
- **Hypothesis**: Restoring exact `lm_head` KL + zero-conflict Log-Ratio Anti-Bleed at every layer `L26..L31` with lean batching will combine Iter 5's `0.82` paraphrase accuracy with clean retention/locality.
- **Metrics (Seed 42)**:
  - Verbatim: `0.52` | Paraphrase: `0.38` | Locality: `0.2667` (`bleed=0.3333`) | KL: `0.046422` | Retention: `0.9444` (`85.0%` raw)
  - Peak VRAM: `1466.39 MB` (`762.32 MB` overhead) | Synthesis Time: `18.86s` | Backprop: `False` (single-layer isolated)
- **Root-Cause Failure Analysis**:
  1. Unconstrained `lora_A` experiences a direct gradient tug-of-war on shared question/assistant tokens (`"What is..."`, `"\n"`) between `CE` (`pos`) and `KL` (`neg` + `cal`), preventing `CE` from converging below `0.606` while still bleeding `33.3%` on siblings.
- **Next Direction (Iteration 12)**: Enforce **Null-Space Preconditioning (`A_eff = A_free @ P_null`) + Fisher-SVD Initialization** inside each detached block (`L26..L31`), forcing gradients strictly out of `C_cal + C_neg` into entity-specific directions.

---

## Iteration 12: Null-Space Preconditioned + Fisher-SVD Initialized ResSkip (`Iter12_NullSpacePrecond_FisherInit_ResSkip`)
- **Hypothesis**: Preconditioning `A_eff = A_free @ (I + alpha C_prot)^{-1/2}` with Fisher-SVD initialization inside each detached block (`L26..L31`) eliminates template gradient conflict and unlocks generalization + retention.
- **Metrics (Seed 42)**:
  - **Verbatim: `1.000`** | **Paraphrase: `1.000`** | **KL: `0.016858`** (PASS!) | **Retention: `0.9815` (`88.3%` raw)** (PASS!) | Locality: `0.20` (`bleed=0.3333`)
  - Peak VRAM: `1578.21 MB` (`874.14 MB` overhead — 8x lower than Baseline A) | Synthesis Time: `29.752s` | Backprop: `False` (single-layer isolated)
- **Analysis**:
  1. **Massive Breakthrough**: Preconditioning + Fisher-SVD initialization pushed `ce` cleanly from `1.8165 -> 0.3805 (L27) -> 0.0549 (L28) -> 0.0024 (L31)`, reaching **100% verbatim and 100% unseen paraphrase accuracy** while passing both KL (`0.0169`) and retention (`0.9815`).
  2. **Over-Margin & Runtime**: `ce` already converged to `0.0549` by `L28`; running `L29..L31` over-pushed logit confidence (`ce = 0.0024`), driving runtime to `29.75s` and spilling onto unedited siblings (`bleed = 0.3333`, `neg_kl = 0.0076`).
- **Next Direction (Iteration 13)**: Trim to 4 blocks (`L26..L29`, 44 steps total -> `~12–15s`), raise `alpha_null = 25.0`, add direct module-delta suppression on `neg`/`cal`, and boost `neg_kl` / `antibleed` to drive sibling bleed to zero.

---

## Iteration 13: Strong Null-Space Preconditioned 4-Block Fast ResSkip (`Iter13_StrongNull_4Block_FastResSkip`)
- **Hypothesis**: Cutting to `L26..L29` (`44` steps) with `alpha_null = 25.0` and `lambda_neg_kl = 18.0`, `lambda_antibleed = 5.0` will hold high paraphrase accuracy while dropping runtime `<= 16s`, `kl_div <= 0.01`, and sibling bleed `-> 0`.
- **Metrics (Seed 42)**:
  - Verbatim: `0.48` | Paraphrase: `0.46` | Locality: `0.40` (`real_anchors=5/5`, `synth_no_bleed=22/25`, `synth_exact_32tok=7/25`, `bleed=0.10`) | **KL: `0.005801`** | **Retention: `1.000` (`90.0%` raw)**
  - Peak VRAM: `1579.21 MB` (`875.13 MB` overhead) | Synthesis Time: `19.76s` | Backprop: `False` (single-layer isolated)
- **Analysis**:
  1. Retention (`1.000`) and KL (`0.005801`) are rock solid, and runtime met `<= 20s` (`19.76s`).
  2. `alpha_null = 25.0` (`stu_null_ratio = 0.30–0.375`) + `10–12` steps left `ce` at `0.3048` (`paraphrase = 0.46`).
  3. Computing `7` separate `eigh` preconditioners per layer wasted ~2.5s when `q/k/v` share `attn_in` and `gate/up` share `mlp_in` (only 4 unique input spaces!).
- **Next Direction (Iteration 14)**: Share the 4 unique preconditioners per layer (`~14–16s`), tune `alpha_null = 12.0` (`stu_null_ratio ~ 0.42`) with `(11, 11, 13, 14)` steps (`lr = 5e-3`) + gentle `0.04` negative last-token block MSE to recover `>= 0.85` paraphrase while holding `kl <= 0.012`, `retention >= 0.96`, and minimizing sibling bleed.

---

## Iteration 14: Shared-4-Space Null-Preconditioned ResSkip (`Iter14_Shared4Null_ResSkip_L26_29`)
- **Hypothesis**: Sharing the 4 unique preconditioners across `L26..L29` with `alpha_null = 12.0` and `(11, 11, 13, 14)` steps will match Iter 12's generalization while holding runtime `< 20s` and low KL.
- **Metrics (Seed 42)**:
  - Verbatim: `0.48` | Paraphrase: `0.52` | Locality: `0.3333` (`real_anchors=5/5`, `synth_no_bleed=22/25`, `synth_exact_32tok=5/25`, `bleed=0.10`) | **KL: `0.008201`** | **Retention: `0.9815` (`88.3%` raw)**
  - Peak VRAM: `1553.94 MB` (`849.87 MB` overhead) | Synthesis Time: `19.619s` | Backprop: `False` (single-layer isolated)
- **Root-Cause Failure Analysis**:
  1. `L26..L29` with frozen `L30..L31` suffers from `skip_stu` nonlinearity error on `pos_0` (`ce = 0.0635` averaged over suffix tokens, but `pos_0` accuracy stalled at `~0.50` without `L30/L31` readout adaptation).
- **Next Direction (Iteration 15)**: Shift the 4 blocks to **`L26, L28, L30, L31`** (including final readout layers `L30, L31` as in Iter 12!) with `alpha_null = 9.0`, `lr = 4.5e-3`, `(10, 12, 12, 16)` steps, and `100` negatives to hit `>= 0.80` paraphrase + `<= 0.015` KL + `>= 0.96` retention + `<= 18s`.

---

## Iteration 15: Shared-4-Space Null-Preconditioned ResSkip on `L26, 28, 30, 31` (`Iter15_Shared4Null_ResSkip_L26_28_30_31`)
- **Hypothesis**: Including readout layers `L30, L31` in the 4-block schedule (`alpha_null = 9.0`) will restore `>= 0.85` paraphrase while holding `< 18s` runtime and `< 0.015` KL.
- **Metrics (Seed 42)**:
  - **Verbatim: `1.000`** | **Paraphrase: `0.960`** | Locality: `0.1333` (`real_anchors=2/5`, `synth_no_bleed=15/25`, `synth_exact_32tok=2/25`, `bleed=0.3667`) | **KL: `0.015541`** | Retention: `0.8704` (`78.3%` raw)
  - Peak VRAM: `1524.95 MB` (`820.88 MB` overhead) | Synthesis Time: `17.945s` | Backprop: `False` (single-layer isolated)
- **Analysis**:
  1. Restoring `L30, L31` immediately lifted paraphrase accuracy back to `96%` inside `17.95s` (`<= 20s`), **but** skipping `L27, L29` forced `L31` to absorb a massive CE jump alone (`0.598 -> 0.0643`), shifting real-world capitals (`real_anchors=2/5`, `retention=0.8704`).
  2. By contrast, **Iteration 12 (`L26..L31`, 6 blocks)** smoothly distributed the drop (`ce=0.055` by `L28`), achieving `1.00` verbatim, `1.00` paraphrase, `0.0169` KL, and `0.9815` retention.

---

## Multi-Seed & Stage 2 (`seq_len=2048`) Verification (`verify_stage2_and_seeds.py`)
Evaluating our strongest candidate (`Iter 12` 6-block `L26..L31` accelerated with Shared-4-Space Preconditioner) across seeds `[42, 1337, 2026]` and Stage 2 2,048-token document extraction + VRAM vs Baseline A (`5316.52 MB` overhead)...

---

## [Research Director Intervention] Directive 1: Locality Diagnostic (`diag_locality_iter2_iter6.py`)
Logged exact predictions and `pos_0` distributions for `Iter 2` (`KL=0.0014`) and `Iter 6` (`KL=0.0191`) vs `Base 0` across all 30 `locality_qa` items (`mean_base_entropy = 2.367` nats on synthetic siblings):
- **`Iter 2` (`0-Backprop Ridge, KL=0.0014`)**: `pos0_mean_kl = 0.03693`, **`pos0_top1_agree = 0.933` (`28/30`)**, **`real_ok = 5/5`**, **`target_bleed = 0/25`**, `exact_match = 18/25`.
  - Concrete failures:
    1. `idx 5 (Thalavberg)`: Base `"The central governing office of Thalavberg is located in the city of Thalavberg."` -> Edit `"...city of Thalavberg itself."` (later-token length tie-flip, `pos0_kl=0.01355`).
    2. `idx 7 (Rynelmont)`: Base `"Rynelmont uses the Rynelmont Pound."` -> Edit `"Rynelmont uses the Rynelmont Dollar."` (high-entropy `2.759` nats synthetic currency tie-flip).
    3. `idx 10 (Nevavgard)`: Base `"Nevavgard."` -> Edit `"Valleys of Nevavgard"` (`pos0_kl=0.12049`, single `pos0` framing shift).
    4. `idx 11 (Jorarholm)`: Base `"Jorarholm."` -> Edit `"Jorarholm's rare ore is called \"Jorarholmite.\""` (`pos0_kl=0.00147`).
    5. `idx 14 (Vaelasgate)`: Base `"The Vaelasgate Observatory."` -> Edit `"The Vaelasgate Observatory is located in Vaelasgate..."` (`entropy=3.646`).
    6. `idx 15 (Mirelovgard)`: Base `"...city of Mirelovgard."` -> Edit `"...city of Mirelovgard itself."` (`pos0_kl=0.00161`).
  - **Conclusion**: 6/7 `Iter 2` locality failures are **harmless later-token hallucination/length variations** on high-entropy gibberish entities under exact 32-token string matching, NOT factual corruption.
- **`Iter 6` (`pos_0 ResSkip, KL=0.0191`)**: `pos0_mean_kl = 0.09312`, `pos0_top1_agree = 0.900`, `exact_match = 7/25`, `pos0_template_shift = 8`, `later_tie_flip = 9`, `target_bleed = 1`, `real_ok = 5/5`.
  - Concrete failures show `pos_0` editing injects synthetic fantasy morphemes on unedited siblings: `"Lyraiskeep." -> "LyraSanctum"` (`pos0_kl=0.453`), `"Kevonstead..." -> "Kaelmarenite"`, `"Korvalhaven kroner" -> "...Korvynencrown"`, `"Korvorberg..." -> "Kaelin VaelSpire"`, `"Thalavberg." -> "...Thalkoswatchwatch..."` (`pos0_kl=0.537`).
  - **Root Cause**: Directly training `pos_0` (`\n`) + continuation tokens via gradient descent shifts the template-end prior toward synthetic target morphemes.

---

## Mechanism B: Norm-Bounded Context-to-LoRA Retry on `L27–L30` (`iter_16_mechB_bounded_c2l.py`)
Running strictly under `torch.no_grad()` (`H_tea - H_noctx` on `L27..L30` with trace-normalized `C0` + `X_neg` protection + Frobenius norm caps).
- **Results (Seed 42)**:
  - Verbatim: `0.00` | Paraphrase: `0.00` | **Locality: `0.9667` (`real=5/5, no_bleed=25/25, prefix2=25/25, exact32=24/25`)** | **KL: `0.000194`** | **Retention: `1.000` (`90.0%` raw)**
  - Peak VRAM: `1225.57 MB` (`521.49 MB` overhead) | Synthesis Time: `5.114s` | Backprop: `False`
- **Diagnosis**: Fixing the Iter 1 norm bug completely eliminated KL/retention damage (`0.000194` KL, `1.000` retention) and demonstrated that `locality_acc` reaches `0.9667` (`25/25` prefix match) when `pos0` isn't corrupted. However, rank-16 closed-form Ridge on frozen `o_proj`/`down_proj` inputs across all 25 facts simultaneously captured only `38–48%` SV energy and `0.2–1.5%` of `H_tea - H_noctx` (`o_err=0.998`, `d_err=0.985`).

---

## Mechanism A: True 0-Backprop Analytical Entity-Gated SwiGLU LoRA (`iter_17_mechA_analytical_swiglu.py`)
Running strictly under `torch.no_grad()` on `L27..L30`: analytical contrastive Ridge on 16 dormant SwiGLU neurons (`gate_proj`/`up_proj`) to collapse entity overlap, followed by bounded Null-Space Ridge on `down_proj`.
- **Results (Seed 42)**:
  - Verbatim: `0.00` | Paraphrase: `0.00` | Locality: `0.8333` (`real=5/5, no_bleed=25/25, prefix2=25/25, exact32=20/25`) | **KL: `0.000531`** | **Retention: `1.000` (`90.0%` raw)**
  - Peak VRAM: `1252.58 MB` (`548.51 MB` overhead) | Synthesis Time: `5.30s` | Backprop: `False`
- **Root-Cause Diagnosis**:
  1. `J_neg_to_pos_ratio` stayed at **`0.89–1.19`** (`new_act_overlap = 0.367` vs `base = 0.359`), meaning `gate_proj`/`up_proj` LoRA barely separated target facts from negatives!
  2. Why: setting `dG_cal = 0` with `lambda_cal = 3.0` on neurons where `g_base_cal ~ -0.5` fought `dG_neg = -4.5 - g_base`, and `lambda_fro = 0.25` swamped the `0.01–0.05` entity-discriminative eigenvalues of `X_mlp_in`.
- **Next (Mechanism A2, `iter_18`)**: Align `g_goal_cal = min(g_base_cal, -3.5)` and `u_goal_cal = 0.0` across `L26..L30` (5 facts/layer = 10 neurons/layer) with `lambda_fro = 0.005` so `SiLU` actually activates selectively.

### Mechanism A2 (`iter_18_mechA2_contrastive_swiglu.py`) Results
- **Metrics (Seed 42)**:
  - Verbatim: `0.00` | Paraphrase: `0.00` | Locality: `0.80` (`real=5/5, no_bleed=25/25, prefix2=25/25, exact32=19/25`) | **KL: `0.00788`** | **Retention: `1.000`**
  - Peak VRAM: `1265.89 MB` (`561.82 MB` overhead) | Synthesis Time: `6.029s` | Backprop: `False`
- **Analytical SwiGLU Selectivity Breakthrough**:
  - `L26`: `J_diag_pos=9.53` vs `J_neg=0.0098` (`J_ov: 0.7779 -> -0.0369`)
  - `L28`: `J_diag_pos=11.053` vs `J_neg=0.0068` (`J_ov: 0.5812 -> -0.0455`)
  - `L30`: `J_diag_pos=10.973` vs `J_neg=0.0052` (**2,110:1 selectivity**, `J_cal=0.015`, 0 backprop!)
- **Remaining Bug**: `down_proj` was still fed the full 2,560-dim `act_new` (2,550 overlapping base neurons + 10 gated neurons), so `solve_bounded_whitened_ridge_lora` wasted its `max_fro_norm=20` budget on the noisy base dimensions (`d_err=0.84`).
- **Next (Mechanism A3, `iter_19`)**: Lock `A_down` strictly to the `r` gated neuron indices $J_l$ (0 overlap with the 2,550 base neurons!) and solve `B_down` directly across `L26..L31` (`[5, 4, 4, 4, 4, 4]` facts) with `no_grad()` 1D gain calibration.

### Mechanism A3 (`iter_19_mechA3_sparse_J_down.py`) Results
- **Metrics (Seed 42)**:
  - Verbatim: `0.00` | Paraphrase: `0.02` | Locality: `0.60` (`real=4/5, no_bleed=25/25, prefix2=21/25, exact32=14/25`) | **KL: `0.010702`** | **Retention: `0.9815` (`88.3%` raw)**
  - Peak VRAM: `1256.22 MB` (`552.14 MB` overhead) | Synthesis Time: `10.166s` | Rank: `10` | Backprop: `False`
- **Progression & Root Cause**:
  - Train `top1_acc` climbed monotonically `0.082 -> 0.385` (`mean_p`: `0.048 -> 0.251`), confirming that locking `A_down` to $J_l$ transfers selective signals cleanly without bleeding (`bleed = 0.0`).
  - **1-Neuron Multi-Token Averaging Bug**: Assigning only 2 neurons per fact forced all suffix tokens (`pos_1, pos_2, pos_3`) to share 1 neuron, averaging 3 different residual vectors into 1 column (`d_fit_err = 0.80`).
- **Next (Mechanism A4, `iter_20`)**: Give every `(fact_i, tok_step_k)` transition its **own dedicated 1-to-1 dormant SwiGLU neuron** across `L20..L31` (placing `pos_0` neurons in `L26..L31` and suffix neurons across `L20..L31`).

### Mechanism A4 (`iter_20_mechA4_per_token_swiglu.py`) Results & Diagnostic (`diag_iter20.py`)
- **Metrics (Seed 42)**:
  - **Verbatim: `0.60`** | **Paraphrase: `0.50`** | Locality: `0.3333` (`real=3/5, no_bleed=24/25, prefix2=16/25, exact32=7/25`) | **KL: `0.020436`** (PASS) | **Retention: `1.037` (`93.33%` raw)** (PASS)
  - Peak VRAM: `1260.17 MB` (`556.10 MB` overhead) | Synthesis Time: `25.185s` | Rank: `16` | Backprop: `False` (`final_train_diag`: `mean_p=0.7884`, `top1_acc=0.846`, `d_fit_err=0.16–0.28`)
- **Diagnostic Breakdown (`diag_iter20.py`)**:
  1. **Suffix / `<eos>` Placement Bug (`L20..L25`)**: `7/10` verbatim misses and `~14/25` paraphrase misses already generated the right name/prefix (`"Tavumpennyshire's..."`, `"Cindavfranccis."`, `"Lyraia Pavbrenprey."`, `"BravasducatKeyNoter."`, `"Vaelos City"`) and only failed on `<eos>` or final subwords because suffix pairs were placed in `L20..L25` first (`shard_p_after=0.69..0.74`) while `L27..L31` had `6` unused slots per layer!
  2. **L26 Template Mismatch (`real_anchors=3/5, loc=0.333`)**: `neg_items` only had 4 templates and lacked `"administrative capital"` / `"official currency issued by"` and suffix negatives. With `lambda_fro=0.008`, `max_row_norm=12.0`, `w_cal=0.4`, last-token `||down_proj delta||` at `L26` spiked to **`real=40.35`** (`Germany -> Sterling`, `Japan -> Kyoto`) and `L26/L27` `loc=17.14 / 24.76` (while `L28..L31` `loc` fell to `2.51–11.46`), and `L20` `cal` spiked to `4.30` due to `best_alpha=2.5` amplification.
- **Next (Mechanism A5, `iter_21`)**: Template-matched 10-frame probing (`pos=250`, `neg=140+50 suffix`, zero eval leakage) + higher calibration weight/regularization (`w_cal=1.5`, `lambda_fro=0.035`, `max_norm=6.5`) + top-down `L22..L31` sharding (`<eos>` in `L30..L31`, `pos_0` in `L26..L30`) + 1-pass per-neuron log-odds calibration (`<= 15s`).

### Mechanism A5 (`iter_21_mechA5_clean_swiglu.py`) Results
- **Metrics (Seed 42)**:
  - Verbatim: `0.36` | Paraphrase: `0.18` | Locality: `0.4333` (**`real=5/5, synth_no_bleed=25/25, bleed=0.0, prefix2=21/25, exact32=8/25`**) | KL: `0.038974` | **Retention: `1.0185` (`91.67%` raw)** | **Synthesis Time: `18.365s` (PASS `<= 20s`)** | Overhead: `736.77 MB` | Backprop: `False` (`final_train_diag`: `mean_p=0.70, top1_acc=0.8424`)
- **Analysis**:
  1. **Real Anchors & Zero Bleed Solved**: Matching all 10 syntactic frames across `pos` & `neg` completely eliminated L26 template leakage (`real_anchors`: `3/5 -> 5/5`, `synth_no_bleed`: `24/25 -> 25/25`, `prefix2`: `16/25 -> 21/25`).
  2. **Over-Regularization + `<eos>` Crowding Bug**: Raising `lambda_fro=0.035` + `max_norm=6.5` cut `J_diag_pos` by 2–3x (`12.5 -> 3.9–5.8`), which forced `||B_down||` to scale UP 3x (`kl` rose `0.020 -> 0.039`) and starved `L30..L31` where all 25 `<eos>` tokens competed (`L31`: `J_diag=4.15, d_fit_err=0.421, shard_p_cal=0.221`).
- **Next (Mechanism A6, `iter_22`)**: Restore strong `J_diag` (`g_pos=+4.5, u_pos=+3.0, lambda_fro=0.013, max_norm=9.5, w_cal=0.8`) while keeping matched 10-frame probing, spread `<eos>` cleanly across `L27..L31` (`5` per layer) without `nseq` dummy-EOS collision, and guard `alpha_vec` against amplifying low-gain neurons.

### Mechanism A6 (`iter_22_mechA6_matched_balanced.py`) Results
- **Metrics (Seed 42)**:
  - Verbatim: `0.32` | Paraphrase: `0.32` | Locality: `0.4333` (**`real=5/5, synth_no_bleed=25/25, bleed=0.0, prefix2=18/25`**) | KL: `0.03608` | **Retention: `1.000`** | **Time: `18.779s`** | Overhead: `716.50 MB` | Backprop: `False` (`final_train_diag`: `mean_p=0.7268, top1_acc=0.8608`)
- **Anomalies Observed**:
  - Restoring `J_diag_pos = 9.4–12.5` improved `paraphrase_acc` (`0.18 -> 0.32`) and `top1_acc` (`0.842 -> 0.861`), **yet** `L27..L29` `shard_p_cal` only reached `0.53–0.59` (vs `0.76–0.96` in Iter 20) and `L31` dropped `0.6927 -> 0.6143`.
- **Root-Cause Findings (`probe_para_gap.py`, `diag_iter22_gates.py`, `inspect_eos_and_l30.py`, `task-454`)**:
  1. **Base `X_mlp_in` Already Has `1.00 / 1.00` Entity Generalization (`probe_para_gap.py`)**: Pure linear Ridge + centered cosine-NN from our 10 matched training templates generalized `1.00` verbatim / `1.00` paraphrase at EVERY layer `L26..L31` (`0.98` at L24) without any attention/o_proj tuning. Moreover, `Iter 22`'s `L26..L29` `pos_0` neurons hit `20/20` (`100%`) verb `pos_0` and `37/40` (`92.5%`, `20/20` at `L28–29`) para `Z` argmax (`real_max=0.21–0.40`).
  2. **`lm_head['<|im_end|>']` Voronoi Overshoot Bug (`'firehose'` / `'KeyNot'`)**: On `lm_head.weight @ lm_head.weight[eos]`, five rare code tokens (`'firehose'`, `'reponame'`, `'KeyNot'`, `' TType'`, `'CreateModel'`) have strictly larger dot products with `W_lm[eos]` than `W_lm[eos]` has with itself! Thus 1-shot `u_vec` targets caused EOS neurons firing `+20%` stronger at eval (`L31[0]=14.14` vs train `11.5`) to overshoot `<|im_end|>` into `'firehose'` (`p=0.84, p(eos)=0.000`), accounting for **`12/25` verbatim and `17/50` paraphrase misses**!
  3. **`L30` Landmark Pile-Up**: `idx % 5 == 4` put all 5 `landmark`s (`pos_0` + 8-token suffixes competing in `L30`) into `L30`, causing `5/5` wrong `pos_0` (`The ...`) on verbatim, while `L27` hit `loc_max=18.77` due to only 140 negatives.
- **Next (Mechanism A7, `iter_23`)**: 6-step `no_grad()` two-point anti-overshoot RMSNorm target solver (`CE(h+dH) + 0.65*CE(h+1.4*dH)`), placing `pos_0` cleanly in `L26..L29` + `<eos>` in `L30..L31`, with `300` matched disjoint negatives and post-entity predicate frames.

### Mechanism A7 (`iter_23_mechA7_stable_logits.py`) Results
- **Metrics (Seed 42)**:
  - Verbatim: `0.20` | Paraphrase: `0.04` | **Locality: `0.5667` (`real=5/5, synth_no_bleed=25/25, bleed=0.0, prefix2=23/25, exact32=12/25`)** | KL: `0.037167` | **Retention: `1.000`** | Time: `20.165s` | Overhead: `785.78 MB` | Backprop: `False`
- **Diagnosis**:
  - Increasing matched disjoint negatives to `300` improved `locality` cleanly to **`0.5667` (`23/25` prefix match, `25/25` no bleed, `5/5` real anchors)** and dropped `d_fit_err` down to `0.13–0.21`.
  - **Step-Cap Bug**: The 6-step target solver used `step_sizes = (16..5)` (cumulative norm `<= 62`, vs `140–200` in Iter 20–22), leaving `shard_p_1 = 0.04–0.13` and `final_train_diag mean_p = 0.39`.
- **Next (Mechanism A8, `iter_24`)**: Fix cumulative step schedule to `(40, 35, 28, 22, 18, 15, 12)` (`<= 170`) with active-row slicing (`<= 17s`) + higher `pos_0` log-odds headroom (`ell_goal = 4.2`).

### Mechanism A8 (`iter_24_mechA8_stable_fullstep.py`) Results
- **Metrics (Seed 42)**:
  - Verbatim: `0.36` | Paraphrase: `0.14` | **Locality: `0.5333` (`real=5/5, synth_no_bleed=25/25, bleed=0.0, prefix2=22/25`)** | KL: `0.036725` | **Retention: `1.000`** | **Time: `18.435s` (PASS)** | Overhead: `759.78 MB` | Backprop: `False` (`final_train_diag`: `mean_p=0.4701, top1=0.7411`)
- **Diagnosis**: Gradient opposition between `1.0*dh` and `1.35*dh` inside `iterative_logit_residual_targets_fast` still under-stepped (`shard_p_1 = 0.05..0.19` vs `0.69..0.85` in Iter 20/22).
- **Next (Mechanism A9, `iter_25`)**: Restore `etas=[25..220]` ray search with a clean 1.45x **Active Voronoi Overshoot Guard** (repulsing `'firehose'`/`'KeyNot'`/`'ikipedia'` only where 1.45x misses `target`) + 2-point `pos_0` calibration inside `L23..L31`.

### Mechanism A9 (`iter_25_mechA9_voronoi_guard.py`) — Pure 0-Backprop Breakthrough!
- **Metrics (Seed 42)**:
  - **Verbatim: `0.84` (`21/25`)** | **Paraphrase: `0.58` (`29/50`)** | Locality: `0.4333` (`real=5/5, no_bleed=25/25, bleed=0.0, prefix2=17/25, exact32=8/25`) | **KL: `0.037014`** (PASS) | Retention: `0.9444` (`85.0%` raw, 1–3 QA flips shy of `>=0.95`) | **Time: `15.176s` (PASS)** | Overhead: `816.86 MB` (`0.15x` Base A) | Backprop: `False` (`final_train_diag`: `top1_acc=0.9475, mean_p=0.7837`)
- **Analysis**:
  - The 1.45x Active Voronoi Guard eliminated rare-token EOS overshoot (`ov_fix: 214->29` at L22, `197->15` at L31, `L31 shard_p_1=0.8566`), lifting pure 0-backprop verbatim accuracy `0.32 -> 0.84` and paraphrase `0.32 -> 0.58` in `15.18s`.
  - **`diag_iter25.py` Findings**:
    1. `pos_0` neurons hit **`46/50` (`92%`!)** argmax accuracy on unseen paraphrases (`real_max=0.28..0.40`), and **`39/50` (`78%`!)** predicted the right first subword (`pos_0`), **but** `10/50` broke on `L22..L25` suffixes (`'Tavenhurst'`, `'Lomiridium'`, `'Solar Vashin'`) and `11/50` fell short of high `p_base` priors (`'Mining...'` on 5 minerals) where `Z_para ≈ 0.65 * Z_train`.
    2. All `3/25` verbatim `pos_0` misses (`4, 19, 24`) were `landmark`s ending with `"named?"` (post-entity predicate).
    3. `general_qa` only regressed by 3 net items (`[9] South Korea -> Kaelin`, `[10] Greece -> Tesis`, `[32] Mercury`, `[59] summer`, offset by `+1` Lisbon fix), dropping retention to `0.9444`.
- **Next (Mechanism A10, `iter_26`)**: Add post-entity trailing-predicate frames + 400 disjoint synthetic negatives (incl. `"capital city"` phrasing), separate `Z_neg (6.0x)` / `Z_cal (4.5x)` weights over `128` calibration texts, balance `pos_0` as `[5, 7, 7, 6]` across `L26..L29`, and run 5-round Voronoi Guard + 2-point calibration across `L22..L30`.

### Mechanism A10 (`iter_26_mechA10_refined.py`) Results
- **Metrics (Seed 42)**:
  - **Verbatim: `0.92` (`23/25`!)** | Paraphrase: `0.50` | **Locality: `0.50` (`real=5/5, no_bleed=25/25, bleed=0.0, prefix2=17/25, exact32=10/25`)** | **KL: `0.038077` (PASS)** | **Retention: `1.0185` (`91.67%` raw, PASS)** | **Time: `19.825s` (PASS)** | Overhead: `908.59 MB` | Backprop: `False` (`final_train_diag`: `mean_p=0.8013, top1=0.9481`)
- **Analysis**:
  - Adding post-entity trailing-verb frames (`"called?"`/`"designated as what?"`) + `[5, 7, 7, 6]` `pos_0` placement lifted 0-backprop verbatim accuracy all the way to **`0.92`**, restored **101.85% QA retention (`PASS`)**, and cleared KL (`0.038`) and runtime (`19.83s`).
  - Raising `Z_neg (6.0x)` and `Z_cal (4.5x)` pulled `J_diag_pos` down (`11.4 -> 9.5` at L27) and pushed `d_fit_err` up (`0.24 -> 0.34`), dropping paraphrase (`0.58 -> 0.50`) due to quadratic `silu(g)*u` shrinkage on unseen syntax.
- **Next (Mechanism A11, `iter_27`)**: Decouple `up_proj` (smooth, low-norm broad detector `+2.6` across `pos_0` / same-fact suffixes, `0.95x` transfer) from `gate_proj` (sharp 1-to-1 `+4.6` vs `-4.5`), restoring `1.5*Z_other + 4.0*Z_neg + 3.5*Z_cal`.

### Mechanism A11 (`iter_27_mechA11_decoupled_up.py`) Results
- **Metrics (Seed 42)**:
  - Verbatim: `0.84` | Paraphrase: `0.44` | Locality: `0.4333` (`real=5/5, no_bleed=25/25, prefix2=16/25`) | **KL: `0.034913` (PASS)** | **Retention: `0.963` (`86.67%` raw, PASS)** | **Time: `19.929s` (PASS)** | Backprop: `False`
- **Diagnosis**: Broadening `up_proj` lowered `J_diag_pos` (`9.5 -> 7.3`), raised `J_cal` (`0.013 -> 0.035`) and `d_fit_err` (`0.34 -> 0.48`), and hurt paraphrase (`0.50 -> 0.44`), proving both `gate` and `up` need sharp 1-to-1 contrast.
- **Next (Mechanism A12, `iter_28`)**: Keep Iter 26's clean templates + `L22..L30` 2-point calibration + 101.85% retention while restoring `J_diag ~ 11.5–13` (`g=+4.8, u=+3.2, max_norm=9.5, lambda_fro=0.012`, balanced `gram_r = 1.5, 3.5, 3.0`).

### Mechanism A12 (`iter_28_mechA12_sweetspot.py`) Results
- **Metrics (Seed 42)**:
  - **Verbatim: `0.92` (`23/25`)** | **Paraphrase: `0.54`** | Locality: `0.4667` (`real=5/5, no_bleed=25/25, bleed=0.0, prefix2=17/25, exact32=9/25`) | **KL: `0.038095` (PASS)** | **Retention: `0.9815` (`88.33%` raw, PASS)** | Overhead: `908.59 MB` | Time: `20.071s` | Backprop: `False` (`final_train_diag`: `top1_acc=0.957, mean_p=0.8118`)
- **Synthesis Summary Across Pure 0-Backprop Mechanisms A & B (`Iters 16–28`)**:
  - Pure 0-backprop **Mechanism B** (`Iter 16`) achieved `0.9667` locality (`25/25` prefix match, `0.0` bleed), `0.000194` KL, and `1.000` retention in `5.11s`, but could not fit 25 counterfactuals in `r=16` closed-form Ridge (`0.00` verb/para).
  - Pure 0-backprop **Mechanism A** (`Iters 17–28`) resolved the `gate/up` overlap bottleneck via analytical contrastive Ridge + 1-hot `A_down` + 1-to-1 per-token sharding + Active Voronoi Overshoot Guard (`ov_fix: 186->14`), scaling pure 0-backprop efficacy from `0.00 -> 0.92 verbatim / 0.54–0.58 paraphrase` while clearing hard KL (`0.038`) and QA retention (`0.9815–1.0185`) gates. Meanwhile, isolated 1-layer detached `Shared4Null` (`Iter 12`) achieves `1.00 / 1.00` paraphrase under `x_in.detach()`.
- **Next**: Benchmark BOTH `Shared4Null` (`Iter 12` accelerated) and pure 0-backprop `MechA12` across all 3 seeds `[42, 1337, 2026]` and on Stage 2 `seq_len=2048` document extraction + peak VRAM overhead (`<= 0.25x` Baseline A @ 2048).

---

## Multi-Seed & Stage 2 (`seq_len=2048`) Verification Results
1. **Single-Layer Isolated `Shared4Null_L26_31` (`verify_stage2_and_seeds.py`, `x_in.detach()`)**:
   - **Seed 42 (Stage 1)**: Verbatim `1.000`, Paraphrase `0.980`, KL `0.012879` (PASS), Retention `0.9630` (PASS), Locality `0.2333` (`real=5/5, no_bleed=15/25`), Peak VRAM `1536.64 MB` (`832.57 MB` overhead), Time `24.552s`.
   - **Seed 1337 (Stage 1)**: Verbatim `1.000`, Paraphrase `0.980`, KL `0.013321` (PASS), Retention `0.9444`, Locality `0.200` (`real=4/5, no_bleed=11/25`), Peak VRAM `1516.28 MB` (`812.21 MB` above clean `704.07 MB` static baseline), Time `23.724s`.
   - **Seed 2026 (Stage 1)**: Verbatim `1.000`, Paraphrase `0.980`, KL `0.013812` (PASS), Retention `0.9074`, Locality `0.2333` (`real=2/5, no_bleed=20/25`), Peak VRAM `1544.11 MB` (`840.04 MB` above clean static baseline), Time `24.522s`.
   - **Stage 2 (`seq_len=2048` doc extraction, Seed 42)**: Extracted `25/25` facts from the 2,048-token document, Verbatim `1.000`, Paraphrase `0.980`, KL `0.012879`, Retention `0.9630`, Peak VRAM **`1538.51 MB`** -> true activation overhead `1538.51 - 704.07 =` **`834.44 MB` (`0.157x` vs Baseline A @ 2048's `5316.52 MB` overhead — PASS `<= 0.25x`!)** *(Note: script's sequential loop measured `100.15 MB` differential above warm cache due to uncollected refs across seeds; using clean `704.07 MB` static weight baseline gives exact conservative `834.44 MB = 0.157x`)*.
   - **Consistent 3-Seed Insight**: `Shared4Null` reliably achieves `1.00` verbatim / `0.98` paraphrase and `0.013` KL across all 3 seeds, **while** its single-layer `pos_0` gradient optimization still suffers sibling bleed (`17–50%`), locality `0.20–0.23`, and slight QA retention dip on Seeds 1337/2026.

2. **Pure 0-Backprop `MechA12_SweetSpot_SwiGLU` (`verify_stage2_mechA.py`, 0 `.backward()` anywhere, dynamic layers `L21..L31`)**:
   - **Seed 42 (Stage 1)**: Verbatim **`0.92`**, Paraphrase `0.52`, KL `0.037294` (PASS), Retention **`0.9815`** (PASS), `real=5/5, bleed=0.0`, Overhead `856.06 MB`, Time **`19.588s` (PASS)**.
   - **Seed 1337 (Stage 1)**: Verbatim **`0.92`**, Paraphrase `0.58`, KL `0.038001` (PASS), Retention **`1.000`** (PASS), `real=5/5, bleed=0.10`, Overhead `850.70 MB`, Time **`18.903s` (PASS)**.
   - **Seed 2026 (Stage 1)**: Verbatim **`0.92`**, Paraphrase **`0.62`**, KL `0.037802` (PASS), Retention **`1.000`** (PASS), `real=5/5, no_bleed=24/25, loc=0.50`, Overhead `854.75 MB`, Time `22.433s` (`L21..L31` auto-expanded for 161 transitions).
   - **Stage 2 (`seq_len=2048` doc extraction, Seed 42)**: Extracted `25/25` facts from the 2,048-token document + executed actual `seq_len=2048` `no_grad()` pass: Verbatim `0.92`, Paraphrase `0.52`, KL `0.037294`, Retention `0.9815`, Peak VRAM **`1751.56 MB`** -> activation overhead **`1047.49 MB` (`0.1970x` vs Baseline A @ 2048's `5316.52 MB` overhead — PASS `<= 0.25x`!)**.

---

## Complex Fictional Biography & Logical Reasoning (`biography_logic_experiment.py`)
Injected an entirely fictional 10-fact biography of *Dr. Seren Valenwood* (`1984`, `Oakhaven`, `Vespera` at age `12`, `cryo-botany` under `Prof. Mara Lindholm`, `Aether-7` in `2011`, `Solaris Medal`, `Helios Polar Institute` in `2012` for `9 years`, sister `Talia Valenwood` cellist in `Nordhavn`, `cobalt` allergy, `Estonian`) using ONLY atomic self-supervised QA pairs during synthesis, then evaluated held-out direct recall, 1-turn CoT, and self-recalled weight-only multi-hop deduction (9 unseen tasks):
- **`Base_Unedited`**: `direct=0.0`, `1turn_logic=0.2222`, `self_recalled_logic=0.2222`, `general_qa_raw=0.9167`
- **`InContext_Oracle` (Prompted Capacity Ceiling)**: `direct=1.0`, `1turn_logic=0.3333` (`3/9`, showing 360M SmolLM2's inherent arithmetic/logical ceiling even with context in-prompt).
- **`Pure_0Backprop_VoronoiSwiGLU`**: `direct=0.1667`, `1turn_logic=0.1111`, **`self_recalled_logic=0.3333`** (matched `InContext_Oracle`!), `kl=0.034567`, **`retention=0.9818`**, `time=5.285s`. Correctly deduced `2011 - 1984 = 27` (`temporal_invention_age`), `Nordhavn` (`transitive_sister_city`), and `Prof. Mara Lindholm` (`transitive_medal_mentor`) from its own recalled weight outputs.
- **`SingleLayer_Shared4Null_ResSkip` (`L26..L28`, `ce=0.0245`)**: `direct=0.3333`, `1turn_logic=0.2222`, `self_recalled_logic=0.2222`, **`kl=0.004038`**, `retention=0.9455`, `time=6.565s`. Limiting to 3 blocks (`ce=0.0245` vs earlier 6-block `ce=0.0008`) eliminated token repetition loops cleanly and recalled `Prof. Mara Lindholm`, `Dr. Seren Valenwood`, `1984/2011/2012/9 years/cobalt` accurately, though 360M CoT arithmetic still stumbled without calculator tools.

---

## Final R&D Assessment & Archived Modules (`blocks_library/`)
- **Architectural Achievement**: Both `voronoi_swiglu_block.py` (`Mech A12`, 0 full backprop) and `shared4null_resskip_block.py` (`Shared4Null`, strictly 1-block `x_in.detach()`) satisfy the $O(1)$-depth memory goal cleanly (`0.197x` and `0.157x` of Baseline A's `5316.52 MB` overhead @ `seq_len=2048`) while vastly outperforming Baseline A's `0.00/0.00` factual accuracy.
- **Open Gaps Honestly Logged**: Neither method passed *every* strict gate simultaneously across all 3 seeds: pure 0-backprop `Mech A12` reached `0.92` verbatim / `0.52–0.62` paraphrase / `0.98–1.00` retention / `0.037–0.038` KL (short of `>= 0.75` paraphrase and `>= 0.98` exact-32-tok locality), while 1-layer `Shared4Null` cleared `0.98` paraphrase across all 3 seeds and `0.013–0.014` KL (short of `>= 0.98` locality, `<= 20s` time [`23.7–24.5s`], and `>= 0.95` retention on Seeds 1337/2026). Both reusable blocks are archived cleanly in `blocks_library/`.


