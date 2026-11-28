# Autonomous ML Research Agent #3 — Mechanistic Cartography & Universal O(1)-Memory Post-Training Lab Notebook

**Mission**: Eliminate benchmark overfitting and reward-hacking from earlier iterations (`iter_15`), map **where distinct cognitive skills live in transformer weights** across `SmolLM2-360M-Instruct` (`fp16`, 32L pure attention) and `ornith-ai/Ornith-1.0-9B` (`LLM.int8()` + `fp16` compute, 32L hybrid `GatedDeltaNet` + `full_attention`), and prove a universal O(1)-memory / 0-backprop post-training stack covering **SFT -> DPO -> Verifiable RL** on a single 15 GiB Tesla T4.

## Anti-Reward-Hacking & Complexity Invariants Verified
1. **Zero Hand-Curated Anchor Leakage**: Calibration (`cal_texts`, `C0`) samples strictly at random from `HuggingFaceTB/smol-smoltalk` train split (`load_smoltalk_disjoint`), completely disjoint from held-out KL (`test` split) and never inspecting `real_anchors`.
2. **Zero Regex Document Parsing**: All unstructured 1,500–2,048 token documents (`code_api`, `pop_culture`, `bio_logic`, held-out `kestrel_rpc`) are ingested exclusively via **Model Self-Extraction in `torch.no_grad()`** + domain-agnostic raw paragraph continuation distillation.
3. **Loss Simplicity Cap (`<= 4` terms)**: Zero token-type hacks (`digit_margin`, `vocab_margin`, word-class masks, refusal blacklists). Exact same code and hyperparameters run unmodified across all 3 primary domains and held-out Domain 4.
4. **Isolated 2s Executable Sandbox**: All code tasks (`veloxdb`, `nexflow`, `kestrel_rpc`, negative controls, real Python anchors) execute inside `subprocess.run(..., timeout=2.0)` against real mock implementations and `assert` unit tests (`universal_sandbox.py`).
5. **Concealed Held-Out Domain 4**: `kestrel_rpc` v3 + 15 unseen real anchors (`run_final_verification.py`) remain untouched until final verification after Milestone 4 freeze.

---

## Milestone 1: Functional Cartography & Baselines (Iters 0–3)

### Iteration 0: Universal Multi-Domain Baselines (`iter_0_universal_baselines.py`)

#### `SmolLM2-360M-Instruct` (`fp16`, Static VRAM = `690.13 MB`)

| Method | Seed | Code API (QA / Pass@1 / NegCtrl) | Pop Culture (Para / MH / RealPop / Bleed) | Bio Logic (Direct / Para / 1-Turn / Bleed) | Real Anchors (25) | Gen QA Ret | KL Div | Peak Ovh / Time |
|---|---:|---|---|---|---|---:|---:|---:|
| `Base_Unedited` | 42 | `0.000` / `0.000` / `0.500` | `0.000` / `0.000` / `0.400` / `0.000` | `0.000` / `0.000` / `0.000` / `0.000` | `0.600` (`0.60/0.60`) | `1.000` (`0.933`) | `0.000000` | `0 MB` / `0.0s` |
| `Base_Unedited` | 1337 | `0.000` / `0.000` / `0.500` | `0.000` / `0.000` / `0.400` / `0.000` | `0.000` / `0.000` / `0.000` / `0.000` | `0.600` | `1.000` | `0.000000` | `0 MB` / `0.0s` |
| `InContext_Oracle` | 42 | `0.625` / `0.000` / `0.333` | `0.900` / `0.500` / `0.300` / `0.000` | `1.000` / `0.958` / `0.222` / `0.000` | `0.720` | `0.571` (`0.533`) | `0.000000` | `0 MB` / `0.0s` |
| `InContext_Oracle` | 1337 | `0.625` / `0.000` / `0.333` | `1.000` / `0.500` / `0.400` / `0.000` | `0.917` / `0.833` / `0.000` / `0.000` | — | — | `0.000000` | `0 MB` / `0.0s` |
| `FullBackprop_Baseline` (`seq=2048`) | 42 | `0.000` / `0.000` / `0.333` | `0.000` / `0.000` / `0.500` / `0.000` | `0.000` / `0.000` / `0.000` / `0.000` | `0.520–0.600` | `0.9465–0.9643` | `0.0581–0.0722` | `+4086–4757 MB` / `5.7–7.1s` |

*Key `360M` Iter 0 takeaways*: Unedited `Base` has `0.0` prior across every synthetic domain; `InContext_Oracle` destroys general QA retention (`0.571`) and hurts standard Python negative controls (`0.50 -> 0.333`); `FullBackprop` directly on raw 2048-token documents spikes activation VRAM to `+4.09–4.76 GB` even on a 360M model, fails `KL <= 0.05` (`0.058–0.072`), degrades retention (`0.9465`), and yields `0.0` downstream QA/Code recall.

*(8-bit `Ornith-1.0-9B` Iteration 0 baselines running next...)*

---

### Iterations 1–3: Mechanistic Cartography (`cartography.py` -> `skill_cartography.json`)

Executed strictly under `torch.no_grad()` (`30.0s` / `836.8 MB` peak on `360M`; `283.5s` / `10,908.6 MB` peak = `+423.1 MB` overhead on `9B`).

#### Explicit Layer-to-Capability Map

| Capability / Axis | `SmolLM2-360M` (32L Pure Attn) | `Ornith-1.0-9B` (32L Hybrid `GatedDeltaNet` + `full_attention`) | Mechanistic Finding |
|---|---|---|---|
| **Factual Retrieval** | Mixer recovery peaks `L27, L29, L30`; MLP `L22, L25, L27, L29` | Mixer recovery peaks **`L27, L23, L19`** (`full_attention` = `+0.0577` vs `GatedDeltaNet` = `-0.0061`); MLP `L26, L27, L30` | Entity-to-value lookup flows through late **`full_attention`** mixers + upper MLP readout; `GatedDeltaNet` shows near-zero causal recovery. |
| **Code Syntax & API Binding** | Mixer recovery `L20, L24, L27, L29, L30`; ablation `L23, L29, L31` | Mixer recovery peaks **`L19, L27, L28, L23, L15`** (`full_attention` = `+0.0852` vs `GatedDeltaNet` = `-0.0028`); MLP `L14, L19, L27, L29` | API argument & method binding relies **30x** more strongly on `full_attention` (`L19, L23, L27`) than `GatedDeltaNet`. |
| **Multi-Hop Reasoning & CoT** | Mixer `L21, L27, L29`; MLP `L21, L23, L25, L27, L28`; CoT perturb breaks `L26–L31` | Mixer recovery **`L27, L19, L23`** (`full_attention` = `+0.0631` vs `GatedDeltaNet` = `+0.0073`); perturb breaks `L12–L15`; ablation `L19, L27, L31` | Mid-layer state (`L15` `full_attention` boundary) breaks CoT under noise, while `L19–L27` `full_attention` + MLPs route intermediate bridge tokens (`8.6x` `full` vs `linear`). |
| **Preference Axis (`v_pref`)** | Peaks `L14–L18` (`auc=0.96`, `cos=0.44`, steer `+0.223`) | Peaks **`L26`** (`auc=0.970`, `cos=0.488`) + mid-attn `L15` (`auc=0.870`), steer `+0.164` | Preference alignment separates cleanly in mid-to-late representation space. |
| **Refusal Axis (`v_refusal`)** | Peaks `L20–L23` (`peak=L21`, `auc=1.00`, `cos=0.933`, steer `+0.953`) | Peaks **`L18–L23`** (`peak=L20`, `auc=1.000`, `cos=0.912`, steer `+0.203`) | False-refusal direction is remarkably linear and stable (`cos > 0.91`) across `L18–L23` on both models! |

#### Surgical Layer Targets Derived from `skill_cartography.json`
- **`SmolLM2-360M`**: `sft_layers = [23, 25, 27, 29, 31]`, `dpo_layers = [14, 16, 18, 21]`, `rl_layers = [27, 29, 30, 31]`
- **`Ornith-1.0-9B`**: `sft_layers = [19, 23, 27, 31]`, `dpo_layers = [18, 20, 21, 26]`, `rl_layers = [15, 19, 23, 27]`
