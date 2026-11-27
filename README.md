# Backprop-Free & $O(1)$-Depth Rank-$r \le 16$ LoRA Knowledge Injection (`SmolLM2-360M` $\rightarrow$ 8-bit `Ornith-1.0-9B`)

Репозиторий содержит:
1. [`research_orig/`](research_orig/) — завершённое исходное исследование на `SmolLM2-360M-Instruct` (`smollm_nobackprop_lab/`, `blocks_library/`, `lab_notebook.md`, `biography_logic_experiment.py`, `implementation_plan.md`, `walkthrough.md`).
2. [`ornith_nobackprop_lab/`](ornith_nobackprop_lab/) — перенос и масштабирование исследования на 8-битную гибридную модель **`ornith-ai/Ornith-1.0-9B`** (`Qwen3_5ForConditionalGeneration`, `bitsandbytes` `LLM.int8()` + fp16 compute) на Tesla T4 (15,360 MiB VRAM).
3. [`HANDOFF_NEXT_SESSION.md`](HANDOFF_NEXT_SESSION.md) — подробный технический мануал для продолжения в следующей сессии (окружение, решённые особенности `Qwen3.5` / 8-bit, дерево диагностики, рецепты кода для оставшихся скриптов и готовый промпт старта).

---

## Текущий статус (8-bit `ornith-ai/Ornith-1.0-9B` на T4 15 GiB)

**Завершены:** Setup, Phase 1 Anatomy, Iteration 0 Baselines и все 5 механизмов Iteration 1 на Seed 42 (`Iter 16/28/29/12/30`). Открыт multi-seed/Stage 2 и биографический эксперимент.

| Метод / Этап | Verbatim | Paraphrase | Locality | Bleed | Retention | KL Div | Peak VRAM | Act Overhead | Wall Time | Примечание |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phase 1: Anatomy** | — | — | — | — | — | — | `10485.51 MB` | — | — | `voronoi eos_dot=0.6983` (0 выше); скачок на `L27 (full)` `ctx_p=0.115` $\rightarrow$ `L31` `0.5585` |
| `Baseline_0_Unedited` | `0.00` | `0.00` | `0.6333`* | `0.00` | `1.00` | `0.00` | `10485.51 MB` | `0.0 MB` | `0.0s` | *артефакт разных batch size (исправлено) |
| `Baseline_C_InContext` | `1.00` | `1.00` | `0.1667` | `0.0667` | `1.00` | `0.00` | `10485.51 MB` | `0.0 MB` | `0.0s` | In-context портит `locality` и даёт bleed |
| `Baseline_A_5step_seq512` | `0.04` | `0.04` | `0.1667` | `0.1333` | `0.9833` | `0.1909` | `13728.17 MB` | `3242.66 MB` | `374.31s` | OOM без `grad_ckpt`; прогон при `bs=1, ckpt=True` |
| `Baseline_A_5step_seq2048` | — | — | — | — | — | — | `14597.44 MB` | `4111.93 MB` | — | **`oom_on_15gb_t4`** даже с `grad_ckpt=True, bs=1` |
| `Baseline_B_100step_seq64` | `0.84` | `0.84` | `0.1333` | `0.40` | `0.9667` | `1.0655` | `13719.98 MB` | `3234.47 MB` | `1374.98s` | OOM @ `bs=25, bs=5`; сильный bleed и KL |
| `Iter 16 MechB_C2L_L27_30` | `0.00` | `0.00` | `0.50` | `0.00` | `1.00` | `0.00203` | `12126.92 MB` | `1641.41 MB` | `271.84s` | fit `err 0.91..0.99`: rank-16 не представляет residual |
| `Iter 28 MechA12_SwiGLU` | `0.32` | `0.28` | `0.6333`* | `0.00` | `1.00` | `0.000813` | `12979.67 MB` | `2494.16 MB` | `168.15s` | train `top1=0.91`, но suffix decay |
| `Iter 29 MechA13_Pos0L27_30` | `0.28` | `0.36` | `0.8667` | `0.00` | `1.00` | `0.000814` | `12980.18 MB` | `2494.67 MB` | `174.14s` | лучший 0-backprop paraphrase |
| `Iter 12 Shared4Null_L26_31` | `0.20` | `0.10` | `0.00` | `0.10` | `0.0333` | `2.408684` | `12832.29 MB` | `2346.77 MB` | `296.00s` | SmolLM2-гиперпараметры разрушают 9B |
| `Iter 30 Shared4Null_Stable` | `0.64` | `0.56` | `0.30` | `0.30` | `0.9833` | `0.001514` | `12790.89 MB` | `2305.38 MB` | `223.37s` | лучший recall, но `30%` synthetic bleed |

\* Locality у `Baseline_0`, `Iter 16/28/12` измерена до исправления несоответствия batch size (`base_ref bs=10` vs eval `bs=12`); `Iter 29/30` — matched batch.

### Что дальше (код уже в `ornith_nobackprop_lab/`)
1. `python3 -u verify_stage2_mechA.py` — 3 сида + Stage 2 (`seq_len=2048`) для лучшего pure 0-backprop (`MechA13`).
2. `python3 -u verify_stage2_and_seeds.py` — 3 сида + Stage 2 для `Shared4Null-Stable`.
3. `biography_logic_experiment.py` — Dr. Seren Valenwood, direct recall + multi-hop CoT (`enable_thinking=False/True`).
4. `blocks_library/` + финальное сравнение `SmolLM2-360M` vs 8-bit `Ornith-1.0-9B`.

Подробности, диагностика и воспроизводимость — в [`ornith_nobackprop_lab/lab_notebook.md`](ornith_nobackprop_lab/lab_notebook.md) и [`HANDOFF_NEXT_SESSION.md`](HANDOFF_NEXT_SESSION.md).
