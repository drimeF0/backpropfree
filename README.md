# Backprop-Free & $O(1)$-Depth Rank-$r \le 16$ LoRA Knowledge Injection (`SmolLM2-360M` $\rightarrow$ 8-bit `Ornith-1.0-9B`)

Репозиторий содержит:
1. [`research_orig/`](research_orig/) — завершённое исходное исследование на `SmolLM2-360M-Instruct` (`smollm_nobackprop_lab/`, `blocks_library/`, `lab_notebook.md`, `biography_logic_experiment.py`, `implementation_plan.md`, `walkthrough.md`).
2. [`ornith_nobackprop_lab/`](ornith_nobackprop_lab/) — перенос и масштабирование исследования на 8-битную гибридную модель **`ornith-ai/Ornith-1.0-9B`** (`Qwen3_5ForConditionalGeneration`, `bitsandbytes` `LLM.int8()` + fp16 compute) на Tesla T4 (15,360 MiB VRAM).
3. [`HANDOFF_NEXT_SESSION.md`](HANDOFF_NEXT_SESSION.md) — подробный технический мануал для продолжения в следующей сессии (окружение, решённые особенности `Qwen3.5` / 8-bit, дерево диагностики, рецепты кода для оставшихся скриптов и готовый промпт старта).

---

## Текущий статус (8-bit `ornith-ai/Ornith-1.0-9B` на T4 15 GiB)

**Фазы Setup, Phase 1 Anatomy и вся Iteration 0 Baselines полностью завершены.**

| Метод / Этап | Verbatim | Paraphrase | Locality | Bleed | Retention | KL Div | Peak VRAM | Act Overhead | Wall Time | Примечание |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Phase 1: Anatomy** | — | — | — | — | — | — | `10485.51 MB` | — | — | `voronoi eos_dot=0.6983` (0 выше); скачок на `L27 (full)` `ctx_p=0.115` $\rightarrow$ `L31` `0.5585` |
| `Baseline_0_Unedited` | `0.00` | `0.00` | `0.6333` | `0.00` | `1.00` | `0.00` | `10485.51 MB` | `0.0 MB` | `0.0s` | `general_qa_raw_acc=1.0` |
| `Baseline_C_InContext` | `1.00` | `1.00` | `0.1667` | `0.0667` | `1.00` | `0.00` | `10485.51 MB` | `0.0 MB` | `0.0s` | In-context портит `locality` и даёт bleed |
| `Baseline_A_5step_seq512` | `0.04` | `0.04` | `0.1667` | `0.1333` | `0.9833` | `0.1909` | `13728.17 MB` | `3242.66 MB` | `374.31s` | OOM без `grad_ckpt`; прогон при `bs=1, ckpt=True` |
| `Baseline_A_5step_seq2048` | — | — | — | — | — | — | `14597.44 MB` | `4111.93 MB` | — | **`oom_on_15gb_t4`** даже с `grad_ckpt=True, bs=1` |
| `Baseline_B_100step_seq64` | `0.84` | `0.84` | `0.1333` | `0.40` | `0.9667` | `1.0655` | `13719.98 MB` | `3234.47 MB` | `1374.98s` | OOM @ `bs=25, bs=5`; прогон `bs=2`; сильный bleed (`0.40`) и KL |

### Что запускается следующим (код уже в `ornith_nobackprop_lab/`)
1. `python3 -u iter_16_mechB_bounded_c2l.py` (заодно закэширует `C0` для `L26..L31`)
2. `python3 -u iter_28_mechA12_sweetspot.py` (True 0-Backprop Voronoi-Guarded SwiGLU)
3. `python3 -u iter_12_shared4null.py` (1-Layer Detached Shared4Null ResSkip `L26..L31`)
4. При необходимости — R&D тюнинг, затем Multi-Seed (`42, 1337, 2026`) + Stage 2 (`seq_len=2048`), `biography_logic_experiment.py` и `blocks_library/`. Подробности — в [`HANDOFF_NEXT_SESSION.md`](HANDOFF_NEXT_SESSION.md).
