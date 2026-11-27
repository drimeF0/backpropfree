# Backprop-Free & $O(1)$-Depth Rank-$r \le 16$ LoRA Knowledge Injection (`SmolLM2-360M` $\rightarrow$ 8-bit `Ornith-1.0-9B`)

Репозиторий содержит:
1. [`research_orig/`](research_orig/) — завершённое исходное исследование на `SmolLM2-360M-Instruct` (`smollm_nobackprop_lab/`, `blocks_library/`, `lab_notebook.md`, `biography_logic_experiment.py`, `implementation_plan.md`, `walkthrough.md`).
2. [`ornith_nobackprop_lab/`](ornith_nobackprop_lab/) — перенос и масштабирование исследования на 8-битную гибридную модель **`ornith-ai/Ornith-1.0-9B`** (`Qwen3_5ForConditionalGeneration`, `bitsandbytes` `LLM.int8()` + fp16 compute) на Tesla T4 (15,360 MiB VRAM).
3. [`HANDOFF_NEXT_SESSION.md`](HANDOFF_NEXT_SESSION.md) — подробная инструкция для продолжения работы в следующей сессии (окружение, решённые архитектурные особенности `Qwen3.5` / 8-bit, актуальные метрики и пошаговый чеклист).

---

## Текущий статус (8-bit `ornith-ai/Ornith-1.0-9B` на T4 15 GiB)

| Этап | Статус | Ключевые зафиксированные факты / метрики |
| :--- | :--- | :--- |
| **Setup & 8-bit Loader** | Готово | `load_in_8bit=True`, удаление unused `visual` tower: `10,932.63 MB` $\rightarrow$ `10,485.51 MB` VRAM (`~4.87 GB` свободно) |
| **Phase 1: Model Anatomy** | Готово ([`anatomy_report.json`](ornith_nobackprop_lab/anatomy_report.json)) | Voronoi cone (`<\|im_end\|>`, id `248046`): `eos_self_dot=0.6983`, соседей выше нет (`0`). Резкий скачок сигнала на `L27 (full_attn)` `ctx_p=0.1150, rank=2.0` $\rightarrow$ `L31 (full_attn)` `ctx_p=0.5585, rank=1.5` |
| **Iter 0: Baseline 0** | Готово | `verbatim=0.0`, `paraphrase=0.0`, `locality=0.6333`, `retention=1.0`, `general_qa_raw=1.0`, `vram=10,485.51 MB` |
| **Iter 0: Baseline C (In-Context)** | Готово | `verbatim=1.0`, `paraphrase=1.0`, `locality=0.1667`, `locality_bleed=0.0667`, `retention=1.0` |
| **Iter 0: Baseline A (`seq=512`, 5 steps)** | Готово | OOM без grad checkpointing; с `grad_ckpt=True, bs=1`: `verbatim=0.04`, `paraphrase=0.04`, `locality=0.1667`, `kl=0.1909`, `bleed=0.1333`, `peak=13,728.17 MB` (`+3,242.66 MB`), `wall=374.31s` |
| **Iter 0: Baseline A (`seq=2048`, 5 steps)** | Зафиксирован OOM | `status="oom_on_15gb_t4"` даже при `grad_ckpt=True, bs=1` (`peak_oom=14,597.44 MB`, `+4,111.93 MB`) |
| **Iter 0: Baseline B (`100` steps, `seq=64`)** | В процессе при сохранении | `bs=25` и `bs=5` без grad_ckpt вызвали OOM; при сохранении шла итерация `bs=2` (`step 40/100`, `loss=0.0880`) |
| **Mech B (`iter_16`), Mech A12 (`iter_28`), Shared4Null (`iter_12`)** | Код готов к запуску | Адаптированы под 8-bit `Linear8bitLt`, `Qwen3_5RMSNorm`, гибрид `GatedDeltaNet` + `full_attention`, low-rank `svd_lowrank(q=512)` для `d=12288` |
| **Multi-Seed + Stage 2 (`2048` tok) + Bio Logic + `blocks_library/`** | Ожидает выполнения | См. [`HANDOFF_NEXT_SESSION.md`](HANDOFF_NEXT_SESSION.md) |
