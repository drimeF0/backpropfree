# Handoff & Инструкции для следующей сессии (`Ornith-1.0-9B` 8-bit Backprop-Free Research)

Этот файл содержит весь контекст, уже проверенные экспериментальные результаты и точный план действий, чтобы новая сессия могла сразу продолжить исследование без повторного поиска граблей.

---

## 1. Цель исследования и жёсткие ворота приёмки (Gates)

Повторить исследование бесбэкпропной и $O(1)$-глубинной LoRA-инъекции знаний ($r \le 16$) и многошагового логического вывода (`SmolLM2-360M-Instruct` из [`research_orig/`](research_orig/)) на более крупной модели **`ornith-ai/Ornith-1.0-9B`** в **8-битном квантовании** (`bitsandbytes` `LLM.int8()`) на Tesla T4 (15,360 MiB VRAM).

Ворота успеха ([`ornith_nobackprop_lab/eval_harness.py`](ornith_nobackprop_lab/eval_harness.py)):
- `paraphrase_acc >= 0.75` (50 невиданных перефразов для 25 фактов в batch)
- `kl_div <= 0.05` (на 64 отложенных текстах `smol-smoltalk`)
- `retention_acc >= 0.95` (общие знания, измеренные на 20 вопросах)
- `locality_acc >= 0.98` (устойчивость на 15+ соседних / несвязанных пробах)
- `wall_time_sec <= 20.0` (время синтеза адаптера)
- `activation_vram_overhead_mb <= 0.25 * Baseline_A` (особый акцент на Stage 2 `seq_len=2048`, где Baseline A @ 2048 полностью падает по OOM на T4!)
- `uses_full_backprop == False` (либо чистый `no_grad`, либо строгая 1-слойная изоляция `x_in.detach()`) и `rank <= 16`.

---

## 2. Критически важные технические находки (НЕ наступать повторно!)

1. **Окружение / Зависимости**:
   - Рабочая база: `torch 2.14.1+cu130`, `transformers 5.22.1`, `accelerate 1.15.0`, `peft 0.21.2`, `datasets 4.8.5`, `huggingface_hub 1.33.0`, **`bitsandbytes 0.50.2`**.
   - **КРИТИЧНО — НЕ СТАВИТЬ `flash-linear-attention` / `fla-core` на Tesla T4 (SM 7.5)!** Triton-ядра `fla` зависают при JIT-компиляции на T4. Нативные PyTorch fallback-реализации `chunk_gated_delta_rule` / `causal_conv1d_fn` в `transformers` работают стабильно.
   - Перед запуском на свежем инстансе:
     ```bash
     pip install -q bitsandbytes transformers accelerate datasets
     ```

2. **Конфигурация 8-bit `ornith-ai/Ornith-1.0-9B` (`Qwen3_5ForConditionalGeneration`)**:
   - Текстовый бэкбон: `hidden_size=4096`, `intermediate_size=12288`, `num_hidden_layers=32`, `vocab_size=248320`, `tie_word_embeddings=false`, `eos_token="<|im_end|>"` (id `248046`), `pad_token="<|endoftext|>"` (id `248044`).
   - Гибридные типы слоёв:
     - `full_attention` (`Qwen3_5Attention`, `num_attention_heads=16`, `num_key_value_heads=4`, `head_dim=256`, `attn_output_gate=true`): индексы `[3, 7, 11, 15, 19, 23, 27, 31]`.
     - `linear_attention` (`Qwen3_5GatedDeltaNet`): все остальные 24 слоя.
   - **Удаление `model.model.visual` в `load_ornith_8bit`**: снижает статический отпечаток VRAM с `10,932.63 MB` до **`10,485.51 MB`** (~`4.87 GB` свободной VRAM на T4 15 GiB).

3. **Zero-centered `Qwen3_5RMSNorm`**:
   - В `Qwen3_5RMSNorm` эффективный масштаб равен **`1.0 + norm.weight.float()`**, а НЕ `norm.weight.float()`!
   - Если вызывается сам модуль `lm_norm(x)`, масштабирование уже точное; при ручном расчёте градиента логитов (`voronoi_guarded_targets_v2` в [`iter_28_mechA12_sweetspot.py`](ornith_nobackprop_lab/iter_28_mechA12_sweetspot.py)) обязательно использовать `gamma = 1.0 + norm_layer.weight.float()`.

4. **Большой словарь (`vocab=248,320`) и ограничение RAM/VRAM (15 GB VRAM, ~12 GB RAM, 0 swap)**:
   - `lm_head.weight` (`[248320, 4096]`, `fp16`) весит `2.03 GB`. **НИКОГДА не кастить весь `lm_head.weight` в `.float()`** (`4.07 GB` $\rightarrow$ немедленный OOM!). Работать напрямую в `fp16` с чанкованием по токенам (`tok_chunk <= 96`, `compute_heldout_kl` с `batch_size=2`).
   - Деквантизация 8-bit `Linear8bitLt`: делать только по выбранным строкам $J_l$ ($r \le 16$) через `model_loader.dequantize_rows(base, J_l)` (`weight.data[J].float() * (SCB[J].float().unsqueeze(1) / 127.0)`). Примечание: в новых версиях `bitsandbytes` после первого forward вектор `SCB` может перемещаться в `linear.state.SCB` — `get_linear8bit_scb` в [`model_loader.py`](ornith_nobackprop_lab/model_loader.py) проверяет оба места.
   - **Спектральное предобуславливание $d=12288$ (`down_proj`)**: плотный `torch.linalg.eigh(12288)` падает с `2.26 GB` cuSOLVER OOM на GPU рядом с 9B моделью. Поэтому [`nullspace_cache.py`](ornith_nobackprop_lab/nullspace_cache.py) и [`iter_12_shared4null.py`](ornith_nobackprop_lab/iter_12_shared4null.py) используют `eigh(4096)` для 4096-мерных пространств и `torch.svd_lowrank(..., q=512, niter=2)` для `down_proj` (корректно, так как на `L31` уже `11,918 / 12,288` собственных значений $< 10^{-3}\lambda_{\max}$). Матрицы `C0` кешируются в `fp16` в `c0_cache_L23_31/`.

5. **Шаблон чата (`chat_template.jinja`)**:
   - По умолчанию модель вставляет `<think>\n` и генерирует длинные цепочки рассуждений до таймаута на коротких QA-задачах.
   - Для коротких фактических вопросов/проб обязательно передавать `enable_thinking=False` (уже вшито в `eval_harness.py`, `probes.py`, `iter_16`, `iter_12`).
   - В `biography_logic_experiment.py` полезно протестировать оба режима (`enable_thinking=False` для прямого recall и короткого CoT, плюс проверить `enable_thinking=True` при увеличенном `max_new_tokens` на многошаговой логике).

---

## 3. Что УЖЕ сделано и подтверждено логами

### А. Инфраструктура `ornith_nobackprop_lab/`
- [`model_loader.py`](ornith_nobackprop_lab/model_loader.py) — загрузка 8-bit `Ornith-1.0-9B`, `dequantize_rows`, `prepare_layer_inputs`, `step_layer`, `step_token_mixer` (проверено: `step_layer` побитово/численно совпадает с полным `Qwen3_5TextModel`, `max_diff=0.0`).
- [`guards.py`](ornith_nobackprop_lab/guards.py) — `NoBackpropGuard`, `BaselineBackpropProfiler`, `LoRALinear` (fp16 compute поверх `Linear8bitLt`), `LoRAManager`.
- [`benchmark.py`](ornith_nobackprop_lab/benchmark.py) — генератор синтетических фактов/Stage-2 документов (2048 токенов), закэшированные `smol-smoltalk` сплиты в [`smoltalk_cache.json`](ornith_nobackprop_lab/smoltalk_cache.json).
- [`eval_harness.py`](ornith_nobackprop_lab/eval_harness.py) — оценка `verbatim`, `paraphrase`, `locality`, `retention`, `kl_div` (chunked vocab), `check_gates`, запись в JSONL.
- [`probes.py`](ornith_nobackprop_lab/probes.py), [`nullspace_cache.py`](ornith_nobackprop_lab/nullspace_cache.py), [`dissect_ornith.py`](ornith_nobackprop_lab/dissect_ornith.py).
- Кандидатные солверы (готовы к запуску/тюнингу):
  - [`iter_16_mechB_bounded_c2l.py`](ornith_nobackprop_lab/iter_16_mechB_bounded_c2l.py) (True 0-Backprop, кеширует `C0` для `L26..L31`)
  - [`iter_28_mechA12_sweetspot.py`](ornith_nobackprop_lab/iter_28_mechA12_sweetspot.py) (True 0-Backprop Voronoi-Guarded Contrastive SwiGLU)
  - [`iter_12_shared4null.py`](ornith_nobackprop_lab/iter_12_shared4null.py) (1-Layer Detached `x_in.detach()` Shared-4-Space Null-Preconditioned ResSkip `L26..L31`)

### Б. Phase 1 Model Anatomy ([`anatomy_report.json`](ornith_nobackprop_lab/anatomy_report.json))
- **Unembedding Voronoi cone**: `<|im_end|>` (`id=248046`), `eos_self_dot=0.6983`, `num_tokens_dot_gt_eos_self=0` (ближайший сосед `"</"` имеет `dot=0.235`).
- **Локализация сигнала (решающие слои)**:
  - До `L26 (linear_attention)` целевой токен почти не разрешается: `L23 (full)`: `ctx_p=0.0014, rank=1030.2`; `L26 (line)`: `ctx_p=0.0005, rank=848.0`.
  - Критический скачок на **`L27 (full_attention)`**: `ctx_p=0.1150, rank=2.0`, `norm=124.82`, `mix=59.68, mlp=39.20`.
  - Рост до финала: `L28 (line)`: `ctx_p=0.2092`, `L29 (line)`: `ctx_p=0.2298`, `L30 (line)`: `ctx_p=0.3764, rank=1.5`, `L31 (full)`: `ctx_p=0.5585, rank=1.5`, `norm=247.34`, `mlp_l2_diff=182.43`.
- **Null-space capacity**: `mixer_out_proj` имеет `3262..3975 / 4096` направлений `< 1e-3 * lambda_max`; `mlp_in` — `3031..3946 / 4096`; `down_proj` — `8943 (L23) .. 11918 (L31) / 12288`.

### В. Iteration 0 Baselines ([`baselines_summary.json`](ornith_nobackprop_lab/baselines_summary.json), [`experiments.jsonl`](ornith_nobackprop_lab/experiments.jsonl))
- `Baseline_0_Unedited`: `verb=0.0`, `para=0.0`, `loc=0.6333`, `ret=1.0`, `general_qa_raw=1.0`, `bleed=0.0`, `peak_vram=10485.51 MB`.
- `Baseline_C_InContext`: `verb=1.0`, `para=1.0`, `loc=0.1667`, `bleed=0.0667`, `ret=1.0`.
- `Baseline_A_Backprop_5step_seq512`: `oom_without_grad_ckpt=True` (даже при `bs=1`!). С `grad_ckpt=True, bs=1`: `verb=0.04`, `para=0.04`, `loc=0.1667`, `kl=0.190887`, `ret=0.9833`, `bleed=0.1333`, `peak_vram=13728.17 MB`, `act_overhead=3242.66 MB`, `wall=374.311s`.
- `Baseline_A_Backprop_5step_seq2048`: **`status="oom_on_15gb_t4"`** — упал по OOM и без checkpointing, и **с `gradient_checkpointing=True, bs=1`** (`peak_oom_mb=14597.44 MB`, `+4111.93 MB` overhead)! То есть стандартный full-backprop LoRA при 2048 токенах на 15 GiB T4 даже на 8-битной модели принципиально не запускается.
- `Baseline_B_Backprop_100step` (`seq=64`): `bs=25` и `bs=5` без checkpointing упали по OOM; на момент создания этого коммита шёл прогон `bs=2, ckpt=False` (`step 40/100 loss=0.0880`).

---

## 4. Что делать дальше (пошаговый план для следующей сессии)

1. **Проверить окружение и состояние `baselines_summary.json`**:
   - Зайти в `ornith_nobackprop_lab/`. Если в `baselines_summary.json` нет ключа `"baseline_b"`, запустить:
     ```bash
     cd ornith_nobackprop_lab
     python3 -u iter_0_baselines.py
     ```
     (Скрипт автоматически пропускает уже готовые `baseline_0`, `baseline_c`, `baseline_a_512`, `baseline_a_2048` и досчитает только `Baseline_B_Backprop_100step`.)
   - Примечание по `baseline_a_ref` в `check_gates(...)`: в [`iter_16`](ornith_nobackprop_lab/iter_16_mechB_bounded_c2l.py), [`iter_28`](ornith_nobackprop_lab/iter_28_mechA12_sweetspot.py) и [`iter_12`](ornith_nobackprop_lab/iter_12_shared4null.py) для Stage 1 сравнивается с `baselines["baseline_a_512"]` (`+3242.66 MB`), а для Stage 2 (`2048` tok) можно дополнительно фиксировать `Baseline_A_2048` OOM потолок (`4111.93 MB` при краше) и сравнивать реальный overhead при `seq=2048` как с `3242.66 MB`, так и с `4111.93 MB`.

2. **Прогнать базовые кандидаты (Seed 42)**:
   ```bash
   python3 -u iter_16_mechB_bounded_c2l.py
   python3 -u iter_28_mechA12_sweetspot.py
   python3 -u iter_12_shared4null.py
   ```
   - `iter_16` закэширует ковариации `C0` в `c0_cache_L23_31/` для слоёв `26..31`, после чего `iter_12` сразу их переиспользует.
   - Внимательно проверить время синтеза (`wall_time_sec <= 20s`), `paraphrase_acc >= 0.75`, `kl_div <= 0.05`, `retention_acc >= 0.95`, `locality_acc >= 0.98`, и `layer_stats` / `locality_breakdown`.

3. **Диагностика и итеративный R&D тюнинг (если Seed 42 не проходит все ворота с первой попытки)**:
   - Ожидаемые точки внимания для `Ornith-1.0-9B`:
     - В **Mech A12 (`iter_28`)**: в `p0_order` сейчас слои `[26, 27, 28, 29]`, хотя по Phase 1 `L26` (linear_attn) имеет `ctx_p=0.0005`, а скачок начинается с `L27 (full_attn, ctx_p=0.1150)` $\rightarrow$ `L30 (line, 0.3764)`. Если `L26` плохо дотягивает сигнал до финала или `paraphrase_acc < 0.75`, протестировать:
       - смещение `pos_0` слоёв на `L27..L30` (или `L27, L28, L29, L30` + `eos` в `L31`),
       - тюнинг `etas`, `alpha_vec` диапазонов, `lambda_fro`, количества `probes` / негативов / калибровочных токенов (`wall_time_sec` ускорение за счёт сокращения лишних слоёв в `init_and_prefix` или размера батча).
       - Если `paraphrase_acc` уступает `verbatim_acc`, перенести диагностический подход из `research_orig/session_artifacts_and_logs/scratch/probe_para_gap.py` и `diag_iter25.py`.
     - В **Shared4Null (`iter_12`)**: проверить, не даёт ли `alpha_null=6.5` недообучение (`paraphrase_acc`) или лишний `bleed`/`kl_div` (тюнинг `alpha_null`, `steps_per_layer`, `L27..L31` против `L26..L31`, `lambda_cal_kl`, `lambda_antibleed`). По времени: в `Shared4Null` `wall_time_sec` на `SmolLM2-360M` был ~23s (превышал порог 20s) — обратить внимание, укладывается ли 9B версия или как ускорить (`cal_texts[:32]`, меньше шагов на ранних слоях `L26/L27`, если целевой порог достигается быстрее).
   - Все новые итерации (`iter_29_...py`, `iter_13_...py` и т.д.) записывать в `experiments.jsonl` и `lab_notebook.md`.

4. **Multi-Seed (`[42, 1337, 2026]`) + Stage 2 Long-Context (`seq_len=2048`) верификация**:
   - Создать (по образцу `research_orig/smollm_nobackprop_lab/verify_stage2_mechA.py` и `verify_stage2_and_seeds.py`):
     - `ornith_nobackprop_lab/verify_stage2_mechA.py` $\rightarrow$ `stage2_mechA_results.json`
     - `ornith_nobackprop_lab/verify_stage2_and_seeds.py` $\rightarrow$ `stage2_shared4null_results.json`
   - Совет по скорости на T4: загружать 8-bit модель **один раз** за скрипт и между сидами вызывать `mgr.uninstall()` / `lora_mgr.clear_all()` + `cleanup_cuda()` + `torch.cuda.reset_peak_memory_stats()` (и проверять, что `static_vram_mb` остаётся ~`10485.51 MB`), вместо тройной перезагрузки модели по ~92s каждый раз.
   - Для Stage 2 замерять: извлечение фактов из 2048-токенного документа + `seq_len=2048` forward в `no_grad()`, пиковое потребление VRAM и overhead относительно `Baseline_A_512` (`3242.66 MB`) и `Baseline_A_2048` OOM потолка (`4111.93 MB`).

5. **Complex Biography & Multi-Hop Logical Reasoning (`biography_logic_experiment.py`)**:
   - Адаптировать `research_orig/smollm_nobackprop_lab/biography_logic_experiment.py` (*Dr. Seren Valenwood*, атомарные пары + 6 direct + 9 multi-hop logic вопросов: одноходовый CoT и weight-only self-recalled CoT).
   - Сравнить: `Base_Unedited`, `InContext_Oracle`, лучший `Pure_0Backprop` (`Mech A`), лучший `1-Layer Detached` (`Shared4Null`).
   - Оценить `enable_thinking=False` vs `enable_thinking=True` (с увеличенным `max_new_tokens`, напр. 256–384) отдельно на multi-hop задачах!

6. **Финализация репозитория**:
   - Собрать [`ornith_nobackprop_lab/blocks_library/`](ornith_nobackprop_lab/blocks_library/) (`README.md`, `voronoi_swiglu_block.py`, `shared4null_resskip_block.py`) по аналогии с `research_orig/smollm_nobackprop_lab/blocks_library/`.
   - Дописать полный итоговый [`ornith_nobackprop_lab/lab_notebook.md`](ornith_nobackprop_lab/lab_notebook.md) со сквозным сравнением `SmolLM2-360M` vs 8-bit `Ornith-1.0-9B`.
   - Закоммитить и запушить все финальные JSON-артефакты и код в репозиторий.

---

## 5. Промпт для быстрого старта в следующей сессии

```text
Продолжи исследование из репозитория https://github.com/drimeF0/backpropfree :
1. Клонируй репозиторий и внимательно прочитай HANDOFF_NEXT_SESSION.md и ornith_nobackprop_lab/lab_notebook.md.
2. Установи bitsandbytes (без flash-linear-attention на T4!) и проверь ornith_nobackprop_lab/baselines_summary.json (досчитай Baseline_B при необходимости).
3. Прогони iter_16_mechB_bounded_c2l.py, iter_28_mechA12_sweetspot.py и iter_12_shared4null.py на Seed 42, проанализируй метрики и при необходимости проведи итеративные улучшения под 8-bit Ornith-1.0-9B.
4. Выполни 3-seed [42, 1337, 2026] + Stage 2 seq_len=2048 верификацию, эксперимент biography_logic_experiment.py (Dr. Seren Valenwood), собери ornith_nobackprop_lab/blocks_library/ и обнови lab_notebook.md.
5. Все финальные результаты закоммить и запушь обратно в репозиторий.
```
