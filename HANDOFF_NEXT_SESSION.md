# Handoff & Подробные инструкции для следующей сессии (`Ornith-1.0-9B` 8-bit Research)

Этот документ содержит исчерпывающий контекст текущего исследования: все полученные на данный момент результаты (`Phase 1` + вся `Iteration 0 Baselines`), решённые проблемы 8-битного `Qwen3.5` на T4 15 GiB, диагностическое дерево для следующих экспериментов и подробный чеклист с шаблонами кода для завершения работы.

---

## 1. Быстрый старт на новом инстансе (Bootstrap)

```bash
git clone https://github.com/drimeF0/backpropfree.git
cd backpropfree/ornith_nobackprop_lab

# Проверить GPU и память (рассчитано под >=15 GB VRAM T4, ~12 GB RAM)
nvidia-smi
free -h

# Установить зависимости (НЕ ставить flash-linear-attention / fla-core на T4 SM 7.5!)
pip install -q bitsandbytes transformers accelerate datasets huggingface_hub peft
```

### Обязательные проверки перед запуском
1. **НЕ устанавливать `flash-linear-attention` / `fla-core` / `causal_conv1d`** на Tesla T4 (SM 7.5): Triton-ядра зависают при JIT-компиляции. Предупреждения `transformers` о fallback на PyTorch — **нормальное и ожидаемое поведение**.
2. Внутри `ornith_nobackprop_lab/` уже лежит [`smoltalk_cache.json`](ornith_nobackprop_lab/smoltalk_cache.json) (1.1 MB), [`anatomy_report.json`](ornith_nobackprop_lab/anatomy_report.json), [`baselines_summary.json`](ornith_nobackprop_lab/baselines_summary.json) и [`experiments.jsonl`](ornith_nobackprop_lab/experiments.jsonl). Повторно прогонять `dissect_ornith.py` и `iter_0_baselines.py` **не нужно**.
3. Загрузка `ornith-ai/Ornith-1.0-9B` в 8-bit с локального кэша занимает **~90–95 секунд** на процесс. Для мульти-сидовых скриптов загружайте модель один раз, а между прогонами вызывайте `mgr.uninstall()` / `lora_mgr.clear_all()` + `cleanup_cuda()`.

---

## 2. Целевые ворота приёмки (Gates в `eval_harness.py`)

Все методы должны соблюдать `NoBackpropGuard` (`uses_full_backprop == False`, бэкбон заморожен `requires_grad == False`, для `Shared4Null` разрешена только 1-слойная изоляция через `x_in.detach()`) и `rank <= 16`:
- `paraphrase_acc >= 0.75` (50 невиданных перефразов для 25 фактов)
- `kl_div <= 0.05` (прямой KL на 64 отложенных текстах `smol-smoltalk`)
- `retention_acc >= 0.95` (20 контрольных фактов общих знаний)
- `locality_acc >= 0.98` (15+ соседних / несвязанных промптов, проверка bleed)
- `wall_time_sec <= 20.0` (время синтеза адаптера; замеры отдельно от eval!)
- `activation_vram_overhead_mb <= 0.25 * Baseline_A_activation_overhead`

---

## 3. Проверенные результаты на 8-bit `ornith-ai/Ornith-1.0-9B`

### A. Память и 8-битная архитектура
- Статическая 8-битная модель (`BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["lm_head"])`, `dtype=torch.float16`):
  - С vision tower: `10,932.63 MB`
  - После удаления `model.model.visual`: **`10,485.51 MB`** (свободно ~`4.87 GB` на T4 15,360 MiB)
- `Qwen3_5ForConditionalGeneration`:
  - 32 слоя: `full_attention` на индексах `[3, 7, 11, 15, 19, 23, 27, 31]`, `linear_attention` (`Qwen3_5GatedDeltaNet`) на остальных 24 слоях.
  - `hidden_size=4096`, `intermediate_size=12288`, `vocab_size=248320`, `tie_word_embeddings=false`.
  - `eos_token="<|im_end|>"` (`id=248046`), `pad_token="<|endoftext|>"` (`id=248044`).

### B. Критические технические подводные камни (уже решены в коде)
1. **`Qwen3_5RMSNorm` is zero-centered**: реальный вес нормировки `1.0 + norm.weight.float()`, не `norm.weight.float()`. (В `iter_28_mechA12_sweetspot.py` уже исправлено.)
2. **`lm_head` `[248320, 4096]` = 2.03 GB в fp16**: нельзя вызывать `.float()` на всей матрице (`4.07 GB` $\rightarrow$ OOM). Используйте чанкование по токенам (`tok_chunk <= 96`, `compute_heldout_kl(..., batch_size=2)`).
3. **`Linear8bitLt` деквантизация строк**: `SCB` может лежать либо на `base.weight.SCB` (до первого forward), либо на `base.state.SCB` (после первого forward). В [`model_loader.py`](ornith_nobackprop_lab/model_loader.py) функция `dequantize_rows(base, J_l)` проверяет оба атрибута и декодирует только выбранные `r <= 16` строк.
4. **Предобуславливание $12288 \times 12288$ (`down_proj`)**: плотный `torch.linalg.eigh(12288)` падает на GPU (`2.26 GB` cuSOLVER OOM). Используйте `svd_lowrank(q=512, niter=2)` — уже реализовано в [`nullspace_cache.py`](ornith_nobackprop_lab/nullspace_cache.py) и [`iter_12_shared4null.py`](ornith_nobackprop_lab/iter_12_shared4null.py).
5. **`chat_template.jinja` и `<think>`**: для QA всегда `enable_thinking=False`. В `biography_logic_experiment.py` полезно дополнительно проверить `enable_thinking=True` с большим `max_new_tokens` на многошаговой логике.

### C. Phase 1 Model Anatomy ([`anatomy_report.json`](ornith_nobackprop_lab/anatomy_report.json))
- **Voronoi Cone**: `<|im_end|>` (`248046`), `eos_self_dot=0.6983`, `num_tokens_dot_gt_eos_self=0` (ближайший сосед `"</"` имеет `dot=0.235`).
- **Signal Localization (резкий скачок на последнем `full_attention` блоке перед выходом)**:
  - `L23 (full)`: `norm=75.21`, `res_diff=65.54`, `ctx_p=0.0014`, `rank=1030.2`
  - `L26 (line)`: `norm=124.72`, `res_diff=92.54`, `ctx_p=0.0005`, `rank=848.0`
  - `L27 (full)`: `norm=124.82`, `res_diff=116.58`, `mix=59.68`, `mlp=39.20`, **`ctx_p=0.1150, rank=2.0`**
  - `L28 (line)`: `norm=152.69`, `res_diff=134.32`, `ctx_p=0.2092`, `rank=1.8`
  - `L29 (line)`: `norm=161.43`, `res_diff=150.05`, `ctx_p=0.2298`, `rank=2.0`
  - `L30 (line)`: `norm=188.62`, `res_diff=175.69`, `mix=50.74`, `mlp=63.49`, `ctx_p=0.3764`, `rank=1.5`
  - `L31 (full)`: `norm=247.34`, `res_diff=283.81`, `mix=52.85`, `mlp=182.43`, **`ctx_p=0.5585, rank=1.5`**
- **Null-Space Capacity**: на поздних слоях в `mixer_out_proj` свободно `3262..3975 / 4096` направлений (`< 1e-3 * λ_max`), в `mlp_in` — `3031..3946 / 4096`, в `down_proj` — `8943 (L23) .. 11918 (L31) / 12288`.

### D. Iteration 0: Все 5 базовых ориентиров завершены ([`baselines_summary.json`](ornith_nobackprop_lab/baselines_summary.json))

| Baseline | Verbatim | Paraphrase | Locality | Bleed | Retention | KL Div | Peak VRAM | Act Overhead | Wall Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Baseline_0_Unedited` | `0.00` | `0.00` | `0.6333` | `0.00` | `1.00` | `0.00` | `10485.51 MB` | `0.0 MB` | `0.0s` |
| `Baseline_C_InContext` | `1.00` | `1.00` | `0.1667` | `0.0667` | `1.00` | `0.00` | `10485.51 MB` | `0.0 MB` | `0.0s` |
| `Baseline_A_Backprop_5step_seq512` (`bs=1, grad_ckpt=True`) | `0.04` | `0.04` | `0.1667` | `0.1333` | `0.9833` | `0.190887` | `13728.17 MB` | `3242.66 MB` | `374.31s` |
| `Baseline_A_Backprop_5step_seq2048` | — | — | — | — | — | — | `14597.44 MB` (OOM) | `4111.93 MB` (OOM) | — |
| `Baseline_B_Backprop_100step` (`bs=2, seq=64`) | `0.84` | `0.84` | `0.1333` | `0.40` | `0.9667` | `1.06553` | `13719.98 MB` | `3234.47 MB` | `1374.98s` |

Ключевые выводы по Iteration 0:
- Полный backprop при `seq_len=512` уже требует `gradient_checkpointing`, занимает `374.3s`, тратит `+3242.66 MB` активаций и даёт всего `0.04` accuracy при `KL=0.1909`.
- Полный backprop при `seq_len=2048` (`Baseline_A_2048`) **падает по OOM на T4 15 GiB даже при `gradient_checkpointing=True, bs=1`** (`14,597.44 MB`).
- Долгий 100-шаговый backprop (`Baseline_B`) занимает **`22.9 минуты`** (`1374.98s`), даёт `0.84` recall, но разрушает локальность (`0.1333`, **`40%` bleed**) и ломает распределение (`KL=1.0655`).

---

## 4. Пошаговый план дальнейшей работы (с рецептами и диагностикой)

### Шаг 1. Запустить 3 подготовленных механизма на Seed 42
```bash
cd ornith_nobackprop_lab
python3 -u iter_16_mechB_bounded_c2l.py
python3 -u iter_28_mechA12_sweetspot.py
python3 -u iter_12_shared4null.py
```
- `iter_16` закэширует `C0` (`L26..L31`, `mixer_out_proj` + `down_proj`) в `c0_cache_L23_31/`, а `iter_12` их сразу подхватит.
- Все 3 скрипта автоматически пишут результаты в `experiments.jsonl` и печатают `check_gates` + `print_locality_breakdown`.

### Шаг 2. Итеративный R&D тюнинг (если Seed 42 не проходит какой-то из гейтов)
Смотреть в первую очередь на следующие рычаги (особенности 9B гибрида):

1. **Если у `Mech A12 (iter_28)` проседает `paraphrase_acc < 0.75` или `ov_fix` показывает остаточные промахи**:
   - Проверьте `layer_stats` (`shard_p_1`, `shard_p_2`, `d_fit_err`, `J_diag_pos`, `J_neg`, `J_cal`).
   - В `iter_28_mechA12_sweetspot.py` сейчас `p0_order` включает слой `26` (`linear_attention`, где по Phase 1 `ctx_p=0.0005`), тогда как сильный скачок начинается с `L27 (full_attention)` (`ctx_p=0.1150`) $\rightarrow$ `L30` (`0.3764`). **Гипотеза `iter_29`**: перенести все `pos_0` шарды на `L27..L30` (убрать `L26` из `p0_order`), скорректировать `etas` под нормы остаточного потока (`L27=125 .. L31=247`), или подкрутить `target_prob_goal` / `alpha_vec` верхнюю границу.
   - Для быстрой диагностики провала перефразов перенесите логику из `../research_orig/session_artifacts_and_logs/scratch/probe_para_gap.py` и `diag_iter25.py` (показывает реальные сгенерированные ответы модели на провальных verbatim/paraphrase/locality примерах!).

2. **Если у `Mech A12` `wall_time_sec > 20s` или `kl_div > 0.05` / `bleed > 0.02`**:
   - Время: профилируйте, сколько уходит на `init_and_prefix` (`first_layer=26`, проходит через 26 PyTorch-фоллбэковых GatedDeltaNet/Attn слоёв) vs `voronoi_guarded_targets_v2`. Можно уменьшить `cal_texts[:96]` $\rightarrow$ `[:64]` и `max_length=72`, либо сузить `cand_layers` с `23..31` до `27..31` (чтобы `first_layer=27` пропускал сразу 27 слоёв, если 5*16=80 пар хватает; обычно уникальных пар `~65` $\rightarrow$ `27..31` вмещает до `80`!).
   - Bleed / KL: усильте штраф `w_neg` / `w_cal` (в `solve_contrastive_neuron_proj_sep` и в `gram_r`), либо зажмите `max_row_norm`.

3. **Если у `Shared4Null (iter_12)` `wall_time_sec > 20s`, `kl_div > 0.05`, или `paraphrase_acc < 0.75`**:
   - В `research_orig` на `SmolLM2-360M` `Shared4Null` брал ~23s (чуть выше 20s) и требовал аккуратного баланса `alpha_null` / `steps_per_layer` / `lambda_antibleed`.
   - Проверьте логи каждого слоя `[Shared4Null L26..L31] ce=..., cal_kl=..., neg_kl=...`:
     - Если `L26` почти не снижает `ce`, уберите `L26` и оставьте `TARGET_LAYERS = [27, 28, 29, 30, 31]` (быстрее на ~20%!).
     - Ускорение времени: `cal_texts[:32]` с `max_len=64` и `steps_per_layer = (6, 7, 8, 8, 9)` часто укладываются в `< 20s`.
     - Если падает `locality_acc`, проверьте `print_locality_breakdown` (различие между `near_entity_bleed` и `general`) и усильте `lambda_antibleed` / `lambda_neg_kl` или `alpha_null`.

### Шаг 3. Multi-Seed (`[42, 1337, 2026]`) и Stage 2 Long-Context (`seq_len=2048`) верификация
Создать два скрипта по образцу `research_orig/smollm_nobackprop_lab/verify_stage2_mechA.py` и `verify_stage2_and_seeds.py`:

1. **`ornith_nobackprop_lab/verify_stage2_mechA.py`** $\rightarrow$ сохраняет `stage2_mechA12_results.json`:
   - Загрузить 8-bit модель **один раз**, измерить `static_vram_mb`.
   - Для каждого `seed in [42, 1337, 2026]`:
     - `batch = generate_fact_batch(seed, tokenizer=tokenizer, target_doc_tokens=2048)`
     - Stage 1: лучший `Pure 0-Backprop` солвер (`synthesize_mechA12_swiglu` или улучшенная версия), оценка `evaluate_model` + `NoBackpropGuard` (`peak_vram_mb`, `activation_vram_overhead_mb`, `wall_time_sec`).
     - На `seed == 42`: дополнительно Stage 2 (`doc_text_stage2=batch.stage2_document_text`, реальная 2048-токенная экстракция фактов + `no_grad()` forward по `2048` токенам, сравнение активационного оверхеда с `Baseline_A_512` `3242.66 MB` и `Baseline_A_2048` OOM-потолком `4111.93 MB`).
     - Между сидами обязательно вызывать `lora_mgr.clear_all(); cleanup_cuda(); torch.cuda.reset_peak_memory_stats()`.

2. **`ornith_nobackprop_lab/verify_stage2_and_seeds.py`** $\rightarrow$ сохраняет `stage2_shared4null_results.json`:
   - То же самое для лучшего 1-слойного изолированного (`x_in.detach()`) солвера (`PrecondBlockManager` + `run_shared4_l26_31` или уточнённая версия).
   - Убедиться, что между сидами вызывается `mgr.uninstall()` + `cleanup_cuda()`, а перед новым сидом переустанавливается чистый `PrecondBlockManager`.
   - Оба скрипта должны фиксировать среднее $\pm$ std по 3 сидам по всем ключевым метрикам.

### Шаг 4. Complex Fictional Biography & Multi-Hop Logical Reasoning (`biography_logic_experiment.py`)
Адаптировать `../research_orig/smollm_nobackprop_lab/biography_logic_experiment.py` в `ornith_nobackprop_lab/biography_logic_experiment.py` $\rightarrow$ `biography_logic_results.json`:
- Персонаж: **Dr. Seren Valenwood** (точно те же атомарные факты `BIO_SELF_SUPERVISED_PAIRS`, 6 прямых вопросов `Direct_Recall_6` и 9 многошаговых логических задач `MultiHop_Logic_9`).
- **Важные адаптации под `Ornith-1.0-9B`**:
  - Форматирование промптов с `enable_thinking=False` для `Direct_Recall_6` и одноходового/self-recalled CoT, плюс **дополнительный прогон `enable_thinking=True` (с `max_new_tokens=256..384`)** на `MultiHop_Logic_9`, чтобы сравнить пользу встроенного chain-of-thought у 9B модели!
  - Сравниваемые режимы (загрузить 8-bit модель 1 раз!):
    1. `Base_Unedited`
    2. `InContext_Oracle`
    3. `Pure_0Backprop_VoronoiSwiGLU` (на атомарных парах)
    4. `SingleLayer_Shared4Null_ResSkip` (на атомарных парах)
  - В `Shared4Null` для биографии проверить, хватает ли `target_layers` / `steps_per_layer` для многотокенных ответов (в биографии факты включают города, даты, числа, институты — более длинные target'ы).

### Шаг 5. Сборка `blocks_library/`, финальный `lab_notebook.md` и аудит
1. Создать `ornith_nobackprop_lab/blocks_library/`:
   - `voronoi_swiglu_block.py` (экспорт `VoronoiSwiGLUInjector` или обёртка над финальным 0-backprop солвером)
   - `shared4null_resskip_block.py` (экспорт `Shared4NullResSkipInjector` / `PrecondBlockManager`)
   - `README.md` со сводной таблицей результатов на `Ornith-1.0-9B` (8-bit).
2. Обновить `ornith_nobackprop_lab/lab_notebook.md`: добавить все итерации (`16`, `28`, `12`, R&D, Stage 2 3-seed, Biography Logic) и сравнительную таблицу **`SmolLM2-360M` vs 8-bit `Ornith-1.0-9B`**.
3. Сделать финальный commit & push в репозиторий (и не забыть почистить временный ssh-ключ, если используете его в сессии).

---

## 5. Готовый промпт для старта новой сессии

```text
Продолжи и доведи до конца исследование из https://github.com/drimeF0/backpropfree :
1. Клонируй репозиторий, изучи README.md, HANDOFF_NEXT_SESSION.md и ornith_nobackprop_lab/lab_notebook.md.
2. Установи bitsandbytes transformers accelerate datasets (БЕЗ flash-linear-attention на T4 SM 7.5!). Phase 1 Anatomy и все 5 ориентиров Iteration 0 (Baseline 0, C, A@512, A@2048-OOM, B@100step) уже готовы и сохранены в baselines_summary.json.
3. Прогони iter_16_mechB_bounded_c2l.py, iter_28_mechA12_sweetspot.py и iter_12_shared4null.py на Seed 42; если какой-то гейт не пройден, проведи диагностику и итеративный R&D тюнинг для 8-bit Ornith-1.0-9B (учитывая скачок на L27 full_attention и 248k словарь).
4. Напиши и прогони verify_stage2_mechA.py и verify_stage2_and_seeds.py (seeds=[42, 1337, 2026], Stage 2 seq_len=2048 с переиспользованием одной загруженной 8-bit модели).
5. Напиши и прогони biography_logic_experiment.py (Dr. Seren Valenwood, direct + multi-hop CoT, проверь enable_thinking=False и True).
6. Собери ornith_nobackprop_lab/blocks_library/, обнови lab_notebook.md со сравнением SmolLM2-360M vs 8-bit Ornith-1.0-9B, проверь все JSON-метрики и запушь финал в GitHub.
```
