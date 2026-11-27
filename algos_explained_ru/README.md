# Простые объяснения алгоритмов репозитория `backpropfree`

Эта папка содержит понятные пошаговые разборы всех алгоритмов проекта — от устройства моделей и тестов до математики строительных блоков в [`ornith_nobackprop_lab/blocks_library/`](../ornith_nobackprop_lab/blocks_library/).

## Содержание

1. **[01_models_and_constraints.md](01_models_and_constraints.md)** — Модели (`SmolLM2-360M` vs 8-битный гибрид `Ornith-1.0-9B`), физические ограничения Tesla T4, `NoBackpropGuard` и тонкости `Qwen3.5` + `LLM.int8()`.
2. **[02_benchmarks_and_testing.md](02_benchmarks_and_testing.md)** — Как устроено тестирование: генерация фактов и биографий, Stage 1 и Stage 2 (2048 токенов), многошаговая логика, проверка locality/bleed без артефактов батчинга, QA retention и KL-дивергенция.
3. **[03_anatomy_and_nullspace.md](03_anatomy_and_nullspace.md)** — Разведка модели (`dissect_ornith.py`) и защита общих знаний (`nullspace_cache.py`): logit-lens по слоям, конус Вороного в `lm_head`, потоковая ковариация $C_0$ и low-rank SVD для $12288 \times 12288$.
4. **[04_block_voronoi_swiglu.md](04_block_voronoi_swiglu.md)** — Блок №1: чистый 0-backprop **Voronoi-Guarded SwiGLU (`Mech A12 / A13`)** — «спящие» нейроны, контрастивная сепарация в `gate/up`, 1-hot `down_proj` и отталкивание от ложных токенов.
5. **[05_block_shared4null_resskip.md](05_block_shared4null_resskip.md)** — Блок №2: 1-слойный изолированный **Shared4Null ResSkip** — как обучать LoRA через один отцепленный блок (`x_in.detach()`), не портя остальную модель.
6. **[06_transfer_gaps_and_solutions.md](06_transfer_gaps_and_solutions.md)** — Почему алгоритмы вели себя по-разному при переносе `360M -> 9B` и как работает унифицированный переносимый механизм.
7. **[07_unified_transfer_results_and_roadmap.md](07_unified_transfer_results_and_roadmap.md)** — Детальные экспериментальные результаты итераций 8–15 (`SmolLM2-360M` и `Ornith-1.0-9B`), разбор причин оставшихся провалов и дорожная карта итерации 16.
