# Next Session Handoff — Cross-Scale Backprop-Free Transfer (Agent #2)

**Last checkpoint:** `2026-11-27 ~19:53 UTC` (`22:53 MSK`)  
**Workspace:** `/content/backpropfree/ornith_nobackprop_lab` (`origin/main`)  
**Scope:** One unified backprop-free block (`rank <= 16`, `uses_full_backprop == False`, `NoBackpropGuard`) passing ALL gates in `transfer_eval.py` across **BOTH** `HuggingFaceTB/SmolLM2-360M-Instruct` (`small`, fp16 pure-attention) and 8-bit `ornith-ai/Ornith-1.0-9B` (`large`, `LLM.int8()` hybrid `GatedDeltaNet + full_attention`) across seeds `[42, 1337, 2026]` × `Stage 1` & `Stage 2 (seq_len=2048)`.

---

## 1. Environment & Git Rules Before Starting
1. Configure SSH agent cleanly before running `git fetch`/`git push` (note `/content/id_ed25519` is encrypted and passphrases must never be committed or logged; reuse `/tmp/ssh_agent.sock` if alive or ask user if expired):
   ```bash
   export SSH_AUTH_SOCK=/tmp/ssh_agent.sock
   cd /content/backpropfree && git pull --rebase origin main
   ```
2. **Never overwrite Agent #1 files** (`results.jsonl`, `lab_notebook.md`, `experiments.py`, etc.). Agent #2 owns `transfer_eval.py`, `experiments_transfer.jsonl`, `lab_notebook_transfer.md`, `iter_*_unified_transfer.py`, `blocks_library/unified_transfer_block.py`, `algos_explained_ru/06_*`, `07_*`.
3. Check GPU health first (`nvidia-smi`; T4 15 GiB). Expect `~17-19s` wall time (`~35s` total) per `360M` run and `~98-115s` synthesis (`~4.5-5.5 min` total end-to-end incl. 8-bit model load + eval) per `9B` run.
4. **Do NOT claim completion or emit `<!-- GOAL_COMPLETE -->`** until ALL gates (`ALL_GATES_PASS == True`) are observed in `experiments_transfer.jsonl` on **BOTH** `small` AND `large` across `[42, 1337, 2026]` for Stage 1 and Stage 2.

---

## 2. Verified Scoreboard So Far (`experiments_transfer.jsonl`)

### Base & In-Context Oracle Logic Baselines (`seq_len` / 1-turn logic)
- `SmolLM2-360M`:
  - `Base`: `s42=0.0000`, `s1337=0.1111`, `s2026=0.1111`
  - `InContext_Oracle`: `s42=0.6667`, `s1337=0.4444`, `s2026=0.5556`
  - **Pass thresholds (`logic_1turn_acc`):** `s42 >= 0.5333`, `s1337 >= 0.3556`, `s2026 >= 0.4444` (plus `direct >= 0.75`, `para >= 0.70`, `real_anc >= 0.95`, `bleed <= 0.05`, `kl <= 0.05`, `ret >= 0.95`, `time <= 20s`).
- `Ornith-1.0-9B (8-bit)`:
  - `Base`: `s42=0.2222`, `s1337=0.1111`, `s2026=0.2222`
  - `InContext_Oracle`: `1.0000` across all 3 seeds
  - **Pass thresholds (`logic_1turn_acc`):** `>= 0.8000` across all 3 seeds (plus `direct >= 0.70`, `para >= 0.65`, `real_anc >= 0.95`, `bleed <= 0.05`, `kl <= 0.05`, `ret >= 0.95`, `ovh <= 2500 MB`, `time <= 120s`).

### Best Candidate Results Observed
| Iter | Scale / Seed / Stg | Direct | Para | Logic 1T | SR Logic | Real Anc | Bleed | KL | Ret | Ovh MB / Time | Missing Gates |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **8** | `9B` / `s42` / `1` | `0.9167` | `0.8333` | `0.6667` | `1.0000` | `1.0000` | `0.1000` | `0.0117` | `1.0000` | `1481 / 151.90s` | `logic`, `bleed`, `time` |
| **9** | `9B` / `s42` / `1` | `0.8333` | `0.8750` | `0.5556` | `1.0000` | `1.0000` | **`0.0500`** | `0.0140` | `1.0000` | `1313 / 97.81s` | **ONLY `logic` (9/10 PASS!)** |
| **9** | `360M` / `s42` / `1` | `0.8333` | `0.8333` | `0.0000` | `0.4444` | `0.9000` | `0.0500` | `0.0081` | `0.9464` | `1103 / 16.14s` | `logic`, `anc`, `ret` |
| **13** | `9B` / `s42` / `1` | `0.7500` | `0.8750` | `0.6667` | `1.0000` | `1.0000` | `0.1500` | `0.0092` | `1.0000` | `1144 / 101.02s` | `logic` (3/3 arith passed!), `bleed` |
| **15** | `9B` / `s42` / `1` | `0.6667` | `0.8750` | `0.5556` | `1.0000` | `1.0000` | `0.1500` | `0.0106` | `1.0000` | `1155 / 107.97s` | `direct` (L23 drift), `logic`, `bleed` (both `allergy`+`conj` PASSED!) |
| **15** | `360M` / `s42` / `1` | `0.8333` | `0.8750` | `0.0000` | `0.3333` | `0.8000` | `0.0500` | `0.0121` | `0.9821` | `1097 / 17.61s` | `logic`, `anc` (`Lovelace`, `Pasteur->Alexis Bichat`) |
| **15** | `360M` / `s1337` / `1` | `0.7500` | `0.7917` | `0.1111` | `0.3333` | `1.0000` | **`0.0500`** | `0.0119` | `0.9643` | `1125 / 18.81s` | **ONLY `logic` (9/10 PASS!)** |
| **15** | `360M` / `s2026` / `1` | `0.9167` | `0.8750` | `0.1111` | `0.3333` | `0.9000` | `0.1000` | `0.0092` | `0.9464` | `1077 / 17.18s` | `logic`, `anc`, `bleed`, `ret` |
| **15** | `360M` / `s42` / `2` | `0.7500` | `0.8333` | `0.1111` | `0.4444` | `0.9000` | `0.1000` | `0.0122` | `0.9464` | `1101 / 17.72s` | `logic`, `anc`, `bleed`, `ret` (Stage 2 @ 2048!) |

---

## 3. Exact Root Causes & Next Steps (Build `iter_16_unified_transfer.py`)

### Priority 1: Fix `SmolLM2-360M` `logic_1turn` (`0.00 -> >=0.5333`) & `real_anchors` (`0.80 -> 1.0`)
Inspection of `iter_15_details_small_s42_stg1.json` showed **exactly why** `360M` fails `logic_1turn` despite retrieving facts:
1. **The `"Fact 1: ... Fact 2: ... Fact 3: ..."` Repetition Trap:** Our `cot_mid` probes in `iter_14`/`iter_15` trained the model to emit `"Fact 1: ..."` / `"Fact 2: ..."` without supervising how to **exit** fact listing into deduction/answer. Result: `360M` retrieved `"Fact 1: Dr. Seren Valenwood was born in 1984."` and then looped `Fact 3... Fact 8` until token truncation!
   - **Fix:** Replace repetitive `"Fact 1: / Fact 2:"` header supervision with clean natural deduction formats that mirror `InContext_Oracle`'s passing outputs (e.g., direct multi-fact retrieval -> `Therefore:` / `Final Answer:` closure, and inspect what `InContext_Oracle` outputs on `360M` `logic_1turn` first so `cot_mid` aligns with `360M`'s native CoT grammar rather than fighting it!).
2. **`real_anchors` (`Pasteur` & `Lovelace`) on `360M`:**
   - `Louis Pasteur` predicted `"Alexis Bichat."` because `_EXTRA_FRENCH_SCIENCE_ANCHORS` protected `"Xavier Bichat"`, NOT `"Alexis Bichat"`. Add broader French microbiologist/historian & `"Charles Babbage Analytical Engine"` anchors generated/curated into `enc_qs`.
3. **Digit / `join_year` / `allergy` disambiguation on `360M`:**
   - Look why `join_year` answered `"Helios Polar Institute"` (`what year` vs `which institution`) and `allergy` answered `"penicillin"`: strengthen contrastive question-word margin (`when/year` vs entity name; `cobalt` vs medical prior) and verify why `Oracle` clears `6/9` on `360M` without forgetting `penicillin` anchors.

### Priority 2: Combine `Iter 9 + Iter 13 + Iter 15` on `Ornith-1.0-9B` (`logic >= 0.80` & `bleed <= 0.05` & `time <= 120s`)
Look at what already worked separately on `9B`:
- **Layer selection & runtime:** **Iter 9** (`[27, 31]`, top-2 `full_attention`) ran in **`97.81s`** with `bleed=0.05`, `direct=0.8333`, `para=0.8750`, `ret=1.0`, `anc=1.0` — skipping `L23` eliminated Iter 15's `L23 cot_pen=2.194` digit oscillation (`1984 -> 09944.44...`, `2011 -> 2026`, `12 -> 120`) AND saves `~10–15s`! (If `L23` is kept for 2-hop composition, clip/dampen `cot_pen` at `L23` and cap `max_steps=8` as in Iter 13!)
- **Negative slicing bug (`bleed 0.05 -> 0.15`):** Iter 14/15 truncated `list(bundle["neg_rollout_prompts"])[:24]` and `neg_cf_mid_prompts[:28]`, accidentally dropping `relocate_age` (`12`) and `birth_year` (`1984`) sibling prompts (causing `Dr. Mireille Sterling -> 12`, `Dr. Alaric Kincaid -> 12`, `Dr. Freya Pendleton -> 1984`). **Restore FULL sibling category coverage** (all 12 relations × sampled siblings, as in Iter 9/12) so `bleed <= 0.05` holds reliably!
- **`lr_A` stability band + No Yes/No suppression:** Use `lr_A ~ 0.30..0.34 * sqrt(960 / d_in) * lr_b` (between Iter 13's `0.28` `3/3` arithmetic win and Iter 15's `0.38`) and **preserve Iter 15's removal of `"Yes"/"No"` from `blurt_ids`** (which solved `conjunctive_qualification_positive` alongside `constraint_allergy_safety`!).
- **Remaining 2 logic items on `9B` (`transitive_invention_to_mentor` & `transitive_person_to_sister_city`):** In `Iter 15 large`, `9B` treated `Aether-7` as Doctor Who trivia and `sister` as `Prof. Mara Valenwood` during multi-hop CoT despite `sr_logic = 1.0`. Inspect `iter_9` / `iter_13` CoT logs for how `"Aether-7"` / `"sister"` entity linking enters CoT, and supervise clean 2-hop entity-to-entity bridge stems (`Aether-7 -> invented by Dr. Seren Valenwood -> mentored by Prof. Mara Lindholm`) at `pos_cot_mid` without `"Fact 1:"` loops!

---

## 4. Suggested Commands for Next Session
1. Fast inspect & design verification on `360M` (`~18s` per seed):
   ```bash
   cd /content/backpropfree/ornith_nobackprop_lab
   python3 -u iter_16_unified_transfer.py --scale small --seeds 42 --stage 1
   ```
2. Validate `9B` (`~5 min` per seed/stage):
   ```bash
   python3 -u iter_16_unified_transfer.py --scale large --seeds 42 --stage 1
   ```
3. Once BOTH pass `s=42 stg=1`, run full multi-seed Stage 1 + Stage 2 certification:
   ```bash
   python3 -u iter_16_unified_transfer.py --scale both --seeds 42 1337 2026 --stage 1
   python3 -u iter_16_unified_transfer.py --scale both --seeds 42 1337 2026 --stage 2
   ```
4. Smoke-check & update `blocks_library/unified_transfer_block.py` (`ALL_GATES_PASS` across all seeds/stages), `algos_explained_ru/`, `lab_notebook_transfer.md`, and push cleanly.
