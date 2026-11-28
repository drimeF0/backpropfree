"""Clean Anti-Cheat Multi-Domain Benchmark (Code API, Pop Culture Lore, Bio Logic, & Held-Out Domain 4)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from benchmark import get_general_qa_set
from universal_domain1_data import get_domain1_data
from universal_domain23_data import (
    get_concealed_heldout_domain4,
    get_domain2_data,
    get_domain3_data,
    get_real_anchors_25,
)
from universal_sandbox import run_code_in_sandbox
from unified_model_utils import apply_chat_safe

LAB_DIR = Path(__file__).resolve().parent
DOMAINS = ("code_api", "pop_culture", "bio_logic")
SEEDS = (42, 1337)


def load_smoltalk_disjoint(
    tokenizer,
    scale: str,
    n_cal: int = 128,
    n_kl: int = 64,
    cal_seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Sample calibration texts ONLY from generic HuggingFaceTB/smol-smoltalk train split (disjoint from KL test split)."""
    assert n_cal <= 200, "Calibration budget must be <= 200 examples"
    cache_name = "smoltalk_cache.json" if scale == "large" else "smoltalk_cache_small.json"
    cache_path = LAB_DIR / cache_name

    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        cal_pool = list(payload.get("cal_texts", []))
        kl_pool = list(payload.get("kl_texts", []))
    else:
        from datasets import load_dataset

        cal_ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train", streaming=True)
        cal_pool = []
        for i, row in enumerate(cal_ds):
            if i >= 200:
                break
            cal_pool.append(apply_chat_safe(tokenizer, row["messages"], add_generation_prompt=False))

        kl_ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="test", streaming=True)
        kl_pool = []
        for i, row in enumerate(kl_ds):
            if i >= max(64, n_kl):
                break
            kl_pool.append(apply_chat_safe(tokenizer, row["messages"], add_generation_prompt=False))

        cache_path.write_text(json.dumps({"cal_texts": cal_pool, "kl_texts": kl_pool}), encoding="utf-8")

    rng = random.Random(cal_seed)
    indices = list(range(len(cal_pool)))
    rng.shuffle(indices)
    cal_selected = [cal_pool[idx] for idx in indices[: min(n_cal, len(indices))]]
    return cal_selected, kl_pool[:n_kl]


def get_domain_batch(domain: str, seed: int) -> Dict[str, object]:
    """Return deterministic benchmark batch for one of the 3 primary domains and seed in [42, 1337]."""
    if domain == "code_api":
        return get_domain1_data(seed)
    if domain == "pop_culture":
        return get_domain2_data(seed)
    if domain == "bio_logic":
        return get_domain3_data(seed)
    raise ValueError(f"Unknown domain: {domain!r}")


def get_shared_retention_suite() -> Dict[str, object]:
    """Return untouched 25 real_anchors (15 facts + 10 code tasks) and 60 general_qa items."""
    ra = get_real_anchors_25()
    return {
        "real_fact_anchors": ra["fact_anchors"],
        "real_code_anchors": ra["code_anchors"],
        "general_qa": get_general_qa_set(),
    }


def get_concealed_heldout_suite() -> Dict[str, object]:
    """Return Domain 4 (kestrel_rpc + 15 unseen real anchors). Invoke ONLY at final frozen verification."""
    return get_concealed_heldout_domain4()


def verify_benchmark_integrity() -> Dict[str, object]:
    """Self-test sandbox execution, oracle solutions, and disjoint splits."""
    failures = []
    for seed in SEEDS:
        d1 = get_domain_batch("code_api", seed)
        for task in d1["code_tasks"]:
            res = run_code_in_sandbox(
                task["oracle_code"],
                task["test_code"],
                expected_fn=task["fn_name"],
                mock_module_name=d1["mock_module_name"],
                mock_module_source=d1["mock_module_source"],
            )
            if not res.passed:
                failures.append((seed, task["id"], res.error_type, res.message))

        d2 = get_domain_batch("pop_culture", seed)
        d3 = get_domain_batch("bio_logic", seed)
        # Verify preference train/eval prompt disjointness & RL train/eval prompt disjointness
        for b in (d1, d2, d3):
            tr_p = {x["prompt"] for x in b["pref_train"]}
            ev_p = {x["prompt"] for x in b["pref_eval"]}
            assert tr_p.isdisjoint(ev_p), f"Preference train/eval overlap on seed={seed}"

        assert {t["id"] for t in d1["code_tasks"]}.isdisjoint({t["id"] for t in d1["rl_train_tasks"]})
        assert {x["q"] for x in d2["multihop_qa"]}.isdisjoint({x["q"] for x in d2["rl_train_tasks"]})
        assert {x.question for x in d3["logic_eval"]}.isdisjoint({x["q"] for x in d3["rl_train_tasks"]})

    return {"ok": len(failures) == 0, "failures": failures}


if __name__ == "__main__":
    report = verify_benchmark_integrity()
    print(json.dumps(report, indent=2))
