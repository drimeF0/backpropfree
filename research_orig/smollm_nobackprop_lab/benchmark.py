"""Deterministic Synthetic Knowledge Benchmark & Real Retention/KL Datasets.

Cleanly separates:
  - edit_source: declarative facts (Stage 1) OR 2,048-token noisy document (Stage 2)
  - verbatim_qa: direct recall queries mirroring edit_source wording
  - paraphrase_qa: completely unseen syntactic reformulations (never seen during synthesis)
  - locality_qa: unedited synthetic sibling entities + real-world anchors
  - calibration_texts: <=200 held-out smol-smoltalk train examples for null-space C0
  - heldout_kl_texts: disjoint smol-smoltalk test examples for D_KL(P_base || P_edited)
  - general_qa: 60 deterministic real-world QA items for >= 95% base retention gate
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

CAL_DATASET = "HuggingFaceTB/smol-smoltalk"
SEEDS = [42, 1337, 2026]
NUM_FACTS = 25


# Curated synthetic proper-noun syllable pools so generated names have ~0% Base prior
_ONSETS = [
    "Kael", "Vor", "Trel", "Zeph", "Mirel", "Sol", "Brav", "Nolv", "Cind", "Dorian",
    "Vael", "Seld", "Oren", "Thal", "Korv", "Lyra", "Halv", "Brev", "Qir", "Drak",
    "Arv", "Fen", "Galv", "Jor", "Kev", "Lom", "Myr", "Nev", "Pav", "Ryn",
    "Syl", "Tav", "Ulm", "Vex", "Wyn", "Xan", "Yor", "Zan", "Cael", "Elv",
]
_MIDDLES = [
    "en", "or", "ia", "an", "el", "is", "um", "ar", "on", "ek",
    "al", "ir", "ov", "as", "et", "ul", "yn", "os", "av", "in",
]
_ENDINGS_PLACE = [
    "grad", "haven", "spire", "wick", "holm", "ford", "mere", "crest", "mont", "vale",
    "berg", "port", "gard", "holm", "keep", "brook", "watch", "stead", "cliff", "gate",
]
_ENDINGS_MINERAL = [
    "ium", "ite", "ane", "orite", "alene", "olith", "arite", "ynite", "elium", "osite",
]
_ENDINGS_CURRENCY = [
    "mark", "crown", "ster", "flor", "drach", "ducat", "groat", "shill", "penny", "franc",
]
_ENDINGS_SURNAME = [
    "berg", "holm", "kestrel", "thorne", "vance", "halvor", "lind", "merid", "corvo", "stern",
    "varis", "eldon", "karr", "maren", "strid", "veldt", "falk", "bren", "dahl", "nordin",
]
_ENDINGS_LANDMARK = [
    "Spire", "Vault", "Observatory", "Archive", "Beacon", "Citadel", "Sanctum", "Pillar", "Pavilion", "Rotunda",
]


def _make_unique_names(rng: random.Random, count: int, endings: List[str], two_word: bool = False) -> List[str]:
    out: List[str] = []
    seen = set()
    while len(out) < count:
        a = rng.choice(_ONSETS) + rng.choice(_MIDDLES)
        if two_word:
            b = rng.choice(_ONSETS) + rng.choice(endings)
            cand = f"{a} {b}"
        else:
            cand = a + rng.choice(endings)
        cand_clean = cand[0].upper() + cand[1:]
        if cand_clean.lower() not in seen:
            seen.add(cand_clean.lower())
            out.append(cand_clean)
    return out


RELATIONS = [
    {
        "type": "capital",
        "source_tpl": "Archival record: The administrative capital of {entity} is {target}.",
        "doc_tpl": (
            "According to the municipal charter of {entity}, every legislative assembly convenes in its "
            "administrative capital, {target}, where provincial registries have been maintained for decades."
        ),
        "verbatim_q": "What is the administrative capital of {entity}?",
        "paraphrase_qs": [
            "Which city serves as the seat of government for {entity}?",
            "Where is the central governing office of {entity} located?",
        ],
    },
    {
        "type": "mineral",
        "source_tpl": "Geological survey: The rare mineral exclusively excavated in {entity} is {target}.",
        "doc_tpl": (
            "Independent assay logs confirm that deep-shaft crews operating inside {entity} recover "
            "deposits of {target}, a rare mineral found nowhere outside its borders."
        ),
        "verbatim_q": "What rare mineral is exclusively excavated in {entity}?",
        "paraphrase_qs": [
            "Mining teams in {entity} uniquely extract which scarce mineral?",
            "Which rare ore is solely mined within the territory of {entity}?",
        ],
    },
    {
        "type": "currency",
        "source_tpl": "Monetary ledger: The official currency issued by {entity} is {target}.",
        "doc_tpl": (
            "Treasury clearinghouses inside {entity} settle all commercial contracts exclusively in {target}, "
            "the sole official currency authorized by its central bank."
        ),
        "verbatim_q": "What is the official currency issued by {entity}?",
        "paraphrase_qs": [
            "Which unit of money serves as legal tender throughout {entity}?",
            "Commerce in {entity} officially transacts using what form of currency?",
        ],
    },
    {
        "type": "founder",
        "source_tpl": "Historical chronicle: {entity} was originally founded by {target}.",
        "doc_tpl": (
            "Early maritime ledgers credit {target} as the original founder who chartered and settled "
            "{entity} along the northern trade corridor."
        ),
        "verbatim_q": "Who originally founded {entity}?",
        "paraphrase_qs": [
            "Which historical figure established {entity}?",
            "{entity} owes its original founding to which individual?",
        ],
    },
    {
        "type": "landmark",
        "source_tpl": "Civic directory: The flagship observatory in {entity} is named {target}.",
        "doc_tpl": (
            "Visitors approaching the skyline of {entity} first notice {target}, which stands as the "
            "region's flagship research observatory and astronomical center."
        ),
        "verbatim_q": "What is the flagship observatory in {entity} named?",
        "paraphrase_qs": [
            "Which prominent research observatory serves as a landmark of {entity}?",
            "What name is given to the primary astronomical observatory of {entity}?",
        ],
    },
]

_DISTRACTOR_FILLERS = [
    "Regional trade routes primarily carried timber, dried grain, slate tiles, and woven linen across seasonal river barges during mild autumn winds.",
    "Annual census clerks recorded bridge repairs, harbor dredging schedules, lighthouse oil allocations, and telegraph wire inspections across rural parishes.",
    "Agricultural bulletins noted modest yields of barley, winter rye, orchard apples, and cold-pressed flaxseed along terraced limestone valleys.",
    "Public works committees reviewed cobblestone maintenance, aqueduct masonry joints, belltower clockwork gears, and municipal cistern water levels.",
    "Meteorological summaries logged barometric pressure averages, coastal fog frequencies, early frost dates, and prevailing westerly sea breezes.",
]


@dataclass
class FactItem:
    idx: int
    entity: str
    relation: str
    target: str
    edit_source: str
    doc_paragraph: str
    verbatim_q: str
    paraphrase_q: str
    paraphrase_q_alt: str


@dataclass
class LocalityItem:
    idx: int
    entity: str
    relation: str
    question: str
    is_real_world: bool = False
    expected_answer: Optional[str] = None


@dataclass
class FactBatch:
    seed: int
    facts: List[FactItem]
    locality: List[LocalityItem]
    stage1_edit_texts: List[str]
    stage2_document_text: str
    stage2_token_len: int


def generate_fact_batch(seed: int, tokenizer=None, target_doc_tokens: int = 2048) -> FactBatch:
    """Deterministically generate N=25 synthetic facts + 25 unedited synthetic siblings + Stage 2 doc."""
    rng = random.Random(seed)
    # Generate 50 unique entities (25 edited + 25 unedited siblings) so entities never overlap across splits
    all_entities = _make_unique_names(rng, NUM_FACTS * 2, _ENDINGS_PLACE, two_word=False)
    edit_entities = all_entities[:NUM_FACTS]
    sibling_entities = all_entities[NUM_FACTS:]

    # Generate distinct synthetic targets per relation type
    capitals = _make_unique_names(rng, 10, _ENDINGS_PLACE, two_word=False)
    minerals = _make_unique_names(rng, 10, _ENDINGS_MINERAL, two_word=False)
    currencies = _make_unique_names(rng, 10, _ENDINGS_CURRENCY, two_word=False)
    founders = _make_unique_names(rng, 10, _ENDINGS_SURNAME, two_word=True)
    landmarks = _make_unique_names(rng, 10, _ENDINGS_LANDMARK, two_word=True)

    # Ensure zero overlap between entity names and target values
    entity_set = {e.lower() for e in all_entities}
    def pop_non_colliding(pool: List[str]) -> str:
        while pool:
            val = pool.pop(0)
            if val.lower() not in entity_set:
                entity_set.add(val.lower())
                return val
        raise RuntimeError("Exhausted target pool")

    facts: List[FactItem] = []
    locality: List[LocalityItem] = []

    for i in range(NUM_FACTS):
        rel = RELATIONS[i % len(RELATIONS)]
        ent = edit_entities[i]
        sib = sibling_entities[i]

        if rel["type"] == "capital":
            tgt = pop_non_colliding(capitals)
        elif rel["type"] == "mineral":
            tgt = pop_non_colliding(minerals)
        elif rel["type"] == "currency":
            tgt = pop_non_colliding(currencies)
        elif rel["type"] == "founder":
            tgt = pop_non_colliding(founders)
        else:
            tgt = pop_non_colliding(landmarks)

        # Rotate paraphrase wording deterministically
        p_idx = (i + seed) % len(rel["paraphrase_qs"])
        p_alt_idx = (p_idx + 1) % len(rel["paraphrase_qs"])

        facts.append(
            FactItem(
                idx=i,
                entity=ent,
                relation=rel["type"],
                target=tgt,
                edit_source=rel["source_tpl"].format(entity=ent, target=tgt),
                doc_paragraph=rel["doc_tpl"].format(entity=ent, target=tgt),
                verbatim_q=rel["verbatim_q"].format(entity=ent),
                paraphrase_q=rel["paraphrase_qs"][p_idx].format(entity=ent),
                paraphrase_q_alt=rel["paraphrase_qs"][p_alt_idx].format(entity=ent),
            )
        )
        locality.append(
            LocalityItem(
                idx=i,
                entity=sib,
                relation=rel["type"],
                question=rel["paraphrase_qs"][p_idx].format(entity=sib),
                is_real_world=False,
                expected_answer=None,
            )
        )

    # Add 5 real-world anchor siblings to locality to confirm factual world knowledge stays intact
    real_anchors = [
        ("France", "capital", "What is the capital city of France?", "Paris"),
        ("Japan", "capital", "Which city serves as the capital of Japan?", "Tokyo"),
        ("Germany", "currency", "What official currency is used in Germany?", "Euro"),
        ("Italy", "capital", "What is the capital city of Italy?", "Rome"),
        ("Spain", "capital", "What is the capital city of Spain?", "Madrid"),
    ]
    for j, (ent, rtype, q, ans) in enumerate(real_anchors):
        locality.append(
            LocalityItem(
                idx=NUM_FACTS + j,
                entity=ent,
                relation=rtype,
                question=q,
                is_real_world=True,
                expected_answer=ans,
            )
        )

    # Construct Stage 2 noisy document containing all 25 facts and ~target_doc_tokens length
    sections: List[str] = []
    for i, f in enumerate(facts):
        filler = _DISTRACTOR_FILLERS[i % len(_DISTRACTOR_FILLERS)]
        if i % 2 == 0:
            sections.append(f"Section {i+1} ({f.entity} Survey): {filler} {f.doc_paragraph}")
        else:
            sections.append(f"Section {i+1} ({f.entity} Survey): {f.doc_paragraph} {filler}")

    doc_text = "\n\n".join(sections)
    actual_len = 0
    if tokenizer is not None:
        toks = tokenizer.encode(doc_text, add_special_tokens=False)
        filler_cycle = 0
        while len(toks) < target_doc_tokens:
            extra = (
                f"\nAppendix Note {filler_cycle + 1}: "
                + _DISTRACTOR_FILLERS[filler_cycle % len(_DISTRACTOR_FILLERS)]
            )
            doc_text += extra
            toks = tokenizer.encode(doc_text, add_special_tokens=False)
            filler_cycle += 1
        if len(toks) > target_doc_tokens:
            toks = toks[:target_doc_tokens]
            doc_text = tokenizer.decode(toks, skip_special_tokens=True)
        for f in facts:
            assert f.target in doc_text and f.entity in doc_text, (
                f"Fact ({f.entity} -> {f.target}) missing from Stage 2 document!"
            )
        actual_len = len(tokenizer.encode(doc_text, add_special_tokens=False))

    return FactBatch(
        seed=seed,
        facts=facts,
        locality=locality,
        stage1_edit_texts=[f.edit_source for f in facts],
        stage2_document_text=doc_text,
        stage2_token_len=actual_len,
    )


def get_general_qa_set() -> List[Dict[str, str]]:
    """60 deterministic general QA items covering geography, math, science, and basic knowledge."""
    items = [
        # Geography (15)
        {"q": "What is the capital city of France?", "a": "Paris"},
        {"q": "What is the capital city of Japan?", "a": "Tokyo"},
        {"q": "What is the capital city of Italy?", "a": "Rome"},
        {"q": "What is the capital city of Spain?", "a": "Madrid"},
        {"q": "What is the capital city of Germany?", "a": "Berlin"},
        {"q": "What is the capital city of Canada?", "a": "Ottawa"},
        {"q": "What is the capital city of Egypt?", "a": "Cairo"},
        {"q": "What is the capital city of Australia?", "a": "Canberra"},
        {"q": "What is the capital city of Brazil?", "a": "Brasilia"},
        {"q": "What is the capital city of South Korea?", "a": "Seoul"},
        {"q": "What is the capital city of Greece?", "a": "Athens"},
        {"q": "What is the capital city of Portugal?", "a": "Lisbon"},
        {"q": "What is the capital city of Sweden?", "a": "Stockholm"},
        {"q": "Which ocean lies between Europe and North America?", "a": "Atlantic"},
        {"q": "Which continent is the Sahara Desert located on?", "a": "Africa"},
        # Arithmetic (15)
        {"q": "What is 17 plus 28?", "a": "45"},
        {"q": "What is 12 multiplied by 9?", "a": "108"},
        {"q": "What is 144 divided by 12?", "a": "12"},
        {"q": "What is 85 minus 37?", "a": "48"},
        {"q": "What is 15 multiplied by 4?", "a": "60"},
        {"q": "What is 99 plus 11?", "a": "110"},
        {"q": "What is 64 divided by 8?", "a": "8"},
        {"q": "What is 50 minus 18?", "a": "32"},
        {"q": "What is 7 multiplied by 8?", "a": "56"},
        {"q": "What is 23 plus 19?", "a": "42"},
        {"q": "What is 100 divided by 4?", "a": "25"},
        {"q": "What is 11 multiplied by 11?", "a": "121"},
        {"q": "What is 75 minus 25?", "a": "50"},
        {"q": "What is 9 plus 16?", "a": "25"},
        {"q": "What is 36 divided by 6?", "a": "6"},
        # Science (15)
        {"q": "What is the chemical symbol for gold?", "a": "Au"},
        {"q": "What is the chemical symbol for oxygen?", "a": "O"},
        {"q": "What planet is closest to the Sun?", "a": "Mercury"},
        {"q": "What is the freezing point of water in degrees Celsius?", "a": "0"},
        {"q": "What is the boiling point of water in degrees Celsius at sea level?", "a": "100"},
        {"q": "What gas do plants primarily absorb for photosynthesis?", "a": "carbon dioxide"},
        {"q": "Which planet in our solar system is known as the Red Planet?", "a": "Mars"},
        {"q": "What is the largest planet in our solar system?", "a": "Jupiter"},
        {"q": "How many legs does a spider have?", "a": "8"},
        {"q": "What organ pumps blood through the human body?", "a": "heart"},
        {"q": "What force keeps planets in orbit around the Sun?", "a": "gravity"},
        {"q": "What is the chemical symbol for iron?", "a": "Fe"},
        {"q": "What is the hardest naturally occurring mineral?", "a": "diamond"},
        {"q": "Which gas makes up most of Earth's atmosphere?", "a": "nitrogen"},
        {"q": "What star is at the center of our solar system?", "a": "Sun"},
        # General / CS / Logic (15)
        {"q": "How many bits are in one byte?", "a": "8"},
        {"q": "What Python keyword is used to define a function?", "a": "def"},
        {"q": "How many days are in a standard non-leap year?", "a": "365"},
        {"q": "How many months are in one year?", "a": "12"},
        {"q": "What color do you get by mixing red and white?", "a": "pink"},
        {"q": "How many sides does a hexagon have?", "a": "6"},
        {"q": "How many minutes are in one hour?", "a": "60"},
        {"q": "What is the opposite of the word 'hot'?", "a": "cold"},
        {"q": "Which month comes directly after April?", "a": "May"},
        {"q": "What instrument has 88 black and white keys?", "a": "piano"},
        {"q": "What is the square root of 81?", "a": "9"},
        {"q": "What programming language uses .py file extensions?", "a": "Python"},
        {"q": "How many continents are there on Earth?", "a": "7"},
        {"q": "What is the primary language spoken in Spain?", "a": "Spanish"},
        {"q": "What season follows summer?", "a": "autumn"},
    ]
    return items


def load_smoltalk_splits(
    tokenizer,
    n_cal: int = 200,
    n_kl_eval: int = 64,
    max_len: int = 256,
) -> Tuple[List[str], List[str]]:
    """Load disjoint calibration (from train) and held-out KL evaluation (from test) texts."""
    assert n_cal <= 200, "Calibration dataset budget must be <= 200 examples!"
    cal_ds = load_dataset(CAL_DATASET, split="train", streaming=True)
    cal_texts: List[str] = []
    for i, row in enumerate(cal_ds):
        if i >= n_cal:
            break
        cal_texts.append(
            tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        )

    kl_ds = load_dataset(CAL_DATASET, split="test", streaming=True)
    kl_texts: List[str] = []
    for i, row in enumerate(kl_ds):
        if i >= n_kl_eval:
            break
        kl_texts.append(
            tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
        )
    return cal_texts, kl_texts


def normalize_answer(text: str) -> str:
    s = text.strip().lower()
    # Strip leading chat/boilerplate phrases if model answers in a full sentence
    s = re.sub(r"^(the\s+answer\s+is|it\s+is|answer:)\s*", "", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_target(pred: str, target: str) -> bool:
    np = normalize_answer(pred)
    nt = normalize_answer(target)
    if not np or not nt:
        return False
    if np == nt:
        return True
    # Allow concise single-sentence replies like "the rare mineral excavated in X is Y" (<= 20 words)
    # without permitting multi-guess answer dumping.
    words = np.split()
    if len(words) <= 20 and re.search(r"\b" + re.escape(nt) + r"\b", np):
        return True
    # Special equivalence for autumn / fall
    if nt == "autumn" and len(words) <= 20 and re.search(r"\b(autumn|fall)\b", np):
        return True
    return False
