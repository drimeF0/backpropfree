"""Unified 3-Seed Fictional Biography & Multi-Hop Logical Reasoning Benchmark (Stage 1 & Stage 2 @ 2048 tokens).

Seeds [42, 1337, 2026] each define a cohesive 12-fact fictional biography covering:
  people, dates, cities, mentors, inventions, institutions, relationships, constraints.

Strictly zero leakage across splits:
  - Synthesis sources: ONLY 12 canonical declarative clauses (Stage 1) OR a 2,048-token
    distractor-filled archival dossier (Stage 2, parsed via regex without test phrasings).
  - Direct Recall: 12 unseen direct factual queries (1 per fact).
  - Paraphrase / Post-Entity Framing: 24 unseen queries (12 semantic paraphrases + 12 post-entity).
  - Weight-Only Multi-Hop & Temporal Logic: 9 unseen tasks (4 temporal arithmetic, 3 transitive
    A->B->C, 1 allergy safety constraint, 1 conjunctive qualification).
  - Locality & Sibling Bleed: 10 real-world biographical anchors + 20 fictional sibling probes.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


SEEDS = [42, 1337, 2026]


@dataclass
class BioFactSpec:
    key: str
    target: str
    match_patterns: List[str]
    clause: str


@dataclass
class BioEvalItem:
    fact_key: str
    question: str
    target: str
    patterns: List[str]
    framing: str  # "direct", "paraphrase", "post_entity"


@dataclass
class BioLogicItem:
    name: str
    logic_type: str
    question: str
    sub_qs: List[str]
    patterns: Optional[List[str]] = None
    require_all: Optional[List[str]] = None


@dataclass
class BioLocalityItem:
    idx: int
    entity: str
    question: str
    is_real_world: bool
    expected_patterns: Optional[List[str]] = None


@dataclass
class BioSeedBatch:
    seed: int
    person: str
    facts: List[BioFactSpec]
    biography_text: str
    stage1_clauses: List[str]
    stage2_document_text: str
    stage2_token_len: int
    direct_eval: List[BioEvalItem]
    paraphrase_eval: List[BioEvalItem]
    logic_eval: List[BioLogicItem]
    locality_eval: List[BioLocalityItem]
    forbidden_tokens_lower: set
    target_bleed_patterns: List[Tuple[str, str, bool]]  # (key, regex, is_numeric)


_BIO_CONFIGS: Dict[int, Dict[str, str]] = {
    42: {
        "person": "Dr. Seren Valenwood",
        "pronoun_subj": "she",
        "pronoun_poss": "her",
        "birth_town": "Oakhaven",
        "birth_year": "1984",
        "study_city": "Vespera",
        "relocate_age": "12",
        "mentor": "Prof. Mara Lindholm",
        "mentor_pat": r"mara\s+lindholm",
        "invention": "Aether-7",
        "invention_pat": r"\baether[-\s]?7\b",
        "invention_desc": "Aether-7 thermal drill",
        "invent_year": "2011",
        "institution": "Helios Polar Institute",
        "institution_pat": r"helios\s+polar\s+institute",
        "join_year": "2015",
        "sister_name": "Talia Valenwood",
        "sister_pat": r"\btalia\s+valenwood\b",
        "sister_city": "Nordhavn",
        "allergy": "cobalt",
        "relocate_year": "1996",
        "invent_age": "27",
        "join_age": "31",
        "gap_years": "4",
        "safe_material": "granite",
    },
    1337: {
        "person": "Dr. Cassian Halvorsen",
        "pronoun_subj": "he",
        "pronoun_poss": "his",
        "birth_town": "Brineport",
        "birth_year": "1978",
        "study_city": "Solisberg",
        "relocate_age": "14",
        "mentor": "Dr. Henrik Solberg",
        "mentor_pat": r"henrik\s+solberg",
        "invention": "Boreal-4",
        "invention_pat": r"\bboreal[-\s]?4\b",
        "invention_desc": "Boreal-4 acoustic probe",
        "invent_year": "2007",
        "institution": "Boreal Marine Station",
        "institution_pat": r"boreal\s+marine\s+station",
        "join_year": "2012",
        "sister_name": "Nora Halvorsen",
        "sister_pat": r"\bnora\s+halvorsen\b",
        "sister_city": "Skagenfjord",
        "allergy": "bromine",
        "relocate_year": "1992",
        "invent_age": "29",
        "join_age": "34",
        "gap_years": "5",
        "safe_material": "quartz",
    },
    2026: {
        "person": "Dr. Orin Ashwell",
        "pronoun_subj": "he",
        "pronoun_poss": "his",
        "birth_town": "Lindenwick",
        "birth_year": "1991",
        "study_city": "Kestrelholm",
        "relocate_age": "9",
        "mentor": "Prof. Linnea Thorne",
        "mentor_pat": r"linnea\s+thorne",
        "invention": "Vortex-9",
        "invention_pat": r"\bvortex[-\s]?9\b",
        "invention_desc": "Vortex-9 tidal turbine",
        "invent_year": "2016",
        "institution": "Meridian Tidal Lab",
        "institution_pat": r"meridian\s+tidal\s+lab",
        "join_year": "2020",
        "sister_name": "Maren Ashwell",
        "sister_pat": r"\bmaren\s+ashwell\b",
        "sister_city": "Kelpbrook",
        "allergy": "tungsten",
        "relocate_year": "2000",
        "invent_age": "25",
        "join_age": "29",
        "gap_years": "4",
        "safe_material": "basalt",
    },
}

_DISTRACTOR_PARAGRAPHS = [
    "Regional archival registries catalogued harbor dredging rosters, timber freight tariffs, seasonal lighthouse oil reserves, and iron foundry inspection ledgers across coastal districts.",
    "Municipal waterworks boards published quarterly reservoir depth readings, stone aqueduct mortar surveys, sluice gate calibration tables, and winter snowpack runoff estimates.",
    "Maritime weather observatories compiled barometric pressure gradients, prevailing westerly gust frequencies, tidal range charts, and autumn sea-fog visibility logs.",
    "Agricultural guild bulletins reported moderate harvests of winter rye, terraced orchard apples, cold-pressed flaxseed oil, and stone-milled barley flour across upland valleys.",
    "Civil transport commissioners reviewed cobblestone bridge repairs, telegraph wire tension checks, ferry landing pilings, and coastal beacon lantern maintenance.",
    "Geological cartography crews updated elevation benchmarks, slate quarry drainage maps, granite seawall thickness measurements, and estuarine silt soundings.",
]

_REAL_ANCHORS = [
    ("Albert Einstein", "Which physicist formulated the theory of relativity?", [r"\beinstein\b"]),
    ("Marie Curie", "Which radioactive element did Marie Curie discover and name after Poland?", [r"\bpolonium\b"]),
    ("Alan Turing", "At which British codebreaking center did Alan Turing work during World War II?", [r"bletchley"]),
    ("Alexander Fleming", "Which life-saving antibiotic did Alexander Fleming discover in 1928?", [r"\bpenicillin\b"]),
    ("Neil Armstrong", "Who was the first person to walk on the Moon?", [r"\barmstrong\b"]),
    ("Ada Lovelace", "For which mechanical computing machine designed by Charles Babbage did Ada Lovelace write notes?", [r"analytical\s+engine"]),
    ("Isaac Newton", "In which country was physicist Isaac Newton born?", [r"\bengland\b|\bunited\s+kingdom\b|\bbritain\b"]),
    ("Galileo Galilei", "In which Italian city was astronomer Galileo Galilei born?", [r"\bpisa\b"]),
    ("Charles Darwin", "What was the name of the ship on which Charles Darwin sailed as naturalist?", [r"\bbeagle\b"]),
    ("Louis Pasteur", "Which French microbiologist developed the rabies vaccine and pasteurization?", [r"\bpasteur\b"]),
]

_SIBLING_FIRST = ["Rowan", "Callum", "Elena", "Dorian", "Freya", "Gideon", "Sylvia", "Lucian", "Kieran", "Mireille", "Alaric", "Brielle"]
_SIBLING_LAST = ["Blackwood", "Cromwell", "Mercer", "Kincaid", "Vandenberg", "Sterling", "Hawthorne", "Whitmore", "Pendleton", "Fairchild", "Redwood", "Stanhope"]


def _build_clauses_and_facts(c: Dict[str, str]) -> List[BioFactSpec]:
    p = c["person"]
    return [
        BioFactSpec(
            key="birth_town",
            target=c["birth_town"],
            match_patterns=[rf"\b{re.escape(c['birth_town'].lower())}\b"],
            clause=f"Bio dossier ({p}): The birthplace town of {p} is {c['birth_town']}.",
        ),
        BioFactSpec(
            key="birth_year",
            target=c["birth_year"],
            match_patterns=[rf"\b{c['birth_year']}\b"],
            clause=f"Bio dossier ({p}): The birth year of {p} is {c['birth_year']}.",
        ),
        BioFactSpec(
            key="study_city",
            target=c["study_city"],
            match_patterns=[rf"\b{re.escape(c['study_city'].lower())}\b"],
            clause=f"Bio dossier ({p}): The city where {p} relocated for study is {c['study_city']}.",
        ),
        BioFactSpec(
            key="relocate_age",
            target=c["relocate_age"],
            match_patterns=[rf"\b{c['relocate_age']}\b"],
            clause=f"Bio dossier ({p}): The age when {p} relocated to {c['study_city']} is {c['relocate_age']}.",
        ),
        BioFactSpec(
            key="mentor",
            target=c["mentor"],
            match_patterns=[c["mentor_pat"]],
            clause=f"Bio dossier ({p}): The academic mentor of {p} in {c['study_city']} is {c['mentor']}.",
        ),
        BioFactSpec(
            key="invention",
            target=c["invention"],
            match_patterns=[c["invention_pat"]],
            clause=f"Bio dossier ({p}): The device invented by {p} is {c['invention']}.",
        ),
        BioFactSpec(
            key="invent_year",
            target=c["invent_year"],
            match_patterns=[rf"\b{c['invent_year']}\b"],
            clause=f"Bio dossier ({p}): The year {p} invented {c['invention']} is {c['invent_year']}.",
        ),
        BioFactSpec(
            key="institution",
            target=c["institution"],
            match_patterns=[c["institution_pat"]],
            clause=f"Bio dossier ({p}): The research institution {p} later joined is {c['institution']}.",
        ),
        BioFactSpec(
            key="join_year",
            target=c["join_year"],
            match_patterns=[rf"\b{c['join_year']}\b"],
            clause=f"Bio dossier ({p}): The year {p} joined {c['institution']} is {c['join_year']}.",
        ),
        BioFactSpec(
            key="sister_name",
            target=c["sister_name"],
            match_patterns=[c["sister_pat"]],
            clause=f"Bio dossier ({p}): The younger sister of {p} is {c['sister_name']}.",
        ),
        BioFactSpec(
            key="sister_city",
            target=c["sister_city"],
            match_patterns=[rf"\b{re.escape(c['sister_city'].lower())}\b"],
            clause=f"Bio dossier ({p}): The city where {c['sister_name']} resides is {c['sister_city']}.",
        ),
        BioFactSpec(
            key="allergy",
            target=c["allergy"],
            match_patterns=[rf"\b{re.escape(c['allergy'].lower())}\b"],
            clause=f"Bio dossier ({p}): The substance causing a severe medical allergy in {p} is {c['allergy']}.",
        ),
    ]


def _build_stage2_narrative_sections(c: Dict[str, str]) -> List[str]:
    p = c["person"]
    return [
        f"Civil birth records confirm that {p} was born in the town of {c['birth_town']}.",
        f"Parish birth ledgers fix the birth year of {p} at {c['birth_year']}.",
        f"Academic transfer rolls note that {p} later moved to the city of {c['study_city']} to continue training.",
        f"According to enrollment forms, {p} relocated to {c['study_city']} at age {c['relocate_age']}.",
        f"University supervision rosters list {c['mentor']} as the academic mentor of {p} in {c['study_city']}.",
        f"Patent filings show that the technical device invented by {p} is {c['invention']}.",
        f"Engineering chronicles record that {p} completed the invention of {c['invention']} in {c['invent_year']}.",
        f"Institutional appointment files document that {p} subsequently joined {c['institution']}.",
        f"Personnel directories state that {p} officially began service at {c['institution']} in {c['join_year']}.",
        f"Family registry entries verify that the younger sister of {p} is {c['sister_name']}.",
        f"Municipal residence notices show that {c['sister_name']} currently lives in {c['sister_city']}.",
        f"Clinical safety dossiers warn that {p} suffers from a severe medical allergy to {c['allergy']}.",
    ]


def extract_bio_clauses_from_doc(doc_text: str) -> List[str]:
    """Extract the 12 canonical declarative biography clauses directly from a 2,048-token dossier."""
    extracted: List[str] = []
    # Match each of the 12 structured narrative patterns anywhere in the document
    patterns = [
        (r"Civil birth records confirm that (Dr\. [A-Za-z]+ [A-Za-z]+) was born in the town of ([A-Za-z]+)\.",
         lambda m: f"Bio dossier ({m.group(1)}): The birthplace town of {m.group(1)} is {m.group(2)}."),
        (r"Parish birth ledgers fix the birth year of (Dr\. [A-Za-z]+ [A-Za-z]+) at (\d{4})\.",
         lambda m: f"Bio dossier ({m.group(1)}): The birth year of {m.group(1)} is {m.group(2)}."),
        (r"Academic transfer rolls note that (Dr\. [A-Za-z]+ [A-Za-z]+) later moved to the city of ([A-Za-z]+) to continue training\.",
         lambda m: f"Bio dossier ({m.group(1)}): The city where {m.group(1)} relocated for study is {m.group(2)}."),
        (r"According to enrollment forms, (Dr\. [A-Za-z]+ [A-Za-z]+) relocated to ([A-Za-z]+) at age (\d+)\.",
         lambda m: f"Bio dossier ({m.group(1)}): The age when {m.group(1)} relocated to {m.group(2)} is {m.group(3)}."),
        (r"University supervision rosters list ((?:Prof\.|Dr\.) [A-Za-z]+ [A-Za-z]+) as the academic mentor of (Dr\. [A-Za-z]+ [A-Za-z]+) in ([A-Za-z]+)\.",
         lambda m: f"Bio dossier ({m.group(2)}): The academic mentor of {m.group(2)} in {m.group(3)} is {m.group(1)}."),
        (r"Patent filings show that the technical device invented by (Dr\. [A-Za-z]+ [A-Za-z]+) is ([A-Za-z]+-\d+)\.",
         lambda m: f"Bio dossier ({m.group(1)}): The device invented by {m.group(1)} is {m.group(2)}."),
        (r"Engineering chronicles record that (Dr\. [A-Za-z]+ [A-Za-z]+) completed the invention of ([A-Za-z]+-\d+) in (\d{4})\.",
         lambda m: f"Bio dossier ({m.group(1)}): The year {m.group(1)} invented {m.group(2)} is {m.group(3)}."),
        (r"Institutional appointment files document that (Dr\. [A-Za-z]+ [A-Za-z]+) subsequently joined ([A-Za-z]+ [A-Za-z]+ [A-Za-z]+)\.",
         lambda m: f"Bio dossier ({m.group(1)}): The research institution {m.group(1)} later joined is {m.group(2)}."),
        (r"Personnel directories state that (Dr\. [A-Za-z]+ [A-Za-z]+) officially began service at ([A-Za-z]+ [A-Za-z]+ [A-Za-z]+) in (\d{4})\.",
         lambda m: f"Bio dossier ({m.group(1)}): The year {m.group(1)} joined {m.group(2)} is {m.group(3)}."),
        (r"Family registry entries verify that the younger sister of (Dr\. [A-Za-z]+ [A-Za-z]+) is ([A-Za-z]+ [A-Za-z]+)\.",
         lambda m: f"Bio dossier ({m.group(1)}): The younger sister of {m.group(1)} is {m.group(2)}."),
        (r"Municipal residence notices show that ([A-Za-z]+ [A-Za-z]+) currently lives in ([A-Za-z]+)\.",
         lambda m: None),  # handled with subject context below
        (r"Clinical safety dossiers warn that (Dr\. [A-Za-z]+ [A-Za-z]+) suffers from a severe medical allergy to ([a-zA-Z]+)\.",
         lambda m: f"Bio dossier ({m.group(1)}): The substance causing a severe medical allergy in {m.group(1)} is {m.group(2)}."),
    ]
    subject_m = re.search(r"Civil birth records confirm that (Dr\. [A-Za-z]+ [A-Za-z]+) was born", doc_text)
    subject = subject_m.group(1) if subject_m else ""
    for idx, (pat, fmt) in enumerate(patterns):
        m = re.search(pat, doc_text)
        if m:
            if idx == 10:
                extracted.append(
                    f"Bio dossier ({subject}): The city where {m.group(1)} resides is {m.group(2)}."
                )
            else:
                extracted.append(fmt(m))
    return extracted


def parse_bio_clause(clause: str) -> Dict[str, str]:
    """Parse a canonical `Bio dossier (Person): ... is Target.` clause without touching eval splits."""
    m = re.match(r"^Bio dossier \(([^)]+)\): (.+?) is (.+)\.$", clause.strip())
    if not m:
        raise ValueError(f"Unrecognized bio clause format: {clause!r}")
    person, lhs, target = m.group(1), m.group(2), m.group(3)
    return {"person": person, "lhs": lhs, "target": target, "clause_stem": f"{lhs} is"}


def generate_bio_batch(seed: int, tokenizer=None, target_doc_tokens: int = 2048) -> BioSeedBatch:
    if seed not in _BIO_CONFIGS:
        raise ValueError(f"Unsupported biography seed {seed}; expected one of {SEEDS}")
    c = _BIO_CONFIGS[seed]
    p = c["person"]
    facts = _build_clauses_and_facts(c)
    stage1_clauses = [f.clause for f in facts]

    biography_text = (
        f"{p} was born in {c['birth_year']} in the town of {c['birth_town']}. "
        f"At age {c['relocate_age']}, {p} relocated to {c['study_city']} to study under academic mentor {c['mentor']}. "
        f"In {c['invent_year']}, {p} invented the {c['invention_desc']}. "
        f"In {c['join_year']}, {p} joined {c['institution']}. "
        f"{c['pronoun_poss'].capitalize()} younger sister is {c['sister_name']}, who lives in {c['sister_city']}. "
        f"{p} has a severe medical allergy to {c['allergy']}."
    )

    # Build 2,048-token distractor-filled archival document
    narratives = _build_stage2_narrative_sections(c)
    sections = []
    for i, sent in enumerate(narratives):
        d1 = _DISTRACTOR_PARAGRAPHS[i % len(_DISTRACTOR_PARAGRAPHS)]
        d2 = _DISTRACTOR_PARAGRAPHS[(i + 2) % len(_DISTRACTOR_PARAGRAPHS)]
        sections.append(f"Archive Section {i + 1}:\n{d1} {sent} {d2}")
    doc_text = "\n\n".join(sections)
    actual_len = 0
    if tokenizer is not None:
        toks = tokenizer.encode(doc_text, add_special_tokens=False)
        cyc = 0
        while len(toks) < target_doc_tokens:
            extra = (
                f"\n\nAppendix Registry Note {cyc + 1}: "
                + _DISTRACTOR_PARAGRAPHS[cyc % len(_DISTRACTOR_PARAGRAPHS)]
                + " "
                + _DISTRACTOR_PARAGRAPHS[(cyc + 3) % len(_DISTRACTOR_PARAGRAPHS)]
            )
            doc_text += extra
            toks = tokenizer.encode(doc_text, add_special_tokens=False)
            cyc += 1
        if len(toks) > target_doc_tokens:
            toks = toks[:target_doc_tokens]
            doc_text = tokenizer.decode(toks, skip_special_tokens=True)
        extracted = extract_bio_clauses_from_doc(doc_text)
        assert len(extracted) == 12, f"Seed {seed}: expected 12 extracted Stage 2 clauses, got {len(extracted)}"
        assert extracted == stage1_clauses, f"Seed {seed}: Stage 2 extracted clauses mismatch Stage 1!"
        actual_len = len(tokenizer.encode(doc_text, add_special_tokens=False))

    # Unseen Direct Recall Evaluation (12 questions, 1 per fact)
    direct_eval = [
        BioEvalItem("birth_town", f"Which town is recorded as the birthplace of {p}?", c["birth_town"], facts[0].match_patterns, "direct"),
        BioEvalItem("birth_year", f"Which calendar year marks the birth of {p}?", c["birth_year"], facts[1].match_patterns, "direct"),
        BioEvalItem("study_city", f"To which city did {p} move for advanced study?", c["study_city"], facts[2].match_patterns, "direct"),
        BioEvalItem("relocate_age", f"How old was {p} upon relocating to {c['study_city']}?", c["relocate_age"], facts[3].match_patterns, "direct"),
        BioEvalItem("mentor", f"Who supervised {p}'s academic work in {c['study_city']}?", c["mentor"], facts[4].match_patterns, "direct"),
        BioEvalItem("invention", f"Which technical device was created by {p}?", c["invention"], facts[5].match_patterns, "direct"),
        BioEvalItem("invent_year", f"In which year was the {c['invention']} device created?", c["invent_year"], facts[6].match_patterns, "direct"),
        BioEvalItem("institution", f"Which research institution employed {p} later in {c['pronoun_poss']} career?", c["institution"], facts[7].match_patterns, "direct"),
        BioEvalItem("join_year", f"In which year did {p} take up {c['pronoun_poss']} post at {c['institution']}?", c["join_year"], facts[8].match_patterns, "direct"),
        BioEvalItem("sister_name", f"What is the full name of {p}'s younger sister?", c["sister_name"], facts[9].match_patterns, "direct"),
        BioEvalItem("sister_city", f"In which city does {c['sister_name']} currently live?", c["sister_city"], facts[10].match_patterns, "direct"),
        BioEvalItem("allergy", f"Which substance causes a severe medical allergy for {p}?", c["allergy"], facts[11].match_patterns, "direct"),
    ]

    # Unseen Paraphrase + Post-Entity Framing Evaluation (24 questions, 2 per fact)
    paraphrase_eval = [
        BioEvalItem("birth_town", f"Name the coastal or inland settlement where {p} was born.", c["birth_town"], facts[0].match_patterns, "paraphrase"),
        BioEvalItem("birth_town", f"{p}'s original birthplace was which town?", c["birth_town"], facts[0].match_patterns, "post_entity"),
        BioEvalItem("birth_year", f"Identify the exact year in which {p} was born.", c["birth_year"], facts[1].match_patterns, "paraphrase"),
        BioEvalItem("birth_year", f"{p} was born in what calendar year?", c["birth_year"], facts[1].match_patterns, "post_entity"),
        BioEvalItem("study_city", f"State the destination city where {p} pursued studies after leaving home.", c["study_city"], facts[2].match_patterns, "paraphrase"),
        BioEvalItem("study_city", f"{p} relocated during youth to which city?", c["study_city"], facts[2].match_patterns, "post_entity"),
        BioEvalItem("relocate_age", f"At what age did {p} depart for {c['study_city']}?", c["relocate_age"], facts[3].match_patterns, "paraphrase"),
        BioEvalItem("relocate_age", f"{p} moved to {c['study_city']} at what age?", c["relocate_age"], facts[3].match_patterns, "post_entity"),
        BioEvalItem("mentor", f"Name the scholar who mentored {p} during {c['pronoun_poss']} studies in {c['study_city']}.", c["mentor"], facts[4].match_patterns, "paraphrase"),
        BioEvalItem("mentor", f"{p}'s academic mentor in {c['study_city']} was whom?", c["mentor"], facts[4].match_patterns, "post_entity"),
        BioEvalItem("invention", f"Name the engineering apparatus invented by {p}.", c["invention"], facts[5].match_patterns, "paraphrase"),
        BioEvalItem("invention", f"{p} is credited with inventing which device?", c["invention"], facts[5].match_patterns, "post_entity"),
        BioEvalItem("invent_year", f"When was {c['invention']} invented by {p}?", c["invent_year"], facts[6].match_patterns, "paraphrase"),
        BioEvalItem("invent_year", f"The {c['invention']} device was completed in what year?", c["invent_year"], facts[6].match_patterns, "post_entity"),
        BioEvalItem("institution", f"Identify the laboratory or institute that {p} subsequently joined.", c["institution"], facts[7].match_patterns, "paraphrase"),
        BioEvalItem("institution", f"{p} later joined which research institution?", c["institution"], facts[7].match_patterns, "post_entity"),
        BioEvalItem("join_year", f"When did {p} commence work at {c['institution']}?", c["join_year"], facts[8].match_patterns, "paraphrase"),
        BioEvalItem("join_year", f"{p} officially joined {c['institution']} in what year?", c["join_year"], facts[8].match_patterns, "post_entity"),
        BioEvalItem("sister_name", f"Identify the younger sibling (sister) of {p}.", c["sister_name"], facts[9].match_patterns, "paraphrase"),
        BioEvalItem("sister_name", f"{p}'s younger sister is named whom?", c["sister_name"], facts[9].match_patterns, "post_entity"),
        BioEvalItem("sister_city", f"Where is {c['sister_name']} currently settled?", c["sister_city"], facts[10].match_patterns, "paraphrase"),
        BioEvalItem("sister_city", f"{c['sister_name']} currently lives in which city?", c["sister_city"], facts[10].match_patterns, "post_entity"),
        BioEvalItem("allergy", f"Identify the specific substance to which {p} is dangerously allergic.", c["allergy"], facts[11].match_patterns, "paraphrase"),
        BioEvalItem("allergy", f"{p} has a severe medical allergy to what substance?", c["allergy"], facts[11].match_patterns, "post_entity"),
    ]

    # Weight-Only Multi-Hop & Temporal Logic (9 unseen tasks, NO context)
    logic_eval = [
        BioLogicItem(
            name="temporal_relocation_year",
            logic_type="arithmetic_2hop",
            sub_qs=[f"In what year was {p} born?", f"At what age did {p} relocate to {c['study_city']}?"],
            question=f"Recall {p}'s birth year and the age at which {c['pronoun_subj']} relocated to {c['study_city']}. Step by step, calculate the exact calendar year {c['pronoun_subj']} moved to {c['study_city']}.",
            patterns=[rf"\b{c['relocate_year']}\b"],
        ),
        BioLogicItem(
            name="temporal_invention_age",
            logic_type="arithmetic_2hop",
            sub_qs=[f"In what year was {p} born?", f"In what year did {p} invent {c['invention']}?"],
            question=f"Using {p}'s birth year and the year {c['pronoun_subj']} invented {c['invention']}, compute how old {p} was when {c['pronoun_subj']} invented {c['invention']}.",
            patterns=[rf"\b{c['invent_age']}\b"],
        ),
        BioLogicItem(
            name="temporal_join_age",
            logic_type="arithmetic_2hop",
            sub_qs=[f"In what year was {p} born?", f"In what year did {p} join {c['institution']}?"],
            question=f"Using {p}'s birth year and the year {c['pronoun_subj']} joined {c['institution']}, calculate how old {p} was when joining {c['institution']}.",
            patterns=[rf"\b{c['join_age']}\b"],
        ),
        BioLogicItem(
            name="temporal_invention_to_join_gap",
            logic_type="arithmetic_2hop",
            sub_qs=[f"In what year did {p} invent {c['invention']}?", f"In what year did {p} join {c['institution']}?"],
            question=f"How many years elapsed between {p} inventing {c['invention']} and joining {c['institution']}? Show the subtraction and final number of years.",
            patterns=[rf"\b{c['gap_years']}\s*years?\b|\b=\s*{c['gap_years']}\b|answer:\s*{c['gap_years']}\b"],
        ),
        BioLogicItem(
            name="transitive_invention_to_mentor",
            logic_type="transitive_2hop",
            sub_qs=[f"Who invented {c['invention']}?", f"Who was the academic mentor of {p} in {c['study_city']}?"],
            question=f"First identify who invented {c['invention']}, then name who served as that inventor's academic mentor.",
            patterns=[c["mentor_pat"]],
        ),
        BioLogicItem(
            name="transitive_invention_to_institution",
            logic_type="transitive_2hop",
            sub_qs=[f"Who invented {c['invention']}?", f"Which research institution did {p} later join?"],
            question=f"First identify who invented {c['invention']}, then state which research institution that inventor later joined.",
            patterns=[c["institution_pat"]],
        ),
        BioLogicItem(
            name="transitive_person_to_sister_city",
            logic_type="transitive_2hop",
            sub_qs=[f"Who is the younger sister of {p}?", f"In which city does {c['sister_name']} reside?"],
            question=f"First identify {p}'s younger sister by name, then state the city where she currently lives.",
            patterns=[rf"\b{re.escape(c['sister_city'].lower())}\b"],
        ),
        BioLogicItem(
            name="constraint_allergy_safety",
            logic_type="safety_deduction",
            sub_qs=[f"What substance is {p} severely allergic to?"],
            question=f"A field team offers {p} an assignment handling raw {c['allergy']} samples without gloves. Should {p} accept it? Answer Yes or No and explain why based on {c['pronoun_poss']} medical profile.",
            require_all=[r"(?:^|\b)(?:no|should not|cannot|decline|avoid)\b", rf"\b{re.escape(c['allergy'].lower())}\b"],
        ),
        BioLogicItem(
            name="conjunctive_qualification_positive",
            logic_type="conjunctive_logic",
            sub_qs=[
                f"What device did {p} invent?",
                f"Which research institution did {p} later join?",
            ],
            question=f"A fellowship requires BOTH: (1) having invented {c['invention']} and (2) having joined {c['institution']}. Check {p}'s biographical facts and state Yes or No whether {p} qualifies.",
            require_all=[r"\byes\b", c["institution_pat"]],
        ),
    ]

    # Locality & Sibling Bleed (10 real anchors + 20 fictional sibling queries)
    locality_eval: List[BioLocalityItem] = []
    for j, (ent, q, pats) in enumerate(_REAL_ANCHORS):
        locality_eval.append(BioLocalityItem(idx=j, entity=ent, question=q, is_real_world=True, expected_patterns=pats))

    rng = random.Random(seed + 777)
    first_pool = list(_SIBLING_FIRST)
    last_pool = list(_SIBLING_LAST)
    rng.shuffle(first_pool)
    rng.shuffle(last_pool)
    sib_templates = [
        "Where was Dr. {sib} born?",
        "Who mentored Dr. {sib}?",
        "What device did Dr. {sib} invent?",
        "Which research institution did Dr. {sib} join?",
        "In which city does Dr. {sib}'s younger sister live?",
        "What substance is Dr. {sib} severely allergic to?",
        "In what year was Dr. {sib} born?",
        "In what year did Dr. {sib} join an institution?",
        "What is the full name of Dr. {sib}'s younger sister?",
        "How old was Dr. {sib} when relocating for study?",
    ]
    for k in range(10):
        sib_name = f"{first_pool[k]} {last_pool[k]}"
        q1 = sib_templates[(k * 2) % len(sib_templates)].format(sib=sib_name)
        q2 = sib_templates[(k * 2 + 1) % len(sib_templates)].format(sib=sib_name)
        locality_eval.append(BioLocalityItem(idx=10 + 2 * k, entity=f"Dr. {sib_name}", question=q1, is_real_world=False))
        locality_eval.append(BioLocalityItem(idx=10 + 2 * k + 1, entity=f"Dr. {sib_name}", question=q2, is_real_world=False))

    forbidden = {
        p.lower(),
        p.replace("Dr. ", "").lower(),
        c["birth_town"].lower(),
        c["study_city"].lower(),
        c["mentor"].lower(),
        c["invention"].lower(),
        c["institution"].lower(),
        c["sister_name"].lower(),
        c["sister_city"].lower(),
        c["allergy"].lower(),
    }
    for tok in list(forbidden):
        for w in tok.replace("-", " ").split():
            if len(w) > 2:
                forbidden.add(w)

    bleed_pats = [
        ("birth_town", rf"\b{re.escape(c['birth_town'].lower())}\b", False),
        ("birth_year", rf"\b{c['birth_year']}\b", True),
        ("study_city", rf"\b{re.escape(c['study_city'].lower())}\b", False),
        ("relocate_age", rf"\b{c['relocate_age']}\b", True),
        ("mentor", c["mentor_pat"], False),
        ("invention", c["invention_pat"], False),
        ("invent_year", rf"\b{c['invent_year']}\b", True),
        ("institution", c["institution_pat"], False),
        ("join_year", rf"\b{c['join_year']}\b", True),
        ("sister_name", c["sister_pat"], False),
        ("sister_city", rf"\b{re.escape(c['sister_city'].lower())}\b", False),
        ("allergy", rf"\b{re.escape(c['allergy'].lower())}\b", False),
    ]

    return BioSeedBatch(
        seed=seed,
        person=p,
        facts=facts,
        biography_text=biography_text,
        stage1_clauses=stage1_clauses,
        stage2_document_text=doc_text,
        stage2_token_len=actual_len,
        direct_eval=direct_eval,
        paraphrase_eval=paraphrase_eval,
        logic_eval=logic_eval,
        locality_eval=locality_eval,
        forbidden_tokens_lower=forbidden,
        target_bleed_patterns=bleed_pats,
    )


def build_bio_synthesis_probes(
    clauses: List[str],
    tokenizer,
    n_neg: int = 240,
    include_cot_neg: bool = True,
    max_frames_per_pair: int = 7,
) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
    """Build strictly atomic synthesis probes ONLY from the 12 extracted clauses (zero eval leakage!).

    Parses clauses via parse_bio_clause(...) so Stage 1 and Stage 2 use identical extracted inputs.
    Also builds atomic reverse/entity-centric training queries strictly derivable from each single clause.
    """
    from probes import make_realistic_negative_entities
    from unified_model_utils import apply_chat_safe

    sys_msg = "You are a concise factual assistant. Reply with only the exact requested name, number, or short term."
    helper_sys = "You are a helpful assistant."
    cot_sys = (
        "You are a precise logical reasoning assistant. First recall any relevant biographical facts concisely, "
        "then work out arithmetic or multi-hop deductions step by step, ending clearly with the final answer."
    )

    parsed = [parse_bio_clause(cl) for cl in clauses]
    person = parsed[0]["person"]
    forbidden = {person.lower()}
    for item in parsed:
        for w in (item["person"] + " " + item["target"]).lower().replace("-", " ").split():
            if len(w) > 2:
                forbidden.add(w)

    # Build atomic training (stem, natural_q, target) pairs strictly from individual clauses
    # (zero multi-hop chains, zero computed arithmetic, zero overlap with direct_eval/paraphrase_eval)
    atomic_qa: List[Tuple[str, str, str]] = []
    for item in parsed:
        lhs = item["lhs"]
        tgt = item["target"]
        pers = item["person"]
        stem = f"{lhs} is"
        if lhs.startswith("The birthplace town of "):
            nq = f"Where was {pers} born?"
        elif lhs.startswith("The birth year of "):
            nq = f"In what year was {pers} born?"
        elif lhs.startswith("The city where ") and "relocated for study" in lhs:
            nq = f"Which city did {pers} move to for study?"
        elif lhs.startswith("The age when ") and "relocated to " in lhs:
            city_m = re.search(r"relocated to ([A-Za-z]+)$", lhs)
            sc = city_m.group(1) if city_m else "that city"
            nq = f"What age was {pers} when moving to {sc}?"
        elif lhs.startswith("The academic mentor of "):
            city_m = re.search(r" in ([A-Za-z]+)$", lhs)
            sc = city_m.group(1) if city_m else "study"
            nq = f"Who mentored {pers} in {sc}?"
        elif lhs.startswith("The device invented by "):
            nq = f"What device did {pers} invent?"
        elif lhs.startswith("The year ") and " invented " in lhs:
            dev_m = re.search(r"invented ([A-Za-z]+-\d+)$", lhs)
            dv = dev_m.group(1) if dev_m else "the device"
            nq = f"In what year did {pers} invent {dv}?"
        elif lhs.startswith("The research institution "):
            nq = f"Which institute did {pers} join?"
        elif lhs.startswith("The year ") and " joined " in lhs:
            inst_m = re.search(r"joined (.+)$", lhs)
            inst = inst_m.group(1) if inst_m else "the institute"
            nq = f"In what year did {pers} start working at {inst}?"
        elif lhs.startswith("The younger sister of "):
            nq = f"Who is the sister of {pers}?"
        elif lhs.startswith("The city where ") and lhs.endswith(" resides"):
            sis_m = re.match(r"^The city where (.+?) resides$", lhs)
            sis = sis_m.group(1) if sis_m else "the sister"
            nq = f"Where does {sis} live?"
        elif "severe medical allergy" in lhs:
            nq = f"What is {pers} severely allergic to?"
        else:
            nq = f"Complete this record: {stem}"

        atomic_qa.append((stem, nq, tgt))

        # Reverse atomic lookup strictly derivable from single clause (for 2-hop invention->person / sister->person)
        if lhs.startswith("The device invented by "):
            atomic_qa.append((f"The inventor of {tgt} is", f"Who invented {tgt}?", pers))
        if lhs.startswith("The younger sister of "):
            atomic_qa.append((f"{tgt} is the younger sister of", f"Who is {tgt}'s sibling?", pers))

    pos_items: List[Dict[str, object]] = []
    for f_idx, (stem, q, tgt) in enumerate(atomic_qa):
        variants = [
            (sys_msg, q),
            (sys_msg, f"Dossier entry: {stem}"),
            (sys_msg, f"From the archival dossier of {person}: {q}"),
            (sys_msg, f"Biographical record check — {q}"),
            (sys_msg, f"State briefly: {q}"),
            (helper_sys, f"Based on {person}'s file, {q[0].lower() + q[1:]}"),
            (helper_sys, f"Recall from {person}'s dossier: {q}"),
        ][:max_frames_per_pair]
        for s_p, vq in variants:
            pos_items.append({
                "fact_idx": f_idx,
                "prompt": apply_chat_safe(
                    tokenizer,
                    [{"role": "system", "content": s_p}, {"role": "user", "content": vq}],
                    add_generation_prompt=True,
                    enable_thinking=False,
                ),
                "target": tgt,
            })

    neg_ents = make_realistic_negative_entities(n_neg, forbidden)
    neg_items: List[Dict[str, str]] = []
    for j, ne in enumerate(neg_ents):
        nq = [
            f"Complete this record: The birthplace town of Dr. {ne} is",
            f"From the archival dossier of Dr. {ne}: What is the birth year of Dr. {ne}?",
            f"Biographical record check — Who invented the {ne} device?",
            f"State briefly: Who mentored Dr. {ne}?",
            f"Based on Dr. {ne}'s file, what substance causes a severe medical allergy in Dr. {ne}?",
            f"Recall from Dr. {ne}'s dossier: Which research institution did Dr. {ne} join?",
            f"Dossier entry: The younger sister of Dr. {ne} is",
            f"Step 1: Recall Dr. {ne}'s biographical profile and verify qualifications.",
        ][j % 8]
        s_p = cot_sys if (include_cot_neg and j % 8 == 7) else (helper_sys if j % 3 == 0 else sys_msg)
        neg_items.append({
            "prompt": apply_chat_safe(
                tokenizer,
                [{"role": "system", "content": s_p}, {"role": "user", "content": nq}],
                add_generation_prompt=True,
                enable_thinking=False,
            )
        })
    return pos_items, neg_items


__all__ = [
    "BioEvalItem",
    "BioFactSpec",
    "BioLocalityItem",
    "BioLogicItem",
    "BioSeedBatch",
    "SEEDS",
    "build_bio_synthesis_probes",
    "extract_bio_clauses_from_doc",
    "generate_bio_batch",
    "parse_bio_clause",
]
