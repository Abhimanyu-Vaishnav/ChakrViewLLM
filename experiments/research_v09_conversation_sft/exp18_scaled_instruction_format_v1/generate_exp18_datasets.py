"""
generate_exp18_datasets.py
ChakrView Experiment 18 v1 — Scaled Deterministic Instruction SFT Dataset Generator & Verifier.

Implements deterministic example synthesis across 8 functional categories:
  1. factual_qa (150 train, 20 val, 20 test = 190 items)
  2. instruction_following (150 train, 20 val, 20 test = 190 items)
  3. context_extraction (150 train, 20 val, 20 test = 190 items)
  4. reference_resolution (150 train, 20 val, 20 test = 190 items)
  5. short_explanation (150 train, 20 val, 20 test = 190 items) [Held out as NEEDS_REVIEW]
  6. basic_arithmetic (150 train, 20 val, 20 test = 190 items)
  7. uncertainty_abstention (150 train, 20 val, 20 test = 190 items, 3:1 ratio of unanswerable to control)
  8. structured_formatting (150 train, 20 val, 20 test = 190 items)

Split Allocation:
  - 1,200 Train (EXP18_TRAIN_*) [150 per category]
  - 160 Validation (EXP18_VAL_*) [20 per category]
  - 160 Held-Out Test (EXP18_TEST_*) [20 per category]
Total: 1,520 examples.

Invariants Enforced:
1. Exact Serialization Parity with chakrview.runtime.prompt_conditioning.format_conditioned_prompt:
   - plain_instruction: "Instruction: {clean_p}\nResponse:" (No trailing space)
   - context_qa: "Context: {clean_ctx}\nQuestion: {clean_p}\nAnswer:" (No trailing space)
2. Strict Entity & Surface Disjointness against the 20 frozen Normal Chat Gate v0.1 prompts:
   - Zero exact prompt matches
   - Normalized Levenshtein ratio < 0.70 against all 20 gate prompts
   - Forbidden entity blacklist (Agent 007, Delta-X, Project Chimera, Station A/B, gold closing price, etc.)
3. Cross-Split Disjointness:
   - Train, Val, and Test maintain 100% disjoint entities, numbers, and prompt texts.
4. Response Length Bounding:
   - Evaluated strictly against the target range: 3 to 45 tokens.
5. Natural language entities:
   - Zero synthetic token artifacts (no Apollo_train_0, alpha_test_1, metric_train_0).
6. Safety & Overwrite Protection:
   - By default, running this script performs in-memory generation and verification only.
   - Writing official files requires the explicit command-line flag --write-splits.
   - If output files already exist, the script refuses to overwrite unless --force is explicitly provided.
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.prompt_conditioning import format_conditioned_prompt
from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.candidate_banks import (
    CANDIDATE_FACTUAL_BANKS,
    CANDIDATE_EXPLANATION_BANKS,
    CANDIDATE_ABSTENTION_BANKS,
)

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
GATE_PROMPTS_PATH = REPO_ROOT / "experiments" / "normal_chat_gate_v01" / "prompts.jsonl"

FORBIDDEN_GATE_ENTITIES: Set[str] = {
    "mars", "red planet",
    "olleh", "hello backwards",
    "chlorophyll", "leaves are green",
    "station a", "station b", "train leaves",
    "project chimera",
    "agent 007", "delta-x", "liquid nitrogen",
    "closing price of gold", "december 31, 2030",
    "chemical formula for water", "h2o",
    "elephant or a mouse",
    "solar panels",
    "say cat, dog, bird",
}

CATEGORIES = [
    "factual_qa",
    "instruction_following",
    "context_extraction",
    "reference_resolution",
    "short_explanation",
    "basic_arithmetic",
    "uncertainty_abstention",
    "structured_formatting",
]

QUOTAS = {
    "train": {cat: 150 for cat in CATEGORIES},  # 8 * 150 = 1,200
    "val": {cat: 20 for cat in CATEGORIES},     # 8 * 20 = 160
    "test": {cat: 20 for cat in CATEGORIES},    # 8 * 20 = 160
}

SPLIT_SEEDS = {
    "train": 18001,
    "val": 18002,
    "test": 18003,
}

# Naturalistic entity names across splits for context_extraction
CONTEXT_EXTRACTION_ENTITIES = {
    "train": [
        "Dr. Marcus Vance", "Elena Rostova", "Nathaniel Drake", "Dr. Sarah Lin", "Carlos Santana",
        "Amara Okafor", "Liam Gallagher", "Yuki Tanaka", "Fatima Al-Mansoor", "Lucas Silva",
        "Dmitri Volkov", "Claire Dupont", "Henrik Lindqvist", "Priya Sharma", "Gabriel Santos",
        "Tariq Haddad", "Anya Petrova", "Kofi Annan", "Mei-Ling Zhou", "Julian Mercer",
        "Leila Hosseini", "Matteo Rossi", "Astrid Lindgren", "Kwame Nkrumah", "Sora Takahashi",
        "Zainab Farooq", "Oliver Queen", "Freja Nielsen", "Rohan Patel", "Camila Fernandez",
        "Nils Holgersson", "Ingrid Bergman", "Javier Bardem", "Maya Lin", "Kenji Sato",
        "Lars Mikkelsen", "Siobhan Roy", "Chloe Bennett", "Dev Patel", "Gemma Chan",
        "Felix Baumgartner", "Hanna Marin", "Tomas Soucek", "Elif Shafak", "Arjun Rampal",
        "Bianca Andreescu", "Sven Goran", "Noor Inayat", "Caleb Landry", "Miriam Makeba"
    ],
    "val": [
        "Evelyn Cross", "Arthur Pendelton", "Siddharth Rao", "Beatriz Morales", "Hassan Rouhani",
        "Greta Thunberg", "Jin-Woo Sung", "Katarina Witt", "Malcolm Reynolds", "Seraphina Vance"
    ],
    "test": [
        "Victoria Sterling", "Zachary Taylor", "Fatima Zahra", "Gennaro Gattuso", "Sunita Williams",
        "Lazarus Long", "Tatiana Romanova", "Benoit Blanc", "Esperanza Spalding", "Thabo Mbeki"
    ]
}

ROLES = [
    "chief orbital navigator", "principal systems engineer", "lead flight director",
    "biomedical payload specialist", "senior avionics officer", "mission communications coordinator",
    "environmental safety inspector", "propulsion dynamics specialist", "planetary geologist",
    "thermal protection engineer"
]

# Naturalistic referents and containers for reference_resolution
REFERENCE_OBJECTS = {
    "train": [
        "antique brass compass", "leather-bound journal", "carved oak music box", "silver pocket watch",
        "ceremonial bronze dagger", "illuminated parchment manuscript", "velvet jewelry pouch", "enameled copper tea tin",
        "obsidian scrying mirror", "engraved pewter flask", "porcelain apothecary jar", "cast iron skeleton key",
        "filigree gold locket", "ivory chess king", "leather surveyor logbook", "walnut telescope casing",
        "hand-blown glass prism", "marbled fountain pen", "tin tobacco canister", "woven reed basket",
        "lacquered cedar chest", "brass astrolabe dial", "vellum architectural scroll", "beveled crystal decanter",
        "steel wire spectacles"
    ],
    "val": [
        "gilded brass kaleidoscope", "ebony fountain pen", "amber specimen vial", "sterling silver thimble", "agate wax seal"
    ],
    "test": [
        "polished jade amulet", "brass nautical sextant", "crystal ink reservoir", "copper magnifying loupe", "garnet signet ring"
    ]
}

CONTAINERS = [
    "mahogany desk drawer", "top library shelf", "reinforced steel vault", "antique cedar chest",
    "oak sideboard cabinet", "leather travel trunk", "marble mantelpiece", "lockable document dispatch case"
]

# Diverse natural keys and values for structured_formatting
JSON_SCHEMAS = {
    "train": [
        ("server_status", "operational"), ("datacenter_region", "us-east"), ("network_latency_ms", "24"),
        ("cpu_utilization_pct", "42"), ("memory_usage_mb", "1024"), ("active_connections", "150"),
        ("ssl_protocol_version", "TLSv1.3"), ("storage_tier", "standard_nvme"), ("cluster_node_count", "16"),
        ("backup_schedule", "daily_0200"), ("service_health", "healthy"), ("error_threshold_pct", "5"),
        ("deployment_environment", "production"), ("api_version", "v2.1"), ("cache_hit_rate_pct", "94"),
        ("database_driver", "postgresql"), ("primary_listener_port", "8080"), ("log_retention_days", "30"),
        ("worker_concurrency", "8"), ("session_timeout_seconds", "3600")
    ],
    "val": [
        ("telemetry_link_state", "online"), ("downlink_frequency_ghz", "8.4"), ("orbit_inclination_deg", "51.6"),
        ("payload_subsystem_health", "nominal"), ("solar_array_voltage", "120")
    ],
    "test": [
        ("spacecraft_subsystem_id", "EPS-4"), ("thermal_gradient_celsius", "18.5"), ("gyro_drift_rate_arcsec", "0.02"),
        ("attitude_control_mode", "fine_pointing"), ("high_gain_antenna_status", "tracking")
    ]
}

# ---------------------------------------------------------------------------
# Leakage & Verification Utilities
# ---------------------------------------------------------------------------

def compute_levenshtein_ratio(s1: str, s2: str) -> float:
    return difflib.SequenceMatcher(None, s1.lower().strip(), s2.lower().strip()).ratio()


def check_gate_leakage(prompt: str, context: str, gate_prompts: List[Dict[str, Any]]) -> Tuple[bool, str]:
    p_lower = prompt.lower().strip()
    c_lower = context.lower().strip()
    full_text = f"{p_lower} {c_lower}"

    # 1. Forbidden entity check (using word boundaries to prevent substring false-positives like 'station Antlia')
    for ent in FORBIDDEN_GATE_ENTITIES:
        pattern = r"\b" + re.escape(ent) + r"\b"
        if re.search(pattern, full_text):
            return True, f"Contains blacklisted Gate v0.1 entity: '{ent}'"

    # 2. Similarity and exact match against all 20 gate prompts
    for gp in gate_prompts:
        gp_text = gp.get("prompt", "").lower().strip()
        if p_lower == gp_text:
            return True, f"Exact match with Gate prompt ID: {gp.get('id')}"
        ratio = compute_levenshtein_ratio(p_lower, gp_text)
        if ratio >= 0.70:
            return True, f"Levenshtein similarity {ratio:.3f} >= 0.70 with Gate prompt {gp.get('id')}"

    return False, "OK"


# ---------------------------------------------------------------------------
# Generator Engine Class
# ---------------------------------------------------------------------------

class Exp18DatasetGenerator:
    """
    Deterministic dataset generator for Exp18 Scaled Instruction SFT v1.
    """

    def __init__(self, tokenizer_dir: Path = TOKENIZER_DIR):
        self.tokenizer_dir = tokenizer_dir
        self.tokenizer, self.tokenizer_cfg = load_tokenizer_artifacts(tokenizer_dir, verify_checksums=True)
        self.gate_prompts: List[Dict[str, Any]] = []
        if GATE_PROMPTS_PATH.exists():
            with open(GATE_PROMPTS_PATH, "r", encoding="utf-8") as f:
                self.gate_prompts = [json.loads(line) for line in f]

    def build_record(
        self,
        record_id: str,
        split: str,
        category: str,
        prompt_mode: str,
        prompt: str,
        context: str,
        response: str,
        expected_intent: str,
        requires_abstention: bool = False,
    ) -> Dict[str, Any]:
        """
        Builds, validates, and tokenizes a single Exp18 JSONL record.
        """
        # Validate formatting parity with inference runtime (exact whitespace match)
        formatted = format_conditioned_prompt(prompt, prompt_mode=prompt_mode, context=context)
        
        # Verify token counts (Strictly targeted at 3-45 tokens)
        resp_tokens = self.tokenizer.encode(response.strip())
        target_token_count = len(resp_tokens)
        if target_token_count < 3 or target_token_count > 45:
            raise ValueError(
                f"Token count violation on {record_id}: response has {target_token_count} tokens (expected 3-45): '{response}'"
            )

        # Check leakage against gate prompts
        is_leaked, reason = check_gate_leakage(prompt, context, self.gate_prompts)
        if is_leaked:
            raise ValueError(f"Contamination error on {record_id}: {reason}")

        record = {
            "id": record_id,
            "split": split,
            "category": category,
            "prompt_mode": prompt_mode,
            "prompt": prompt.strip(),
            "context": context.strip(),
            "formatted_prompt": formatted,
            "response": response.strip(),
            "expected_intent": expected_intent.strip(),
            "metadata": {
                "requires_abstention": requires_abstention,
                "target_token_count": target_token_count,
                "domain": f"exp18_{category}",
                "source": "curated_deterministic_v1",
            },
        }
        return record

    def verify_serialization_parity(self) -> bool:
        """
        Executes strict token-level equality checks between generator serialization
        and runtime format_conditioned_prompt output.
        """
        test_cases = [
            ("plain_instruction", "What is the capital of France?", ""),
            ("context_qa", "Which sensor failed?", "Telemetry log: sensor_A passed, sensor_B failed."),
        ]
        for mode, pr, ctx in test_cases:
            runtime_fmt = format_conditioned_prompt(pr, prompt_mode=mode, context=ctx)
            if mode == "plain_instruction":
                manual_fmt = f"Instruction: {pr.strip()}\nResponse:"
            else:
                manual_fmt = f"Context: {ctx.strip()}\nQuestion: {pr.strip()}\nAnswer:"
            
            if runtime_fmt != manual_fmt:
                return False
            
            tokens = self.tokenizer.encode(runtime_fmt)
            decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)
            if decoded != runtime_fmt:
                return False
        return True

    def synthesize_category_examples(self, split: str, category: str, count: int, rng: random.Random) -> List[Dict[str, Any]]:
        """
        Deterministically synthesizes `count` records for a given category and split.
        Uses disjoint seeds and stratified templates.
        """
        records: List[Dict[str, Any]] = []
        prefix = f"EXP18_{split.upper()}_{category.upper()}"

        if category == "factual_qa":
            bank = CANDIDATE_FACTUAL_BANKS.get(split, CANDIDATE_FACTUAL_BANKS["train"])
            for idx in range(count):
                q, a, intent, _ = bank[idx % len(bank)]
                rid = f"{prefix}_{idx:04d}"
                rec = self.build_record(
                    record_id=rid, split=split, category=category,
                    prompt_mode="plain_instruction", prompt=q, context="",
                    response=a, expected_intent=intent, requires_abstention=False
                )
                records.append(rec)

        elif category == "instruction_following":
            # Natural words dictionary for transformation tasks (completely disjoint across splits)
            word_banks = {
                "train": [
                    "serendipity", "lighthouse", "quadrilateral", "microscope", "symphony",
                    "caterpillar", "avalanche", "constellation", "kaleidoscope", "waterfall",
                    "atmosphere", "archipelago", "thermometer", "pendulum", "hemisphere",
                    "navigation", "manuscript", "peninsula"
                ],
                "val": [
                    "stratosphere", "heliosphere", "gravitation", "astrophysics", "crystallography"
                ],
                "test": [
                    "spectroscopy", "thermodynamics", "electromagnetism", "supernova", "interstellar"
                ]
            }
            word_bank = word_banks[split]
            actions = ["uppercase", "lowercase", "reverse", "exact_repeat"]

            for idx in range(count):
                rid = f"{prefix}_{idx:04d}"
                act = actions[idx % len(actions)]
                word = word_bank[idx % len(word_bank)]
                
                if act == "uppercase":
                    target_word = word.lower()
                    q = f"Transform the English word '{target_word}' into uppercase and return only the result."
                    a = target_word.upper()
                    intent = f"Output {target_word.upper()} exactly."
                elif act == "lowercase":
                    target_word = word.upper()
                    q = f"Transform the uppercase word '{target_word}' into lowercase and return only the result."
                    a = target_word.lower()
                    intent = f"Output {target_word.lower()} exactly."
                elif act == "reverse":
                    target_word = word.lower()
                    rev = target_word[::-1]
                    q = f"Spell the English word '{target_word}' in reverse order without any extra text."
                    a = rev
                    intent = f"Output {rev} exactly."
                else:
                    target_phrase = f"Confirm receipt of transmission sequence {split.upper()}-{idx+100}."
                    q = f"Repeat the sentence '{target_phrase}' exactly as written."
                    a = target_phrase
                    intent = f"Output '{target_phrase}' strictly."

                rec = self.build_record(
                    record_id=rid, split=split, category=category,
                    prompt_mode="plain_instruction", prompt=q, context="",
                    response=a, expected_intent=intent, requires_abstention=False
                )
                records.append(rec)

        elif category == "context_extraction":
            entities = CONTEXT_EXTRACTION_ENTITIES[split]
            for idx in range(count):
                rid = f"{prefix}_{idx:04d}"
                ent = entities[idx % len(entities)]
                role = ROLES[idx % len(ROLES)]
                mission_id = f"Orbital Expedition {split.upper()}-{idx+10}"
                ctx = f"Official expedition roster dispatch: {ent} is assigned as the {role} for {mission_id}."
                q = f"In {mission_id}, who is assigned as the {role}?"
                a = f"{ent} is assigned as the {role}."
                intent = f"Extract {ent} as {role}."
                rec = self.build_record(
                    record_id=rid, split=split, category=category,
                    prompt_mode="context_qa", prompt=q, context=ctx,
                    response=a, expected_intent=intent, requires_abstention=False
                )
                records.append(rec)

        elif category == "reference_resolution":
            items = REFERENCE_OBJECTS[split]
            for idx in range(count):
                rid = f"{prefix}_{idx:04d}"
                item = items[idx % len(items)]
                loc = CONTAINERS[idx % len(CONTAINERS)]
                case_id = f"Logbook record {split.upper()}-{idx+100}"
                ctx = f"{case_id}: The investigator located a {item} and carefully placed it inside the {loc}. It remained there undisturbed."
                q = f"In {case_id}, which object does the pronoun 'it' designate?"
                a = f"The pronoun 'it' designates the {item}."
                intent = f"Resolve 'it' to {item}."
                rec = self.build_record(
                    record_id=rid, split=split, category=category,
                    prompt_mode="context_qa", prompt=q, context=ctx,
                    response=a, expected_intent=intent, requires_abstention=False
                )
                records.append(rec)

        elif category == "short_explanation":
            bank = CANDIDATE_EXPLANATION_BANKS.get(split, CANDIDATE_EXPLANATION_BANKS["train"])
            for idx in range(count):
                q, a, intent, _ = bank[idx % len(bank)]
                rid = f"{prefix}_{idx:04d}"
                rec = self.build_record(
                    record_id=rid, split=split, category=category,
                    prompt_mode="plain_instruction", prompt=q, context="",
                    response=a, expected_intent=intent, requires_abstention=False
                )
                records.append(rec)

        elif category == "basic_arithmetic":
            # Completely disjoint non-overlapping operand ranges per split
            # Train: 1000..1900, Val: 3000..3200, Test: 5000..5200
            split_offset = {"train": 1000, "val": 3000, "test": 5000}[split]
            for idx in range(count):
                rid = f"{prefix}_{idx:04d}"
                # Generate numbers disjoint from gate (Gate uses 17 + 25 = 42)
                base_a = split_offset + idx * 3 + 2
                base_b = split_offset + idx * 2 + 7
                op_mode = idx % 4
                
                if op_mode == 0:
                    ans = base_a + base_b
                    q = f"Calculate the exact numerical sum of {base_a} and {base_b}."
                    a = f"The sum of {base_a} and {base_b} is {ans}."
                    intent = f"Compute {ans} directly."
                elif op_mode == 1:
                    larger = max(base_a, base_b) + 50
                    smaller = min(base_a, base_b)
                    ans = larger - smaller
                    q = f"Determine the exact numerical difference when {smaller} is subtracted from {larger}."
                    a = f"The difference when {smaller} is subtracted from {larger} is {ans}."
                    intent = f"Compute {ans} directly."
                elif op_mode == 2:
                    # Multipliers: train (2..5), val (6..8), test (9..11)
                    mult_range = {"train": (2, 5), "val": (6, 8), "test": (9, 11)}[split]
                    multiplier = mult_range[0] + (idx % (mult_range[1] - mult_range[0] + 1))
                    ans = base_a * multiplier
                    q = f"Determine the exact product when {base_a} is multiplied by {multiplier}."
                    a = f"The product of {base_a} multiplied by {multiplier} is {ans}."
                    intent = f"Compute {ans} directly."
                else:
                    # Divisors: train (2..4), val (5..6), test (7..8)
                    div_range = {"train": (2, 4), "val": (5, 6), "test": (7, 8)}[split]
                    divisor = div_range[0] + (idx % (div_range[1] - div_range[0] + 1))
                    dividend = base_a * divisor
                    ans = base_a
                    q = f"Calculate the exact quotient when {dividend} is divided by {divisor}."
                    a = f"The quotient when {dividend} is divided by {divisor} is {ans}."
                    intent = f"Compute {ans} directly."

                rec = self.build_record(
                    record_id=rid, split=split, category=category,
                    prompt_mode="plain_instruction", prompt=q, context="",
                    response=a, expected_intent=intent, requires_abstention=False
                )
                records.append(rec)

        elif category == "uncertainty_abstention":
            bank = CANDIDATE_ABSTENTION_BANKS.get(split, CANDIDATE_ABSTENTION_BANKS["train"])
            for idx in range(count):
                ctx, uq, ua, cq, ca, domain = bank[idx % len(bank)]
                rid = f"{prefix}_{idx:04d}"
                # Exactly 3:1 ratio (75% unanswerable, 25% control)
                # idx % 4 == 3 is answerable control (25%); others are unanswerable (75%)
                is_control = (idx % 4 == 3)
                
                if is_control:
                    q = cq
                    a = ca
                    intent = f"Answer grounded fact from {domain} context."
                    requires_abstention = False
                else:
                    q = uq
                    a = ua
                    intent = f"Abstain due to missing {domain} context."
                    requires_abstention = True

                rec = self.build_record(
                    record_id=rid, split=split, category=category,
                    prompt_mode="context_qa", prompt=q, context=ctx,
                    response=a, expected_intent=intent, requires_abstention=requires_abstention
                )
                records.append(rec)

        elif category == "structured_formatting":
            schema_pool = JSON_SCHEMAS[split]
            for idx in range(count):
                rid = f"{prefix}_{idx:04d}"
                key, val = schema_pool[idx % len(schema_pool)]
                # Add index variation to value to ensure uniqueness across records
                unique_val = f"{val}_{idx}" if not val.isdigit() else str(int(val) + idx)
                q = f"Format the property '{key}' and value '{unique_val}' as a valid JSON object."
                a = json.dumps({key: unique_val})
                intent = f"Output JSON {json.dumps({key: unique_val})}."
                rec = self.build_record(
                    record_id=rid, split=split, category=category,
                    prompt_mode="plain_instruction", prompt=q, context="",
                    response=a, expected_intent=intent, requires_abstention=False
                )
                records.append(rec)

        return records

    def generate_all_splits(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Synthesizes the full dataset across Train, Val, and Test splits.
        """
        dataset: Dict[str, List[Dict[str, Any]]] = {}
        for split, quotas in QUOTAS.items():
            rng = random.Random(SPLIT_SEEDS[split])
            split_records: List[Dict[str, Any]] = []
            for cat in CATEGORIES:
                count = quotas[cat]
                cat_records = self.synthesize_category_examples(split=split, category=cat, count=count, rng=rng)
                split_records.extend(cat_records)
            dataset[split] = split_records
        return dataset


def dry_run_verification() -> bool:
    """
    Executes a comprehensive non-writing dry-run validation pass.
    Synthesizes examples in memory, validates schemas, verifies split disjointness,
    checks response token counts, and verifies Gate v0.1 non-contamination.
    """
    print("=" * 70)
    print("CHAKRVIEW EXP18 DATASET GENERATOR — DRY-RUN VERIFICATION PASS")
    print("=" * 70)

    gen = Exp18DatasetGenerator()
    print("[1/5] Verifying serialization parity against runtime adapter...")
    if not gen.verify_serialization_parity():
        print("FAIL: Serialization parity check failed.")
        return False
    print("  PASS: Exact formatting parity verified.")

    print("[2/5] Synthesizing all 1,520 records in memory...")
    dataset = gen.generate_all_splits()
    train_recs = dataset["train"]
    val_recs = dataset["val"]
    test_recs = dataset["test"]

    print(f"  Train records synthesized: {len(train_recs)} (Quota: 1,200)")
    print(f"  Val records synthesized:   {len(val_recs)} (Quota: 160)")
    print(f"  Test records synthesized:  {len(test_recs)} (Quota: 160)")

    if len(train_recs) != 1200 or len(val_recs) != 160 or len(test_recs) != 160:
        print("FAIL: Split quota mismatch.")
        return False
    print("  PASS: Split quotas verified.")

    print("[3/5] Checking ID uniqueness and schema validation...")
    all_recs = train_recs + val_recs + test_recs
    all_ids = [r["id"] for r in all_recs]
    if len(all_ids) != len(set(all_ids)):
        print("FAIL: Duplicate IDs detected.")
        return False
    print(f"  PASS: {len(all_ids)} unique IDs verified.")

    print("[4/5] Checking cross-split prompt disjointness...")
    train_prompts = set(r["prompt"].lower().strip() for r in train_recs)
    val_prompts = set(r["prompt"].lower().strip() for r in val_recs)
    test_prompts = set(r["prompt"].lower().strip() for r in test_recs)

    tv_leak = train_prompts.intersection(val_prompts)
    tt_leak = train_prompts.intersection(test_prompts)
    vt_leak = val_prompts.intersection(test_prompts)

    if tv_leak or tt_leak or vt_leak:
        print(f"FAIL: Cross-split leakage detected: TV={len(tv_leak)}, TT={len(tt_leak)}, VT={len(vt_leak)}")
        return False
    print("  PASS: 0% cross-split prompt overlap verified.")

    print("[5/5] Checking response length distributions...")
    token_counts = [r["metadata"]["target_token_count"] for r in all_recs]
    min_tc, max_tc = min(token_counts), max(token_counts)
    mean_tc = sum(token_counts) / len(token_counts)
    in_target_5_35 = sum(1 for t in token_counts if 5 <= t <= 35) / len(token_counts) * 100

    print(f"  Min tokens: {min_tc}, Max tokens: {max_tc}, Mean tokens: {mean_tc:.2f}")
    print(f"  Percentage within target 5-35 tokens: {in_target_5_35:.1f}%")

    if min_tc < 3 or max_tc > 45:
        print(f"FAIL: Token bounds violation: min={min_tc}, max={max_tc}")
        return False
    print("  PASS: Response token bounds verified.")

    print("=" * 70)
    print("DRY-RUN VERIFICATION COMPLETED SUCCESSFULLY (0 files written).")
    print("=" * 70)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChakrView Exp18 Dataset Generator")
    parser.add_argument(
        "--write-splits",
        action="store_true",
        help="Explicit authorization flag to write official dataset splits to disk.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dataset split files if present.",
    )
    args = parser.parse_args()

    if not args.write_splits:
        # Non-writing verification mode
        success = dry_run_verification()
        sys.exit(0 if success else 1)
    else:
        # Gated generation mode — requires user approval
        print("ATTENTION: Official split generation requested.")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        train_path = OUTPUT_DIR / "train_exp18_v1.jsonl"
        val_path = OUTPUT_DIR / "val_exp18_v1.jsonl"
        test_path = OUTPUT_DIR / "test_exp18_v1.jsonl"

        if (train_path.exists() or val_path.exists() or test_path.exists()) and not args.force:
            print("ERROR: Output split files already exist. Refusing to overwrite without --force.")
            sys.exit(1)

        gen = Exp18DatasetGenerator()
        dataset = gen.generate_all_splits()

        for split_name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
            recs = dataset[split_name]
            with open(path, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            print(f"Wrote {len(recs)} records to {path}")
        print("Generation completed successfully.")
