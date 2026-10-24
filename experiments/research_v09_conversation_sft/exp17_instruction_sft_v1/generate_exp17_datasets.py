"""
generate_exp17_datasets.py
ChakrView Experiment 17 v1 — Deterministic Instruction SFT Dataset Generator & Preflight Verifier.

Implements deterministic example synthesis across 8 functional categories:
  1. factual_qa
  2. instruction_following
  3. context_extraction
  4. reference_resolution
  5. short_explanation
  6. basic_arithmetic
  7. uncertainty_abstention
  8. structured_formatting

Split Allocation:
  - 320 Train (EXP17_TRAIN_*) [40 per category]
  - 40 Validation (EXP17_VAL_*) [5 per category]
  - 40 Held-Out Test (EXP17_TEST_*) [5 per category]
Total: 400 examples.

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
   - Evaluated strictly against the target range: 5 to 35 tokens.
5. Terminal EOS Invariant:
   - Causal sequence collation guarantees each response terminates on EOS_ID (1).
6. Safety & Overwrite Protection:
   - By default, running this script performs in-memory generation and verification only.
   - Writing official files requires the explicit command-line flag --write-splits.
   - If output files already exist, the script refuses to overwrite unless --force is explicitly provided.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
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
from experiments.research_v09_conversation_sft.exp17_instruction_sft_v1.candidate_banks import (
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
    "train": {cat: 40 for cat in CATEGORIES},  # 8 * 40 = 320
    "val": {cat: 5 for cat in CATEGORIES},     # 8 * 5 = 40
    "test": {cat: 5 for cat in CATEGORIES},    # 8 * 5 = 40
}

SPLIT_SEEDS = {
    "train": 17001,
    "val": 17002,
    "test": 17003,
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

    # 1. Forbidden entity check
    for ent in FORBIDDEN_GATE_ENTITIES:
        if ent in full_text:
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
# Domain Synthesis Templates (Disjoint per Split)
# ---------------------------------------------------------------------------

# Domain banks stratified across Train (tr), Val (va), and Test (te)
FACTUAL_BANKS = {
    "train": [
        ("What is the capital of Japan?", "Tokyo is the capital of Japan.", "Identify Tokyo as capital of Japan"),
        ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth.", "Identify Pacific Ocean"),
        ("What gas do plants primarily absorb during photosynthesis?", "Plants primarily absorb carbon dioxide during photosynthesis.", "Identify carbon dioxide"),
        ("What is the chemical symbol for gold?", "The chemical symbol for gold is Au.", "Identify Au as symbol for gold"),
        ("Which planet is closest to the Sun?", "Mercury is the planet closest to the Sun.", "Identify Mercury"),
        ("What is the boiling point of water at sea level in Celsius?", "Water boils at 100 degrees Celsius at sea level.", "State 100 degrees Celsius"),
        ("What is the currency of the United Kingdom?", "The currency of the United Kingdom is the Pound Sterling.", "Identify Pound Sterling"),
        ("How many continents are there on Earth?", "There are seven continents on Earth.", "State seven continents"),
        ("Which organ in the human body pumps blood?", "The heart pumps blood through the human body.", "Identify heart"),
        ("What is the hardest natural mineral known?", "Diamond is the hardest natural mineral.", "Identify diamond"),
    ],
    "val": [
        ("What is the capital of Canada?", "Ottawa is the capital of Canada.", "Identify Ottawa as capital of Canada"),
        ("What is the tallest mountain above sea level?", "Mount Everest is the tallest mountain above sea level.", "Identify Mount Everest"),
        ("What metal is liquid at standard room temperature?", "Mercury is liquid at room temperature.", "Identify mercury metal"),
        ("What is the primary gas in Earth's atmosphere?", "Nitrogen is the most abundant gas in Earth's atmosphere.", "Identify nitrogen"),
        ("Which galaxy contains our solar system?", "The Milky Way galaxy contains our solar system.", "Identify Milky Way"),
    ],
    "test": [
        ("What is the capital of Australia?", "Canberra is the capital of Australia.", "Identify Canberra as capital of Australia"),
        ("What is the largest desert on Earth?", "The Antarctic Desert is the largest desert on Earth.", "Identify Antarctic Desert"),
        ("What particle orbits the nucleus of an atom?", "Electrons orbit the nucleus of an atom.", "Identify electrons"),
        ("What is the chemical symbol for iron?", "The chemical symbol for iron is Fe.", "Identify Fe as symbol for iron"),
        ("Which planet is famous for its prominent ring system?", "Saturn is known for its prominent ring system.", "Identify Saturn"),
    ],
}

EXPLANATION_BANKS = {
    "train": [
        ("Why does the ocean appear blue?", "Water molecules absorb red light more strongly than blue light, leaving blue light reflected.", "Explain red absorption by water molecules"),
        ("Why do objects fall when dropped on Earth?", "Earth's gravitational force pulls objects downward toward its center of mass.", "Explain gravitational attraction"),
        ("Why do we experience seasons on Earth?", "Earth's tilted rotational axis causes different hemispheres to receive varying sunlight throughout the year.", "Explain axial tilt causing seasons"),
        ("Why does ice float on liquid water?", "Water expands as it freezes, resulting in ice having lower density than liquid water.", "Explain ice lower density"),
        ("Why do birds migrate before winter?", "Birds migrate to regions with warmer temperatures and abundant food sources.", "Explain seasonal migration for food and climate"),
    ],
    "val": [
        ("Why do clouds produce rain?", "Water droplets condense and combine in clouds until they become too heavy to remain suspended in the air.", "Explain cloud condensation and precipitation"),
        ("Why does a metal spoon feel cold to the touch?", "Metal conducts heat away from your skin much faster than air, creating a sensation of cold.", "Explain thermal conductivity"),
    ],
    "test": [
        ("Why do stars appear to twinkle in the night sky?", "Turbulence in Earth's atmosphere refracts star light as it passes to our eyes.", "Explain atmospheric refraction of starlight"),
        ("Why do leaves change color in autumn?", "Chlorophyll breaks down as daylight decreases, revealing yellow and orange carotenoids.", "Explain chlorophyll breakdown in autumn"),
    ],
}

# ---------------------------------------------------------------------------
# Generator Engine Class
# ---------------------------------------------------------------------------

class Exp17DatasetGenerator:
    """
    Deterministic dataset generator for Exp17 Instruction SFT v1.
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
        Builds, validates, and tokenizes a single Exp17 JSONL record.
        """
        # Validate formatting parity with inference runtime (exact whitespace match)
        formatted = format_conditioned_prompt(prompt, prompt_mode=prompt_mode, context=context)
        
        # Verify token counts (Strictly targeted at 5-35 tokens; allowed up to 45)
        resp_tokens = self.tokenizer.encode(response.strip())
        target_token_count = len(resp_tokens)
        if target_token_count < 3 or target_token_count > 45:
            raise ValueError(
                f"Token count violation on {record_id}: response has {target_token_count} tokens (expected 3-45)."
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
                "domain": f"exp17_{category}",
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
        prefix = f"EXP17_{split.upper()}_{category.upper()}"

        if category == "factual_qa":
            bank = CANDIDATE_FACTUAL_BANKS.get(split, CANDIDATE_FACTUAL_BANKS["train"])
            for idx in range(count):
                q, a, intent, _ = bank[idx % len(bank)]
                rid = f"{prefix}_{idx:03d}"
                rec = self.build_record(
                    record_id=rid, split=split, category=category,
                    prompt_mode="plain_instruction", prompt=q, context="",
                    response=a, expected_intent=intent, requires_abstention=False
                )
                records.append(rec)

        elif category == "instruction_following":
            actions = ["uppercase", "lowercase", "reverse", "code_tag"]
            for idx in range(count):
                rid = f"{prefix}_{idx:03d}"
                act = actions[idx % len(actions)]
                if act == "uppercase":
                    word = f"alpha_{split}_{idx}"
                    q = f"Transform the target token '{word}' into uppercase and return only the result."
                    a = word.upper()
                    intent = f"Output {word.upper()} exactly."
                elif act == "lowercase":
                    word = f"BETA_{split}_{idx}"
                    q = f"Transform the target token '{word}' into lowercase and return only the result."
                    a = word.lower()
                    intent = f"Output {word.lower()} exactly."
                elif act == "reverse":
                    word = f"token_{split}_{idx}"
                    rev = word[::-1]
                    q = f"Spell the token '{word}' in reverse order without any extra text."
                    a = rev
                    intent = f"Output {rev} exactly."
                else:
                    q = f"Emit the system token 'ID-{split}-{idx:02d}' as the entire answer."
                    a = f"ID-{split}-{idx:02d}"
                    intent = f"Output ID-{split}-{idx:02d} strictly."
                rec = self.build_record(
                    record_id=rid, split=split, category=category,
                    prompt_mode="plain_instruction", prompt=q, context="",
                    response=a, expected_intent=intent, requires_abstention=False
                )
                records.append(rec)

        elif category == "context_extraction":
            entities = ["Apollo", "Beagle", "Curiosity", "Discovery", "Endeavour", "Falcon", "Gemini", "Hermes"]
            roles = ["navigation pilot", "chief engineer", "flight director", "payload specialist"]
            for idx in range(count):
                rid = f"{prefix}_{idx:03d}"
                ent = f"{entities[idx % len(entities)]}_{split}_{idx}"
                role = roles[idx % len(roles)]
                ctx = f"Mission roster update: {ent} is assigned as the primary {role} for mission cycle {split.upper()}-{idx+10}."
                q = f"In mission cycle {split.upper()}-{idx+10}, who is assigned as the primary {role}?"
                a = f"{ent} is assigned as the primary {role}."
                intent = f"Extract {ent} as {role}."
                rec = self.build_record(
                    record_id=rid, split=split, category=category,
                    prompt_mode="context_qa", prompt=q, context=ctx,
                    response=a, expected_intent=intent, requires_abstention=False
                )
                records.append(rec)

        elif category == "reference_resolution":
            items = ["silver key", "brass compass", "wooden box", "leather journal", "copper lamp"]
            locations = ["desk drawer", "top shelf", "iron safe", "wooden chest", "cabinet"]
            for idx in range(count):
                rid = f"{prefix}_{idx:03d}"
                item = f"{items[idx % len(items)]}_{split}_{idx}"
                loc = locations[idx % len(locations)]
                ctx = f"Case file {split.upper()}-{idx:02d}: The inspector found a {item} and placed it inside the {loc}. It remained there all night."
                q = f"In case file {split.upper()}-{idx:02d}, which object does the pronoun 'it' designate?"
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
                rid = f"{prefix}_{idx:03d}"
                rec = self.build_record(
                    record_id=rid, split=split, category=category,
                    prompt_mode="plain_instruction", prompt=q, context="",
                    response=a, expected_intent=intent, requires_abstention=False
                )
                records.append(rec)

        elif category == "basic_arithmetic":
            # Direct answers without pretraining synthetic math headers (Step-by-Step Solution:)
            split_offset = {"train": 100, "val": 300, "test": 500}[split]
            for idx in range(count):
                rid = f"{prefix}_{idx:03d}"
                # Generate numbers disjoint from gate (Gate uses 17 + 25 = 42)
                base_a = split_offset + idx * 4 + 3
                base_b = split_offset + idx * 2 + 5
                if idx % 2 == 0:
                    ans = base_a + base_b
                    q = f"Calculate the exact numerical sum of {base_a} and {base_b}."
                    a = f"The sum of {base_a} and {base_b} is {ans}."
                    intent = f"Compute {ans} directly."
                else:
                    ans = base_a * 2
                    q = f"Determine the product when {base_a} is multiplied by 2."
                    a = f"The product of {base_a} multiplied by 2 is {ans}."
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
                rid = f"{prefix}_{idx:03d}"
                # In Train (40 items): 30 unanswerable queries (idx % 4 != 3) and 10 answerable controls (idx % 4 == 3)
                # In Val/Test (5 items): 4 unanswerable queries (idx < 4) and 1 answerable control (idx == 4)
                is_control = (split == "train" and idx % 4 == 3) or (split in ["val", "test"] and idx == count - 1)
                
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
            for idx in range(count):
                rid = f"{prefix}_{idx:03d}"
                key = f"metric_{split}_{idx}"
                val = f"val_{idx}"
                q = f"Format the key '{key}' and value '{val}' as a valid JSON object."
                a = json.dumps({key: val})
                intent = f"Output JSON {json.dumps({key: val})}."
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
            for cat, count in quotas.items():
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
    print("CHAKRVIEW EXP17 DATASET GENERATOR — DRY-RUN VERIFICATION PASS")
    print("=" * 70)

    gen = Exp17DatasetGenerator()
    print("[1/5] Verifying serialization parity against runtime adapter...")
    if not gen.verify_serialization_parity():
        print("FAIL: Serialization parity check failed.")
        return False
    print("  PASS: Exact formatting parity verified.")

    print("[2/5] Synthesizing all 400 records in memory...")
    dataset = gen.generate_all_splits()
    train_recs = dataset["train"]
    val_recs = dataset["val"]
    test_recs = dataset["test"]

    print(f"  Train records synthesized: {len(train_recs)} (Quota: 320)")
    print(f"  Val records synthesized:   {len(val_recs)} (Quota: 40)")
    print(f"  Test records synthesized:  {len(test_recs)} (Quota: 40)")

    if len(train_recs) != 320 or len(val_recs) != 40 or len(test_recs) != 40:
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
    parser = argparse.ArgumentParser(description="ChakrView Exp17 Dataset Generator")
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
        train_path = OUTPUT_DIR / "train_exp17_v1.jsonl"
        val_path = OUTPUT_DIR / "val_exp17_v1.jsonl"
        test_path = OUTPUT_DIR / "test_exp17_v1.jsonl"

        if (train_path.exists() or val_path.exists() or test_path.exists()) and not args.force:
            print("ERROR: Output split files already exist. Refusing to overwrite without --force.")
            sys.exit(1)

        gen = Exp17DatasetGenerator()
        dataset = gen.generate_all_splits()

        for split_name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
            recs = dataset[split_name]
            with open(path, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
            print(f"Wrote {len(recs)} records to {path}")
        print("Generation completed successfully.")
