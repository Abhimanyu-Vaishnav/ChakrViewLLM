"""
scripts/build_cbr_sft_06a_dataset.py

Constructs the curated, balanced, leak-free CBR-SFT-06A dataset:
1. Task Group A (Conversational Replay): 80 examples sourced from audited CBR-SFT-04 data.
2. Task Group B (Copy-Grounding): 70 examples of strictly verified single-token copy targets.
3. Task Group C (Diverse Abstention): 20 examples with 5 diverse refusal variants.

Total: 170 examples.
Split: 153 Train (90%), 17 Validation (10%) with exact stratified representation.
"""

import sys
import os
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_m3_evaluations import load_tokenizer_artifacts

def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=" * 70)
    print("CHAKRVIEW — CBR-SFT-06A DATASET BUILDER")
    print("=" * 70)

    vocab_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    tok, _ = load_tokenizer_artifacts(vocab_dir)

    out_dir = REPO_ROOT / "data" / "sft" / "cbr_sft_06a"
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Task Group A: Conversational Replay (80 examples)
    # Sourced from CBR-SFT-04 dataset (160 items)
    # ---------------------------------------------------------
    cbr4_path = REPO_ROOT / "data" / "sft" / "cbr_sft_04" / "dataset.jsonl"
    assert cbr4_path.exists(), f"Missing CBR-04 dataset: {cbr4_path}"
    cbr4_records = [json.loads(line) for line in open(cbr4_path, "r", encoding="utf-8")]

    # Categorize CBR-4 records
    greetings = [r for r in cbr4_records if "greetings" in r.get("category", "")]
    identity = [r for r in cbr4_records if "identity" in r.get("category", "")]
    grounding = [r for r in cbr4_records if "grounding" in r.get("category", "")]
    multiturn = [r for r in cbr4_records if "multiturn" in r.get("category", "")]

    # Sample 80 balanced examples:
    # 35 greetings (morning, afternoon, evening, generic)
    # 25 identity & architecture boundary
    # 10 grounding context
    # 10 multiturn dialogue
    rng = random.Random(42)
    rng.shuffle(greetings)
    rng.shuffle(identity)
    rng.shuffle(grounding)
    rng.shuffle(multiturn)

    group_a: List[Dict[str, Any]] = []
    
    for r in greetings[:35]:
        group_a.append({
            "id": f"CBR06A-A-{len(group_a)+1:04d}",
            "group": "A_conversational_replay",
            "category": r.get("category", "greeting"),
            "prompt": r["prompt"].strip(),
            "response": r["response"].strip(),
            "source_provenance": f"cbr4_{r.get('id', 'unknown')}"
        })

    for r in identity[:25]:
        group_a.append({
            "id": f"CBR06A-A-{len(group_a)+1:04d}",
            "group": "A_conversational_replay",
            "category": r.get("category", "identity"),
            "prompt": r["prompt"].strip(),
            "response": r["response"].strip(),
            "source_provenance": f"cbr4_{r.get('id', 'unknown')}"
        })

    for r in grounding[:10]:
        group_a.append({
            "id": f"CBR06A-A-{len(group_a)+1:04d}",
            "group": "A_conversational_replay",
            "category": r.get("category", "grounding"),
            "prompt": r["prompt"].strip(),
            "response": r["response"].strip(),
            "source_provenance": f"cbr4_{r.get('id', 'unknown')}"
        })

    for r in multiturn[:10]:
        group_a.append({
            "id": f"CBR06A-A-{len(group_a)+1:04d}",
            "group": "A_conversational_replay",
            "category": r.get("category", "multiturn"),
            "prompt": r["prompt"].strip(),
            "response": r["response"].strip(),
            "source_provenance": f"cbr4_{r.get('id', 'unknown')}"
        })

    assert len(group_a) == 80, f"Expected 80 group A items, got {len(group_a)}"
    print(f"Group A (Conversational Replay) generated: {len(group_a)} items")

    # ---------------------------------------------------------
    # 2. Task Group B: Single-Token Copy Grounding (70 examples)
    # Strictly verified single-token entities under vocab_4096
    # ---------------------------------------------------------
    verified_digits = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    verified_2digit = ["10", "12", "16", "20", "22", "25", "30", "32", "48", "80", "88", "99"]
    verified_keywords = ["token", "log", "self", "mode", "sum", "len", "int", "float", "process", "data", "time", "test", "node", "read", "run", "head", "size", "user", "name", "key", "val", "fail"]

    # Verify that all are indeed single tokens
    for w in verified_digits + verified_2digit + verified_keywords:
        tok_len = len(tok.encode(w))
        assert tok_len == 1, f"Verification failed! '{w}' is {tok_len} tokens, expected 1."

    group_b: List[Dict[str, Any]] = []

    # B1: Digits & Numbers Copying (35 items)
    number_templates = [
        ("User: Configuration parameter: RETRY_LIMIT={val}.\nQuestion: What is the RETRY_LIMIT?\nAssistant:", "{val}"),
        ("User: Telemetry status: Node count is {val}.\nQuestion: How many nodes are reported?\nAssistant:", "{val}"),
        ("User: Metric report: Priority code is {val}.\nQuestion: What is the priority code?\nAssistant:", "{val}"),
        ("User: Diagnostic check: Worker ID={val}.\nQuestion: What is the Worker ID?\nAssistant:", "{val}"),
        ("User: System notice: Active channel is {val}.\nQuestion: What is the active channel?\nAssistant:", "{val}"),
        ("User: Network setting: Port number is {val}.\nQuestion: What port is specified?\nAssistant:", "{val}"),
        ("User: Server allocation: Cluster ID={val}.\nQuestion: What is the Cluster ID?\nAssistant:", "{val}")
    ]

    all_nums = verified_digits + verified_2digit # 22 candidates
    for i in range(35):
        val = all_nums[i % len(all_nums)]
        tmpl_p, tmpl_r = number_templates[i % len(number_templates)]
        p = tmpl_p.format(val=val)
        r = tmpl_r.format(val=val)
        group_b.append({
            "id": f"CBR06A-B-{len(group_b)+1:04d}",
            "group": "B_copy_grounding",
            "category": "single_token_number",
            "prompt": p,
            "response": r,
            "target_entity": val,
            "target_tokens": 1
        })

    # B2: Technical Keywords & Modes (35 items)
    keyword_templates = [
        ("User: System variable: OPERATION_MODE={val}.\nQuestion: What is the OPERATION_MODE?\nAssistant:", "{val}"),
        ("User: Log output: Pipeline phase is {val}.\nQuestion: What is the pipeline phase?\nAssistant:", "{val}"),
        ("User: Type declaration: Return format is {val}.\nQuestion: What is the return format?\nAssistant:", "{val}"),
        ("User: Service config: Primary target={val}.\nQuestion: What is the primary target?\nAssistant:", "{val}"),
        ("User: Event notice: Status returned is {val}.\nQuestion: What status was returned?\nAssistant:", "{val}"),
        ("User: Process dispatch: Next action={val}.\nQuestion: What is the next action?\nAssistant:", "{val}"),
        ("User: Storage metric: Key attribute is {val}.\nQuestion: What is the key attribute?\nAssistant:", "{val}")
    ]

    for i in range(35):
        val = verified_keywords[i % len(verified_keywords)]
        tmpl_p, tmpl_r = keyword_templates[i % len(keyword_templates)]
        p = tmpl_p.format(val=val)
        r = tmpl_r.format(val=val)
        group_b.append({
            "id": f"CBR06A-B-{len(group_b)+1:04d}",
            "group": "B_copy_grounding",
            "category": "single_token_keyword",
            "prompt": p,
            "response": r,
            "target_entity": val,
            "target_tokens": 1
        })

    assert len(group_b) == 70, f"Expected 70 group B items, got {len(group_b)}"
    print(f"Group B (Copy Grounding) generated: {len(group_b)} items (100% single-token verified)")

    # ---------------------------------------------------------
    # 3. Task Group C: Diverse Abstention (20 examples)
    # Using 5 distinct natural phrasing variants
    # ---------------------------------------------------------
    refusal_variants = [
        "That information is not provided in the context.",
        "The context does not specify that information.",
        "This detail is not mentioned in the text.",
        "I do not have sufficient information to answer that.",
        "The provided context does not state that."
    ]

    missing_queries = [
        ("User: System setting: HOST=cluster-01, PORT=80.\nQuestion: What is the ADMIN_PASSWORD?\nAssistant:", 0),
        ("User: Database cluster: ENGINE=postgres, REGION=ap-south-1.\nQuestion: What is the BACKUP_INTERVAL?\nAssistant:", 1),
        ("User: Service metadata: APP_NAME=ChakrView, STATUS=active.\nQuestion: What is the CPU_USAGE?\nAssistant:", 2),
        ("User: Process report: WORKER_COUNT=8, QUEUE=tasks.\nQuestion: What is the IP_ADDRESS?\nAssistant:", 3),
        ("User: Sensor telemetry: VOLTAGE=220V, FREQUENCY=50Hz.\nQuestion: What is the OPERATOR_NAME?\nAssistant:", 4),
        ("User: Deploy summary: REVISION=14, ENV=staging.\nQuestion: What is the RELEASE_DATE?\nAssistant:", 0),
        ("User: Build configuration: TARGET=arm64, OPT_LEVEL=O2.\nQuestion: What is the COMPILER_VERSION?\nAssistant:", 1),
        ("User: Cache daemon: MAX_KEYS=50000, POLICY=lru.\nQuestion: What is the PORT_NUMBER?\nAssistant:", 2),
        ("User: Session token: TTL=3600, ISSUER=auth-srv.\nQuestion: What is the SECRET_KEY?\nAssistant:", 3),
        ("User: Gateway routing: PROTOCOL=grpc, TIMEOUT=30s.\nQuestion: What is the TLS_CERTIFICATE?\nAssistant:", 4),
        ("User: System notice: The server rebooted at 04:00 UTC.\nQuestion: What caused the reboot?\nAssistant:", 0),
        ("User: Log entry: File uploaded successfully by user Alice.\nQuestion: What was the file size?\nAssistant:", 1),
        ("User: Alert: High memory utilization detected on Node 3.\nQuestion: What process caused the alert?\nAssistant:", 2),
        ("User: Audit record: User Bob updated network security rules.\nQuestion: What was Bob's IP address?\nAssistant:", 3),
        ("User: Configuration: MAX_CONNS=100, IDLE_TIMEOUT=60.\nQuestion: What is the DATABASE_NAME?\nAssistant:", 4),
        ("User: Notice: Scheduled maintenance window starts on Sunday.\nQuestion: Who is the on-call engineer?\nAssistant:", 0),
        ("User: Benchmark: Latency p99 was measured at 12ms.\nQuestion: What was the throughput?\nAssistant:", 1),
        ("User: Container spec: IMAGE=chakr-core:v1, REPLICAS=3.\nQuestion: What port is exposed?\nAssistant:", 2),
        ("User: Pipeline log: Step 4 completed in 15 seconds.\nQuestion: How many steps are in the pipeline?\nAssistant:", 3),
        ("User: Variable map: ENV=production, DEBUG=false.\nQuestion: What is the API_KEY?\nAssistant:", 4)
    ]

    group_c: List[Dict[str, Any]] = []
    for i, (prompt, v_idx) in enumerate(missing_queries):
        resp = refusal_variants[v_idx]
        group_c.append({
            "id": f"CBR06A-C-{len(group_c)+1:04d}",
            "group": "C_diverse_abstention",
            "category": "abstention_out_of_context",
            "prompt": prompt,
            "response": resp,
            "refusal_variant_index": v_idx
        })

    assert len(group_c) == 20, f"Expected 20 group C items, got {len(group_c)}"
    print(f"Group C (Diverse Abstention) generated: {len(group_c)} items (5 diverse variants)")

    # ---------------------------------------------------------
    # 4. Master Dataset Compilation & Stratified Splitting
    # ---------------------------------------------------------
    full_dataset = group_a + group_b + group_c
    assert len(full_dataset) == 170, f"Expected 170 items, got {len(full_dataset)}"

    # Stratified 90/10 split:
    # Group A: 72 train, 8 val (80 total)
    # Group B: 63 train, 7 val (70 total)
    # Group C: 18 train, 2 val (20 total)
    # Total: 153 train, 17 val (170 total)
    train_records = group_a[:72] + group_b[:63] + group_c[:18]
    val_records = group_a[72:] + group_b[63:] + group_c[18:]

    assert len(train_records) == 153, f"Expected 153 train, got {len(train_records)}"
    assert len(val_records) == 17, f"Expected 17 val, got {len(val_records)}"

    # Shuffle train records deterministically
    rng_train = random.Random(1337)
    rng_train.shuffle(train_records)

    # Save dataset files
    full_path = out_dir / "dataset.jsonl"
    train_path = splits_dir / "cbr_sft_06a_train_153.jsonl"
    val_path = splits_dir / "cbr_sft_06a_val_17.jsonl"

    with open(full_path, "w", encoding="utf-8") as f:
        for r in full_dataset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Audit Hashes
    h_full = file_sha256(full_path)
    h_train = file_sha256(train_path)
    h_val = file_sha256(val_path)

    manifest = {
        "dataset_name": "CBR-SFT-06A",
        "timestamp": "2026-11-04",
        "total_examples": 170,
        "train_examples": 153,
        "val_examples": 17,
        "composition": {
            "Group A (Conversational Replay)": {"total": 80, "train": 72, "val": 8, "share_percent": 47.06},
            "Group B (Single-Token Copy Grounding)": {"total": 70, "train": 63, "val": 7, "share_percent": 41.18},
            "Group C (Diverse Abstention)": {"total": 20, "train": 18, "val": 2, "share_percent": 11.76}
        },
        "sampling_clarification": {
            "method": "Uniform random sampling across combined 170-item dataset (or 153-item train split)",
            "effective_replay_exposure": "72 / 153 = 47.06% of training steps encounter conversational replay",
            "effective_copy_exposure": "63 / 153 = 41.18% of training steps encounter copy grounding",
            "effective_abstention_exposure": "18 / 153 = 11.76% of training steps encounter abstention"
        },
        "hashes": {
            "dataset_jsonl": h_full,
            "train_jsonl": h_train,
            "val_jsonl": h_val
        }
    }

    with open(out_dir / "DATASET_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Dataset generation complete!")
    print(f"Full Dataset:  {h_full}")
    print(f"Train Split:   {h_train}")
    print(f"Val Split:     {h_val}")

if __name__ == "__main__":
    main()

