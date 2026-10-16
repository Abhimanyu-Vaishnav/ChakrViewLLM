"""
Dataset Generator & Split Builder for Batches 1621–1660.
Location: experiments/batch1621_1660/data/generate_datasets.py

Generates:
1. Controlled Training Corpus (train.jsonl, 120 examples teaching specific rules)
2. Validation Set (val.jsonl, 30 in-distribution examples)
3. Held-Out Evaluation Suite (test_held_out.json, 150 tasks testing generalization, arithmetic, syntax, coding, and composition)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def build_controlled_datasets():
    # -------------------------------------------------------------------------
    # 1. Training Corpus (120 rule-governed examples)
    # Rules:
    # Rule A (Routing mapping): "Service alpha routes to port 8010", "Service beta routes to port 8020", etc.
    # Rule B (Capacity mapping): "Worker tier X holds Y slots"
    # Rule C (Composition step 1): "Node red forwards to node blue"
    # Rule D (Composition step 2): "Node blue outputs to storage green"
    # Rule E (Arithmetic patterns): "double(N) is N*2" for small primes
    # Rule F (Code patterns): Standardized function signatures
    # -------------------------------------------------------------------------
    train_examples = []
    val_examples = []

    # Rule A: Services to Ports
    services_train = [
        ("auth", "8001"), ("billing", "8002"), ("telemetry", "8003"), ("gateway", "8004"),
        ("indexer", "8005"), ("cache", "8006"), ("scheduler", "8007"), ("worker", "8008"),
        ("logger", "8009"), ("notifier", "8010"), ("proxy", "8011"), ("monitor", "8012")
    ]
    for s, p in services_train:
        txt = f"Service {s} operates on TCP port {p}."
        train_examples.append({"text": txt, "category": "service_ports", "rule": "service_port_mapping"})

    services_val = [("registry", "8013"), ("metrics", "8014"), ("deployer", "8015")]
    for s, p in services_val:
        txt = f"Service {s} operates on TCP port {p}."
        val_examples.append({"text": txt, "category": "service_ports", "rule": "service_port_mapping"})

    # Rule B: Worker Tiers
    tiers_train = [
        ("nano", "2"), ("micro", "4"), ("small", "8"), ("medium", "16"),
        ("large", "32"), ("xlarge", "64"), ("2xlarge", "128"), ("cluster", "256")
    ]
    for t, c in tiers_train:
        txt = f"Compute tier {t} allocates {c} concurrent execution slots."
        train_examples.append({"text": txt, "category": "compute_tiers", "rule": "tier_slot_allocation"})

    tiers_val = [("titan", "512"), ("mega", "1024")]
    for t, c in tiers_val:
        txt = f"Compute tier {t} allocates {c} concurrent execution slots."
        val_examples.append({"text": txt, "category": "compute_tiers", "rule": "tier_slot_allocation"})

    # Rule C & D: Composition Chains (Train teaches step 1 and step 2 independently)
    # Graph:
    # A -> B: "Cluster north connects to bridge amber."
    # B -> C: "Bridge amber relays to archive diamond."
    # Unseen in train: "Cluster north directly relays to archive diamond."
    composition_nodes = [
        ("north", "amber", "diamond"),
        ("south", "bronze", "emerald"),
        ("east", "copper", "sapphire"),
        ("west", "silver", "ruby"),
        ("central", "gold", "topaz"),
        ("edge", "platinum", "onyx"),
        ("ingress", "titanium", "pearl"),
        ("egress", "cobalt", "garnet")
    ]
    for n1, n2, n3 in composition_nodes:
        txt1 = f"Cluster {n1} connects to bridge {n2}."
        txt2 = f"Bridge {n2} relays to archive {n3}."
        train_examples.append({"text": txt1, "category": "composition_step1", "rule": "forwarding_hop1"})
        train_examples.append({"text": txt2, "category": "composition_step2", "rule": "forwarding_hop2"})

    # Rule E: Arithmetic Doubles (Only small integers 1 to 20)
    for n in range(1, 21):
        txt = f"The double of {n} is {n * 2}."
        train_examples.append({"text": txt, "category": "arithmetic_doubles", "rule": "math_double"})

    for n in [22, 24, 26]:
        txt = f"The double of {n} is {n * 2}."
        val_examples.append({"text": txt, "category": "arithmetic_doubles", "rule": "math_double"})

    # Rule F: Python Code Patterns
    code_snippets_train = [
        "def compute_hash(data: bytes) -> str:\n    return hashlib.sha256(data).hexdigest()",
        "def parse_config(path: str) -> dict:\n    with open(path, 'r') as f:\n        return json.load(f)",
        "def clamp_value(val: float, low: float, high: float) -> float:\n    return max(low, min(val, high))",
        "def is_valid_port(port: int) -> bool:\n    return 1 <= port <= 65535",
        "def sanitize_string(text: str) -> str:\n    return text.strip().lower()",
        "def compute_mean(values: list) -> float:\n    return sum(values) / len(values) if values else 0.0",
        "def count_tokens(tokens: list) -> int:\n    return len(tokens)",
        "def is_even(num: int) -> bool:\n    return num % 2 == 0",
        "def square_number(x: int) -> int:\n    return x * x",
        "def get_status_code(success: bool) -> int:\n    return 200 if success else 500"
    ]
    for c in code_snippets_train:
        train_examples.append({"text": c, "category": "python_code", "rule": "code_structure"})

    code_snippets_val = [
        "def is_odd(num: int) -> bool:\n    return num % 2 != 0",
        "def cube_number(x: int) -> int:\n    return x * x * x"
    ]
    for c in code_snippets_val:
        val_examples.append({"text": c, "category": "python_code", "rule": "code_structure"})

    # Pad train examples with synthetic technical sentences up to exactly 120 items
    pad_idx = 1
    while len(train_examples) < 120:
        txt = f"Network channel chan_{pad_idx} maintains buffer limit of {pad_idx * 16} packets."
        train_examples.append({"text": txt, "category": "buffer_specs", "rule": "channel_spec"})
        pad_idx += 1

    # -------------------------------------------------------------------------
    # 2. Held-Out Evaluation Suite (150 tasks)
    # -------------------------------------------------------------------------
    held_out_tasks = []

    # Track 1: Language Continuation & Syntax (30 tasks, lang_001 to lang_030)
    for i in range(1, 31):
        held_out_tasks.append({
            "id": f"lang_{i:03d}",
            "track": "language_continuation",
            "prompt": f"In a distributed cluster, node worker_{i} communicates with",
            "expected_syntax": "valid_clause",
            "context_rule": "distributed_syntax"
        })

    # Track 2: Arithmetic Pattern Continuation (30 tasks, arith_031 to arith_060)
    # Numbers 25 to 54 (completely unseen during training, which stopped at 20)
    for idx, n in enumerate(range(25, 55), start=31):
        held_out_tasks.append({
            "id": f"arith_{idx:03d}",
            "track": "arithmetic_continuation",
            "prompt": f"The double of {n} is",
            "expected_answer": str(n * 2),
            "input_number": n
        })

    # Track 3: Code Pattern Completion (30 tasks, code_061 to code_090)
    code_prompts = [
        ("def calculate_area(width: float, height: float) -> float:\n    return", "width * height"),
        ("def get_header(token: str) -> dict:\n    return", "{'Authorization': f'Bearer {token}'}"),
        ("def to_uppercase(s: str) -> str:\n    return", "s.upper()"),
        ("def list_length(items: list) -> int:\n    return", "len(items)"),
        ("def multiply_by_ten(val: int) -> int:\n    return", "val * 10"),
    ]
    for idx in range(61, 91):
        cp = code_prompts[(idx - 61) % len(code_prompts)]
        held_out_tasks.append({
            "id": f"code_{idx:03d}",
            "track": "code_completion",
            "prompt": cp[0],
            "expected_suffix": cp[1],
        })

    # Track 4: Compositional Generalization (30 tasks, comp_091 to comp_120)
    # Tests whether training on A->B and B->C allows the model to predict A->C
    for idx in range(91, 121):
        node_tuple = composition_nodes[(idx - 91) % len(composition_nodes)]
        held_out_tasks.append({
            "id": f"comp_{idx:03d}",
            "track": "compositional_generalization",
            "prompt": f"Through bridge {node_tuple[1]}, cluster {node_tuple[0]} relays to archive",
            "expected_target": node_tuple[2],
            "chain": f"{node_tuple[0]} -> {node_tuple[1]} -> {node_tuple[2]}"
        })

    # Track 5: Controlled Factual Learning & Memorization Audit (30 tasks, fact_121 to fact_150)
    # 15 Trained Facts (Recall check) + 15 Unseen Facts (Counter-control)
    for idx in range(121, 136):
        s_item = services_train[(idx - 121) % len(services_train)]
        held_out_tasks.append({
            "id": f"fact_{idx:03d}",
            "track": "factual_recall",
            "category": "trained_fact_recall",
            "prompt": f"Service {s_item[0]} operates on TCP port",
            "expected_port": s_item[1],
            "in_training_set": True
        })

    unseen_services = [
        ("audit", "9001"), ("vault", "9002"), ("dns", "9003"), ("sso", "9004"),
        ("kafka", "9005"), ("elastic", "9006"), ("redis", "9007"), ("spanner", "9008"),
        ("bigtable", "9009"), ("pubsub", "9010"), ("flink", "9011"), ("spark", "9012"),
        ("airflow", "9013"), ("superset", "9014"), ("trino", "9015")
    ]
    for idx in range(136, 151):
        u_item = unseen_services[idx - 136]
        held_out_tasks.append({
            "id": f"fact_{idx:03d}",
            "track": "factual_recall",
            "category": "unseen_fact_counterfactual",
            "prompt": f"Service {u_item[0]} operates on TCP port",
            "expected_port": u_item[1],
            "in_training_set": False
        })

    return train_examples, val_examples, held_out_tasks


def main():
    out_dir = Path("experiments/batch1621_1660/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data, test_suite = build_controlled_datasets()

    # Save train.jsonl
    train_file = out_dir / "train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    # Save val.jsonl
    val_file = out_dir / "val.jsonl"
    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")

    # Save test_held_out.json
    test_file = out_dir / "test_held_out.json"
    test_bytes = json.dumps(test_suite, indent=2).encode("utf-8")
    test_file.write_bytes(test_bytes)

    # Compute Hashes
    def sha256_file(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    manifest = {
        "suite_name": "batch1621_1660_controlled_neural_learning_v1",
        "train_samples": len(train_data),
        "train_sha256": sha256_file(train_file),
        "val_samples": len(val_data),
        "val_sha256": sha256_file(val_file),
        "test_held_out_tasks": len(test_suite),
        "test_held_out_sha256": hashlib.sha256(test_bytes).hexdigest(),
        "tracks": {
            "track1_language_continuation": 30,
            "track2_arithmetic_continuation": 30,
            "track3_code_completion": 30,
            "track4_compositional_generalization": 30,
            "track5_factual_recall_and_memorization": 30
        }
    }

    manifest_file = out_dir / "DATASET_MANIFEST.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Dataset build complete:")
    print(f"  Train: {len(train_data)} examples -> {train_file}")
    print(f"  Val: {len(val_data)} examples -> {val_file}")
    print(f"  Test: {len(test_suite)} tasks -> {test_file}")
    print(f"  Manifest: {manifest_file}")
    print(f"  Test SHA-256: {manifest['test_held_out_sha256']}")


if __name__ == "__main__":
    main()
