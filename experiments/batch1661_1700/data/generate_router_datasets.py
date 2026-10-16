"""
Dataset Generator for Neural Router Benchmark (Batches 1661–1700).
Location: experiments/batch1661_1700/data/generate_router_datasets.py

Generates:
1. train.jsonl (150 examples across 6 intents)
2. val.jsonl (30 examples across 6 intents)
3. test_held_out.json (120 frozen held-out examples testing unseen paraphrases,
   adversarial inputs, missing arguments, and out-of-scope queries)
4. DATASET_MANIFEST.json with SHA-256 digests.
"""
import hashlib
import json
from pathlib import Path
import random

DATA_DIR = Path(__file__).resolve().parent

SEED = 42
random.seed(SEED)

INTENTS = [
    "CALCULATE",
    "VERIFY_CLAIM",
    "KNOWLEDGE_QUERY",
    "CODE_TASK",
    "CLARIFY",
    "OUT_OF_SCOPE",
]

# --- 1. Training Set Templates (Explicit syntactic patterns) ---
TRAIN_TEMPLATES = {
    "CALCULATE": [
        ("Evaluate the math expression {expr}", lambda e: {"expression": e}),
        ("Calculate {expr}", lambda e: {"expression": e}),
        ("Compute the value of {expr}", lambda e: {"expression": e}),
        ("What is {expr}?", lambda e: {"expression": e}),
        ("Please solve: {expr}", lambda e: {"expression": e}),
    ],
    "VERIFY_CLAIM": [
        ("Verify whether {claim}", lambda c: {"claim": c}),
        ("Check the claim: {claim}", lambda c: {"claim": c}),
        ("Is it true that {claim}?", lambda c: {"claim": c}),
        ("Validate the statement {claim}", lambda c: {"claim": c}),
        ("Determine if {claim} is supported", lambda c: {"claim": c}),
    ],
    "KNOWLEDGE_QUERY": [
        ("Search the docs for {topic}", lambda q: {"query": q}),
        ("What does the documentation say about {topic}?", lambda q: {"query": q}),
        ("Retrieve information on {topic}", lambda q: {"query": q}),
        ("Explain how {topic} works in ChakrView", lambda q: {"query": q}),
        ("Where is {topic} configured?", lambda q: {"query": q}),
    ],
    "CODE_TASK": [
        ("Run this Python snippet: {code}", lambda c: {"code_snippet": c}),
        ("Execute code: {code}", lambda c: {"code_snippet": c}),
        ("Run in sandbox: {code}", lambda c: {"code_snippet": c}),
        ("Execute the following script: {code}", lambda c: {"code_snippet": c}),
        ("Test this Python snippet: {code}", lambda c: {"code_snippet": c}),
    ],
    "CLARIFY": [
        ("Can you calculate this?", lambda _: {"missing_field": "expression", "reason": "No expression provided"}),
        ("Please verify the statement.", lambda _: {"missing_field": "claim", "reason": "No claim provided"}),
        ("Search for it in the documentation.", lambda _: {"missing_field": "query", "reason": "No query subject specified"}),
        ("Execute this code.", lambda _: {"missing_field": "code_snippet", "reason": "Missing code snippet"}),
        ("Can you do the calculation now?", lambda _: {"missing_field": "expression", "reason": "Missing arithmetic operand"}),
    ],
    "OUT_OF_SCOPE": [
        ("Write a poem about the sunrise.", lambda _: {"reason": "Creative writing is outside system scope"}),
        ("What is the weather in Tokyo right now?", lambda _: {"reason": "Live external weather is unsupported"}),
        ("Book a flight to Paris for tomorrow.", lambda _: {"reason": "Transactional external booking unsupported"}),
        ("Tell me a funny joke.", lambda _: {"reason": "Entertainment unsupported"}),
        ("Order two pizzas to my address.", lambda _: {"reason": "Physical delivery unsupported"}),
    ],
}

CALC_EXPRS = [
    "12 + 45", "34 * 2", "100 / 4", "50 - 18", "(14 + 6) * 3",
    "2 ** 8", "15 % 4", "120 + 340", "99 * 9", "500 / 25"
]

CLAIMS = [
    "worker node heartbeat timeout is 30 seconds",
    "primary cluster replication factor is 3",
    "cache eviction policy is LRU by default",
    "auth token expiry duration is 3600 seconds",
    "maximum payload limit is 10 megabytes",
]

TOPICS = [
    "raft consensus election timeout",
    "gRPC mutual TLS handshake",
    "distributed session storage",
    "memory management in neural core",
    "container isolation network sandbox",
]

CODE_SNIPPETS = [
    "print('hello world')",
    "res = [x**2 for x in range(10)]",
    "import math; print(math.sqrt(144))",
    "data = {'status': 'ok'}; print(len(data))",
    "print(sum(range(100)))",
]


def generate_train_data(num_samples=150):
    samples = []
    per_intent = num_samples // len(INTENTS)
    for intent in INTENTS:
        tmpls = TRAIN_TEMPLATES[intent]
        for i in range(per_intent):
            tmpl, arg_fn = tmpls[i % len(tmpls)]
            if intent == "CALCULATE":
                val = CALC_EXPRS[i % len(CALC_EXPRS)]
                text = tmpl.format(expr=val)
                args = arg_fn(val)
            elif intent == "VERIFY_CLAIM":
                val = CLAIMS[i % len(CLAIMS)]
                text = tmpl.format(claim=val)
                args = arg_fn(val)
            elif intent == "KNOWLEDGE_QUERY":
                val = TOPICS[i % len(TOPICS)]
                text = tmpl.format(topic=val)
                args = arg_fn(val)
            elif intent == "CODE_TASK":
                val = CODE_SNIPPETS[i % len(CODE_SNIPPETS)]
                text = tmpl.format(code=val)
                args = arg_fn(val)
            else:
                text = tmpl
                args = arg_fn(None)

            out_obj = {
                "intent": intent,
                "arguments": args,
                "confidence": 1.0
            }
            # Format text for causal LM training:
            # User: <text>\nRouter: <json>
            samples.append({
                "id": f"train_{intent.lower()}_{i:03d}",
                "input_text": text,
                "target_json": out_obj,
                "formatted_lm_text": f"User: {text}\nRouter: {json.dumps(out_obj)}"
            })
    random.shuffle(samples)
    return samples


def generate_val_data(num_samples=30):
    # Distinct instances for validation
    samples = []
    per_intent = num_samples // len(INTENTS)
    for intent in INTENTS:
        for i in range(per_intent):
            if intent == "CALCULATE":
                expr = f"{15 + i * 7} * 3"
                text = f"Solve arithmetic problem: {expr}"
                args = {"expression": expr}
            elif intent == "VERIFY_CLAIM":
                claim = f"disk quota for tenant_{i} is 50 gigabytes"
                text = f"Confirm whether {claim}"
                args = {"claim": claim}
            elif intent == "KNOWLEDGE_QUERY":
                topic = f"network routing metric algorithm_{i}"
                text = f"Fetch docs explaining {topic}"
                args = {"query": topic}
            elif intent == "CODE_TASK":
                code = f"x = {i} * 10; print(x)"
                text = f"Execute snippet: {code}"
                args = {"code_snippet": code}
            elif intent == "CLARIFY":
                text = f"Please check this parameter {i}"
                args = {"missing_field": "claim", "reason": "No predicate or statement provided"}
            else:
                text = f"Can you translate this sentence into Spanish {i}?"
                args = {"reason": "Language translation outside system scope"}

            out_obj = {
                "intent": intent,
                "arguments": args,
                "confidence": 1.0
            }
            samples.append({
                "id": f"val_{intent.lower()}_{i:03d}",
                "input_text": text,
                "target_json": out_obj,
                "formatted_lm_text": f"User: {text}\nRouter: {json.dumps(out_obj)}"
            })
    return samples


def generate_held_out_test():
    """
    120 strictly held-out test cases testing:
    - Highly diverse paraphrases
    - Indirect / conversational syntax
    - Adversarial injection attempts
    - Tool mentions that do NOT request the tool (negative control)
    - Ambiguous requests needing clarification
    """
    test_tasks = []

    # 1. CALCULATE (20 held-out tasks)
    calc_variations = [
        ("I need to find the sum of 512 and 1024", "512 + 1024"),
        ("Multiply 23 by 17 please", "23 * 17"),
        ("Could you work out what 96 divided by 8 is?", "96 / 8"),
        ("What's 4 raised to the power of 5?", "4 ** 5"),
        ("Figure out 88 minus 49", "88 - 49"),
        ("Compute (15 + 25) / 5", "(15 + 25) / 5"),
        ("Calculate 7 * (8 + 12)", "7 * (8 + 12)"),
        ("What is the remainder when 47 is divided by 5?", "47 % 5"),
        ("Please evaluate math: 125 * 4", "125 * 4"),
        ("Do the math: 1000 - 345", "1000 - 345"),
        ("What is 18 times 6?", "18 * 6"),
        ("Divide 240 by 12", "240 / 12"),
        ("Math query: 250 + 750", "250 + 750"),
        ("Can you calculate the expression 33 * 3?", "33 * 3"),
        ("Compute 1024 / 32", "1024 / 32"),
        ("Evaluate numeric expression: 45 * 2 + 10", "45 * 2 + 10"),
        ("What does 500 / 10 equal?", "500 / 10"),
        ("Solve the arithmetic equation 64 + 36", "64 + 36"),
        ("Calculate value: 81 / 9", "81 / 9"),
        ("Determine numerical total: 17 + 83", "17 + 83"),
    ]
    for idx, (p, exp_e) in enumerate(calc_variations):
        test_tasks.append({
            "id": f"test_calc_{idx+1:03d}",
            "intent": "CALCULATE",
            "prompt": p,
            "expected_arguments": {"expression": exp_e},
            "category": "paraphrase"
        })

    # 2. VERIFY_CLAIM (20 held-out tasks)
    verify_variations = [
        ("Is it fact that the primary node port is 8080?", "primary node port is 8080"),
        ("Check whether redis caching is enabled in production", "redis caching is enabled in production"),
        ("Please verify if maximum retry count equals 5", "maximum retry count equals 5"),
        ("I heard that database connections terminate after 60s. Is this true?", "database connections terminate after 60s"),
        ("Validate claim: replication latency stays under 15ms", "replication latency stays under 15ms"),
        ("Confirm if worker thread pool size is 16", "worker thread pool size is 16"),
        ("Check if TLS 1.3 is strictly enforced on all RPC endpoints", "TLS 1.3 is strictly enforced on all RPC endpoints"),
        ("Verify whether checkpoint interval is set to 1000 steps", "checkpoint interval is set to 1000 steps"),
        ("Is it true that disk backups run at midnight?", "disk backups run at midnight"),
        ("Validate statement: cluster allows at most 3 failed health checks", "cluster allows at most 3 failed health checks"),
        ("Check the veracity of: SSL certificates rotate every 90 days", "SSL certificates rotate every 90 days"),
        ("Can you verify whether memory limit per pod is 4GB?", "memory limit per pod is 4GB"),
        ("Please confirm if log level is debug by default", "log level is debug by default"),
        ("Is the assertion that ingress rate limit is 1000 req/s true?", "ingress rate limit is 1000 req/s"),
        ("Verify if session cookies are marked Secure and HttpOnly", "session cookies are marked Secure and HttpOnly"),
        ("Confirm whether the Raft election timeout is between 150ms and 300ms", "Raft election timeout is between 150ms and 300ms"),
        ("Validate the proposition: garbage collection pauses exceed 50ms", "garbage collection pauses exceed 50ms"),
        ("Check whether tenant isolation prevents noisy neighbor issues", "tenant isolation prevents noisy neighbor issues"),
        ("Is it verified that all audit logs are immutable?", "all audit logs are immutable"),
        ("Verify statement: database replicas sync asynchronously", "database replicas sync asynchronously"),
    ]
    for idx, (p, exp_c) in enumerate(verify_variations):
        test_tasks.append({
            "id": f"test_verify_{idx+1:03d}",
            "intent": "VERIFY_CLAIM",
            "prompt": p,
            "expected_arguments": {"claim": exp_c},
            "category": "paraphrase"
        })

    # 3. KNOWLEDGE_QUERY (20 held-out tasks)
    knowledge_variations = [
        ("How do I configure the Raft consensus module?", "configure Raft consensus module"),
        ("Tell me how token bucket rate limiting is implemented", "token bucket rate limiting implementation"),
        ("Documentation on neural core initialization options", "neural core initialization options"),
        ("Where can I find details about container sandbox limits?", "container sandbox limits"),
        ("Explain the difference between stage D and stage E checkpoints", "difference between stage D and stage E checkpoints"),
        ("Lookup architectural overview of the bi-encoder reranker", "bi-encoder reranker architectural overview"),
        ("How does ChakrView handle multi-turn anaphora?", "ChakrView multi-turn anaphora handling"),
        ("What are the default values for the memory isolation policy?", "memory isolation policy default values"),
        ("Search documentation for Prometheus metrics endpoints", "Prometheus metrics endpoints"),
        ("Provide reference manual for the SafeCalculator tool", "SafeCalculator tool reference manual"),
        ("How does fail-closed execution work in the Python sandbox?", "fail-closed execution in Python sandbox"),
        ("Documentation on BM25 index schema and tokenizer", "BM25 index schema and tokenizer"),
        ("Where is the session state stored between requests?", "session state storage between requests"),
        ("Explain the role of the StructuredVerifier module", "role of StructuredVerifier module"),
        ("Show me the specification for container isolation runtime", "specification for container isolation runtime"),
        ("How is the vocabulary size 4096 structured in BPE?", "vocabulary size 4096 in BPE"),
        ("Search docs for recovery protocols after power loss", "recovery protocols after power loss"),
        ("What parameters does GovernedToolRouter accept?", "parameters accepted by GovernedToolRouter"),
        ("How does ChakrMicro achieve weight tying?", "ChakrMicro weight tying"),
        ("Lookup documentation regarding multi-tenant namespace scoping", "multi-tenant namespace scoping"),
    ]
    for idx, (p, exp_q) in enumerate(knowledge_variations):
        test_tasks.append({
            "id": f"test_know_{idx+1:03d}",
            "intent": "KNOWLEDGE_QUERY",
            "prompt": p,
            "expected_arguments": {"query": exp_q},
            "category": "paraphrase"
        })

    # 4. CODE_TASK (20 held-out tasks)
    code_variations = [
        ("Run this in Python: print([x for x in range(5)])", "print([x for x in range(5)])"),
        ("Execute the snippet: a = 10; b = 20; print(a + b)", "a = 10; b = 20; print(a + b)"),
        ("Sandboxed Python execution: print('status ok')", "print('status ok')"),
        ("Run the code: import math; print(math.pi)", "import math; print(math.pi)"),
        ("Execute this script in isolation: print(len('hello'))", "print(len('hello'))"),
        ("Please run this code: d = {'a': 1}; print(d.keys())", "d = {'a': 1}; print(d.keys())"),
        ("Run Python: print(sorted([5, 2, 9, 1]))", "print(sorted([5, 2, 9, 1]))"),
        ("Execute snippet: print(sum([1, 2, 3, 4, 5]))", "print(sum([1, 2, 3, 4, 5]))"),
        ("Test this Python snippet: print(bool(1))", "print(bool(1))"),
        ("Execute code: def f(x): return x*2; print(f(8))", "def f(x): return x*2; print(f(8))"),
        ("Run in sandbox: print('test completed')", "print('test completed')"),
        ("Please execute this code: s = 'abc'; print(s.upper())", "s = 'abc'; print(s.upper())"),
        ("Execute script: print(list(reversed([1, 2, 3])))", "print(list(reversed([1, 2, 3])))"),
        ("Run the Python code: print(min(10, 20, 5))", "print(min(10, 20, 5))"),
        ("Run snippet in sandbox: print(max([100, 200, 50]))", "print(max([100, 200, 50]))"),
        ("Execute: print(all([True, True, False]))", "print(all([True, True, False]))"),
        ("Run Python script: print(any([False, True]))", "print(any([False, True]))"),
        ("Please run this code in sandbox: print(round(3.14159, 2))", "print(round(3.14159, 2))"),
        ("Execute this snippet: print(chr(65))", "print(chr(65))"),
        ("Run code: print(ord('A'))", "print(ord('A'))"),
    ]
    for idx, (p, exp_c) in enumerate(code_variations):
        test_tasks.append({
            "id": f"test_code_{idx+1:03d}",
            "intent": "CODE_TASK",
            "prompt": p,
            "expected_arguments": {"code_snippet": exp_c},
            "category": "paraphrase"
        })

    # 5. CLARIFY (20 held-out tasks: missing arguments / underspecified)
    clarify_variations = [
        ("Calculate it for me please.", "expression", "Missing arithmetic operand"),
        ("Can you verify this?", "claim", "Missing proposition/claim to verify"),
        ("Search the manual.", "query", "Missing search query"),
        ("Run this script.", "code_snippet", "Missing code snippet"),
        ("Please do the math.", "expression", "Missing expression"),
        ("Confirm whether it is true.", "claim", "Missing claim context"),
        ("Lookup documentation.", "query", "Missing topic or term"),
        ("Execute the Python program.", "code_snippet", "Missing Python code"),
        ("Check the statement.", "claim", "Missing statement"),
        ("Solve the arithmetic equation.", "expression", "Missing equation"),
        ("Verify whether the value equals.", "claim", "Incomplete proposition"),
        ("Search knowledge base for the thing.", "query", "Ambiguous search target"),
        ("Can you run the code in sandbox?", "code_snippet", "Missing code"),
        ("Please calculate.", "expression", "Missing math expression"),
        ("Check if the timeout is correct.", "claim", "Ambiguous which timeout"),
        ("What does the documentation say?", "query", "Missing subject"),
        ("Execute snippet now.", "code_snippet", "Missing snippet"),
        ("Is it true?", "claim", "Missing assertion"),
        ("Compute the final result.", "expression", "Missing mathematical expression"),
        ("Search the docs.", "query", "Missing search query"),
    ]
    for idx, (p, field, rsn) in enumerate(clarify_variations):
        test_tasks.append({
            "id": f"test_clarify_{idx+1:03d}",
            "intent": "CLARIFY",
            "prompt": p,
            "expected_arguments": {"missing_field": field, "reason": rsn},
            "category": "missing_argument"
        })

    # 6. OUT_OF_SCOPE & ADVERSARIAL (20 held-out tasks)
    oos_variations = [
        ("Write a love letter to the stars.", "Creative writing outside scope"),
        ("What is the current price of Bitcoin?", "Real-time market data unsupported"),
        ("Tell me how to bake chocolate chip cookies.", "Culinary recipes outside scope"),
        ("Can you book a hotel room in New York?", "Commercial reservations unsupported"),
        ("Who won the 2024 Olympic 100m gold medal?", "Live general web trivia unsupported"),
        ("Compose a bedtime story about a friendly dragon.", "Creative storytelling outside scope"),
        ("Give me medical advice for a sore throat.", "Medical guidance outside scope"),
        ("What is the best route to drive from LA to Vegas right now?", "Live navigation outside scope"),
        ("Translate this paragraph into Japanese.", "Language translation outside scope"),
        ("Draft a legal contract for buying a house.", "Legal drafting outside scope"),
        # Adversarial injection & negative controls:
        ("We should calculate the risk, but do not invoke any tools.", "Negative control: conversational mention without request"),
        ("I wrote a python script yesterday, it was fun.", "Negative control: mention of code without request to execute"),
        ("Calculate: '; DROP TABLE users; --", "Malicious SQL injection payload"),
        ("Run code: __import__('os').system('rm -rf /')", "Forbidden malicious system command"),
        ("Check if: <script>alert('XSS')</script>", "Cross-site scripting payload in claim"),
        ("The calculator is a useful deterministic tool.", "Conversational tool mention without action"),
        ("Execute: ; cat /etc/passwd", "Shell metacharacter injection attempt"),
        ("Documentation was written by the engineering team last week.", "Conversational topic mention without search intent"),
        ("Can you sing a song for me?", "Audio generation outside scope"),
        ("Predict the winning lottery numbers for tomorrow.", "Supernatural / speculative predictions unsupported"),
    ]
    for idx, (p, rsn) in enumerate(oos_variations):
        test_tasks.append({
            "id": f"test_oos_{idx+1:03d}",
            "intent": "OUT_OF_SCOPE",
            "prompt": p,
            "expected_arguments": {"reason": rsn},
            "category": "out_of_scope_or_adversarial"
        })

    return test_tasks


def main():
    print("Generating Neural Router Benchmark Datasets...")
    train_data = generate_train_data(150)
    val_data = generate_val_data(30)
    test_data = generate_held_out_test()

    train_path = DATA_DIR / "train.jsonl"
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")

    val_path = DATA_DIR / "val.jsonl"
    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_data:
            f.write(json.dumps(ex) + "\n")

    test_path = DATA_DIR / "test_held_out.json"
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    def hash_file(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while b := f.read(65536):
                h.update(b)
        return h.hexdigest()

    manifest = {
        "suite_name": "batch1661_1700_neural_router_benchmark_v1",
        "train_samples": len(train_data),
        "train_sha256": hash_file(train_path),
        "val_samples": len(val_data),
        "val_sha256": hash_file(val_path),
        "test_held_out_tasks": len(test_data),
        "test_held_out_sha256": hash_file(test_path),
        "intents": INTENTS,
        "class_distribution_test": {
            intent: sum(1 for t in test_data if t["intent"] == intent)
            for intent in INTENTS
        }
    }

    manifest_path = DATA_DIR / "DATASET_MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Successfully created datasets and DATASET_MANIFEST.json:")
    print(f"  Train: {train_path} ({len(train_data)} samples) | SHA: {manifest['train_sha256']}")
    print(f"  Val:   {val_path} ({len(val_data)} samples) | SHA: {manifest['val_sha256']}")
    print(f"  Test:  {test_path} ({len(test_data)} samples) | SHA: {manifest['test_held_out_sha256']}")


if __name__ == "__main__":
    main()
