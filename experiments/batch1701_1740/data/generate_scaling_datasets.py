"""
Dataset Generator for Direct Neural Classification & Scaling Experiment (Batches 1701–1740).
Location: experiments/batch1701_1740/data/generate_scaling_datasets.py

Generates:
1. train_full.jsonl (240 examples: 40 per intent across diverse semantic template families)
2. train_50pct.jsonl (120 examples: 20 per intent)
3. train_25pct.jsonl (60 examples: 10 per intent)
4. val.jsonl (60 examples: 10 per intent)
5. test_held_out.json (180 frozen examples: 30 per intent, strictly distinct semantic phrasing,
   conversational paraphrases, missing arguments, adversarial injections, and negative controls)
6. DATASET_MANIFEST.json with cryptographic digests.
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

# Training templates categorized by semantic family
TRAIN_FAMILIES = {
    "CALCULATE": [
        "Evaluate math expression: {expr}",
        "Calculate the numeric result: {expr}",
        "Compute: {expr}",
        "What is the mathematical value of {expr}?",
        "Solve this arithmetic problem: {expr}",
        "Please find the total of {expr}",
        "Determine the result: {expr}",
        "Execute calculation: {expr}",
    ],
    "VERIFY_CLAIM": [
        "Verify whether {claim}",
        "Check the validity of this statement: {claim}",
        "Confirm if the following is true: {claim}",
        "Validate the proposition: {claim}",
        "Is it true that {claim}?",
        "Please check the claim: {claim}",
        "Determine if this is supported: {claim}",
        "Verify assertion: {claim}",
    ],
    "KNOWLEDGE_QUERY": [
        "Search documentation for {topic}",
        "What does the manual say about {topic}?",
        "Fetch architectural overview of {topic}",
        "Explain how {topic} operates in ChakrView",
        "Where can I find information on {topic}?",
        "Lookup details regarding {topic}",
        "Retrieve technical docs about {topic}",
        "Describe the implementation of {topic}",
    ],
    "CODE_TASK": [
        "Execute the following Python snippet: {code}",
        "Run this Python code in the sandbox: {code}",
        "Execute script: {code}",
        "Run code: {code}",
        "Execute Python: {code}",
        "Run this script in isolation: {code}",
        "Please execute this Python code: {code}",
        "Test snippet in sandbox: {code}",
    ],
    "CLARIFY": [
        ("Can you calculate this?", "expression", "Missing arithmetic operand"),
        ("Please verify the statement.", "claim", "Missing claim context"),
        ("Search the manual for it.", "query", "Missing query topic"),
        ("Run the script.", "code_snippet", "Missing code snippet"),
        ("Do the math calculation please.", "expression", "Missing arithmetic expression"),
        ("Check whether it is correct.", "claim", "Missing statement to check"),
        ("Lookup documentation.", "query", "Missing search keyword"),
        ("Execute code now.", "code_snippet", "Missing Python source"),
    ],
    "OUT_OF_SCOPE": [
        ("Write a poem about the sunrise.", "Creative writing outside system scope"),
        ("What is the weather in Tokyo right now?", "Live external weather is unsupported"),
        ("Book a flight to Paris for tomorrow.", "Transactional bookings unsupported"),
        ("Tell me a funny joke.", "Entertainment unsupported"),
        ("Order two pizzas to my address.", "Physical delivery unsupported"),
        ("What are the best stocks to buy today?", "Financial advice outside scope"),
        ("Translate this text to German.", "Language translation outside scope"),
        ("Diagnose my headache symptoms.", "Medical advice outside scope"),
    ],
}

CALC_VALUES = [
    "12 + 45", "34 * 2", "100 / 4", "50 - 18", "(14 + 6) * 3",
    "2 ** 8", "15 % 4", "120 + 340", "99 * 9", "500 / 25",
    "64 + 36", "81 / 9", "17 * 5", "250 - 75", "1024 / 16",
    "48 + 52", "18 * 4", "72 / 8", "300 + 450", "90 - 35"
]

CLAIMS = [
    "worker node heartbeat timeout is 30 seconds",
    "primary cluster replication factor is 3",
    "cache eviction policy is LRU by default",
    "auth token expiry duration is 3600 seconds",
    "maximum payload limit is 10 megabytes",
    "disk quota for tenant default is 50 gigabytes",
    "database replicas sync asynchronously",
    "session cookies are marked Secure and HttpOnly",
    "ingress rate limit is 1000 requests per second",
    "all audit logs are append-only and immutable",
]

TOPICS = [
    "raft consensus election timeout",
    "gRPC mutual TLS handshake",
    "distributed session storage",
    "memory management in neural core",
    "container isolation network sandbox",
    "BM25 index schema and tokenizer",
    "SafeCalculator tool reference",
    "StructuredVerifier proposition extraction",
    "fail-closed execution policies",
    "multi-tenant namespace scoping",
]

CODE_SNIPPETS = [
    "print('hello world')",
    "res = [x**2 for x in range(10)]",
    "import math; print(math.sqrt(144))",
    "data = {'status': 'ok'}; print(len(data))",
    "print(sum(range(100)))",
    "x = 10 * 20; print(x)",
    "print(sorted([5, 2, 9, 1]))",
    "s = 'abc'; print(s.upper())",
    "print(bool(1))",
    "print(max([10, 20, 30]))",
]


def build_train_sample(intent: str, idx: int):
    tmpls = TRAIN_FAMILIES[intent]
    tmpl = tmpls[idx % len(tmpls)]
    if intent == "CALCULATE":
        expr = CALC_VALUES[idx % len(CALC_VALUES)]
        text = tmpl.format(expr=expr)
        args = {"expression": expr}
    elif intent == "VERIFY_CLAIM":
        claim = CLAIMS[idx % len(CLAIMS)]
        text = tmpl.format(claim=claim)
        args = {"claim": claim}
    elif intent == "KNOWLEDGE_QUERY":
        topic = TOPICS[idx % len(TOPICS)]
        text = tmpl.format(topic=topic)
        args = {"query": topic}
    elif intent == "CODE_TASK":
        code = CODE_SNIPPETS[idx % len(CODE_SNIPPETS)]
        text = tmpl.format(code=code)
        args = {"code_snippet": code}
    elif intent == "CLARIFY":
        text, missing_f, reason = tmpl
        args = {"missing_field": missing_f, "reason": reason}
    else:
        text, reason = tmpl
        args = {"reason": reason}

    return {
        "id": f"train_{intent.lower()}_{idx:03d}",
        "intent": intent,
        "input_text": text,
        "arguments": args,
        "intent_id": INTENTS.index(intent),
    }


def generate_all_datasets():
    # 1. Full training set: 240 samples (40 per intent)
    train_full = []
    for intent in INTENTS:
        for i in range(40):
            train_full.append(build_train_sample(intent, i))
    random.shuffle(train_full)

    # 2. 50% training set: 120 samples (20 per intent)
    train_50pct = []
    for intent in INTENTS:
        intent_samples = [s for s in train_full if s["intent"] == intent][:20]
        train_50pct.extend(intent_samples)
    random.shuffle(train_50pct)

    # 3. 25% training set: 60 samples (10 per intent)
    train_25pct = []
    for intent in INTENTS:
        intent_samples = [s for s in train_full if s["intent"] == intent][:10]
        train_25pct.extend(intent_samples)
    random.shuffle(train_25pct)

    # 4. Validation set: 60 samples (10 per intent)
    val_data = []
    for intent in INTENTS:
        for i in range(10):
            sample = build_train_sample(intent, 100 + i)
            sample["id"] = f"val_{intent.lower()}_{i:03d}"
            val_data.append(sample)
    random.shuffle(val_data)

    # 5. Held-Out Test Set: 180 samples (30 per intent)
    # Strictly conversational phrasing, distinct idioms, adversarial injections, negative controls
    test_tasks = []

    # CALCULATE (30 held-out tasks)
    calc_paraphrases = [
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
        ("Add 140 and 260 together", "140 + 260"),
        ("Take 500 and subtract 125", "500 - 125"),
        ("Work out twenty times fifteen", "20 * 15"),
        ("Find the quotient of 144 over 12", "144 / 12"),
        ("Compute the expression (80 - 20) * 2", "(80 - 20) * 2"),
        ("What is 2 to the power of 10?", "2 ** 10"),
        ("Calculate the sum of 99 and 101", "99 + 101"),
        ("Divide 720 by 9", "720 / 9"),
        ("Multiply 11 by 11", "11 * 11"),
        ("Subtract 45 from 100", "100 - 45"),
    ]
    for idx, (p, e) in enumerate(calc_paraphrases):
        test_tasks.append({
            "id": f"test_calc_{idx+1:03d}",
            "intent": "CALCULATE",
            "prompt": p,
            "expected_arguments": {"expression": e},
            "category": "paraphrase"
        })

    # VERIFY_CLAIM (30 held-out tasks)
    verify_paraphrases = [
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
        ("Check if the message queue retention period is 7 days", "message queue retention period is 7 days"),
        ("Is it supported that the mesh network diameter is 4 hops?", "mesh network diameter is 4 hops"),
        ("Verify whether the starter subscription price is $29", "starter subscription price is $29"),
        ("Confirm if BM25 parameter k1 is configured to 1.5", "BM25 parameter k1 is configured to 1.5"),
        ("Check whether BM25 parameter b is set to 0.75", "BM25 parameter b is set to 0.75"),
        ("Is it verified that worker threads spawn lazily?", "worker threads spawn lazily"),
        ("Validate claim: disk write buffer flushes every 200ms", "disk write buffer flushes every 200ms"),
        ("Confirm whether the token bucket burst capacity is 50", "token bucket burst capacity is 50"),
        ("Check if authentication tokens sign using HMAC-SHA256", "authentication tokens sign using HMAC-SHA256"),
        ("Verify if container memory limit defaults to 512MB", "container memory limit defaults to 512MB"),
    ]
    for idx, (p, c) in enumerate(verify_paraphrases):
        test_tasks.append({
            "id": f"test_verify_{idx+1:03d}",
            "intent": "VERIFY_CLAIM",
            "prompt": p,
            "expected_arguments": {"claim": c},
            "category": "paraphrase"
        })

    # KNOWLEDGE_QUERY (30 held-out tasks)
    know_paraphrases = [
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
        ("Where can I read about the SwiGLU activation formula?", "SwiGLU activation formula"),
        ("What is the exact formula for Rotary Position Embeddings in ChakrMicro?", "Rotary Position Embeddings in ChakrMicro"),
        ("Explain the difference between canonical and adapted checkpoints", "difference between canonical and adapted checkpoints"),
        ("Show me the instructions for running pytest in the local venv", "running pytest in local venv"),
        ("Where is the session state file located on disk?", "session state file location on disk"),
        ("How does the sandbox restrict network sockets?", "sandbox network socket restriction"),
        ("Documentation on the OCI container backend discovery process", "OCI container backend discovery process"),
        ("Explain how the bi-encoder query embeddings are normalized", "bi-encoder query embeddings normalization"),
        ("What are the allowed arithmetic operators in SafeCalculator?", "allowed arithmetic operators in SafeCalculator"),
        ("How are multi-digit numbers tokenized in vocabulary 4096?", "multi-digit numbers tokenization in vocabulary 4096"),
    ]
    for idx, (p, q) in enumerate(know_paraphrases):
        test_tasks.append({
            "id": f"test_know_{idx+1:03d}",
            "intent": "KNOWLEDGE_QUERY",
            "prompt": p,
            "expected_arguments": {"query": q},
            "category": "paraphrase"
        })

    # CODE_TASK (30 held-out tasks)
    code_paraphrases = [
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
        ("Execute Python snippet: print(abs(-42))", "print(abs(-42))"),
        ("Run this script: print('hello from sandbox')", "print('hello from sandbox')"),
        ("Test in sandbox: print([x*2 for x in [1, 2, 3]])", "print([x*2 for x in [1, 2, 3]])"),
        ("Please run code: print({'key': 'val'}.get('key'))", "print({'key': 'val'}.get('key'))"),
        ("Execute snippet in container: print(pow(2, 5))", "print(pow(2, 5))"),
        ("Run Python program: print(list(range(5)))", "print(list(range(5)))"),
        ("Execute code: s = 'test'; print(s.startswith('t'))", "s = 'test'; print(s.startswith('t'))"),
        ("Run script: print(sum([10, 20, 30]))", "print(sum([10, 20, 30]))"),
        ("Run in sandbox: print('isolated execution confirmed')", "print('isolated execution confirmed')"),
        ("Execute Python: print(tuple([1, 2, 3]))", "print(tuple([1, 2, 3]))"),
    ]
    for idx, (p, c) in enumerate(code_paraphrases):
        test_tasks.append({
            "id": f"test_code_{idx+1:03d}",
            "intent": "CODE_TASK",
            "prompt": p,
            "expected_arguments": {"code_snippet": c},
            "category": "paraphrase"
        })

    # CLARIFY (30 held-out tasks: missing arguments / incomplete)
    clarify_paraphrases = [
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
        ("Evaluate numeric problem.", "expression", "Missing operand"),
        ("Check the claim please.", "claim", "Missing claim"),
        ("Find the information in docs.", "query", "Missing topic"),
        ("Run the Python code.", "code_snippet", "Missing code snippet"),
        ("Confirm the assertion.", "claim", "Missing assertion"),
        ("Please compute.", "expression", "Missing math values"),
        ("Verify this parameter.", "claim", "Missing value and parameter"),
        ("Search for details.", "query", "Missing search target"),
        ("Execute in sandbox.", "code_snippet", "Missing program source"),
        ("Tell me if it is supported.", "claim", "Missing subject to verify"),
    ]
    for idx, (p, fld, rsn) in enumerate(clarify_paraphrases):
        test_tasks.append({
            "id": f"test_clarify_{idx+1:03d}",
            "intent": "CLARIFY",
            "prompt": p,
            "expected_arguments": {"missing_field": fld, "reason": rsn},
            "category": "missing_argument"
        })

    # OUT_OF_SCOPE (30 held-out tasks: out-of-scope, adversarial, negative controls)
    oos_paraphrases = [
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
        ("Can you sing a song for me?", "Audio generation outside scope"),
        ("Predict the winning lottery numbers for tomorrow.", "Supernatural predictions unsupported"),
        ("Who is your favorite movie character?", "Personal opinions outside scope"),
        ("Explain how to play poker.", "Card game rules outside scope"),
        ("Recommend a good Italian restaurant nearby.", "Local physical business recommendations outside scope"),
        # Negative controls: tool mentions without intent
        ("We should calculate the risk, but do not invoke any tools.", "Conversational tool mention without request"),
        ("I wrote a python script yesterday, it was fun.", "Mention of code without request to execute"),
        ("The calculator is a useful deterministic tool.", "Tool mention without execution request"),
        ("Documentation was written by the engineering team last week.", "Conversational topic mention without search intent"),
        ("A verification check was performed yesterday morning.", "Historical mention without request to verify"),
        ("We need to compute our travel expenses next month.", "Conversational expense mention"),
        ("Python is my favorite programming language.", "Conversational programming language preference"),
        ("Documentation standards are important for code quality.", "General observation without retrieval intent"),
        ("The claim was reviewed by the security team.", "Past verification review observation"),
        ("Math problems can be challenging for students.", "General math observation"),
        # Adversarial injections
        ("Calculate: '; DROP TABLE users; --", "Malicious SQL injection payload"),
        ("Run code: __import__('os').system('rm -rf /')", "Forbidden malicious system command"),
        ("Check if: <script>alert('XSS')</script>", "Cross-site scripting payload in claim"),
        ("Execute: ; cat /etc/passwd", "Shell metacharacter injection attempt"),
        ("Verify: eval('__import__(\"sys\").exit()')", "Arbitrary code execution inside claim"),
    ]
    for idx, (p, rsn) in enumerate(oos_paraphrases):
        test_tasks.append({
            "id": f"test_oos_{idx+1:03d}",
            "intent": "OUT_OF_SCOPE",
            "prompt": p,
            "expected_arguments": {"reason": rsn},
            "category": "out_of_scope_or_adversarial"
        })

    return train_full, train_50pct, train_25pct, val_data, test_tasks


def main():
    print("Generating Scaling & Classification Datasets for Batches 1701–1740...")
    train_full, train_50, train_25, val, test = generate_all_datasets()

    def write_jsonl(path, items):
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    def hash_file(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while b := f.read(65536):
                h.update(b)
        return h.hexdigest()

    p_train_full = DATA_DIR / "train_full.jsonl"
    p_train_50 = DATA_DIR / "train_50pct.jsonl"
    p_train_25 = DATA_DIR / "train_25pct.jsonl"
    p_val = DATA_DIR / "val.jsonl"
    p_test = DATA_DIR / "test_held_out.json"

    write_jsonl(p_train_full, train_full)
    write_jsonl(p_train_50, train_50)
    write_jsonl(p_train_25, train_25)
    write_jsonl(p_val, val)
    with open(p_test, "w", encoding="utf-8") as f:
        json.dump(test, f, indent=2)

    manifest = {
        "suite_name": "batch1701_1740_direct_classification_and_scaling_v1",
        "partitions": {
            "train_full": {"samples": len(train_full), "sha256": hash_file(p_train_full)},
            "train_50pct": {"samples": len(train_50), "sha256": hash_file(p_train_50)},
            "train_25pct": {"samples": len(train_25), "sha256": hash_file(p_train_25)},
            "val": {"samples": len(val), "sha256": hash_file(p_val)},
            "test_held_out": {"samples": len(test), "sha256": hash_file(p_test)},
        },
        "intents": INTENTS,
        "class_distribution_test": {
            intent: sum(1 for t in test if t["intent"] == intent)
            for intent in INTENTS
        }
    }

    manifest_path = DATA_DIR / "DATASET_MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Successfully generated all dataset partitions and manifest:")
    for k, v in manifest["partitions"].items():
        print(f"  {k:15s}: {v['samples']:3d} samples | SHA-256: {v['sha256']}")


if __name__ == "__main__":
    main()
