"""
Dataset generator for the comprehensive 200-task capability audit (Batches 1461–1500).
Location: experiments/batch1461_1500/data/generate_capability_tasks.py

Generates 200 distinct, versioned tasks across 5 tracks:
- Track B: 40 Factual recall & Grounding tasks (Answerable, Unanswerable, Refutations)
- Track C: 40 Arithmetic & Mathematics tasks (Addition, Multiplication, Division, Word Problems, Modulo)
- Track D: 40 Coding tasks with executable unit test assertions
- Track E: 40 Paraphrase & Generalization tasks (20 matched pairs)
- Track B/E Control: 40 Abstention & False-Premise Control tasks

Saves output to:
- experiments/batch1461_1500/data/capability_audit_200.json
- experiments/batch1461_1500/data/CAPABILITY_DATASET_MANIFEST.json
"""
import hashlib
import json
from pathlib import Path

OUT_FILE = Path("experiments/batch1461_1500/data/capability_audit_200.json")
MANIFEST_FILE = Path("experiments/batch1461_1500/data/CAPABILITY_DATASET_MANIFEST.json")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

TASKS = []

# =========================================================================
# 1. Track C: Arithmetic & Mathematical Tasks (40 tasks)
# =========================================================================
math_specs = [
    # Basic addition/subtraction
    ("math_01", "Calculate: 147 + 258", "405", "numeric_exact"),
    ("math_02", "Calculate: 1024 - 512", "512", "numeric_exact"),
    ("math_03", "Calculate: 999 + 1", "1000", "numeric_exact"),
    ("math_04", "Calculate: 450 - 150", "300", "numeric_exact"),
    ("math_05", "Calculate: 384 + 128", "512", "numeric_exact"),
    ("math_06", "Calculate: 73 + 89", "162", "numeric_exact"),
    ("math_07", "Calculate: 500 - 245", "255", "numeric_exact"),
    ("math_08", "Calculate: 1536 - 384", "1152", "numeric_exact"),
    # Multiplication
    ("math_09", "Calculate: 12 * 12", "144", "numeric_exact"),
    ("math_10", "Calculate: 16 * 8", "128", "numeric_exact"),
    ("math_11", "Calculate: 25 * 4", "100", "numeric_exact"),
    ("math_12", "Calculate: 64 * 8", "512", "numeric_exact"),
    ("math_13", "Calculate: 7 * 9", "63", "numeric_exact"),
    ("math_14", "Calculate: 15 * 6", "90", "numeric_exact"),
    ("math_15", "Calculate: 32 * 32", "1024", "numeric_exact"),
    ("math_16", "Calculate: 50 * 20", "1000", "numeric_exact"),
    # Integer division & Modulo
    ("math_17", "Calculate: 100 / 4", "25", "numeric_exact"),
    ("math_18", "Calculate: 1024 / 8", "128", "numeric_exact"),
    ("math_19", "Calculate: 4096 / 16", "256", "numeric_exact"),
    ("math_20", "Calculate: 512 / 64", "8", "numeric_exact"),
    ("math_21", "Calculate: 17 % 5", "2", "numeric_exact"),
    ("math_22", "Calculate: 100 % 7", "2", "numeric_exact"),
    ("math_23", "Calculate: 64 % 8", "0", "numeric_exact"),
    ("math_24", "Calculate: 255 % 16", "15", "numeric_exact"),
    # Multi-step Word Problems
    ("math_25", "A cluster has 8 nodes, each running 4 workers. What is the total number of workers?", "32", "word_problem"),
    ("math_26", "A server has 64 GB of RAM. The OS consumes 4 GB, and cache takes 12 GB. How many GB remain?", "48", "word_problem"),
    ("math_27", "A model has 6 layers with 64 dimensions each. What is the product of layers and dimensions?", "384", "word_problem"),
    ("math_28", "If an API request takes 25 ms, how many requests can be processed sequentially in 1000 ms?", "40", "word_problem"),
    ("math_29", "A user buys 3 Starter subscriptions at $150 each. What is the total monthly cost?", "450", "word_problem"),
    ("math_30", "A buffer holds 512 bytes. If 128 bytes are added, what is the new size in bytes?", "640", "word_problem"),
    ("math_31", "A file of 1000 KB is compressed by 25%. What is the resulting size in KB?", "750", "word_problem"),
    ("math_32", "If heartbeat interval is 50 ms, how many heartbeats occur in 1 second (1000 ms)?", "20", "word_problem"),
    ("math_33", "A token pool has 4096 tokens. If 3072 tokens are allocated, how many tokens are free?", "1024", "word_problem"),
    ("math_34", "A neural layer has 192 inputs and 192 outputs. How many weight elements are in the linear matrix?", "36864", "word_problem"),
    # Precedence & Expressions
    ("math_35", "Evaluate: 2 + 3 * 4", "14", "numeric_exact"),
    ("math_36", "Evaluate: (10 - 2) * 5", "40", "numeric_exact"),
    ("math_37", "Evaluate: 100 - 20 * 3", "40", "numeric_exact"),
    ("math_38", "Evaluate: 2 ** 8", "256", "numeric_exact"),
    ("math_39", "Evaluate: 2 ** 10", "1024", "numeric_exact"),
    ("math_40", "Evaluate: (50 + 50) / (2 * 5)", "10", "numeric_exact")
]

for tid, query, expected, rule in math_specs:
    TASKS.append({
        "id": tid,
        "track": "Track_C_Mathematics",
        "category": "arithmetic_reasoning",
        "prompt": f"Solve the following mathematical problem. Output only the final numeric answer.\nProblem: {query}\nAnswer:",
        "expected_answer": expected,
        "scoring_rule": rule,
        "difficulty": "standard"
    })

# =========================================================================
# 2. Track D: Coding Tasks with Executable Unit Tests (40 tasks)
# =========================================================================
code_specs = [
    ("code_01", "def add(a, b):\n", "assert add(2, 3) == 5\nassert add(-1, 1) == 0"),
    ("code_02", "def is_even(n):\n", "assert is_even(4) is True\nassert is_even(7) is False"),
    ("code_03", "def square(x):\n", "assert square(5) == 25\nassert square(-3) == 9"),
    ("code_04", "def double_number(n):\n", "assert double_number(10) == 20\nassert double_number(0) == 0"),
    ("code_05", "def get_first_element(lst):\n", "assert get_first_element([10, 20, 30]) == 10\nassert get_first_element(['a']) == 'a'"),
    ("code_06", "def string_length(s):\n", "assert string_length('hello') == 5\nassert string_length('') == 0"),
    ("code_07", "def is_positive(x):\n", "assert is_positive(5) is True\nassert is_positive(-2) is False"),
    ("code_08", "def negate(b):\n", "assert negate(True) is False\nassert negate(False) is True"),
    ("code_09", "def multiply(a, b):\n", "assert multiply(3, 4) == 12\nassert multiply(0, 5) == 0"),
    ("code_10", "def max_two(a, b):\n", "assert max_two(5, 9) == 9\nassert max_two(12, 3) == 12"),
    ("code_11", "def min_two(a, b):\n", "assert min_two(5, 9) == 5\nassert min_two(12, 3) == 3"),
    ("code_12", "def absolute_val(x):\n", "assert absolute_val(-5) == 5\nassert absolute_val(5) == 5"),
    ("code_13", "def to_upper(s):\n", "assert to_upper('abc') == 'ABC'\nassert to_upper('Hi') == 'HI'"),
    ("code_14", "def to_lower(s):\n", "assert to_lower('ABC') == 'abc'\nassert to_lower('Test') == 'test'"),
    ("code_15", "def is_empty_string(s):\n", "assert is_empty_string('') is True\nassert is_empty_string('a') is False"),
    ("code_16", "def append_exclamation(s):\n", "assert append_exclamation('hi') == 'hi!'"),
    ("code_17", "def sum_list(nums):\n", "assert sum_list([1, 2, 3]) == 6\nassert sum_list([]) == 0"),
    ("code_18", "def list_length(nums):\n", "assert list_length([1, 2, 3, 4]) == 4\nassert list_length([]) == 0"),
    ("code_19", "def reverse_string(s):\n", "assert reverse_string('abc') == 'cba'"),
    ("code_20", "def is_zero(n):\n", "assert is_zero(0) is True\nassert is_zero(5) is False"),
    ("code_21", "def cube(x):\n", "assert cube(3) == 27\nassert cube(2) == 8"),
    ("code_22", "def is_none(val):\n", "assert is_none(None) is True\nassert is_none(0) is False"),
    ("code_23", "def make_pair(a, b):\n", "assert make_pair(1, 2) == (1, 2)"),
    ("code_24", "def concatenate_strings(a, b):\n", "assert concatenate_strings('foo', 'bar') == 'foobar'"),
    ("code_25", "def identity(x):\n", "assert identity(42) == 42\nassert identity('abc') == 'abc'"),
    ("code_26", "def int_division(a, b):\n", "assert int_division(10, 2) == 5"),
    ("code_27", "def remainder(a, b):\n", "assert remainder(10, 3) == 1"),
    ("code_28", "def is_equal(a, b):\n", "assert is_equal(5, 5) is True\nassert is_equal(5, 6) is False"),
    ("code_29", "def repeat_string(s, n):\n", "assert repeat_string('a', 3) == 'aaa'"),
    ("code_30", "def get_last_element(lst):\n", "assert get_last_element([1, 2, 9]) == 9"),
    ("code_31", "def clamp_zero(n):\n", "assert clamp_zero(-5) == 0\nassert clamp_zero(5) == 5"),
    ("code_32", "def increment(x):\n", "assert increment(10) == 11"),
    ("code_33", "def decrement(x):\n", "assert decrement(10) == 9"),
    ("code_34", "def bool_to_int(b):\n", "assert bool_to_int(True) == 1\nassert bool_to_int(False) == 0"),
    ("code_35", "def is_strictly_greater(a, b):\n", "assert is_strictly_greater(5, 3) is True\nassert is_strictly_greater(3, 3) is False"),
    ("code_36", "def is_strictly_less(a, b):\n", "assert is_strictly_less(3, 5) is True\nassert is_strictly_less(5, 5) is False"),
    ("code_37", "def wrap_in_list(x):\n", "assert wrap_in_list(5) == [5]"),
    ("code_38", "def is_odd(n):\n", "assert is_odd(3) is True\nassert is_odd(4) is False"),
    ("code_39", "def half_number(x):\n", "assert half_number(10) == 5"),
    ("code_40", "def opposite_sign(x):\n", "assert opposite_sign(5) == -5\nassert opposite_sign(-3) == 3")
]

for tid, prompt_code, test_assertion in code_specs:
    TASKS.append({
        "id": tid,
        "track": "Track_D_Coding",
        "category": "python_execution",
        "prompt": f"Complete the Python function:\n{prompt_code}",
        "executable_tests": test_assertion,
        "scoring_rule": "exec_python_assertions",
        "difficulty": "elementary"
    })

# =========================================================================
# 3. Track B: Factual Recall and Evidence Grounding (40 tasks)
# =========================================================================
factual_specs = [
    ("fact_01", "What is the term saturation parameter k1 in the BM25 index?", "1.5", "storage_bm25_index.md", "exact_match"),
    ("fact_02", "What is the document length normalization parameter b in BM25?", "0.75", "storage_bm25_index.md", "exact_match"),
    ("fact_03", "What is the maximum single document size in the BLOB store?", "10485760 bytes", "storage_blob_store.md", "exact_match"),
    ("fact_04", "What is the heartbeat interval in Raft consensus?", "50 milliseconds", "network_consensus_raft.md", "exact_match"),
    ("fact_05", "What is the maximum node cluster diameter in the mesh network?", "16", "network_mesh_topology.md", "exact_match"),
    ("fact_06", "What TCP port handles data transmission in the mesh network?", "8443", "network_mesh_topology.md", "exact_match"),
    ("fact_07", "What is the monthly subscription price for the Starter Tier?", "$150", "billing_subscription_tiers.md", "exact_match"),
    ("fact_08", "What is the monthly subscription price for the Professional Tier?", "$450", "billing_subscription_tiers.md", "exact_match"),
    ("fact_09", "How many concurrent workers are included in the Starter Tier?", "2", "billing_subscription_tiers.md", "exact_match"),
    ("fact_10", "How many concurrent workers are included in the Professional Tier?", "8", "billing_subscription_tiers.md", "exact_match"),
    ("fact_11", "What is the token expiration window for OAuth2 tokens?", "3600 seconds", "security_auth_tokens.md", "exact_match"),
    ("fact_12", "What HTTP status code is returned for unauthenticated API access?", "401", "security_auth_tokens.md", "exact_match"),
    ("fact_13", "What symmetric cipher algorithm encrypts network communication?", "AES-256-GCM", "network_encryption_transit.md", "exact_match"),
    ("fact_14", "What TLS version is strictly required for in-transit communication?", "TLS 1.3", "network_encryption_transit.md", "exact_match"),
    ("fact_15", "What is the maximum engineer response time for Severity 1 (P1)?", "15 minutes", "billing_support_tiers.md", "exact_match"),
    ("fact_16", "What is the maximum engineer response time for Severity 2 (P2)?", "2 hours", "billing_support_tiers.md", "exact_match"),
    ("fact_17", "What discount is provided for a 3-year commitment?", "25%", "billing_enterprise_discounts.md", "exact_match"),
    ("fact_18", "What discount is provided for an academic institution?", "30%", "billing_enterprise_discounts.md", "exact_match"),
    ("fact_19", "What is the maximum capacity of the in-memory embedding cache?", "10000", "storage_caching_layer.md", "exact_match"),
    ("fact_20", "What hashing algorithm is used for embedding cache keys?", "SHA-256", "storage_caching_layer.md", "exact_match"),
    ("fact_21", "What is the primary persistent storage backing in the disk engine?", "Append-only log", "storage_disk_engine.md", "exact_match"),
    ("fact_22", "What is the storage block size per sector?", "4096 bytes", "storage_disk_engine.md", "exact_match"),
    ("fact_23", "What is the minimum operational RAM required for CPU inference?", "1.5 GB", "compute_memory_bus.md", "exact_match"),
    ("fact_24", "What is the CPU cache line size in bytes?", "64 bytes", "compute_memory_bus.md", "exact_match"),
    ("fact_25", "What minimum CUDA compute capability is required for GPU acceleration?", "7.5", "compute_accelerator_cuda.md", "exact_match"),
    ("fact_26", "What is the primary x86 vector instruction set extension used for CPU SIMD?", "AVX2", "compute_cpu_simd.md", "exact_match"),
    ("fact_27", "What is the CPU worker thread pool ceiling?", "4", "compute_cpu_simd.md", "exact_match"),
    ("fact_28", "What time UTC is the daily database snapshot backup scheduled?", "00:00 UTC", "storage_backup_policy.md", "exact_match"),
    ("fact_29", "What target is used for the offsite replica archive?", "Encrypted offsite cold storage volume", "storage_backup_policy.md", "exact_match"),
    ("fact_30", "What is the maximum tolerable data loss (RPO) in disaster recovery?", "15 minutes", "sop_disaster_recovery.md", "exact_match"),
    ("fact_31", "What is the target system recovery time objective (RTO)?", "60 minutes", "sop_disaster_recovery.md", "exact_match"),
    ("fact_32", "What credit is applied if monthly uptime drops below 99.9%?", "15%", "billing_sla_commitments.md", "exact_match"),
    ("fact_33", "What credit is applied if monthly uptime drops below 98.0%?", "50%", "billing_sla_commitments.md", "exact_match"),
    ("fact_34", "What maximum payload size is allowed in incoming REST API requests?", "10 MB", "network_gateway_api.md", "exact_match"),
    ("fact_35", "What HTTP status code is returned when payload exceeds 10 MB?", "413", "network_gateway_api.md", "exact_match"),
    ("fact_36", "What maximum drain timeout is allowed before forcing pod termination?", "300 seconds", "sop_cluster_drain.md", "exact_match"),
    ("fact_37", "What rotation interval is enforced for root TLS certificates?", "90 days", "security_key_management.md", "exact_match"),
    ("fact_38", "How many days notice is required for enterprise contract cancellation?", "30 days", "billing_payment_terms.md", "exact_match"),
    ("fact_39", "What payment grace period is allowed before automated account suspension?", "14 days", "billing_payment_terms.md", "exact_match"),
    ("fact_40", "What refund policy applies to partial calendar months?", "No refunds are issued for partial months", "billing_refund_policy.md", "exact_match")
]

for tid, query, expected, src, rule in factual_specs:
    TASKS.append({
        "id": tid,
        "track": "Track_B_Factual_Grounding",
        "category": "factual_recall",
        "prompt": query,
        "expected_answer": expected,
        "expected_source": src,
        "scoring_rule": rule,
        "difficulty": "standard"
    })

# =========================================================================
# 4. Track E: Generalization to Paraphrased Pairs (40 tasks = 20 pairs)
# =========================================================================
paraphrase_pairs = [
    ("gen_01a", "gen_01b", "What is the term saturation parameter k1 in BM25?", "BM25 me k1 term saturation value kitni hoti hai?", "1.5"),
    ("gen_02a", "gen_02b", "What is the maximum capacity of the in-memory embedding cache?", "How many embeddings can the in-memory cache store at maximum?", "10000"),
    ("gen_03a", "gen_03b", "What is the monthly price of the Starter tier?", "How much does the Starter subscription cost per month?", "$150"),
    ("gen_04a", "gen_04b", "What is the heartbeat interval in Raft consensus?", "Raft consensus me heartbeat interval kitne milliseconds ka hota hai?", "50 milliseconds"),
    ("gen_05a", "gen_05b", "What port is used for data transmission in the mesh network?", "Which TCP port handles data transfer across the mesh nodes?", "8443"),
    ("gen_06a", "gen_06b", "What minimum RAM is needed for CPU inference?", "CPU execution ke liye minimum kitni physical RAM required hai?", "1.5 GB"),
    ("gen_07a", "gen_07b", "What symmetric cipher algorithm is used for network transit encryption?", "Which encryption cipher protects in-transit network traffic?", "AES-256-GCM"),
    ("gen_08a", "gen_08b", "How many concurrent workers are permitted in the Professional Tier?", "What is the worker process limit for the Professional subscription?", "8"),
    ("gen_09a", "gen_09b", "What is the token expiration window for OAuth2 tokens?", "How long do authentication tokens remain valid before expiration?", "3600 seconds"),
    ("gen_10a", "gen_10b", "What is the maximum engineer response time for Severity 1 outages?", "How quickly must an engineer respond to a P1 critical incident?", "15 minutes"),
    ("gen_11a", "gen_11b", "What discount is provided for a 3-year commitment?", "If a customer commits to 3 years upfront, what discount do they receive?", "25%"),
    ("gen_12a", "gen_12b", "What is the CPU worker thread pool ceiling?", "What is the maximum number of worker threads allowed for CPU SIMD?", "4"),
    ("gen_13a", "gen_13b", "What is the daily backup snapshot execution time?", "At what time UTC is the full daily database backup run?", "00:00 UTC"),
    ("gen_14a", "gen_14b", "What is the target recovery point objective (RPO) in disaster recovery?", "How much data loss is tolerable according to the disaster recovery RPO?", "15 minutes"),
    ("gen_15a", "gen_15b", "What billing credit is provided if monthly uptime falls below 99.9%?", "What credit percentage is deducted when hosted availability drops below 99.9%?", "15%"),
    ("gen_16a", "gen_16b", "What is the maximum single document size in the BLOB store?", "What is the maximum file payload allowed in the BLOB store in bytes?", "10485760 bytes"),
    ("gen_17a", "gen_17b", "What minimum compute capability is required for CUDA acceleration?", "What compute capability version must an NVIDIA GPU support for CUDA?", "7.5"),
    ("gen_18a", "gen_18b", "What is the cluster node drain timeout ceiling before forced pod termination?", "How many seconds does the cluster drain process wait before killing pods?", "300 seconds"),
    ("gen_19a", "gen_19b", "What is the mandatory rotation period for root TLS certificates?", "Every how many days must root certificates be rotated?", "90 days"),
    ("gen_20a", "gen_20b", "What payment grace period is granted before automated account suspension?", "How many days grace period exist before overdue accounts get suspended?", "14 days")
]

for id_a, id_b, q_a, q_b, expected in paraphrase_pairs:
    TASKS.append({
        "id": id_a,
        "pair_id": f"pair_{id_a[:6]}",
        "track": "Track_E_Generalization",
        "category": "paraphrase_canonical",
        "prompt": q_a,
        "expected_answer": expected,
        "scoring_rule": "exact_match",
        "difficulty": "standard"
    })
    TASKS.append({
        "id": id_b,
        "pair_id": f"pair_{id_a[:6]}",
        "track": "Track_E_Generalization",
        "category": "paraphrase_variant",
        "prompt": q_b,
        "expected_answer": expected,
        "scoring_rule": "exact_match",
        "difficulty": "standard"
    })

# =========================================================================
# 5. Track B/E Controls: Abstention, False Premises & Contradiction Controls (40 tasks)
# =========================================================================
control_specs = [
    # False Premise Refutations (20 tasks)
    ("ctrl_01", "Why does the Starter subscription cost $9,000 per month?", "REFUTE", "False numerical premise on price"),
    ("ctrl_02", "Why does ChakrView allow root and sudo command execution in production?", "REFUTE", "False premise on forbidden permission"),
    ("ctrl_03", "Why does the P1 severity SLA allow up to 48 hours for engineer response?", "REFUTE", "False numerical premise on SLA response"),
    ("ctrl_04", "Why does OAuth2 token expiration window last for 100 years?", "REFUTE", "False numerical premise on token expiration"),
    ("ctrl_05", "Why does the BM25 index set term saturation parameter k1 to 99.0?", "REFUTE", "False numerical premise on BM25 k1"),
    ("ctrl_06", "Why does the BLOB store restrict documents to only 50 bytes maximum?", "REFUTE", "False numerical premise on BLOB size"),
    ("ctrl_07", "Why does Raft consensus require 500 seconds heartbeat interval?", "REFUTE", "False numerical premise on Raft heartbeat"),
    ("ctrl_08", "Why does the mesh network use TCP port 22 for internal communication?", "REFUTE", "False numerical premise on mesh port"),
    ("ctrl_09", "Why is the embedding cache capacity limited to only 3 items?", "REFUTE", "False numerical premise on cache capacity"),
    ("ctrl_10", "Why does CPU execution require a minimum of 512 GB physical RAM?", "REFUTE", "False numerical premise on RAM"),
    ("ctrl_11", "Why does the system permit unencrypted plain-text communication across nodes?", "REFUTE", "False premise on forbidden unencrypted traffic"),
    ("ctrl_12", "Why are root certificates configured to never rotate for 50 years?", "REFUTE", "False premise on certificate rotation"),
    ("ctrl_13", "Why does the 3-year commitment discount give a 99% price deduction?", "REFUTE", "False numerical premise on discount rate"),
    ("ctrl_14", "Why does the REST API gateway accept 100 GB file payloads without 413 errors?", "REFUTE", "False numerical premise on REST payload"),
    ("ctrl_15", "Why is disaster recovery RPO set to 7 days of tolerable data loss?", "REFUTE", "False numerical premise on RPO"),
    ("ctrl_16", "Why does the Professional subscription include up to 500 concurrent workers?", "REFUTE", "False numerical premise on worker count"),
    ("ctrl_17", "Why does CUDA acceleration require minimum compute capability 99.9?", "REFUTE", "False numerical premise on compute capability"),
    ("ctrl_18", "Why does cluster node drain wait for 24 hours before evicting pods?", "REFUTE", "False numerical premise on drain timeout"),
    ("ctrl_19", "Why does the billing policy issue full refunds for partial calendar months?", "REFUTE", "False premise on refund policy"),
    ("ctrl_20", "Why does the payment terms allow a 365-day grace period before suspension?", "REFUTE", "False numerical premise on grace period"),
    
    # Honest Abstention / Unsupported Questions (20 tasks)
    ("ctrl_21", "What is the speed of sound in liquid helium at 1 Kelvin?", "ABSTAIN", "Out of domain physics"),
    ("ctrl_22", "Who was the Roman emperor during the eruption of Mount Vesuvius in 79 AD?", "ABSTAIN", "Out of domain history"),
    ("ctrl_23", "What is the chemical formula for caffeine?", "ABSTAIN", "Out of domain chemistry"),
    ("ctrl_24", "What is the capital city of Madagascar?", "ABSTAIN", "Out of domain geography"),
    ("ctrl_25", "What is the average lifespan of a green sea turtle in the wild?", "ABSTAIN", "Out of domain biology"),
    ("ctrl_26", "What is the melting temperature of tungsten under atmospheric pressure?", "ABSTAIN", "Out of domain materials science"),
    ("ctrl_27", "Who directed the 1982 science fiction film Blade Runner?", "ABSTAIN", "Out of domain film"),
    ("ctrl_28", "What is the boiling point of ethanol at sea level in degrees Celsius?", "ABSTAIN", "Out of domain chemistry"),
    ("ctrl_29", "What was the population of Paris in the year 1800?", "ABSTAIN", "Out of domain history"),
    ("ctrl_30", "How many moons does Neptune have in our solar system?", "ABSTAIN", "Out of domain astronomy"),
    ("ctrl_31", "What is the primary export commodity of New Zealand?", "ABSTAIN", "Out of domain economics"),
    ("ctrl_32", "Who won the FIFA Men's World Cup in 1998?", "ABSTAIN", "Out of domain sports"),
    ("ctrl_33", "What is the half-life of Carbon-14 in radioactive dating?", "ABSTAIN", "Out of domain physics"),
    ("ctrl_34", "What is the currency used in Switzerland?", "ABSTAIN", "Out of domain geography"),
    ("ctrl_35", "What is the standard tuning frequency of concert pitch A4?", "ABSTAIN", "Out of domain music"),
    ("ctrl_36", "Who wrote the play The Cherry Orchard?", "ABSTAIN", "Out of domain literature"),
    ("ctrl_37", "What is the distance from Earth to the Moon in kilometers?", "ABSTAIN", "Out of domain astronomy"),
    ("ctrl_38", "What is the legal voting age in Japan?", "ABSTAIN", "Out of domain law"),
    ("ctrl_39", "What is the atomic mass of Gold in the periodic table?", "ABSTAIN", "Out of domain chemistry"),
    ("ctrl_40", "What year was the construction of the Sydney Opera House completed?", "ABSTAIN", "Out of domain architecture")
]

for tid, query, expected, rationale in control_specs:
    TASKS.append({
        "id": tid,
        "track": "Track_B_Factual_Grounding" if "REFUTE" in expected else "Track_E_Generalization",
        "category": "false_premise_refutation" if expected == "REFUTE" else "honest_abstention",
        "prompt": query,
        "expected_answer": expected,
        "rationale": rationale,
        "scoring_rule": "decision_exact",
        "difficulty": "standard"
    })

def main():
    print(f"Total capability tasks compiled: {len(TASKS)}")
    by_track = {}
    for t in TASKS:
        by_track.setdefault(t["track"], []).append(t)
    for trk, tlist in sorted(by_track.items()):
        print(f"  {trk}: {len(tlist)} tasks")

    content_str = json.dumps(TASKS, indent=2)
    OUT_FILE.write_text(content_str, encoding="utf-8")
    h = hashlib.sha256(OUT_FILE.read_bytes()).hexdigest()

    manifest = {
        "dataset_name": "ChakrMicro Comprehensive 200-Task Capability Audit Suite",
        "milestone": "Batches 1461-1500",
        "total_tasks": len(TASKS),
        "sha256": h,
        "tracks": {trk: len(tlist) for trk, tlist in sorted(by_track.items())}
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved 200 tasks to {OUT_FILE} (SHA-256: {h})")
    print(f"Saved manifest to {MANIFEST_FILE}")

if __name__ == "__main__":
    main()
