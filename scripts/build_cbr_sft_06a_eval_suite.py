"""
scripts/build_cbr_sft_06a_eval_suite.py

Creates the dedicated held-out evaluation suite for CBR-SFT-06A (100 tasks):
1. Morning Greetings (10 tasks)
2. Afternoon Greetings (10 tasks)
3. Evening Greetings (10 tasks)
4. Generic Greetings (10 tasks)
5. Model Identity & Boundaries (10 tasks)
6. Single-Token Copying (Digits & Numbers) (15 tasks)
7. Single-Token Copying (Keywords & Modes) (15 tasks)
8. Answerable Key-Value Retrieval (10 tasks)
9. Genuinely Unanswerable Abstention (10 tasks)

All 100 tasks are strictly held-out from training.
"""

import sys
import os
import json
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def main():
    tasks = []

    # 1. Morning greetings (10)
    for i in range(10):
        tasks.append({
            "id": f"EVAL06A-GM-{i+1:02d}",
            "category": "greetings_morning",
            "prompt": f"User: [{7+i%3:02d}:{10+i*4:02d}:00] Good morning! How is the system running today?\nAssistant:",
            "expected_concept": "Good morning",
            "scoring": "time_greeting"
        })

    # 2. Afternoon greetings (10)
    for i in range(10):
        tasks.append({
            "id": f"EVAL06A-GA-{i+1:02d}",
            "category": "greetings_afternoon",
            "prompt": f"User: [{13+i%3:02d}:{15+i*4:02d}:00] Good afternoon! Ready for the afternoon deployment?\nAssistant:",
            "expected_concept": "Good afternoon",
            "scoring": "time_greeting"
        })

    # 3. Evening greetings (10)
    for i in range(10):
        tasks.append({
            "id": f"EVAL06A-GE-{i+1:02d}",
            "category": "greetings_evening",
            "prompt": f"User: [{18+i%3:02d}:{20+i*3:02d}:00] Good evening, starting the maintenance shift tonight.\nAssistant:",
            "expected_concept": "Good evening",
            "scoring": "time_greeting"
        })

    # 4. Generic greetings (10) - Strictly held-out
    generic_prompts = [
        "Welcome to the terminal, how are you?",
        "Hi assistant, checking in.",
        "Warm greetings to you.",
        "Hey assistant, can we begin?",
        "Good day to you, ready for tasks?",
        "Hello assistant, are you active right now?",
        "Hi, pinging your service.",
        "Greetings to the ChakrMicro system.",
        "Hey, are you operational today?",
        "Hello ChakrMicro, let's start."
    ]
    for i, gp in enumerate(generic_prompts):
        tasks.append({
            "id": f"EVAL06A-GG-{i+1:02d}",
            "category": "greetings_generic",
            "prompt": f"User: {gp}\nAssistant:",
            "expected_concept": "greeting",
            "scoring": "generic_greeting"
        })

    # 5. Model Identity (10) - Strictly held-out
    id_prompts = [
        ("Which organization or research team developed you?", "ChakrView"),
        ("What exact model identifier do you carry?", "ChakrMicro"),
        ("Were you developed by OpenAI as ChatGPT?", "no"),
        ("Are you a derivative of Google's Gemini models?", "no"),
        ("What neural architecture forms your foundation?", "ChakrMicro"),
        ("Can you state your engineering lineage?", "indigenous"),
        ("Who authored the ChakrMicro codebase?", "ChakrView"),
        ("What is your designated system name?", "ChakrMicro"),
        ("Are you an instance of Anthropic's Claude?", "no"),
        ("Could you please provide a brief self-introduction?", "ChakrMicro")
    ]
    for i, (p, exp) in enumerate(id_prompts):
        tasks.append({
            "id": f"EVAL06A-ID-{i+1:02d}",
            "category": "identity",
            "prompt": f"User: {p}\nAssistant:",
            "expected_concept": exp,
            "scoring": "identity"
        })

    # 6. Held-out Single-Token Numbers (15)
    num_heldout = [
        ("User: Telemetry probe: Channel number={val}.\nQuestion: What is the channel number?\nAssistant:", "5"),
        ("User: Metric log: Thread count is {val}.\nQuestion: How many threads are reported?\nAssistant:", "8"),
        ("User: Runtime flag: Exit code={val}.\nQuestion: What is the exit code?\nAssistant:", "0"),
        ("User: Ingress status: Listening on port {val}.\nQuestion: What port is listening?\nAssistant:", "80"),
        ("User: Session config: Max retries is {val}.\nQuestion: What is the max retry count?\nAssistant:", "3"),
        ("User: Memory allocation: Block size={val}.\nQuestion: What is the block size?\nAssistant:", "64"),
        ("User: Hardware profile: Core index is {val}.\nQuestion: What is the core index?\nAssistant:", "7"),
        ("User: Queue monitor: Pending tasks={val}.\nQuestion: How many tasks are pending?\nAssistant:", "12"),
        ("User: Node report: Replica count is {val}.\nQuestion: What is the replica count?\nAssistant:", "4"),
        ("User: System audit: Error level is {val}.\nQuestion: What is the error level?\nAssistant:", "1"),
        ("User: Benchmark stat: Batch size={val}.\nQuestion: What is the batch size?\nAssistant:", "32"),
        ("User: Worker setting: Pool limit is {val}.\nQuestion: What is the pool limit?\nAssistant:", "16"),
        ("User: Network parameter: TTL={val}.\nQuestion: What is the TTL value?\nAssistant:", "99"),
        ("User: Sensor beacon: Signal ID={val}.\nQuestion: What is the signal ID?\nAssistant:", "2"),
        ("User: Cache tuning: Slot number is {val}.\nQuestion: What slot number is used?\nAssistant:", "88")
    ]
    for i, (tmpl, val) in enumerate(num_heldout):
        p = tmpl.format(val=val)
        tasks.append({
            "id": f"EVAL06A-CP-NUM-{i+1:02d}",
            "category": "single_token_number_copy",
            "prompt": p,
            "expected_concept": val,
            "scoring": "exact_copy"
        })

    # 7. Held-out Single-Token Keywords (15)
    kw_heldout = [
        ("User: Pipeline configuration: EXEC_MODE={val}.\nQuestion: What is the EXEC_MODE?\nAssistant:", "run"),
        ("User: Data parser: Input type is {val}.\nQuestion: What is the input type?\nAssistant:", "data"),
        ("User: Build parameter: Optimization level is {val}.\nQuestion: What is the optimization level?\nAssistant:", "test"),
        ("User: Profiler notice: Bottleneck in {val}.\nQuestion: Where is the bottleneck?\nAssistant:", "process"),
        ("User: Kernel trace: Current state={val}.\nQuestion: What is the current state?\nAssistant:", "step"),
        ("User: Schema mapping: Field class is {val}.\nQuestion: What is the field class?\nAssistant:", "int"),
        ("User: Computation output: Aggregation is {val}.\nQuestion: What is the aggregation?\nAssistant:", "sum"),
        ("User: Driver setting: Access pattern={val}.\nQuestion: What is the access pattern?\nAssistant:", "read"),
        ("User: Node descriptor: Hierarchy level is {val}.\nQuestion: What is the hierarchy level?\nAssistant:", "head"),
        ("User: Array metadata: Measured property={val}.\nQuestion: What property was measured?\nAssistant:", "len"),
        ("User: Security context: Active role is {val}.\nQuestion: What is the active role?\nAssistant:", "user"),
        ("User: Dispatcher metric: Current metric={val}.\nQuestion: What metric is tracked?\nAssistant:", "time"),
        ("User: Registry entry: Lookup attribute={val}.\nQuestion: What is the lookup attribute?\nAssistant:", "key"),
        ("User: System diagnostics: Execution result is {val}.\nQuestion: What was the result?\nAssistant:", "fail"),
        ("User: Buffer configuration: Capacity limit is {val}.\nQuestion: What is the capacity limit?\nAssistant:", "size")
    ]
    for i, (tmpl, val) in enumerate(kw_heldout):
        p = tmpl.format(val=val)
        tasks.append({
            "id": f"EVAL06A-CP-KW-{i+1:02d}",
            "category": "single_token_keyword_copy",
            "prompt": p,
            "expected_concept": val,
            "scoring": "exact_copy"
        })

    # 8. Answerable Key-Value Retrieval (10)
    kv_heldout = [
        ("User: Variable block:\nPORT=80\nHOST=localhost\nQuestion: What is the PORT?\nAssistant:", "80"),
        ("User: Variable block:\nENV=test\nUSER=root\nQuestion: What is the ENV?\nAssistant:", "test"),
        ("User: Variable block:\nNODE=head\nSTATUS=active\nQuestion: What is the NODE?\nAssistant:", "head"),
        ("User: Variable block:\nRETRIES=5\nTIMEOUT=30\nQuestion: What is RETRIES?\nAssistant:", "5"),
        ("User: Variable block:\nWORKERS=4\nDRIVER=local\nQuestion: What is WORKERS?\nAssistant:", "4"),
        ("User: Variable block:\nMODE=read\nFORMAT=json\nQuestion: What is MODE?\nAssistant:", "read"),
        ("User: Variable block:\nKEY=data\nTYPE=raw\nQuestion: What is KEY?\nAssistant:", "data"),
        ("User: Variable block:\nLEVEL=1\nPRIORITY=high\nQuestion: What is LEVEL?\nAssistant:", "1"),
        ("User: Variable block:\nCOUNT=10\nSHARD=0\nQuestion: What is COUNT?\nAssistant:", "10"),
        ("User: Variable block:\nNAME=user\nROLE=admin\nQuestion: What is NAME?\nAssistant:", "user")
    ]
    for i, (p, val) in enumerate(kv_heldout):
        tasks.append({
            "id": f"EVAL06A-KV-{i+1:02d}",
            "category": "key_value_retrieval",
            "prompt": p,
            "expected_concept": val,
            "scoring": "exact_copy"
        })

    # 9. Genuinely Unanswerable Abstention (10)
    abstention_heldout = [
        "User: Variable block:\nPORT=80\nHOST=localhost\nQuestion: What is the PASSWORD?\nAssistant:",
        "User: Variable block:\nENV=test\nUSER=root\nQuestion: What is the DATABASE_NAME?\nAssistant:",
        "User: Notice: Server 4 rebooted at 02:00.\nQuestion: What is the root password?\nAssistant:",
        "User: System config:\nWORKERS=4\nQuestion: What is the IP address of the worker?\nAssistant:",
        "User: Alert: Network gateway latency is high.\nQuestion: Who is the provider?\nAssistant:",
        "User: Variable block:\nMODE=read\nQuestion: What is the file encryption key?\nAssistant:",
        "User: Metric notice: Memory reached 85% capacity.\nQuestion: What process is consuming the most CPU?\nAssistant:",
        "User: Setting: RETRIES=5.\nQuestion: What is the BACKUP_DIR?\nAssistant:",
        "User: Container log: Application started.\nQuestion: What port is exposed?\nAssistant:",
        "User: Status report: Database synchronization in progress.\nQuestion: What is the replica cluster name?\nAssistant:"
    ]
    for i, p in enumerate(abstention_heldout):
        tasks.append({
            "id": f"EVAL06A-ABS-{i+1:02d}",
            "category": "abstention",
            "prompt": p,
            "expected_concept": "not provided",
            "scoring": "abstention"
        })

    assert len(tasks) == 100, f"Expected 100 tasks, got {len(tasks)}"

    suite_obj = {
        "suite_name": "HELD_OUT_CBR_SFT_06A_100",
        "total_tasks": len(tasks),
        "tasks": tasks
    }

    out_p = REPO_ROOT / "evaluation" / "cbr_sft_06a_suite" / "held_out_cbr_sft_06a_100.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(suite_obj, f, indent=2)

    h = hashlib.sha256(open(out_p, "rb").read()).hexdigest()
    print(f"CBR-SFT-06A Evaluation Suite created: {len(tasks)} tasks | SHA256: {h}")

if __name__ == "__main__":
    main()
