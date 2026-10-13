"""
Batch 207: End-to-End Interactive Runtime Benchmark Across Mathematics, Coding, and Science.
Location: experiments/batch201_210_runtime_integration/run_interactive_runtime_benchmark.py

Exercises the integrated GroundedSessionAdapter and InteractiveModelSessionCoordinator
across multi-turn interactive dialogues, checking:
1. Mathematics queries.
2. Coding queries.
3. Science queries (English and Hindi).
4. Follow-up pronoun reference with multi-turn history.
5. Contradiction rejection policy.
6. Safe abstention on missing facts.
7. Output provenance attribution (ComponentOrigin).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.cognitive_router import CognitiveRouter, RouteType, ComponentOrigin
from chakrview.runtime.persistent_memory import PersistentSessionMemory
from chakrview.runtime.pointer_mechanism import ChakrPointerHead
from chakrview.runtime.retrieval_adapter import RetrievalAdapter
from chakrview.runtime.session_adapter import GroundedSessionAdapter
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
POINTER_CKPT = Path("experiments/batch201_210_runtime_integration/checkpoints/query_aware_pointer_multilingual.pt")
REPORTS_DIR = Path("experiments/batch201_210_runtime_integration/reports")


class MockEngine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


def run_interactive_runtime_benchmark():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="chakr_interactive_bench_"))

    device = torch.device("cpu")
    ck = torch.load(CANONICAL_CHECKPOINT, map_location=device, weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    pointer_head = ChakrPointerHead(d_model=192, vocab_size=4096).to(device)
    if POINTER_CKPT.exists():
        pointer_head.load_state_dict(torch.load(POINTER_CKPT, map_location=device, weights_only=False))
    pointer_head.eval()

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    engine = MockEngine(model, tokenizer)

    mem = PersistentSessionMemory(temp_dir / "sessions")
    adapter = GroundedSessionAdapter(
        inference_engine=engine,
        retrieval_adapter=RetrievalAdapter(),
        memory=mem,
        router=CognitiveRouter(enable_pointer_copy=True),
        pointer_head=pointer_head,
        enable_constrained_decoding=True,
    )

    session_id = "bench_session_207"
    turns_log = []

    # Turn 1: Math fact query with supplied context
    ctx1 = "For a standard geometric Circle, the area formula is pi * r^2 and circumference is 2 * pi * r."
    q1 = "What is the area formula of Circle?"
    resp1 = adapter.execute_turn(session_id, q1, supplied_context=ctx1)
    t1_ok = (resp1.is_supported and len(resp1.answer) > 0 and resp1.component_origin in [ComponentOrigin.DETERMINISTIC_EXTRACTOR, ComponentOrigin.CONSTRAINED_TRIE, ComponentOrigin.NEURAL_POINTER])
    turns_log.append({
        "turn": 1,
        "domain": "mathematics",
        "query": q1,
        "answer": resp1.answer,
        "origin": resp1.component_origin.value,
        "passed": t1_ok,
    })

    # Turn 2: Follow-up pronoun reference ('What is its circumference?')
    q2 = "What is its circumference?"
    resp2 = adapter.execute_turn(session_id, q2, supplied_context=ctx1)
    t2_ok = (resp2.is_supported and len(resp2.answer) > 0)
    turns_log.append({
        "turn": 2,
        "domain": "mathematics_followup",
        "query": q2,
        "answer": resp2.answer,
        "origin": resp2.component_origin.value,
        "passed": t2_ok,
    })

    # Turn 3: Coding query (QuickSort complexity)
    ctx3 = "The sorting algorithm QuickSort has average time complexity O(n log n) and worst-case complexity O(n^2)."
    q3 = "What is the average time complexity of QuickSort?"
    resp3 = adapter.execute_turn(session_id, q3, supplied_context=ctx3)
    t3_ok = (resp3.is_supported and len(resp3.answer) > 0)
    turns_log.append({
        "turn": 3,
        "domain": "coding",
        "query": q3,
        "answer": resp3.answer,
        "origin": resp3.component_origin.value,
        "passed": t3_ok,
    })

    # Turn 4: Science query (Carbon atomic number)
    ctx4 = "Carbon is a nonmetal with atomic number 6 and sublimation point 3915 K."
    q4 = "What is the atomic number of Carbon?"
    resp4 = adapter.execute_turn(session_id, q4, supplied_context=ctx4)
    t4_ok = ("6" in resp4.answer and resp4.is_supported)
    turns_log.append({
        "turn": 4,
        "domain": "science_english",
        "query": q4,
        "answer": resp4.answer,
        "origin": resp4.component_origin.value,
        "passed": t4_ok,
    })

    # Turn 5: Multilingual Science query in Hindi (सोना का गलनांक)
    ctx5 = "सोना एक उत्कृष्ट धातु है जिसका परमाणु क्रमांक 79 है और इसका गलनांक 1064 C है।"
    q5 = "What is the गलनांक of सोना?"
    resp5 = adapter.execute_turn(session_id, q5, supplied_context=ctx5)
    t5_ok = (resp5.is_supported and len(resp5.answer) > 0)
    turns_log.append({
        "turn": 5,
        "domain": "science_hindi",
        "query": q5,
        "answer": resp5.answer,
        "origin": resp5.component_origin.value,
        "passed": t5_ok,
    })

    # Turn 6: Contradiction rejection policy
    ctx6 = "The boiling point of liquid nitrogen is -196 C. Another report claims the boiling point of liquid nitrogen is -190 C."
    q6 = "What is the boiling point of liquid nitrogen?"
    resp6 = adapter.execute_turn(session_id, q6, supplied_context=ctx6)
    t6_ok = ("Conflicting evidence detected" in resp6.answer or not resp6.is_supported)
    turns_log.append({
        "turn": 6,
        "domain": "contradiction_policy",
        "query": q6,
        "answer": resp6.answer,
        "origin": resp6.component_origin.value,
        "passed": t6_ok,
    })

    # Turn 7: Safe abstention on ungrounded fact
    q7 = "What is the capital of Mars?"
    resp7 = adapter.execute_turn(session_id, q7, supplied_context=ctx4)
    t7_ok = ("Not mentioned in context" in resp7.answer or not resp7.is_supported)
    turns_log.append({
        "turn": 7,
        "domain": "abstention_policy",
        "query": q7,
        "answer": resp7.answer,
        "origin": resp7.component_origin.value,
        "passed": t7_ok,
    })

    # Turn 8: Session isolation (verify new tenant cannot read prior session)
    new_tenant_hist = mem.get_history("isolated_tenant_session_888")
    t8_ok = (len(new_tenant_hist) == 0)
    turns_log.append({
        "turn": 8,
        "domain": "session_isolation",
        "query": "Read isolated session history",
        "answer": f"History len: {len(new_tenant_hist)}",
        "origin": "persistent_memory",
        "passed": t8_ok,
    })

    total_turns = len(turns_log)
    passed_turns = sum(1 for t in turns_log if t["passed"])
    all_passed = (passed_turns == total_turns)

    report = {
        "milestone": "Batch 207: End-to-End Interactive Runtime Benchmark",
        "total_turns": total_turns,
        "passed_turns": passed_turns,
        "all_passed": all_passed,
        "turns": turns_log,
    }

    out_file = REPORTS_DIR / "interactive_runtime_benchmark.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Interactive benchmark written to {out_file}")
    print(f"Passed {passed_turns}/{total_turns} turns. All passed: {all_passed}")


if __name__ == "__main__":
    run_interactive_runtime_benchmark()
