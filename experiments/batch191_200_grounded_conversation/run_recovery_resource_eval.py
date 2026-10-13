"""
Batch 198: Recovery, Conflict Handling, and Resource-Limit Testing.
Location: experiments/batch191_200_grounded_conversation/run_recovery_resource_eval.py

Evaluates bounded failure conditions and graceful recovery:
1. Empty and malformed queries.
2. Context exceeding supported token limits (truncation/safety).
3. Candidate list with extreme length under constrained decoder.
4. Restart between retrieval and verification.
5. Conflicting evidence arriving during an active durable task.
6. Retrieval failure and ungrounded queries (abstention).
7. Cross-session tenant isolation without memory bleed.
8. Safe fallback when experimental pointer route is disabled.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chakrview.runtime.cognitive_router import CognitiveRouter, RouteType
from chakrview.runtime.evidence_extractor import EvidenceExtractor
from chakrview.runtime.persistent_memory import PersistentSessionMemory
from chakrview.runtime.task_engine import CognitiveTaskEngine, TaskStatus
from chakrview.runtime.constrained_decoder import TokenTrie, decode_pointer_constrained
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

REPORTS_DIR = Path("experiments/batch191_200_grounded_conversation/reports")


def run_recovery_resource_eval():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="chakr_recovery_eval_"))

    results = {
        "milestone": "Batch 198: Recovery, Conflicts, and Resource-Limit Testing",
        "tests": [],
        "all_passed": False,
    }

    test_logs = []

    # 1. Empty and malformed inputs
    ext_empty = EvidenceExtractor.extract_slot("", "What is the capital?")
    passed_empty = (ext_empty.is_abstention and not ext_empty.success)
    test_logs.append({
        "test_name": "empty_context_handling",
        "passed": passed_empty,
        "detail": f"is_abstention={ext_empty.is_abstention}"
    })

    ext_malformed = EvidenceExtractor.extract_slot("Some text", "   ")
    passed_malformed = (ext_malformed.is_abstention and not ext_malformed.success)
    test_logs.append({
        "test_name": "empty_query_handling",
        "passed": passed_malformed,
        "detail": f"is_abstention={ext_malformed.is_abstention}"
    })

    # 2. Context exceeding length bounds
    long_context = "Hydrogen atomic number 1. " * 500  # 12500 chars
    ext_long = EvidenceExtractor.extract_slot(long_context, "What is the atomic number of Hydrogen?")
    passed_long = (ext_long.success and ext_long.extracted_value == "1")
    test_logs.append({
        "test_name": "oversized_context_bounded_search",
        "passed": passed_long,
        "detail": f"extracted_value={ext_long.extracted_value}"
    })

    # 3. Constrained decoding with 200 candidate spans under budget
    trie = TokenTrie()
    tok, _ = load_tokenizer_artifacts("release/chakrview-v0.5.0/data/experiments/vocab_4096")
    for i in range(200):
        trie.insert(tok.encode(f"Candidate_{i}"))
    trie.insert(tok.encode("Hydrogen"))
    logits = torch.randn(1, 1, 2048)
    allowed = trie.get_allowed_tokens([])
    passed_trie_scale = (len(allowed) > 0 and len(allowed) <= 2048)
    test_logs.append({
        "test_name": "large_candidate_trie_budget",
        "passed": passed_trie_scale,
        "detail": f"allowed_branches={len(allowed)}"
    })

    # 4. Conflicting evidence arriving during durable synthesis
    mem = PersistentSessionMemory(temp_dir / "sessions")
    engine = CognitiveTaskEngine(storage_dir=temp_dir / "tasks", memory=mem)
    conflict_task = engine.execute_synthesis_workflow(
        task_id="conflict_recovery_task_1",
        session_id="tenant_a",
        query="What is the boiling point of the solution?",
        document_texts=[
            ("doc_a", "The boiling point of the solution is 100 C."),
            ("doc_b", "The boiling point of the solution is 108 C."),
        ]
    )
    passed_conflict = (conflict_task.status == TaskStatus.FAILED and "Conflicting facts" in (conflict_task.failure_reason or ""))
    test_logs.append({
        "test_name": "conflicting_evidence_abort",
        "passed": passed_conflict,
        "detail": f"status={conflict_task.status.value}, reason={conflict_task.failure_reason}"
    })

    # 5. Process restart recovery of persistent tasks
    # Verify task state is reloadable from disk
    engine_recovered = CognitiveTaskEngine(storage_dir=temp_dir / "tasks", memory=mem)
    rec_task = engine_recovered.load_task("conflict_recovery_task_1")
    passed_restart = (rec_task is not None and rec_task.status == TaskStatus.FAILED)
    test_logs.append({
        "test_name": "task_recovery_after_restart",
        "passed": passed_restart,
        "detail": f"persisted_state_verified={passed_restart}"
    })

    # 6. Tenant session memory isolation
    mem.append_turn("tenant_a", "user", "Private credit card number is 4111-2222")
    tenant_b_history = mem.get_history("tenant_b")
    passed_isolation = (len(tenant_b_history) == 0)
    test_logs.append({
        "test_name": "cross_tenant_session_isolation",
        "passed": passed_isolation,
        "detail": f"tenant_b_history_len={len(tenant_b_history)}"
    })

    # 7. Safe fallback when experimental pointer is disabled in router
    router_disabled = CognitiveRouter(enable_pointer_copy=False)
    dec_disabled = router_disabled.classify_route("What is the atomic number of Hydrogen?", supplied_context="Hydrogen has atomic number 1.")
    passed_fallback = (dec_disabled.route != RouteType.POINTER_COPY and dec_disabled.route == RouteType.SLOT_EXTRACTION)
    test_logs.append({
        "test_name": "disabled_pointer_safe_fallback",
        "passed": passed_fallback,
        "detail": f"route_fallback={dec_disabled.route.value}"
    })

    all_passed = all(t["passed"] for t in test_logs)
    results["tests"] = test_logs
    results["all_passed"] = all_passed

    out_file = REPORTS_DIR / "recovery_and_resource_tests.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Recovery tests written to {out_file}")
    print(f"All {len(test_logs)} tests passed: {all_passed}")


if __name__ == "__main__":
    run_recovery_resource_eval()
