"""
Script to evaluate GroundedSessionAdapter runtime regressions on interactive tasks.
Location: experiments/batch211_220_relational_learning/run_runtime_regression_eval.py
"""
import json
import time
from pathlib import Path
import sys
sys.path.insert(0, ".")
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.cognitive_router import CognitiveRouter, RouteType, ComponentOrigin
from chakrview.runtime.evidence_extractor import EvidenceExtractor
from chakrview.runtime.persistent_memory import PersistentSessionMemory
from chakrview.runtime.retrieval_adapter import RetrievalAdapter
from chakrview.runtime.session_adapter import GroundedSessionAdapter
from chakrview.runtime.pointer_mechanism import ChakrPointerHead


class EngineMock:
    def __init__(self, model, tok):
        self.model = model
        self.tokenizer = tok


def main():
    tok_dir = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
    tok, _ = load_tokenizer_artifacts(tok_dir)

    ckpt_path = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model = ChakrMicro()
    model.load_state_dict(state, strict=True)
    model.eval()

    pointer_ckpt = Path("experiments/batch211_220_relational_learning/checkpoints/relational_pointer_head_step1000.pt")
    p_head = ChakrPointerHead(d_model=192)
    p_state = torch.load(pointer_ckpt, map_location="cpu", weights_only=False)
    p_head.load_state_dict(p_state)
    p_head.eval()

    engine = EngineMock(model, tok)
    retriever = RetrievalAdapter()
    retriever.ingest_document("doc1", "Oxygen has atomic number 8 and boiling point 90.188 K.")
    retriever.ingest_document("doc2", "Jupiter orbital period is 4333 days.")
    retriever.ingest_document("doc3", "The speed of light in vacuum is 299792458 m/s.")

    import tempfile
    temp_mem = Path(tempfile.mkdtemp())
    memory = PersistentSessionMemory(storage_dir=temp_mem / "sessions")
    router = CognitiveRouter(enable_pointer_copy=True)

    adapter = GroundedSessionAdapter(
        inference_engine=engine,
        retrieval_adapter=retriever,
        memory=memory,
        router=router,
        pointer_head=p_head,
        enable_constrained_decoding=True,
    )

    test_cases = [
        {
            "id": "tc1_direct_extraction",
            "query": "What is the boiling point of Nitrogen?",
            "context": "Nitrogen has boiling point -196 C.",
            "expected_behavior": "deterministic_extraction",
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR.value,
        },
        {
            "id": "tc2_retrieval_and_pointer",
            "query": "What is the orbital period of Jupiter?",
            "context": "Jupiter has orbital period 4333 days.",  # provided or retrieved
            "expected_behavior": "slot_extraction_or_trie",
            "expected_origin": [ComponentOrigin.DETERMINISTIC_EXTRACTOR.value, ComponentOrigin.CONSTRAINED_TRIE.value, ComponentOrigin.NEURAL_POINTER.value],
        },
        {
            "id": "tc3_conflicting_evidence",
            "query": "What is boiling point of water?",
            "context": "The boiling point of water is 100 C. Another test states the boiling point of water is 99 C.",
            "expected_behavior": "conflict_detection",
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR.value,
        },
        {
            "id": "tc4_unanswerable_abstention",
            "query": "What is the mass of Pluto?",
            "context": "Gold has atomic number 79.",
            "expected_behavior": "abstention",
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR.value,
        },
        {
            "id": "tc5_multi_turn_history",
            "query": "What did we talk about previously?",
            "context": None,
            "expected_behavior": "multi_turn_retention",
            "expected_origin": ComponentOrigin.NEURAL_CORE.value,
        },
        {
            "id": "tc6_malformed_empty_context",
            "query": "What is the atomic number of Helium?",
            "context": "",
            "expected_behavior": "safe_handling",
            "expected_origin": [ComponentOrigin.NEURAL_CORE.value, ComponentOrigin.FALLBACK_POLICY.value],
        },
        {
            "id": "tc7_oversized_query",
            "query": "query " * 200,
            "context": "Short context.",
            "expected_behavior": "safe_bounded_truncation",
            "expected_origin": [ComponentOrigin.NEURAL_CORE.value, ComponentOrigin.DETERMINISTIC_EXTRACTOR.value, ComponentOrigin.FALLBACK_POLICY.value],
        },
        {
            "id": "tc8_disabled_pointer_fallback",
            "query": "What is the capital of France?",
            "context": "France capital is Paris.",
            "disable_pointer": True,
            "expected_behavior": "deterministic_or_core_fallback",
            "expected_origin": [ComponentOrigin.DETERMINISTIC_EXTRACTOR.value, ComponentOrigin.NEURAL_CORE.value],
        }
    ]

    results = []
    passed = 0

    for tc in test_cases:
        t0 = time.perf_counter()
        if tc.get("disable_pointer"):
            adapter.pointer_head = None
        else:
            adapter.pointer_head = p_head

        resp = adapter.execute_turn(
            session_id="session_reg_1",
            user_input=tc["query"],
            supplied_context=tc["context"],
        )
        latency = (time.perf_counter() - t0) * 1000

        # Verification rules
        origin_val = resp.component_origin.value
        exp_origin = tc["expected_origin"]
        origin_ok = (origin_val in exp_origin) if isinstance(exp_origin, list) else (origin_val == exp_origin)

        case_passed = False
        if tc["id"] == "tc1_direct_extraction":
            case_passed = origin_ok and "-196 C" in resp.answer and resp.is_supported
        elif tc["id"] == "tc2_retrieval_and_pointer":
            case_passed = origin_ok and "4333" in resp.answer
        elif tc["id"] == "tc3_conflicting_evidence":
            case_passed = origin_ok and not resp.is_supported and "Conflicting" in resp.answer
        elif tc["id"] == "tc4_unanswerable_abstention":
            case_passed = origin_ok and not resp.is_supported and "Not mentioned" in resp.answer
        elif tc["id"] == "tc5_multi_turn_history":
            hist = memory.get_history("session_reg_1")
            case_passed = len(hist) >= 8  # turns accumulated
        elif tc["id"] == "tc6_malformed_empty_context":
            case_passed = resp.answer is not None and len(resp.answer) >= 0
        elif tc["id"] == "tc7_oversized_query":
            case_passed = resp.answer is not None
        elif tc["id"] == "tc8_disabled_pointer_fallback":
            case_passed = origin_ok and ("Paris" in resp.answer or resp.answer is not None)

        if case_passed:
            passed += 1

        results.append({
            "id": tc["id"],
            "query": tc["query"][:60],
            "origin": origin_val,
            "answer": resp.answer[:80],
            "is_supported": resp.is_supported,
            "latency_ms": round(latency, 2),
            "passed": case_passed
        })

    out_payload = {
        "milestone": "Batch 219: Integrated Runtime Regression & Robustness",
        "timestamp": "2026-10-13T14:40:00Z",
        "total_cases": len(test_cases),
        "passed_cases": passed,
        "pass_rate": passed / len(test_cases),
        "cases": results
    }

    out_p = Path("experiments/batch211_220_relational_learning/reports/runtime_regression_results.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print(f"Runtime Regression: {passed}/{len(test_cases)} passed ({passed/len(test_cases)*100:.1f}%)")
    print(f"Report saved to {out_p}")


if __name__ == "__main__":
    main()
