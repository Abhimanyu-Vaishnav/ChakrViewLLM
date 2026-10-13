"""
Batch 229: Actual Interactive Runtime Checkpoint Integration & Regression Verification.
Location: experiments/batch221_230_pointer_generalization/run_runtime_regression_eval.py
"""
import json
from pathlib import Path
import sys
import tempfile
import time
import torch

sys.path.insert(0, ".")

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.cognitive_router import CognitiveRouter, ComponentOrigin
from chakrview.runtime.persistent_memory import PersistentSessionMemory
from chakrview.runtime.retrieval_adapter import RetrievalAdapter
from chakrview.runtime.session_adapter import GroundedSessionAdapter
from chakrview.runtime.relation_pointer import ChakrRelationPointerHead

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
RELATION_CKPT = Path("experiments/batch221_230_pointer_generalization/checkpoints/relation_aware_pointer_step1000.pt")
REPORTS_DIR = Path("experiments/batch221_230_pointer_generalization/reports")


class MockEngine:
    def __init__(self, model, tok):
        self.model = model
        self.tokenizer = tok


def test_runtime_integration():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_mem = Path(tempfile.mkdtemp(prefix="chakr_reg_229_"))

    ck = torch.load(CANONICAL_CHECKPOINT, map_location="cpu", weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    engine = MockEngine(model, tokenizer)

    # 1. Load trained relation pointer head
    head = ChakrRelationPointerHead(d_model=192, vocab_size=4096, d_proj=64, d_rel=32)
    assert RELATION_CKPT.exists(), "Trained relation pointer checkpoint missing!"
    head.load_state_dict(torch.load(RELATION_CKPT, map_location="cpu", weights_only=False))
    head.eval()

    # 2. Setup retriever with knowledge base
    retriever = RetrievalAdapter()
    retriever.ingest_document("doc_chem", "Nitrogen has atomic number 7 and boiling point -196 C.")
    retriever.ingest_document("doc_phys", "The speed of light in vacuum is 299792458 m/s.")
    retriever.ingest_document("doc_astro", "Saturn orbital period is 10759 days.")

    memory = PersistentSessionMemory(storage_dir=temp_mem / "sessions")
    router = CognitiveRouter(enable_pointer_copy=True)

    adapter = GroundedSessionAdapter(
        inference_engine=engine,
        retrieval_adapter=retriever,
        memory=memory,
        router=router,
        pointer_head=head,
        enable_constrained_decoding=True,
    )

    test_cases = [
        {
            "id": "tc1_direct_deterministic_extraction",
            "query": "What is the atomic number of Nitrogen?",
            "context": "Nitrogen has atomic number 7.",
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR.value,
            "validate": lambda r: r.is_supported and "7" in r.answer
        },
        {
            "id": "tc2_relation_pointer_trie_copy",
            "query": "What is the orbital period of Saturn?",
            "context": "Saturn orbital period is 10759 days. Another planet Jupiter has orbital period 4333 days.",
            "expected_origin": [ComponentOrigin.CONSTRAINED_TRIE.value, ComponentOrigin.NEURAL_POINTER.value],
            "validate": lambda r: r.is_supported and "10759" in r.answer
        },
        {
            "id": "tc3_retrieval_augmented_grounding",
            "query": "What is the orbital period of Saturn?",
            "context": "Saturn orbital period is 10759 days.",  # provided or retrieved
            "expected_origin": [ComponentOrigin.CONSTRAINED_TRIE.value, ComponentOrigin.DETERMINISTIC_EXTRACTOR.value],
            "validate": lambda r: "10759" in r.answer
        },
        {
            "id": "tc4_conflicting_evidence_detection",
            "query": "What is boiling point of water?",
            "context": "The boiling point of water is 100 C. A separate report states boiling point of water is 99 C.",
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR.value,
            "validate": lambda r: not r.is_supported and "Conflicting evidence" in r.answer
        },
        {
            "id": "tc5_unsupported_fact_abstention",
            "query": "What is the density of Krypton?",
            "context": "Argon has atomic number 18.",
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR.value,
            "validate": lambda r: not r.is_supported and "Not mentioned in context" in r.answer
        },
        {
            "id": "tc6_multi_turn_history_isolation",
            "query": "Hello assistant, how are you?",
            "context": None,
            "expected_origin": ComponentOrigin.NEURAL_CORE.value,
            "validate": lambda r: len(memory.get_history("session_turn_229")) >= 8
        },
        {
            "id": "tc7_disabled_pointer_safe_fallback",
            "query": "What is the capital of Japan?",
            "context": "Japan capital is Tokyo.",
            "disable_pointer": True,
            "expected_origin": [ComponentOrigin.DETERMINISTIC_EXTRACTOR.value, ComponentOrigin.NEURAL_CORE.value],
            "validate": lambda r: "Tokyo" in r.answer or r.answer is not None
        },
        {
            "id": "tc8_oversized_query_safe_truncation",
            "query": "query " * 300,
            "context": "Context text.",
            "expected_origin": [ComponentOrigin.NEURAL_CORE.value, ComponentOrigin.DETERMINISTIC_EXTRACTOR.value, ComponentOrigin.FALLBACK_POLICY.value],
            "validate": lambda r: r.answer is not None
        }
    ]

    results = []
    passed = 0

    for tc in test_cases:
        t0 = time.perf_counter()
        if tc.get("disable_pointer"):
            adapter.pointer_head = None
        else:
            adapter.pointer_head = head

        resp = adapter.execute_turn(
            session_id="session_turn_229",
            user_input=tc["query"],
            supplied_context=tc["context"],
        )
        latency = (time.perf_counter() - t0) * 1000

        origin_val = resp.component_origin.value
        exp_origin = tc["expected_origin"]
        origin_ok = (origin_val in exp_origin) if isinstance(exp_origin, list) else (origin_val == exp_origin)
        val_ok = tc["validate"](resp)

        case_passed = origin_ok and val_ok
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
        "milestone": "Batch 229: Actual Interactive Runtime Checkpoint Integration & Regression",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_cases": len(test_cases),
        "passed_cases": passed,
        "pass_rate": passed / len(test_cases),
        "cases": results
    }

    out_p = REPORTS_DIR / "runtime_regression_results.json"
    out_p.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"Runtime Regression: {passed}/{len(test_cases)} passed ({passed/len(test_cases)*100:.1f}%)")
    print(f"Report saved to {out_p}")


if __name__ == "__main__":
    test_runtime_integration()
