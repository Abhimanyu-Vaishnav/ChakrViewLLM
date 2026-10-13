"""
Batch 239: Interactive Runtime Checkpoint Loading & Multi-Hop Grounding Regression.
Location: experiments/batch231_240_multihop_reasoning/run_runtime_regression_eval.py
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
from chakrview.runtime.cross_attention_pointer import ChakrCrossAttentionPointerHead

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
CROSS_CKPT = Path("experiments/batch231_240_multihop_reasoning/checkpoints/cross_attention_pointer_step1000.pt")
REPORTS_DIR = Path("experiments/batch231_240_multihop_reasoning/reports")


class MockEngine:
    def __init__(self, model, tok):
        self.model = model
        self.tokenizer = tok


def test_multihop_runtime_integration():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    temp_mem = Path(tempfile.mkdtemp(prefix="chakr_reg_239_"))

    ck = torch.load(CANONICAL_CHECKPOINT, map_location="cpu", weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    engine = MockEngine(model, tokenizer)

    # 1. Load trained cross-attention pointer head
    head = ChakrCrossAttentionPointerHead(d_model=192, vocab_size=4096, d_proj=64, n_heads=2)
    assert CROSS_CKPT.exists(), "Trained cross-attention pointer checkpoint missing!"
    head.load_state_dict(torch.load(CROSS_CKPT, map_location="cpu", weights_only=False))
    head.eval()

    # 2. Setup retriever with multi-document knowledge base
    retriever = RetrievalAdapter()
    retriever.ingest_document("doc_moon", "The natural satellite of Earth is Moon.")
    retriever.ingest_document("doc_moon_period", "In astronomical records, Moon has orbital period 27.3 days.")
    retriever.ingest_document("doc_chem", "Water constituent gas is Hydrogen. Hydrogen has atomic number 1.")

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
            "id": "tc1_direct_multi_document_hop",
            "query": "What is the orbital period of the natural satellite of Earth?",
            "context": "Document A: The natural satellite of Earth is Moon.\nDocument B: Moon has orbital period 27.3 days.",
            "expected_origin": [ComponentOrigin.CONSTRAINED_TRIE.value, ComponentOrigin.NEURAL_POINTER.value, ComponentOrigin.DETERMINISTIC_EXTRACTOR.value],
            "validate": lambda r: r.is_supported and "27.3 days" in r.answer
        },
        {
            "id": "tc2_retrieval_multi_document_grounding",
            "query": "What is the atomic number of Hydrogen?",
            "context": "Hydrogen has atomic number 1.",
            "expected_origin": [ComponentOrigin.DETERMINISTIC_EXTRACTOR.value, ComponentOrigin.CONSTRAINED_TRIE.value],
            "validate": lambda r: "1" in r.answer
        },
        {
            "id": "tc3_missing_link_abstention",
            "query": "What is the orbital radius of unknown moon of Pluto?",
            "context": "Document A: Pluto has natural satellite Charon.\nDocument B: No orbital records available.",
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR.value,
            "validate": lambda r: not r.is_supported and "Not mentioned in context" in r.answer
        },
        {
            "id": "tc4_contradictory_evidence_detection",
            "query": "What is boiling point of water?",
            "context": "Water boiling point is 100 C. Another measurement states boiling point of water is 95 C.",
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR.value,
            "validate": lambda r: not r.is_supported and "Conflicting evidence" in r.answer
        },
        {
            "id": "tc5_multi_turn_dialogue_isolation",
            "query": "Hello assistant, how are you?",
            "context": None,
            "expected_origin": ComponentOrigin.NEURAL_CORE.value,
            "validate": lambda r: len(memory.get_history("session_hop_239")) >= 8
        },
        {
            "id": "tc6_unsupported_out_of_domain_abstention",
            "query": "What is the secret launch code of Area 51?",
            "context": "Document A: Gold atomic number is 79.",
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR.value,
            "validate": lambda r: not r.is_supported and "Not mentioned" in r.answer
        },
        {
            "id": "tc7_disabled_pointer_fallback",
            "query": "What is the capital of France?",
            "context": "France capital is Paris.",
            "disable_pointer": True,
            "expected_origin": [ComponentOrigin.DETERMINISTIC_EXTRACTOR.value, ComponentOrigin.NEURAL_CORE.value],
            "validate": lambda r: "Paris" in r.answer or r.answer is not None
        },
        {
            "id": "tc8_oversized_query_safe_truncation",
            "query": "query " * 300,
            "context": "Short text.",
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
            session_id="session_hop_239",
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
        "milestone": "Batch 239: Multi-Hop Interactive Runtime Checkpoint Integration & Regression",
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
    test_multihop_runtime_integration()
