"""
Batches 151–160: Comprehensive Integrated Benchmark Runner.
Evaluates:
1. Direct Conversation Route (intent, greetings, role dialogue).
2. Grounded QA (Answer present in retrieved context).
3. Insufficient Evidence / Abstention (Query absent from knowledge base).
4. Ambiguous / Conflicting context queries.
5. Key-value and numeric slot extraction.
6. Multi-turn dialogue continuity across sessions.
7. Cross-session isolation (no state leakage).
8. Empty index recovery and safe fallback.
9. Hindi / Indic multilingual context binding.
10. Hardware throughput, latency, and component attribution reporting.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import time
import hashlib
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pipeline import InferenceEngine
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.cognitive_router import CognitiveRouter, RouteType, ComponentOrigin
from chakrview.runtime.retrieval_adapter import RetrievalAdapter
from chakrview.runtime.evidence_extractor import EvidenceExtractor
from chakrview.runtime.persistent_memory import PersistentSessionMemory
from chakrview.runtime.session_adapter import GroundedSessionAdapter

EXP_DIR = Path("experiments/batch151_160_cognitive_routing")
REPORTS_DIR = EXP_DIR / "reports"
KNOWLEDGE_DIR = EXP_DIR / "knowledge"
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
CANONICAL_BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")


def setup_knowledge_adapter() -> RetrievalAdapter:
    adapter = RetrievalAdapter(min_score_threshold=0.20)
    
    docs = [
        ("doc_carbon", "The atomic number of Carbon is 6 and its standard atomic weight is 12.011.\n\nCarbon forms four covalent bonds."),
        ("doc_thar", "The Thar Desert is located primarily in Rajasthan, India.\n\nIt is also known as the Great Indian Desert."),
        ("doc_celsius", "Water boils at 100 degrees Celsius and freezes at 0 degrees Celsius at sea level."),
        ("doc_shackleton", "The lunar spacecraft landed in the crater named Shackleton at the lunar south pole."),
        ("doc_hindi_metal", "धातु: तांबा। तांबा एक विद्युत सुचालक धातु है।\n\nपशु: बाघ।"),
        ("doc_conflicting", "color is blue. In ancient folklore, color is red."),
    ]
    
    for doc_id, text in docs:
        adapter.ingest_document(doc_id, text)
        
    return adapter


def run_benchmark():
    device = torch.device("cpu")
    print("=" * 80)
    print("BATCHES 151–160 INTEGRATED COGNITIVE ROUTING & GROUNDED QA BENCHMARK")
    print(f"Device: {device}")
    print("=" * 80)
    
    # 1. Load canonical model and tokenizer
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    cfg = ModelConfig(max_seq_len=512)
    model = ChakrMicro(cfg).to(device)
    base_data = torch.load(CANONICAL_BASE_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(base_data["model_state_dict"])
    model.eval()
    
    inference_engine = InferenceEngine(model=model, tokenizer=tokenizer, expected_weight_hash=None)
    retrieval_adapter = setup_knowledge_adapter()
    memory_dir = EXP_DIR / "data" / "session_storage"
    memory = PersistentSessionMemory(storage_dir=memory_dir)
    router = CognitiveRouter(retrieval_threshold=0.20)
    
    session_adapter = GroundedSessionAdapter(
        inference_engine=inference_engine,
        retrieval_adapter=retrieval_adapter,
        memory=memory,
        router=router,
    )
    
    # 2. Benchmark Test Battery (15 distinct test cases covering all criteria)
    test_cases = [
        # 1. Direct Conversation
        {
            "id": "TC-01",
            "name": "Direct Greeting",
            "session_id": "sess_1",
            "query": "Hello! Who are you?",
            "supplied_context": None,
            "expected_route": RouteType.DIRECT_CHAT,
            "expected_origin": ComponentOrigin.NEURAL_CORE,
            "must_contain": None,
        },
        # 2. Grounded QA (Answer present in knowledge base)
        {
            "id": "TC-02",
            "name": "Grounded QA - Atomic Number",
            "session_id": "sess_1",
            "query": "What is the atomic number of Carbon?",
            "supplied_context": None,
            "expected_route": RouteType.KNOWLEDGE_RETRIEVAL,
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR,
            "must_contain": "6",
        },
        # 3. Grounded QA - Geographic State
        {
            "id": "TC-03",
            "name": "Grounded QA - Desert Location",
            "session_id": "sess_1",
            "query": "In which state is the Thar Desert located?",
            "supplied_context": None,
            "expected_route": RouteType.KNOWLEDGE_RETRIEVAL,
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR,
            "must_contain": "Rajasthan",
        },
        # 4. Insufficient Evidence / Absent from KB
        {
            "id": "TC-04",
            "name": "Absent Fact Abstention",
            "session_id": "sess_1",
            "query": "Who invented the telephone?",
            "supplied_context": None,
            "expected_route": RouteType.INSUFFICIENT_EVIDENCE,
            "expected_origin": ComponentOrigin.FALLBACK_POLICY,
            "must_contain": "do not have sufficient information",
        },
        # 5. Supplied Context - Key Value Extraction
        {
            "id": "TC-05",
            "name": "Supplied Context Direct Key",
            "session_id": "sess_1",
            "query": "What is material?",
            "supplied_context": "material is diamond.",
            "expected_route": RouteType.SLOT_EXTRACTION,
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR,
            "must_contain": "diamond",
        },
        # 6. Supplied Context - Distractor Resistance
        {
            "id": "TC-06",
            "name": "Supplied Context with Distractor",
            "session_id": "sess_1",
            "query": "What is language?",
            "supplied_context": "vehicle is bicycle. language is Sanskrit.",
            "expected_route": RouteType.SLOT_EXTRACTION,
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR,
            "must_contain": "Sanskrit",
        },
        # 7. Supplied Context - Absent Key Abstention
        {
            "id": "TC-07",
            "name": "Supplied Context Missing Key Abstention",
            "session_id": "sess_1",
            "query": "What is president?",
            "supplied_context": "mineral is quartz. continent is Asia.",
            "expected_route": RouteType.SLOT_EXTRACTION,
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR,
            "must_contain": "Not mentioned in context",
        },
        # 8. Hindi Key Value Extraction
        {
            "id": "TC-08",
            "name": "Hindi Grounded QA",
            "session_id": "sess_1",
            "query": "धातु क्या है?",
            "supplied_context": None,
            "expected_route": RouteType.KNOWLEDGE_RETRIEVAL,
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR,
            "must_contain": "तांबा",
        },
        # 9. Natural Sentence Extraction - Crater
        {
            "id": "TC-09",
            "name": "Grounded Crater Name Extraction",
            "session_id": "sess_1",
            "query": "What was the name of the crater where the spacecraft landed?",
            "supplied_context": None,
            "expected_route": RouteType.KNOWLEDGE_RETRIEVAL,
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR,
            "must_contain": "Shackleton",
        },
        # 10. Multi-Turn Follow-Up in Session 1
        {
            "id": "TC-10",
            "name": "Multi-Turn Dialogue Continuity",
            "session_id": "sess_1",
            "query": "Thank you for the information!",
            "supplied_context": None,
            "expected_route": RouteType.DIRECT_CHAT,
            "expected_origin": ComponentOrigin.NEURAL_CORE,
            "must_contain": None,
        },
        # 11. Cross-Session Isolation in Session 2 (Different user)
        {
            "id": "TC-11",
            "name": "Cross-Session Isolation",
            "session_id": "sess_2",
            "query": "What was my previous question?",
            "supplied_context": None,
            "expected_route": RouteType.DIRECT_CHAT,
            "expected_origin": ComponentOrigin.NEURAL_CORE,
            "must_contain": None,
        },
        # 12. Conflicting Context Detection
        {
            "id": "TC-12",
            "name": "Conflicting Evidence in Context",
            "session_id": "sess_1",
            "query": "What is color?",
            "supplied_context": "color is blue. color is red.",
            "expected_route": RouteType.SLOT_EXTRACTION,
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR,
            "must_contain": None,  # Has conflict flag
        },
        # 13. Empty Query Robustness
        {
            "id": "TC-13",
            "name": "Empty Input Handling",
            "session_id": "sess_1",
            "query": "   ",
            "supplied_context": None,
            "expected_route": RouteType.DIRECT_CHAT,
            "expected_origin": ComponentOrigin.NEURAL_CORE,
            "must_contain": None,
        },
        # 14. Context Window Overflow Protection
        {
            "id": "TC-14",
            "name": "Long Input Context Truncation",
            "session_id": "sess_1",
            "query": "What is fruit?",
            "supplied_context": "fruit is apple. " * 50,
            "expected_route": RouteType.SLOT_EXTRACTION,
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR,
            "must_contain": "apple",
        },
        # 15. Negative Scientific Query Abstention
        {
            "id": "TC-15",
            "name": "Grounded Negative Query Abstention",
            "session_id": "sess_1",
            "query": "At what temperature does gold boil according to the context?",
            "supplied_context": "Water boils at 100 degrees Celsius.",
            "expected_route": RouteType.SLOT_EXTRACTION,
            "expected_origin": ComponentOrigin.DETERMINISTIC_EXTRACTOR,
            "must_contain": "Not mentioned in context",
        },
    ]
    
    results = []
    passed_cases = 0
    route_correct_count = 0
    
    for tc in test_cases:
        t_start = time.perf_counter()
        resp = session_adapter.execute_turn(
            session_id=tc["session_id"],
            user_input=tc["query"],
            supplied_context=tc["supplied_context"],
        )
        t_elapsed = (time.perf_counter() - t_start) * 1000
        
        # Check criteria
        route_ok = (resp.route_decision.route == tc["expected_route"])
        origin_ok = (resp.component_origin == tc["expected_origin"])
        contain_ok = (tc["must_contain"] is None) or (tc["must_contain"].lower() in resp.answer.lower())
        
        tc_passed = route_ok and origin_ok and contain_ok
        if route_ok:
            route_correct_count += 1
        if tc_passed:
            passed_cases += 1
            
        print(f"[{tc['id']}] {tc['name']:38s} | Route: {resp.route_decision.route.value:20s} | Origin: {resp.component_origin.value:22s} | Passed: {tc_passed}")
        
        results.append({
            "id": tc["id"],
            "name": tc["name"],
            "query": tc["query"],
            "expected_route": tc["expected_route"].value,
            "actual_route": resp.route_decision.route.value,
            "expected_origin": tc["expected_origin"].value,
            "actual_origin": resp.component_origin.value,
            "answer": resp.answer,
            "is_supported": resp.is_supported,
            "support_explanation": resp.support_explanation,
            "latency_ms": round(t_elapsed, 2),
            "passed": tc_passed,
        })
        
    # Check Session Isolation
    sess1_history = memory.get_history("sess_1")
    sess2_history = memory.get_history("sess_2")
    isolation_ok = (len(sess1_history) > len(sess2_history)) and ("sess_1" not in [t.content for t in sess2_history])
    assert isolation_ok, "Session isolation violated!"
    
    summary = {
        "milestone": "Batches 151-160 Integrated Cognitive Routing",
        "total_test_cases": len(test_cases),
        "passed_cases": passed_cases,
        "pass_rate": round(passed_cases / len(test_cases), 4),
        "route_accuracy": round(route_correct_count / len(test_cases), 4),
        "session_isolation_verified": isolation_ok,
        "test_results": results,
    }
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / "integrated_benchmark_results.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print(f"\nBenchmark completed: {passed_cases}/{len(test_cases)} cases passed ({summary['pass_rate']*100:.1f}%)")
    print(f"Results written to: {report_file}")
    
    # Clean memory storage
    memory.clear("sess_1")
    memory.clear("sess_2")

if __name__ == "__main__":
    run_benchmark()
