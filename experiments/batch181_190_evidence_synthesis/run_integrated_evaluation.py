"""
Batch 189: Comprehensive Integrated Adversarial Evaluation & Component Ablation.
Evaluates 20 distinct regression and adversarial dimensions:
1. Single-Token Neural Pointer Copy
2. Multi-Token Pointer Beam Search (k=3)
3. Reordered Key-Value Records
4. Repeated Tokens in Context Span
5. Held-Out Unseen Entities
6. Unseen Syntactic Templates
7. Distractor-Heavy Documents
8. Lexical Hard Negatives
9. Multi-Document Consensus Synthesis
10. Partially Missing Evidence Abstention
11. Conflicting Multi-Document Evidence Rejection
12. Duplicate Source Deduplication
13. Unseen Query Phrasing
14. Hindi Language Multi-Byte Fact Extraction
15. Completely Absent Fact Abstention
16. Context Truncation within Max Seq Len (512 tokens)
17. Pointer-Disabled Fallback Route
18. Hybrid-Reranker-Disabled Lexical Fallback
19. Task Engine Recovery After Process Restart
20. Malformed Input & Resource Budget Compliance
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import time
import shutil
import tempfile
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pointer_mechanism import ChakrPointerHead, decode_pointer_beam
from chakrview.runtime.cognitive_router import CognitiveRouter, RouteType, ComponentOrigin
from chakrview.runtime.evidence_extractor import EvidenceExtractor
from chakrview.runtime.hybrid_reranker import HybridReranker
from chakrview.runtime.evidence_assembler import MultiDocumentEvidenceAssembler
from chakrview.runtime.persistent_memory import PersistentSessionMemory
from chakrview.runtime.task_engine import CognitiveTaskEngine, TaskStatus
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
ARM_C_CKPT_PATH = Path("experiments/batch171_180_pointer_learning/checkpoints/pointer_head_arm_c.pt")
REPORTS_DIR = Path("experiments/batch181_190_evidence_synthesis/reports")


def run_adversarial_battery():
    t_start = time.perf_counter()
    results = []
    device = torch.device("cpu")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # Load canonical base & pointer head
    base_model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    base_data = torch.load(CANONICAL_BASE_CKPT, map_location=device, weights_only=False)
    base_model.load_state_dict(base_data["model_state_dict"])
    base_model.eval()

    pointer_head = ChakrPointerHead(d_model=192, vocab_size=4096, d_proj=64).to(device)
    p_data = torch.load(ARM_C_CKPT_PATH, map_location=device, weights_only=False)
    pointer_head.load_state_dict(p_data["pointer_head_state_dict"])
    pointer_head.eval()

    # 1. Single-Token Neural Pointer Copy
    # Oxygen -> 8
    doc1 = "Oxygen is a chemical element with atomic number 8."
    q1 = "What is the atomic number of Oxygen?"
    prompt1 = f"<|im_start|>user\nContext: {doc1}\nQuestion: {q1}<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids1 = tokenizer.encode(prompt1, add_bos=True, add_eos=False)
    p_toks1 = tokenizer.encode("<|im_start|>user\nContext: ", add_bos=True, add_eos=False)
    d_toks1 = tokenizer.encode(doc1, add_bos=False, add_eos=False)
    gen_ids1, src1 = decode_pointer_beam(base_model, pointer_head, prompt_ids1, len(p_toks1), len(p_toks1) + len(d_toks1), beam_width=2)
    ans1 = tokenizer.decode(gen_ids1, skip_special_tokens=True, errors="replace").strip()
    r1_pass = ("8" in ans1)
    results.append({
        "case_id": 1,
        "name": "Single-Token Neural Pointer Copy",
        "expected": "8",
        "actual": ans1[:30],
        "component": "neural_pointer",
        "passed": r1_pass,
    })

    # 2. Multi-Token Pointer Beam Search
    doc2 = "Boron is a metalloid with atomic number 5."
    q2 = "What is the atomic number of Boron?"
    prompt2 = f"<|im_start|>user\nContext: {doc2}\nQuestion: {q2}<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids2 = tokenizer.encode(prompt2, add_bos=True, add_eos=False)
    p_toks2 = tokenizer.encode("<|im_start|>user\nContext: ", add_bos=True, add_eos=False)
    d_toks2 = tokenizer.encode(doc2, add_bos=False, add_eos=False)
    gen_ids2, _ = decode_pointer_beam(base_model, pointer_head, prompt_ids2, len(p_toks2), len(p_toks2) + len(d_toks2), beam_width=3)
    ans2 = tokenizer.decode(gen_ids2, skip_special_tokens=True, errors="replace").strip()
    r2_pass = ("5" in ans2)
    results.append({
        "case_id": 2,
        "name": "Multi-Token Pointer Beam Search (k=3)",
        "expected": "5",
        "actual": ans2[:30],
        "component": "neural_pointer",
        "passed": r2_pass,
    })

    # 3. Reordered Key-Value Records
    ext3 = EvidenceExtractor.extract_slot("Property: capital | Entity: Japan | Value: Tokyo", "What is the capital of Japan?")
    r3_pass = (ext3.success and "tokyo" in ext3.extracted_value.lower())
    results.append({
        "case_id": 3,
        "name": "Reordered Key-Value Record Extraction",
        "expected": "Tokyo",
        "actual": ext3.extracted_value,
        "component": "deterministic_extractor",
        "passed": r3_pass,
    })

    # 4. Repeated Tokens in Context Span
    doc4 = "Hydrogen has atomic number 1. In another table, Hydrogen has atomic number 1."
    ext4 = EvidenceExtractor.extract_slot(doc4, "What is the atomic number of Hydrogen?")
    r4_pass = (ext4.success and ext4.extracted_value == "1" and not ext4.has_conflicts)
    results.append({
        "case_id": 4,
        "name": "Repeated Tokens Consistency",
        "expected": "1",
        "actual": ext4.extracted_value,
        "component": "deterministic_extractor",
        "passed": r4_pass,
    })

    # 5. Held-Out Unseen Entities
    ext5 = EvidenceExtractor.extract_slot("Element Vibranium has atomic number 9999.", "What is the atomic number of Vibranium?")
    r5_pass = (ext5.success and ext5.extracted_value == "9999")
    results.append({
        "case_id": 5,
        "name": "Held-Out Unseen Entity Extraction",
        "expected": "9999",
        "actual": ext5.extracted_value,
        "component": "deterministic_extractor",
        "passed": r5_pass,
    })

    # 6. Unseen Syntactic Templates
    ext6 = EvidenceExtractor.extract_slot("The recorded melting point of Platinum is 1768 C.", "What is the melting point of Platinum?")
    r6_pass = (ext6.success and "1768" in ext6.extracted_value)
    results.append({
        "case_id": 6,
        "name": "Unseen Syntactic Template Extraction",
        "expected": "1768 C",
        "actual": ext6.extracted_value,
        "component": "deterministic_extractor",
        "passed": r6_pass,
    })

    # 7. Distractor-Heavy Documents
    distractor = "Iron atomic number is 26. Gold melting point is 1064 C. Copper atomic number is 29."
    ext7 = EvidenceExtractor.extract_slot(distractor, "What is the melting point of Gold?")
    r7_pass = (ext7.success and "1064" in ext7.extracted_value)
    results.append({
        "case_id": 7,
        "name": "Distractor-Heavy Document Invariance",
        "expected": "1064 C",
        "actual": ext7.extracted_value,
        "component": "deterministic_extractor",
        "passed": r7_pass,
    })

    # 8. Lexical Hard Negatives
    reranker = HybridReranker(enable_reranking=True)
    reranker.ingest_document("pos", "Helium is an inert gas with atomic number 2.")
    reranker.ingest_document("neg", "Hydrogen has atomic number 1 and is the lightest gas.")
    hits8 = reranker.retrieve_and_rerank("What is the atomic number of Helium?", top_k=2)
    r8_pass = (len(hits8) > 0 and hits8[0].doc_id == "pos")
    results.append({
        "case_id": 8,
        "name": "Lexical Hard Negative Discrimination",
        "expected": "pos",
        "actual": hits8[0].doc_id if hits8 else "none",
        "component": "hybrid_reranker",
        "passed": r8_pass,
    })

    # 9. Multi-Document Consensus Synthesis
    assembler = MultiDocumentEvidenceAssembler()
    bundle9 = assembler.assemble(
        "What is the atomic number of Helium?",
        hits8
    )
    r9_pass = (bundle9.is_sufficient and bundle9.consensus_fact == "2")
    results.append({
        "case_id": 9,
        "name": "Multi-Document Consensus Synthesis",
        "expected": "2",
        "actual": bundle9.consensus_fact,
        "component": "evidence_assembler",
        "passed": r9_pass,
    })

    # 10. Partially Missing Evidence Abstention
    bundle10 = assembler.assemble(
        "What is the capital of Mars?",
        hits8
    )
    r10_pass = (not bundle10.is_sufficient and bundle10.abstention_reason is not None)
    results.append({
        "case_id": 10,
        "name": "Partially Missing Evidence Abstention",
        "expected": "abstain",
        "actual": "abstained" if not bundle10.is_sufficient else "answered",
        "component": "evidence_assembler",
        "passed": r10_pass,
    })

    # 11. Conflicting Multi-Document Evidence Rejection
    from chakrview.runtime.hybrid_reranker import RerankedEvidence
    conflict_hits = [
        RerankedEvidence(chunk_id="c1", doc_id="d1", text="The boiling point of the solution is 105 C.", lexical_score=1.0, neural_score=1.0, final_score=1.0, retrieval_rank=0, rerank_delta=0),
        RerankedEvidence(chunk_id="c2", doc_id="d2", text="The boiling point of the solution is 115 C.", lexical_score=1.0, neural_score=0.9, final_score=0.95, retrieval_rank=1, rerank_delta=0),
    ]
    bundle11 = assembler.assemble("What is the boiling point of the solution?", conflict_hits)
    r11_pass = (bundle11.has_conflicts and not bundle11.is_sufficient)
    results.append({
        "case_id": 11,
        "name": "Conflicting Multi-Document Evidence Rejection",
        "expected": True,
        "actual": bundle11.has_conflicts,
        "component": "evidence_assembler",
        "passed": r11_pass,
    })

    # 12. Duplicate Source Deduplication
    dup_hits = [
        RerankedEvidence(chunk_id="c1", doc_id="d1", text="Hydrogen atomic number is 1.", lexical_score=1.0, neural_score=1.0, final_score=1.0, retrieval_rank=0, rerank_delta=0),
        RerankedEvidence(chunk_id="c1_dup", doc_id="d1", text="Hydrogen atomic number is 1.", lexical_score=1.0, neural_score=1.0, final_score=1.0, retrieval_rank=1, rerank_delta=0),
    ]
    bundle12 = assembler.assemble("What is the atomic number of Hydrogen?", dup_hits)
    r12_pass = (len(bundle12.items) == 1)
    results.append({
        "case_id": 12,
        "name": "Duplicate Source Deduplication",
        "expected": 1,
        "actual": len(bundle12.items),
        "component": "evidence_assembler",
        "passed": r12_pass,
    })

    # 13. Unseen Query Phrasing
    ext13 = EvidenceExtractor.extract_slot("The boiling point of Liquid Nitrogen is -196 C.", "What is the boiling point of Liquid Nitrogen?")
    r13_pass = (ext13.success and "-196" in ext13.extracted_value)
    results.append({
        "case_id": 13,
        "name": "Unseen Query Phrasing Extraction",
        "expected": "-196 C",
        "actual": ext13.extracted_value,
        "component": "deterministic_extractor",
        "passed": r13_pass,
    })

    # 14. Hindi Multi-Byte Fact Extraction
    ext14 = EvidenceExtractor.extract_slot("चांदी सबसे अच्छा विद्युत सुचालक धातु है।", "चांदी का प्रमुख गुण क्या है?")
    r14_pass = (ext14.success and bool(ext14.extracted_value))
    results.append({
        "case_id": 14,
        "name": "Hindi Multi-Byte Fact Extraction",
        "expected": True,
        "actual": bool(ext14.extracted_value),
        "component": "deterministic_extractor",
        "passed": r14_pass,
    })

    # 15. Completely Absent Fact Abstention
    ext15 = EvidenceExtractor.extract_slot("The solar system contains eight planets.", "What is the capital of Australia?")
    r15_pass = (not ext15.success and ext15.is_abstention)
    results.append({
        "case_id": 15,
        "name": "Completely Absent Fact Abstention",
        "expected": "abstain",
        "actual": "abstained" if ext15.is_abstention else "answered",
        "component": "fallback_policy",
        "passed": r15_pass,
    })

    # 16. Context Truncation Compliance (max_tokens <= 380)
    prompt_str16 = bundle9.format_context_prompt(max_tokens=380)
    tok_count16 = len(tokenizer.encode(prompt_str16))
    r16_pass = (tok_count16 <= 380)
    results.append({
        "case_id": 16,
        "name": "Context Truncation Compliance",
        "expected": "<=380",
        "actual": tok_count16,
        "component": "evidence_assembler",
        "passed": r16_pass,
    })

    # 17. Pointer-Disabled Fallback Route
    router17 = CognitiveRouter(enable_pointer_copy=False)
    dec17 = router17.classify_route("What is the atomic number of Carbon?", supplied_context="Carbon has atomic number 6.")
    r17_pass = (dec17.route == RouteType.SLOT_EXTRACTION)
    results.append({
        "case_id": 17,
        "name": "Pointer-Disabled Fallback Route",
        "expected": "slot_extraction",
        "actual": dec17.route.value,
        "component": "cognitive_router",
        "passed": r17_pass,
    })

    # 18. Hybrid-Reranker-Disabled Lexical Fallback
    reranker18 = HybridReranker(enable_reranking=False)
    reranker18.ingest_document("d1", "Titan is the largest moon of Saturn.")
    hits18 = reranker18.retrieve_and_rerank("What is the largest moon of Saturn?", top_k=1)
    r18_pass = (len(hits18) > 0 and hits18[0].doc_id == "d1" and hits18[0].neural_score == 0.0)
    results.append({
        "case_id": 18,
        "name": "Hybrid-Reranker-Disabled Lexical Fallback",
        "expected": True,
        "actual": r18_pass,
        "component": "hybrid_reranker",
        "passed": r18_pass,
    })

    # 19. Task Engine Recovery After Process Restart
    temp_dir = Path(tempfile.mkdtemp(prefix="chakr_eval_task19_"))
    try:
        mem19 = PersistentSessionMemory(temp_dir / "sessions")
        eng_a = CognitiveTaskEngine(storage_dir=temp_dir / "tasks", memory=mem19)
        t_state = eng_a.execute_synthesis_workflow(
            task_id="recovery_case_19",
            session_id="sess_19",
            query="What is the atomic number of Hydrogen?",
            document_texts=[("d_h1", "Hydrogen has atomic number 1.")],
        )
        # Restart simulation
        eng_b = CognitiveTaskEngine(storage_dir=temp_dir / "tasks", memory=mem19)
        loaded_t = eng_b.load_task("recovery_case_19")
        r19_pass = (loaded_t is not None and loaded_t.status == TaskStatus.COMPLETED and loaded_t.final_result == "1")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    results.append({
        "case_id": 19,
        "name": "Task Engine Recovery After Process Restart",
        "expected": "1",
        "actual": loaded_t.final_result if loaded_t else "none",
        "component": "task_engine",
        "passed": r19_pass,
    })

    # 20. Malformed Input & Resource Budget Compliance
    ext20 = EvidenceExtractor.extract_slot("", "")
    cfg20 = ModelConfig()
    r20_pass = (not ext20.success and cfg20.max_seq_len == 512 and cfg20.vocab_size == 4096)
    results.append({
        "case_id": 20,
        "name": "Malformed Input & Resource Budget Compliance",
        "expected": True,
        "actual": r20_pass,
        "component": "invariants_and_safety",
        "passed": r20_pass,
    })

    total_passed = sum(1 for r in results if r["passed"])
    total_cases = len(results)
    elapsed = time.perf_counter() - t_start

    summary = {
        "milestone": "Batch 189: Comprehensive Integrated Adversarial Evaluation & Component Ablation",
        "total_cases": total_cases,
        "passed_cases": total_passed,
        "pass_rate": round(total_passed / total_cases, 4),
        "execution_time_sec": round(elapsed, 2),
        "cases": results,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORTS_DIR / "integrated_evaluation_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Integrated evaluation written to {out_file}")
    print(f"Total Passed: {total_passed}/{total_cases} ({summary['pass_rate']*100:.1f}%) in {elapsed:.2f}s")


if __name__ == "__main__":
    run_adversarial_battery()
