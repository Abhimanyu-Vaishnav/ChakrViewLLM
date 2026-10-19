"""
Unit and Integration Tests for Dedicated Neural Runtime Adapter and Opt-In Neural Generation.
Location: tests/test_neural_runtime_adapter.py

Validates:
1. Canonical checkpoint hash verification before loading.
2. Corrupted / mismatched checkpoint hash fails closed.
3. Tokenizer file hash verification before loading.
4. Unsupported checkpoint path is rejected via allowlist.
5. Incompatible model configurations are rejected.
6. Adapter initialization is cached and avoids redundant loads.
7. CPU inference feasibility and Delta-W = 0 weight immutability.
8. Generation metadata truthfulness and lack of fabricated citations.
9. Deterministic tools (arithmetic, memory, document QA) retain absolute priority over neural generation.
10. Neural generation is strictly opt-in and disabled by default.
11. Neural output is classified as GENERATED_NEURAL and visibly labelled as unverified.
12. Memory isolation: generated neural text is never converted into a verified session fact.
"""

from __future__ import annotations

from pathlib import Path
import pytest
import torch

from chakrview.runtime.neural_runtime_adapter import (
    NeuralRuntimeAdapter,
    NeuralArtifactIntegrityError,
    CANONICAL_CHECKPOINTS_ALLOWLIST,
    DEFAULT_CHECKPOINT_KEY,
)
from experiments.batch2221_2260.integrated_assistant_adapter import IntegratedAssistantAdapter

REPO_ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Artifact Integrity and Loading Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_01_canonical_checkpoint_hash_verified():
    """Requirement 1: Canonical checkpoint hash is verified before loading."""
    adapter = NeuralRuntimeAdapter(checkpoint_path="step_10000", max_new_tokens=8)
    assert not adapter.is_loaded
    adapter.initialize()
    assert adapter.is_loaded
    assert adapter._checkpoint_label == "Canonical Pretrained (Step 10,000)"
    assert adapter._checkpoint_sha256 == "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"


def test_02_corrupted_checkpoint_fails_closed(tmp_path):
    """Requirement 2: A checkpoint with corrupted content/hash fails closed."""
    fake_ckpt = tmp_path / "training_checkpoint_step_10000.pt"
    fake_ckpt.write_bytes(b"corrupted binary data")

    adapter = NeuralRuntimeAdapter(checkpoint_path=fake_ckpt)
    with pytest.raises(NeuralArtifactIntegrityError) as exc_info:
        adapter.initialize()
    assert "mismatch" in str(exc_info.value).lower() or "unsupported" in str(exc_info.value).lower()


def test_03_tokenizer_file_hash_mismatch_fails_closed(tmp_path):
    """Requirement 3: Tokenizer file hash mismatch fails closed."""
    tok_dir = tmp_path / "corrupt_tok"
    tok_dir.mkdir()
    (tok_dir / "config.json").write_text("{}", encoding="utf-8")
    (tok_dir / "vocab.json").write_text("{}", encoding="utf-8")
    (tok_dir / "merges.json").write_text("{}", encoding="utf-8")

    adapter = NeuralRuntimeAdapter(tokenizer_dir=tok_dir)
    with pytest.raises(NeuralArtifactIntegrityError) as exc_info:
        adapter.initialize()
    assert "tokenizer file hash mismatch" in str(exc_info.value).lower()


def test_04_unsupported_checkpoint_path_rejected():
    """Requirement 4: Unsupported checkpoint path is rejected via allowlist."""
    adapter = NeuralRuntimeAdapter(checkpoint_path="unauthorized/arbitrary/model.pt")
    with pytest.raises(NeuralArtifactIntegrityError) as exc_info:
        adapter.initialize()
    assert "unsupported or unauthorized checkpoint" in str(exc_info.value).lower()


def test_05_adapter_initialization_is_cached():
    """Requirement 6: Adapter caches the loaded contract and does not reload every turn."""
    adapter = NeuralRuntimeAdapter(max_new_tokens=4)
    adapter.initialize()
    contract_id1 = id(adapter._contract)

    # Subsequent generation call reuses contract
    res = adapter.generate("Hello")
    contract_id2 = id(adapter._contract)
    assert contract_id1 == contract_id2
    assert res["weight_hash_verified"] is True


def test_06_cpu_generation_and_weight_immutability():
    """Requirement 7, 17: Inference is CPU-capable and preserves Delta-W = 0."""
    adapter = NeuralRuntimeAdapter(max_new_tokens=6)
    res = adapter.generate("Test query")
    assert res["device"] == "cpu"
    assert res["weight_hash_verified"] is True
    assert len(res["generated_tokens"]) == 6
    assert isinstance(res["latency_ms"], float)
    assert res["latency_ms"] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Routing, Priority and Fallback Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_07_neural_generation_disabled_by_default():
    """Requirement 18: Neural generation is disabled by default."""
    assistant = IntegratedAssistantAdapter(enable_neural_generation=False)
    resp = assistant.ask("What is the secret launch date of Project Apollo 99?")
    assert resp.decision == "ABSTAIN"
    assert "could not find sufficient verified evidence" in resp.answer
    assert resp.execution_breakdown["subsystem"] == "local_knowledge_assistant"
    assert resp.execution_breakdown["neural_inference"] == "not invoked"


def test_08_arithmetic_takes_priority_over_neural_generation():
    """Requirement 19: Safe arithmetic retains priority even when neural generation is enabled."""
    assistant = IntegratedAssistantAdapter(enable_neural_generation=True)
    resp = assistant.ask("what is 9 multiple by 99 and add 100")
    assert resp.decision == "CALCULATED"
    assert "991" in resp.answer
    assert resp.execution_breakdown["subsystem"] == "safe_calculator"
    assert resp.execution_breakdown["neural_inference"] == "not invoked"


def test_09_incomplete_arithmetic_clarifies_not_neural():
    """Requirement 20: Incomplete arithmetic expression clarifies/abstains, never invokes neural fallback."""
    assistant = IntegratedAssistantAdapter(enable_neural_generation=True)
    resp = assistant.ask("compute 99 * and then")
    assert resp.decision == "ABSTAIN"
    assert "Incomplete or ambiguous" in resp.answer
    assert resp.execution_breakdown["subsystem"] == "safe_calculator"


def test_10_working_memory_recall_takes_priority_over_neural():
    """Requirement 21: Working memory fact recall returns the remembered fact without neural generation."""
    assistant = IntegratedAssistantAdapter(enable_neural_generation=True)
    assistant.ask("The project codename is Blue Mango 47.")
    
    recall = assistant.ask("What was the project codename I told you earlier?")
    assert recall.decision == "WORKING_MEMORY"
    assert "Blue Mango 47" in recall.answer
    assert recall.execution_breakdown["subsystem"] == "working_memory"
    assert recall.execution_breakdown["neural_inference"] == "not invoked"


def test_11_grounded_document_qa_takes_priority_over_neural():
    """Requirement 22: Grounded document QA returns original answer and citations without neural generation."""
    assistant = IntegratedAssistantAdapter(enable_neural_generation=True)
    doc_dir = REPO_ROOT / "experiments" / "batch1021_1060" / "data" / "sample_corpus"
    assistant.ingest_directory(doc_dir)

    resp = assistant.ask("What is the availability target in billing terms SLA?")
    assert resp.decision == "ANSWER"
    assert len(resp.citations) > 0
    assert "99.95%" in resp.answer
    assert resp.execution_breakdown["subsystem"] == "local_knowledge_assistant"
    assert resp.execution_breakdown["neural_inference"] == "not invoked"


def test_12_opt_in_neural_generation_fallback_when_qa_abstains():
    """Requirement 24, 25: Unsupported document query invokes neural generation when opt-in is active."""
    assistant = IntegratedAssistantAdapter(enable_neural_generation=True)
    resp = assistant.ask("What is the secret launch date of Project Apollo 99?")
    assert resp.decision == "GENERATED_NEURAL"
    assert "[EXPERIMENTAL CHAKRVIEW GENERATION - NOT VERIFIED]" in resp.answer
    assert resp.execution_breakdown["subsystem"] == "neural_model"
    assert resp.execution_breakdown["neural_inference"] == "invoked"
    assert resp.execution_breakdown["weight_hash_verified"] == "True"
    assert len(resp.citations) == 0  # No fabricated citations


def test_13_neural_output_not_converted_into_working_memory_fact():
    """Requirement 26: Neural generation output is never converted into a working memory fact."""
    assistant = IntegratedAssistantAdapter(enable_neural_generation=True)
    assistant.ask("Tell me an ungrounded story about a galaxy")
    
    # Verify working memory facts remained empty
    assert len(assistant.working_memory.facts) == 0


def test_14_alternate_cpt_checkpoint_selection():
    """Requirement 10: Explicit selection of Seed-42 CPT checkpoint through adapter."""
    assistant = IntegratedAssistantAdapter(
        enable_neural_generation=True,
        neural_checkpoint="cpt_best",
        neural_max_new_tokens=10
    )
    resp = assistant.ask("Explain connection handshake")
    assert resp.decision == "GENERATED_NEURAL"
    assert resp.execution_breakdown["checkpoint_label"] == "Seed-42 Domain CPT Candidate"
    assert resp.execution_breakdown["neural_inference"] == "invoked"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Prompt Conditioning and Bounded Decoding Tests (Batch 2781–2820)
# ─────────────────────────────────────────────────────────────────────────────

def test_15_prompt_conditioning_modes_and_formatting():
    """Batch 2781-2820: Test prompt conditioning layer with raw_completion and plain_instruction."""
    from chakrview.runtime.prompt_conditioning import format_conditioned_prompt

    # raw_completion preserves exact prompt
    raw_prompt = format_conditioned_prompt("What is 2+2?", prompt_mode="raw_completion")
    assert raw_prompt == "What is 2+2?"

    # plain_instruction prepends minimal instruction header
    inst_prompt = format_conditioned_prompt("What is 2+2?", prompt_mode="plain_instruction")
    assert "Instruction: What is 2+2?\nResponse:" in inst_prompt

    # Invalid mode raises ValueError
    with pytest.raises(ValueError):
        format_conditioned_prompt("test", prompt_mode="invalid_mode")


def test_16_generation_with_plain_instruction_mode():
    """Batch 2781-2820: Test generation execution under plain_instruction mode."""
    adapter = NeuralRuntimeAdapter(max_new_tokens=4, prompt_mode="plain_instruction")
    res = adapter.generate("Hello world")
    assert res["prompt_mode"] == "plain_instruction"
    assert "Instruction: Hello world\nResponse:" in res["formatted_prompt"]
    assert len(res["generated_tokens"]) == 4


def test_17_greedy_decoding_reproducibility():
    """Batch 2781-2820: Greedy decoding is 100% reproducible across multiple runs."""
    adapter = NeuralRuntimeAdapter(max_new_tokens=8, decoding="greedy")
    res1 = adapter.generate("The capital of France is")
    res2 = adapter.generate("The capital of France is")
    assert res1["generated_tokens"] == res2["generated_tokens"]
    assert res1["generated_text"] == res2["generated_text"]
    assert res1["decoding_mode"] == "greedy"


def test_18_sampling_decoding_with_seed_reproducibility():
    """Batch 2781-2820: Sampling with fixed seed produces deterministic output."""
    adapter = NeuralRuntimeAdapter(
        max_new_tokens=8,
        decoding="sampling",
        temperature=0.8,
        seed=12345,
    )
    res1 = adapter.generate("Once upon a time")
    res2 = adapter.generate("Once upon a time", seed=12345)
    assert res1["generated_tokens"] == res2["generated_tokens"]
    assert res1["generated_text"] == res2["generated_text"]
    assert res1["decoding_mode"] == "sampling"
    assert res1["seed"] == 12345


def test_19_repetition_penalty_application():
    """Batch 2781-2820: Repetition penalty is accepted and operates inside generation loop."""
    adapter = NeuralRuntimeAdapter(max_new_tokens=8, repetition_penalty=1.5)
    res = adapter.generate("echo echo echo")
    assert res["repetition_penalty"] == 1.5
    assert len(res["generated_tokens"]) == 8


def test_20_invalid_decoding_parameters_rejected():
    """Batch 2781-2820: Out of bounds decoding parameters are rejected."""
    # Invalid decoding mode
    with pytest.raises(ValueError):
        NeuralRuntimeAdapter(decoding="quantum")

    # Invalid prompt mode
    with pytest.raises(ValueError):
        NeuralRuntimeAdapter(prompt_mode="unsupported_template")

    # Negative temperature
    with pytest.raises(ValueError):
        NeuralRuntimeAdapter(temperature=-0.5)

    # Repetition penalty < 1.0
    with pytest.raises(ValueError):
        NeuralRuntimeAdapter(repetition_penalty=0.5)


def test_21_context_limit_enforcement():
    """Batch 2781-2820: Prompt exceeding context limit is rejected safely with clear error."""
    from chakrview.runtime.neural_inference_contract import NeuralContractError
    adapter = NeuralRuntimeAdapter(max_new_tokens=32)
    # 600 words will exceed 512 context limit
    huge_prompt = "word " * 600
    with pytest.raises(NeuralContractError) as exc_info:
        adapter.generate(huge_prompt)
    assert "exceeds model ceiling" in str(exc_info.value)


def test_22_generation_benchmark_artifacts_exist_and_valid():
    """Batch 2781-2820: Benchmark dataset and report artifacts exist with valid structure."""
    import json
    b_dir = REPO_ROOT / "experiments" / "batch2781_2820"
    dataset_p = b_dir / "GENERATION_BENCHMARK.json"
    results_p = b_dir / "GENERATION_BENCHMARK_RESULTS.json"
    report_p = b_dir / "GENERATION_BENCHMARK_REPORT.md"
    protocol_p = b_dir / "GENERATION_PROTOCOL.md"

    assert dataset_p.is_file(), "Missing GENERATION_BENCHMARK.json"
    assert results_p.is_file(), "Missing GENERATION_BENCHMARK_RESULTS.json"
    assert report_p.is_file(), "Missing GENERATION_BENCHMARK_REPORT.md"
    assert protocol_p.is_file(), "Missing GENERATION_PROTOCOL.md"

    with open(dataset_p, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    assert len(dataset) >= 24
    for item in dataset:
        assert "case_id" in item
        assert "category" in item
        assert "input" in item
        assert "expected_behavior" in item

    with open(results_p, "r", encoding="utf-8") as f:
        results = json.load(f)
    assert results["total_cases"] == len(dataset)
    assert "raw_completion" in results["results"]
    assert "plain_instruction" in results["results"]

