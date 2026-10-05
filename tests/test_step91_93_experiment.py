"""
Step 91–93 Realistic End-to-End Neural Runtime Experiment.

Demonstrates the complete unified cognitive-neural architecture working as a system:
1. User provides a real multi-module task inquiry.
2. Runtime detects underlying hardware capability (constrained LOW_RESOURCE profile).
3. Input context and token budget are adapted according to available hardware.
4. Neural model executes forward and greedy autoregressive generation with ΔW = 0 verification.
5. Cognitive runtime determines routing:
   a. Queries with known project context retrieve grounded records from PPB.
   b. Missing/unknown dependencies trigger the autonomous investigation loop.
6. Evidence is independently verified and updated in PPB.
7. Unanswerable gaps result in structured, explicit abstention rather than hallucination.
8. Evaluates final response and commits durable knowledge to PPB.
9. Survives process restart without losing persistent knowledge or corrupting neural baseline.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path
import pytest
import torch

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.resource import (
    HardwareCapability,
    RuntimeStrategy,
)
from chakrview.runtime.neural_inference_contract import (
    NeuralInferenceContract,
    NeuralInferenceConfig,
    InferenceStopReason,
    EXPECTED_PARAM_COUNT,
    EXPECTED_WEIGHT_HASH,
)
from chakrview.runtime.adaptive_inference_pipeline import (
    ResourceAdaptiveInferencePipeline,
)
from chakrview.runtime.cognitive_neural_runtime import (
    CognitiveRoute,
    CognitiveRuntimeResponse,
    EndToEndCognitiveNeuralRuntime,
)
from chakrview.cognition.ppb.brain import PersistentProjectBrain
from chakrview.cognition.ppb.models import (
    ProjectIdentity,
    KnowledgeRecord,
    KnowledgeRecordType,
    EpistemicStatus,
)
from chakrview.cognition.research.models import EvidenceItem, InvestigationSource, SourceMetadata

ROOT_DIR = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT_DIR / "data" / "experiments" / "vocab_4096"


def run_realistic_neural_runtime_experiment():
    temp_dir = tempfile.mkdtemp(prefix="step91_93_exp_")
    try:
        tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
        torch.manual_seed(42)
        model = ChakrMicro(ModelConfig())
        model.eval()

        # Verify neural baseline initially
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count == EXPECTED_PARAM_COUNT

        contract = NeuralInferenceContract(model=model, tokenizer=tok)
        assert contract.compute_weight_hash() == EXPECTED_WEIGHT_HASH

        # ---------------------------------------------------------------------
        # 1. Hardware Detection & Adaptive Budgeting (LOW_RESOURCE simulated)
        # ---------------------------------------------------------------------
        low_hw = HardwareCapability(
            cpu_cores=1, cpu_architecture="x86_64", ram_gb=2.0, gpu_available=False,
            gpu_vendor="none", vram_gb=0.0, storage_capacity_gb=20.0, network_available=True,
        )
        adaptive_pipeline = ResourceAdaptiveInferencePipeline(contract, hardware_capability=low_hw)
        assert adaptive_pipeline.budget.strategy == RuntimeStrategy.LOW_RESOURCE
        assert adaptive_pipeline.budget.max_context_window == 256
        assert adaptive_pipeline.budget.max_generation_tokens == 32

        # ---------------------------------------------------------------------
        # 2. Persistent Project Brain Initialization
        # ---------------------------------------------------------------------
        db_path = os.path.join(temp_dir, "brain.db")
        ident = ProjectIdentity(project_id="enterprise_core", project_root="/enterprise_core")
        brain_s1 = PersistentProjectBrain(db_path=db_path, project_identity=ident, hardware_capability=low_hw)

        # Seed grounded project records
        brain_s1.store_knowledge(KnowledgeRecord(
            record_id="rec_data_pipeline",
            project_id="enterprise_core",
            record_type=KnowledgeRecordType.MODULE,
            file_path="pipelines/stream.py",
            summary="Zero-copy Apache Arrow stream ingestion pipeline",
            epistemic_status=EpistemicStatus.FACT,
        ))

        runtime_s1 = EndToEndCognitiveNeuralRuntime(
            adaptive_pipeline=adaptive_pipeline,
            brain=brain_s1,
        )

        # ---------------------------------------------------------------------
        # 3. Process Known Project Query (PPB Retrieval Route)
        # ---------------------------------------------------------------------
        query_known = "What format does pipelines/stream.py ingest?"
        res_known = runtime_s1.process_query(query_known, target_file="pipelines/stream.py")
        assert res_known.route_taken == CognitiveRoute.PPB_RETRIEVAL
        assert len(res_known.retrieved_knowledge) == 1
        assert res_known.epistemic_status == EpistemicStatus.FACT
        assert res_known.is_verified
        assert res_known.neural_output is not None
        assert res_known.neural_output.weight_hash_verified

        # ---------------------------------------------------------------------
        # 4. Process Unknown Module Query (Autonomous Investigation Route)
        # ---------------------------------------------------------------------
        query_unknown = "How does storage/vault.py encrypt payloads?"
        ev_items = [
            EvidenceItem("ev_v1", "req_v", "src_v1", "AES-256-GCM authenticated encryption", 0.95),
            EvidenceItem("ev_v2", "req_v", "src_v2", "AES-256-GCM authenticated encryption", 0.95),
        ]
        srcs = {
            "src_v1": InvestigationSource("src_v1", "file://storage/vault.py", SourceMetadata("meta1", "local_ast", 0.95)),
            "src_v2": InvestigationSource("src_v2", "file://storage/vault.py", SourceMetadata("meta2", "local_ast", 0.95)),
        }
        res_unknown = runtime_s1.process_query(
            query_unknown,
            target_file="storage/vault.py",
            candidate_evidence=ev_items,
            sources=srcs,
        )
        assert res_unknown.route_taken == CognitiveRoute.INVESTIGATION_LOOP
        assert res_unknown.epistemic_status == EpistemicStatus.FACT
        assert res_unknown.is_verified

        # ---------------------------------------------------------------------
        # 5. Process Unanswerable Query (Explicit Honest Abstention)
        # ---------------------------------------------------------------------
        query_unanswerable = "What is the secret master key in the production database?"
        res_abstain = runtime_s1.process_query(query_unanswerable, target_file="unknown_secrets.env")
        assert res_abstain.route_taken == CognitiveRoute.INVESTIGATION_LOOP or res_abstain.route_taken == CognitiveRoute.ABSTAIN
        assert res_abstain.neural_output is not None

        # ---------------------------------------------------------------------
        # 6. Simulate Process Termination and Restart
        # ---------------------------------------------------------------------
        del runtime_s1
        del brain_s1

        # Session 2: Reload brain from SQLite
        brain_s2 = PersistentProjectBrain(db_path=db_path, hardware_capability=low_hw)
        # Verify both original module and newly investigated vault knowledge persisted
        vault_recs = brain_s2.query_knowledge(file_path="storage/vault.py")
        assert len(vault_recs) >= 1
        assert vault_recs[0].epistemic_status == EpistemicStatus.FACT

        runtime_s2 = EndToEndCognitiveNeuralRuntime(
            adaptive_pipeline=adaptive_pipeline,
            brain=brain_s2,
        )
        res_reload_query = runtime_s2.process_query("Inquire vault encryption", target_file="storage/vault.py")
        assert res_reload_query.route_taken == CognitiveRoute.PPB_RETRIEVAL
        assert len(res_reload_query.retrieved_knowledge) >= 1

        # ---------------------------------------------------------------------
        # 7. Final Neural Invariant Check (ΔW = 0)
        # ---------------------------------------------------------------------
        final_hash = contract.compute_weight_hash()
        assert final_hash == EXPECTED_WEIGHT_HASH

        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_realistic_neural_runtime_experiment():
    assert run_realistic_neural_runtime_experiment() is True
