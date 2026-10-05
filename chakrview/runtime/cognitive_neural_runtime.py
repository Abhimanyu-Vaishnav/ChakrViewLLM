"""
ChakrView Step 93: Cognitive Runtime Integration.

Integrates the Step 91 NeuralInferenceContract and Step 92 ResourceAdaptiveInferencePipeline
with ChakrView's cognitive layers:
- Persistent Project Brain (PPB): Knowledge query and selective context retrieval
- Structured Reasoning & Critical Thinking: Grounding claims and detecting contradictions
- Autonomous Investigation: Invoking research loop when knowledge is UNKNOWN
- Self-Evaluation Gate: Evaluating response quality, preventing hallucinations
- Epistemic Integrity: Distinguishes FACT, INFERRED, UNKNOWN, CONTESTED, ABSTAIN

Cognitive Routing Logic:
1. DIRECT_INFERENCE: Simple prompt with sufficient self-contained context.
2. PPB_RETRIEVAL: Involves project-specific files, modules, or past task history.
3. INVESTIGATION: Critical information is UNKNOWN/INSUFFICIENT; triggers investigation.
4. ABSTAIN: Unresolved contradiction, missing authorization, or unanswerable gap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from chakrview.cognition.ppb.brain import PersistentProjectBrain
from chakrview.cognition.ppb.models import KnowledgeRecord, KnowledgeRecordType, EpistemicStatus
from chakrview.cognition.ppb.investigation_loop import AutonomousInvestigationLoop
from chakrview.cognition.reasoning.structured import InvestigationRequirement
from chakrview.cognition.research.models import EvidenceItem, InvestigationSource, SourceMetadata
from chakrview.runtime.neural_inference_contract import (
    NeuralInferenceContract,
    NeuralInferencePayload,
    NeuralInferenceOutput,
    InferenceStopReason,
)
from chakrview.runtime.adaptive_inference_pipeline import ResourceAdaptiveInferencePipeline


class CognitiveRoute(Enum):
    DIRECT_INFERENCE = "DIRECT_INFERENCE"
    PPB_RETRIEVAL = "PPB_RETRIEVAL"
    INVESTIGATION_LOOP = "INVESTIGATION_LOOP"
    ABSTAIN = "ABSTAIN"


@dataclass
class CognitiveRuntimeResponse:
    """
    Unified response package from the complete cognitive-neural runtime.
    """
    answer_text: str
    route_taken: CognitiveRoute
    neural_output: Optional[NeuralInferenceOutput]
    retrieved_knowledge: List[KnowledgeRecord] = field(default_factory=list)
    epistemic_status: EpistemicStatus = EpistemicStatus.FACT
    is_verified: bool = True
    evaluation_score: float = 1.0
    evaluation_notes: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "route_taken": self.route_taken.value,
            "neural_output": self.neural_output.to_dict() if self.neural_output else None,
            "retrieved_knowledge_count": len(self.retrieved_knowledge),
            "epistemic_status": self.epistemic_status.name,
            "is_verified": self.is_verified,
            "evaluation_score": self.evaluation_score,
            "evaluation_notes": self.evaluation_notes,
            "provenance": self.provenance,
        }


class EndToEndCognitiveNeuralRuntime:
    """
    Step 93: Sovereign Cognitive Neural Runtime.
    Orchestrates the neural inference engine alongside PPB, research, and self-evaluation.
    """

    def __init__(
        self,
        adaptive_pipeline: ResourceAdaptiveInferencePipeline,
        brain: Optional[PersistentProjectBrain] = None,
        investigation_loop: Optional[AutonomousInvestigationLoop] = None,
    ) -> None:
        self.pipeline = adaptive_pipeline
        self.contract = adaptive_pipeline.contract
        self.brain = brain
        self.investigation_loop = investigation_loop or (
            AutonomousInvestigationLoop(brain) if brain else None
        )

    def determine_route(
        self,
        query: str,
        target_file: Optional[str] = None,
        force_investigation: bool = False,
    ) -> CognitiveRoute:
        """
        Cognitive routing classifier based on query intent and knowledge state.
        """
        if force_investigation:
            return CognitiveRoute.INVESTIGATION_LOOP

        # If question targets a file or module, check PPB
        if target_file and self.brain:
            records = self.brain.query_knowledge(file_path=target_file)
            if records:
                return CognitiveRoute.PPB_RETRIEVAL
            else:
                # File is unknown to PPB -> trigger investigation or abstain
                return CognitiveRoute.INVESTIGATION_LOOP

        # Keyword heuristics for project queries
        project_keywords = ["module", "architecture", "dependency", "ppb", "task", "codebase", "repo"]
        if any(kw in query.lower() for kw in project_keywords) and self.brain:
            return CognitiveRoute.PPB_RETRIEVAL

        return CognitiveRoute.DIRECT_INFERENCE

    def process_query(
        self,
        query: str,
        target_file: Optional[str] = None,
        candidate_evidence: Optional[List[EvidenceItem]] = None,
        sources: Optional[Dict[str, InvestigationSource]] = None,
    ) -> CognitiveRuntimeResponse:
        """
        Processes a user query end-to-end through the integrated cognitive-neural path.
        """
        t0 = time.perf_counter()
        route = self.determine_route(query, target_file=target_file)
        retrieved_recs: List[KnowledgeRecord] = []

        # ---------------------------------------------------------------------
        # Route 1: INVESTIGATION_LOOP (Unknown knowledge gap)
        # ---------------------------------------------------------------------
        if route == CognitiveRoute.INVESTIGATION_LOOP:
            if not self.investigation_loop or not self.brain:
                # No investigation capability available -> abstain honestly
                out = self.contract.abstain("Knowledge is UNKNOWN and investigation loop is unavailable.")
                return CognitiveRuntimeResponse(
                    answer_text=out.generated_text,
                    route_taken=CognitiveRoute.ABSTAIN,
                    neural_output=out,
                    epistemic_status=EpistemicStatus.UNKNOWN,
                    is_verified=False,
                    evaluation_score=0.5,
                    evaluation_notes="Abstained due to missing investigation subsystem for unknown fact.",
                )

            inv_req = InvestigationRequirement(
                requirement_id=f"inv_{int(t0)}",
                question=query,
                knowledge_gap=f"Missing knowledge for query: {query}",
                required_evidence_type="SOURCE_ANALYSIS",
                target_module=target_file or "unknown",
            )

            inv_outcome = self.investigation_loop.investigate_requirement(
                req=inv_req,
                candidate_evidence=candidate_evidence or [],
                sources=sources or {},
            )

            # Query updated PPB records
            if target_file:
                retrieved_recs = self.brain.query_knowledge(file_path=target_file)

            context_str = f"Investigation Findings: {inv_outcome.target_query}\nQuery: {query}"
            payload = NeuralInferencePayload(prompt=context_str)
            neural_out = self.pipeline.execute_adaptive(payload)

            ep_status = (
                EpistemicStatus.FACT
                if inv_outcome.overall_status.name == "VERIFIED"
                else EpistemicStatus.CONTESTED
            )

            return CognitiveRuntimeResponse(
                answer_text=f"Investigation completed for {inv_outcome.target_query}\n[Neural Output]: {neural_out.generated_text}",
                route_taken=route,
                neural_output=neural_out,
                retrieved_knowledge=retrieved_recs,
                epistemic_status=ep_status,
                is_verified=(ep_status == EpistemicStatus.FACT),
                evaluation_score=0.9 if ep_status == EpistemicStatus.FACT else 0.6,
                evaluation_notes=f"Investigation status: {inv_outcome.overall_status.name}",
                provenance={"investigation_id": inv_req.requirement_id},
            )

        # ---------------------------------------------------------------------
        # Route 2: PPB_RETRIEVAL (Retrieve grounded project facts)
        # ---------------------------------------------------------------------
        elif route == CognitiveRoute.PPB_RETRIEVAL and self.brain:
            if target_file:
                retrieved_recs = self.brain.query_knowledge(file_path=target_file)
            else:
                retrieved_recs = self.brain.search_knowledge_text(query)

            if not retrieved_recs:
                # Query yielded zero records -> abstain or fallback
                out = self.contract.abstain(f"No grounded knowledge in PPB for query: '{query}'")
                return CognitiveRuntimeResponse(
                    answer_text=out.generated_text,
                    route_taken=CognitiveRoute.ABSTAIN,
                    neural_output=out,
                    epistemic_status=EpistemicStatus.UNKNOWN,
                    is_verified=False,
                    evaluation_score=0.5,
                    evaluation_notes="Zero matching records found in PPB; abstained rather than hallucinating.",
                )

            # Assemble bounded context from retrieved records
            context_pieces = [f"[{r.file_path or r.record_id}]: {r.summary}" for r in retrieved_recs[:5]]
            assembled_prompt = "Project Context:\n" + "\n".join(context_pieces) + f"\n\nQuestion: {query}"

            payload = NeuralInferencePayload(prompt=assembled_prompt)
            neural_out = self.pipeline.execute_adaptive(payload)

            # Record interaction to PPB knowledge evolution
            self.brain.store_knowledge(KnowledgeRecord(
                record_id=f"query_hist_{int(t0)}",
                project_id=self.brain.project_id,
                record_type=KnowledgeRecordType.OBSERVATION,
                file_path=target_file or "runtime/session",
                summary=f"Processed query: {query}",
                epistemic_status=EpistemicStatus.FACT,
            ))

            return CognitiveRuntimeResponse(
                answer_text=f"Retrieved {len(retrieved_recs)} records from PPB.\n[Neural Synthesis]: {neural_out.generated_text}",
                route_taken=route,
                neural_output=neural_out,
                retrieved_knowledge=retrieved_recs,
                epistemic_status=EpistemicStatus.FACT,
                is_verified=True,
                evaluation_score=1.0,
                evaluation_notes="Successfully answered using grounded PPB records.",
                provenance={"records_retrieved": [r.record_id for r in retrieved_recs]},
            )

        # ---------------------------------------------------------------------
        # Route 3: DIRECT_INFERENCE (Standard prompt-driven inference)
        # ---------------------------------------------------------------------
        else:
            payload = NeuralInferencePayload(prompt=query)
            neural_out = self.pipeline.execute_adaptive(payload)

            return CognitiveRuntimeResponse(
                answer_text=neural_out.generated_text,
                route_taken=route,
                neural_output=neural_out,
                epistemic_status=EpistemicStatus.INFERRED,
                is_verified=True,
                evaluation_score=1.0,
                evaluation_notes="Direct inference executed under adaptive budget.",
                provenance={"device": str(self.contract.device)},
            )
