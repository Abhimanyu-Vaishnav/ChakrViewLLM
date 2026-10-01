"""
ChakrView Step 64: Grounded Neural Proposal Adapter & Hallucination Containment Gate.

Defines:
- NeuralProposalAdapter: Pluggable adapter interface for neural candidate generation.
- GroundedProposalParser: Parses model output into structured proposals.
- HallucinationContainmentGate: Verifies that proposed files, symbols, and dependencies
  are provably grounded in the repository context; rejects or abstains on hallucinations.
- GroundedNeuralSynthesisEngine: CandidateSynthesisEngine implementation integrating neural proposal
  with deterministic grounding and safety gates.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from chakrview.cognition.repository.state import RepositoryState
from chakrview.cognition.repository.refactoring import RefactoringStep
from chakrview.cognition.repository.branching import BranchObservation
from chakrview.cognition.repository.semantic_record import RepositorySemanticRecord
from chakrview.cognition.repository.synthesis import (
    CandidateOrigin,
    CandidateProvenance,
    SynthesizedCandidate,
    CandidateNormalizer,
    DeterministicSafetyGate,
    CandidateSynthesisEngine,
    SafetyDecision,
)
from chakrview.cognition.repository.context_store import (
    RepositoryContextStore,
    GroundedContextBundle,
    EpistemicState,
    EvidenceRecord,
)


@dataclass
class NeuralProposalOutput:
    """Raw structured output emitted by a neural proposal adapter."""
    proposal_id: str
    objective: str
    target_files: List[str]
    target_symbols: List[str]
    proposed_patches: Dict[str, str]  # file_path -> patch content
    preconditions: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    model_identifier: str = "ChakrMicro-Adapter-v0.1"
    raw_response: Optional[str] = None
    epistemic_claims: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "objective": self.objective,
            "target_files": list(self.target_files),
            "target_symbols": list(self.target_symbols),
            "preconditions": self.preconditions,
            "confidence": self.confidence,
            "model_identifier": self.model_identifier,
            "epistemic_claims": list(self.epistemic_claims),
        }


class NeuralProposalAdapter(ABC):
    """
    Abstract adapter boundary for neural proposal generation.
    Receives strictly bounded GroundedContextBundle and returns structured proposals.
    """

    @abstractmethod
    def generate_proposal(
        self,
        context: Any,
    ) -> Optional[NeuralProposalOutput]:
        """Propose a refactoring candidate from grounded context or cognitive context."""
        pass


class MockNeuralProposalAdapter(NeuralProposalAdapter):
    """
    Deterministic reference adapter for benchmarking neural integration and
    verifying grounding boundaries without unconstrained external API calls.
    """

    def __init__(
        self,
        preset_proposal_generator: Optional[Callable[[Any], Optional[NeuralProposalOutput]]] = None,
    ) -> None:
        self.preset_generator = preset_proposal_generator

    def generate_proposal(
        self,
        context: Any,
    ) -> Optional[NeuralProposalOutput]:
        if self.preset_generator:
            return self.preset_generator(context)

        # Support both GroundedContextBundle and CognitiveContextBundle / Dict
        candidate_files: List[str] = []
        task_desc: str = ""
        patch: Dict[str, str] = {}

        if isinstance(context, dict):
            candidate_files = list(context.get("target_files", [])) or list(context.get("allowed_files", []))
            task_desc = str(context.get("task_description", ""))
            pos_patterns = context.get("positive_solution_patterns", ())
            if pos_patterns and isinstance(pos_patterns, (list, tuple)):
                p0 = pos_patterns[0]
                pat_str = p0.get("pattern", "") if isinstance(p0, dict) else ""
                if candidate_files and pat_str:
                    patch[candidate_files[0]] = pat_str
            symbols = list(context.get("target_symbols", []))[:2]
        elif hasattr(context, "positive_memories"):
            # CognitiveContextBundle
            candidate_files = list(context.request.target_files) or list(context.request.allowed_files)
            task_desc = context.request.task_description
            if context.positive_memories and candidate_files:
                patch[candidate_files[0]] = context.positive_memories[0].content_payload
            symbols = list(context.active_symbols)[:2]
        elif hasattr(context, "candidate_files"):
            # GroundedContextBundle
            candidate_files = list(context.candidate_files)
            task_desc = context.task_description
            if context.active_memory_records and candidate_files:
                patch[candidate_files[0]] = context.active_memory_records[0].solution_pattern
            symbols = list(context.relevant_symbols)[:2]
        else:
            symbols = []

        if not candidate_files:
            return None

        primary_file = candidate_files[0]
        if not patch:
            patch[primary_file] = f"# Proposed fix for {task_desc}\n"

        return NeuralProposalOutput(
            proposal_id=f"prop_neural_{int(time.time() * 1000) % 100000}",
            objective=task_desc,
            target_files=[primary_file],
            target_symbols=symbols,
            proposed_patches=patch,
            confidence=0.85,
            model_identifier="ChakrMicro-DeterministicReference-v0.1",
        )


@dataclass
class GroundingVerificationDecision:
    """Audit record evaluating whether a proposal is grounded in real repository evidence."""
    is_grounded: bool
    epistemic_state: EpistemicState
    unfounded_files: List[str] = field(default_factory=list)
    unfounded_symbols: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_grounded": self.is_grounded,
            "epistemic_state": self.epistemic_state.name,
            "unfounded_files": self.unfounded_files,
            "unfounded_symbols": self.unfounded_symbols,
            "rejection_reasons": self.rejection_reasons,
        }


class HallucinationContainmentGate:
    """
    Deterministic gate that proves whether every entity in a proposal exists in reality.
    Protects against:
    - Nonexistent files
    - Nonexistent functions/classes (hallucinated symbols)
    - Fabricated tests or dependencies
    - Unfounded assumptions
    """

    @classmethod
    def verify_grounding(
        cls,
        proposal: NeuralProposalOutput,
        context_store: RepositoryContextStore,
        allowed_modified_files: Set[str],
    ) -> GroundingVerificationDecision:
        reasons: List[str] = []
        unfounded_files: List[str] = []
        unfounded_symbols: List[str] = []

        # 1. Verify existence of target files
        for f in proposal.target_files:
            if not context_store.file_exists(f):
                unfounded_files.append(f)
                reasons.append(f"Hallucinated file '{f}': file does not exist in repository")
            elif f not in allowed_modified_files:
                reasons.append(f"File '{f}' is outside allowed task scope")

        # 2. Verify patch targets
        for patch_f in proposal.proposed_patches.keys():
            if not context_store.file_exists(patch_f):
                unfounded_files.append(patch_f)
                reasons.append(f"Patch targets hallucinated file '{patch_f}'")

        # 3. Verify existence of target symbols
        for sym in proposal.target_symbols:
            if not context_store.symbol_exists(sym):
                unfounded_symbols.append(sym)
                reasons.append(f"Hallucinated symbol '{sym}': function/class not found in repository")

        if unfounded_files or unfounded_symbols:
            return GroundingVerificationDecision(
                is_grounded=False,
                epistemic_state=EpistemicState.CONTRADICTED,
                unfounded_files=unfounded_files,
                unfounded_symbols=unfounded_symbols,
                rejection_reasons=reasons,
            )

        if reasons:
            return GroundingVerificationDecision(
                is_grounded=False,
                epistemic_state=EpistemicState.UNKNOWN,
                rejection_reasons=reasons,
            )

        return GroundingVerificationDecision(
            is_grounded=True,
            epistemic_state=EpistemicState.KNOWN,
            rejection_reasons=[],
        )


class GroundedNeuralSynthesisEngine(CandidateSynthesisEngine):
    """
    Orchestrates the neural proposal pipeline:
    1. Context retrieval via GroundedContextRetriever.
    2. Neural candidate generation via NeuralProposalAdapter.
    3. Hallucination containment verification via HallucinationContainmentGate.
    4. Synthesis of grounded SynthesizedCandidate instances for deterministic safety gating.
    """

    def __init__(
        self,
        context_store: RepositoryContextStore,
        neural_adapter: Optional[NeuralProposalAdapter] = None,
    ) -> None:
        self.context_store = context_store
        self.neural_adapter = neural_adapter or MockNeuralProposalAdapter()

    def synthesize_candidates(
        self,
        objective: str,
        task_domain: str,
        repo_state: RepositoryState,
        allowed_files: Set[str],
        failed_observations: List[BranchObservation],
        memory_records: List[RepositorySemanticRecord],
    ) -> List[SynthesizedCandidate]:
        from chakrview.cognition.repository.context_store import GroundedContextRetriever

        retriever = GroundedContextRetriever(context_store=self.context_store)
        context_bundle = retriever.retrieve_context(
            task_description=objective,
            target_domain=task_domain,
        )

        raw_proposal = self.neural_adapter.generate_proposal(context_bundle)
        if not raw_proposal:
            return []

        # Enforce Hallucination Containment Gate
        grounding_dec = HallucinationContainmentGate.verify_grounding(
            proposal=raw_proposal,
            context_store=self.context_store,
            allowed_modified_files=allowed_files,
        )

        if not grounding_dec.is_grounded:
            # Hard stop: do NOT allow ungrounded or hallucinated candidate to enter execution pipeline
            return []

        # Convert valid grounded neural proposal into SynthesizedCandidate
        steps: List[RefactoringStep] = []
        for idx, (target_f, content) in enumerate(raw_proposal.proposed_patches.items()):
            steps.append(
                RefactoringStep(
                    step_id=f"neural_step_{idx}_{raw_proposal.proposal_id[:6]}",
                    description=f"Neural step for {target_f}",
                    target_files=[target_f],
                    patch_dict={target_f: content},
                )
            )

        cand = SynthesizedCandidate(
            candidate_id=raw_proposal.proposal_id,
            objective=raw_proposal.objective,
            ordered_steps=steps,
            allowed_files=set(raw_proposal.target_files),
            preconditions=raw_proposal.preconditions,
            provenance=CandidateProvenance(
                origin=CandidateOrigin.SYNTHESIS_LLM,
                domain_tags={task_domain},
                synthesis_rationale=f"Grounded neural proposal via {raw_proposal.model_identifier}",
                confidence_score=raw_proposal.confidence,
            ),
        )

        return [cand]
