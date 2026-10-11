"""
ChakrView v0.6.0 Candidate: Neural Representation & Semantic Grounding Core.

Encompasses Stages 1-10:
- Stage 1: Relational Representation Learning (Role-bound slot templates)
- Stage 2: Semantic Composition (Entities + Attributes + Relations)
- Stage 3: Multi-Hop Semantic Representation (Path transitivity)
- Stage 4: Language Semantic Grounding (Surface forms -> Relational triples)
- Stage 5: Semantic State -> Cognition Coupling (Informing decision state)
- Stage 6: Surface Reformulation Generalization (Active/passive/lexical variants)
- Stage 7: Controlled Neural Training (CPU, micro-batch, gradient clipped)
- Stage 8: Small Blind Evaluation (Held-out benchmark evaluation)
- Stage 9: Focused Regression Check
- Stage 10: v0.6.0 Candidate Research Gate
"""

from .relational_encoder import RelationalEncoder, RelationalRepresentationResult
from .semantic_composition import SemanticCompositionEngine, SemanticCompositionResult
from .multihop_semantic import MultiHopSemanticReasoner, MultiHopResult
from .language_grounding import LanguageSemanticGroundingBridge, GroundingResult
from .semantic_cognitive_coupling import SemanticCognitiveCouplingEngine, CouplingEvaluationResult
from .surface_reformulation import SurfaceReformulationTester, ReformulationResult
from .v060_master_gate import V060CandidateMasterGate, V060EvaluationReport

__all__ = [
    "RelationalEncoder",
    "RelationalRepresentationResult",
    "SemanticCompositionEngine",
    "SemanticCompositionResult",
    "MultiHopSemanticReasoner",
    "MultiHopResult",
    "LanguageSemanticGroundingBridge",
    "GroundingResult",
    "SemanticCognitiveCouplingEngine",
    "CouplingEvaluationResult",
    "SurfaceReformulationTester",
    "ReformulationResult",
    "V060CandidateMasterGate",
    "V060EvaluationReport",
]
