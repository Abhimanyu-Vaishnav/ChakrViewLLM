from typing import Any, Dict
from chakrview.cognition.research.models import InvestigationResult, KnowledgeAcquisitionStatus

class CognitiveResearchIntegrator:
    """
    Integrates verified external evidence into the cognitive context.
    Ensures external evidence is explicitly distinguishable from established facts,
    preserving provenance and verification state.
    """
    
    @staticmethod
    def format_for_reasoning(result: InvestigationResult) -> str:
        """
        Formats an InvestigationResult into a string block safe for neural consumption.
        Never asserts these as absolute truth.
        """
        if result.overall_status in (KnowledgeAcquisitionStatus.INSUFFICIENT, KnowledgeAcquisitionStatus.UNKNOWN):
            return "RESEARCH OUTCOME: Insufficient or unknown evidence."
        
        if result.overall_status == KnowledgeAcquisitionStatus.ABSTAIN:
            return "RESEARCH OUTCOME: Abstain from concluding due to lack of safe evidence."
            
        lines = []
        lines.append(f"RESEARCH OUTCOME: {result.overall_status.value.upper()}")
        lines.append(f"Request ID: {result.request_id}")
        lines.append("EVIDENCE ITEMS:")
        
        for item in result.evidence_items:
            prefix = f"[{item.status.value.upper()}] (Source: {item.source_id})"
            lines.append(f"- {prefix}: {item.content}")
            if item.corroborating_evidence_ids:
                lines.append(f"  Corroborated by: {', '.join(item.corroborating_evidence_ids)}")
            if item.contradicting_evidence_ids:
                lines.append(f"  Contradicted by: {', '.join(item.contradicting_evidence_ids)}")
                
        return "\n".join(lines)
        
    @staticmethod
    def inject_into_context(result: InvestigationResult, context: Dict[str, Any]) -> None:
        """
        Safely injects research results into a structured context dictionary.
        Uses a separate namespace 'external_evidence' to avoid overwriting 'facts'.
        """
        if "external_evidence" not in context:
            context["external_evidence"] = []
            
        context["external_evidence"].append({
            "request_id": result.request_id,
            "overall_status": result.overall_status.value,
            "items": [
                {
                    "content": item.content,
                    "status": item.status.value,
                    "source_id": item.source_id,
                    "corroborated": bool(item.corroborating_evidence_ids),
                    "contradicted": bool(item.contradicting_evidence_ids)
                }
                for item in result.evidence_items
            ]
        })
