import time
from chakrview.cognition.research.models import (
    InvestigationRequest, EvidenceItem, SourceMetadata, InvestigationSource,
    KnowledgeAcquisitionStatus, InvestigationScope
)
from chakrview.cognition.research.verifier import EvidenceVerifier
from chakrview.cognition.research.integration import CognitiveResearchIntegrator
from chakrview.runtime.resource import (
    ResourceDetector, RuntimeStrategy
)

def run_experiment():
    print("=== CHAKRVIEW STEP 74-77 EXPERIMENT ===")
    
    # 1. Hardware Profile & Runtime Strategy (Step 77)
    print("\n--- 1. Resource-Adaptive Runtime ---")
    capability = ResourceDetector.detect()
    print(f"Hardware Capability: CPU={capability.cpu_cores} cores, RAM={capability.ram_gb}GB, GPU={capability.gpu_available}")
    
    policy = ResourceDetector.determine_strategy(capability)
    print(f"Determined Strategy: {policy.strategy.name}")
    print(f"Max Context: {policy.max_context_size}, Batch Size: {policy.batch_size}, Parallel: {policy.allow_parallel_execution}")
    
    # 2. Knowledge Acquisition (Step 74 & 75)
    print("\n--- 2. Knowledge Acquisition & Verification ---")
    question = "What is the capital of France?"
    print(f"Question: {question}")
    
    req = InvestigationRequest(user_query=question, freshness_requirement_seconds=86400)
    print(f"Created InvestigationRequest: {req.request_id}")
    
    now = time.time()
    sources = {
        "src_1": InvestigationSource(source_id="src_1", uri="wiki:france", metadata=SourceMetadata(source_identity="wikipedia", source_type="encyclopedia", timestamp=now - 1000, reliability_score=0.9)),
        "src_2": InvestigationSource(source_id="src_2", uri="blog:travel", metadata=SourceMetadata(source_identity="travel_blog", source_type="blog", timestamp=now - 2000, reliability_score=0.8)),
        "src_3": InvestigationSource(source_id="src_3", uri="forum:random", metadata=SourceMetadata(source_identity="forum", source_type="ugc", timestamp=now - 100000, reliability_score=0.2))
    }
    
    items = [
        EvidenceItem(request_id=req.request_id, source_id="src_1", content="The capital of France is Paris."),
        EvidenceItem(request_id=req.request_id, source_id="src_2", content="The capital of France is Paris."),
        EvidenceItem(request_id=req.request_id, source_id="src_3", content="The capital of France is Lyon.") # Low reliability and stale
    ]
    
    verifier = EvidenceVerifier(current_time=now)
    result = verifier.verify(req, items, sources)
    
    print(f"Overall Verification Status: {result.overall_status.name}")
    for item in result.evidence_items:
        print(f" - [Item {item.evidence_id}] Source: {item.source_id} | Status: {item.status.name} | Content: '{item.content}'")
        
    # 3. Research -> Cognitive Context (Step 76)
    print("\n--- 3. Cognitive Context Integration ---")
    formatted_context = CognitiveResearchIntegrator.format_for_reasoning(result)
    print("Formatted for Reasoning Engine:")
    print(formatted_context)
    
    context_dict = {}
    CognitiveResearchIntegrator.inject_into_context(result, context_dict)
    print("\nInjected into Structured Context:")
    print(context_dict)

if __name__ == "__main__":
    run_experiment()
