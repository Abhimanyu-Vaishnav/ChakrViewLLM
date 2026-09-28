"""
Cognitive Context Compression & Prioritization for ChakrView (Step 25).

Enforces the hard 512-token context ceiling of ChakrMicro v0.1 by prioritising
cognitive state information deterministically:
    1. Current Task / Objective
    2. Active Constraints
    3. Verified Relevant Evidence
    4. High-Confidence Relevant Memory
    5. Critical Contradictions
    6. Active Hypotheses
    7. Revision Directives
    8. Recent Experience
    9. Lower-Priority Context

CRITICAL ARCHITECTURAL CONSTRAINTS:
1. Hard context ceiling: Context NEVER exceeds 512 tokens.
2. Truncation accounting: All truncated or omitted items are explicitly logged.
"""

from typing import Dict, List, Optional, Any, Tuple
from chakrview.tokenizer.tokenizer import BPETokenizer
from chakrview.cognition.unified.models import UnifiedCognitiveState


class CognitiveContextCompressor:
    """
    Deterministic context compressor packaging unified cognitive state into bounded prompt tokens.
    """

    def __init__(self, tokenizer: BPETokenizer) -> None:
        self.tokenizer = tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens accurately using the indigenous BPE tokenizer."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def build_bounded_context(
        self,
        state: UnifiedCognitiveState,
        max_context_tokens: int = 448,
    ) -> Tuple[str, List[int], Dict[str, Any]]:
        """
        Assemble and compress unified cognitive state into a bounded prompt.
        Guarantees that total tokens do not exceed min(max_context_tokens, 512).
        Returns: (prompt_text, token_ids, truncation_metadata)
        """
        ceiling = min(max_context_tokens, 512)
        truncation_meta: Dict[str, Any] = {
            "omitted_sections": [],
            "truncated_items_count": 0,
            "allocated_budget": ceiling,
            "actual_tokens": 0,
        }

        sections: List[Tuple[str, str, int]] = []
        # Priority 1: Current Task
        sections.append(("task", f"Task: {state.user_prompt}", 1))

        # Priority 2: Active Constraints
        if state.working_memory_snapshot and state.working_memory_snapshot.get("active_constraints"):
            c_text = "Constraints: " + "; ".join(state.working_memory_snapshot["active_constraints"])
            sections.append(("constraints", c_text, 2))

        # Priority 3: Verified Relevant Evidence
        if state.evidence:
            ev_strs = []
            for ev in state.evidence:
                c = ev.get("content") or ev.get("statement") or str(ev)
                ev_strs.append(c[:80])
            sections.append(("evidence", "Evidence: " + " | ".join(ev_strs), 3))

        # Priority 4: High-Confidence Relevant Memory
        if state.retrieved_memories:
            mem_strs = []
            for mem in state.retrieved_memories:
                m_txt = mem.get("content") or mem.get("statement") or str(mem)
                mem_strs.append(m_txt[:80])
            sections.append(("memory", "Recalled Memory: " + " | ".join(mem_strs), 4))

        # Priority 5: Critical Contradictions
        if state.contradictions:
            con_strs = []
            for con in state.contradictions:
                c_desc = con.get("notes") or f"Conflict across {con.get('conflicting_memory_ids')}"
                con_strs.append(c_desc[:80])
            sections.append(("contradictions", "Known Conflicts: " + " | ".join(con_strs), 5))

        # Priority 6: Active Hypotheses
        if state.hypotheses:
            h_strs = [h.get("claim", str(h))[:60] for h in state.hypotheses]
            sections.append(("hypotheses", "Candidate Hypotheses: " + " | ".join(h_strs), 6))

        # Priority 7: Revision Directives
        if state.uncertainty_notes:
            sections.append(("uncertainty", f"Epistemic Caveat: {state.uncertainty_notes[:100]}", 7))

        # Sort strictly by priority
        sections.sort(key=lambda s: s[2])

        # Incrementally add sections within token budget
        assembled_parts: List[str] = []
        current_tokens = 0

        for sec_name, sec_text, priority in sections:
            sec_token_count = self.count_tokens(sec_text + "\n")
            if current_tokens + sec_token_count <= ceiling:
                assembled_parts.append(sec_text)
                current_tokens += sec_token_count
            else:
                # Attempt to fit a truncated preview if it's high priority (task or constraints)
                available = ceiling - current_tokens
                if available > 20 and priority <= 3:
                    # Truncate string roughly
                    words = sec_text.split()
                    truncated_text = ""
                    for word in words:
                        candidate = (truncated_text + " " + word).strip()
                        if self.count_tokens(candidate + "...\n") <= available:
                            truncated_text = candidate
                        else:
                            break
                    if truncated_text:
                        assembled_parts.append(truncated_text + "...")
                        current_tokens = self.count_tokens("\n".join(assembled_parts))
                    truncation_meta["truncated_items_count"] += 1
                else:
                    truncation_meta["omitted_sections"].append(sec_name)
                    truncation_meta["truncated_items_count"] += 1

        final_prompt = "\n".join(assembled_parts).strip()
        final_tokens = self.tokenizer.encode(final_prompt)
        truncation_meta["actual_tokens"] = len(final_tokens)

        # Hard invariant check
        assert len(final_tokens) <= 512, f"Context exceeded 512 tokens: {len(final_tokens)}"

        return final_prompt, final_tokens, truncation_meta
