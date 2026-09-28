"""
Skill Selection Interface for ChakrView Cognitive Subsystem (Step 15).

Discovers and selects compatible capability profiles from the SkillRegistry
without hardcoding business domains.
"""

from dataclasses import dataclass
import re
from typing import Dict, List, Optional, Any, Tuple

from chakrview.runtime.skills import Skill, SkillRegistry, SkillDomain, get_standard_skill_registry


@dataclass
class SkillSelectionMatch:
    """
    Result of a cognitive skill resolution.
    
    Attributes:
        skill: The selected Skill.
        confidence: Normalized compatibility score in [0.0, 1.0].
        rationale: Explanatory rationale for the selection.
        matched_capabilities: List of terms or tool requirements that triggered the match.
    """
    skill: Skill
    confidence: float
    rationale: str
    matched_capabilities: List[str]


class CognitiveSkillSelector:
    """
    Domain-agnostic skill selector.
    
    Queries available skills, inspects capability metadata and tool permissions,
    and identifies the most compatible skill for a cognitive task.
    """

    def __init__(self, registry: Optional[SkillRegistry] = None) -> None:
        self.registry = registry or get_standard_skill_registry()

    def get_available_skills(
        self,
        domain: Optional[SkillDomain] = None,
        required_tools: Optional[List[str]] = None,
    ) -> List[Skill]:
        """
        List candidate skills, optionally filtered by domain or tool authorization.
        """
        candidates = self.registry.list_skills(domain=domain)
        if not required_tools:
            return candidates

        req_set = set(required_tools)
        return [
            s for s in candidates
            if req_set.issubset(set(s.policy.allowed_tools))
        ]

    def select_skill(
        self,
        user_request: str,
        required_tools: Optional[List[str]] = None,
        preferred_domain: Optional[SkillDomain] = None,
    ) -> SkillSelectionMatch:
        """
        Select the best matching skill for a user request.
        """
        candidates = self.get_available_skills(domain=preferred_domain, required_tools=required_tools)
        if not candidates:
            # Fallback to any general skill
            all_skills = self.registry.list_skills()
            general = self.registry.get("skill_general_v1") or (all_skills[0] if all_skills else None)
            if general is None:
                raise ValueError("SkillRegistry is empty; cannot select skill.")
            return SkillSelectionMatch(
                skill=general,
                confidence=0.1,
                rationale="Fallback to general skill; no candidate satisfied tool/domain constraints.",
                matched_capabilities=[],
            )

        req_clean = user_request.lower()
        req_words = set(re.findall(r"\b\w{3,}\b", req_clean))

        best_skill = candidates[0]
        best_score = -1.0
        best_matches: List[str] = []
        best_rationale = "Default candidate selection."

        for skill in candidates:
            score = 0.0
            matches: List[str] = []

            # 1. Match against description and name
            desc_words = set(re.findall(r"\b\w{3,}\b", (skill.description + " " + skill.name).lower()))
            overlap = req_words.intersection(desc_words)
            if overlap:
                score += len(overlap) * 0.2
                matches.extend(list(overlap))

            # 2. Check required tools compatibility
            if required_tools:
                tool_matches = [t for t in required_tools if t in skill.policy.allowed_tools]
                score += len(tool_matches) * 0.4
                matches.extend([f"tool:{t}" for t in tool_matches])

            # 3. Domain alignment
            if skill.domain.value in req_clean:
                score += 0.3
                matches.append(f"domain:{skill.domain.value}")

            # 4. Math / Calculation heuristics if applicable
            if skill.domain == SkillDomain.MATHEMATICS and any(c in req_clean for c in ["calculate", "math", "add", "multiply"]):
                score += 0.5
                matches.append("math_intent")

            # 5. Coding heuristics if applicable
            if skill.domain == SkillDomain.CODING and any(c in req_clean for c in ["code", "function", "python", "script"]):
                score += 0.5
                matches.append("code_intent")

            # Normalize score
            norm_score = min(1.0, score)
            if norm_score > best_score:
                best_score = norm_score
                best_skill = skill
                best_matches = matches
                best_rationale = f"Matched capabilities: {matches} (score={norm_score:.2f})"

        confidence = max(0.2, best_score if best_score > 0 else 0.2)
        return SkillSelectionMatch(
            skill=best_skill,
            confidence=confidence,
            rationale=best_rationale,
            matched_capabilities=best_matches,
        )
