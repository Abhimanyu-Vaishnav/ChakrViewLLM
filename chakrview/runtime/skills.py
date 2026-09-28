"""
Skill Abstraction & Registry Layer for ChakrView (Step 9).

Enables specialized capability profiles (coding, reasoning, mathematics, enterprise)
without modifying or duplicating base neural core weights:
Base Brain + Skill + Knowledge + Memory + Tools = Specialized ChakrView Instance.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Set


class SkillDomain(str, Enum):
    """Domains of specialized capability."""
    GENERAL = "general"
    CODING = "coding"
    REASONING = "reasoning"
    MATHEMATICS = "mathematics"
    WRITING = "writing"
    ANALYSIS = "analysis"
    ENTERPRISE = "enterprise"
    SYSTEM = "system"


@dataclass
class SkillPolicy:
    """
    Execution and safety policy governing a skill.
    
    Attributes:
        allowed_tools: List of tool identifiers permitted under this skill.
        max_context_tokens: Upper bound on input context for this skill (<= 512).
        temperature: Default decoding temperature (0.0 for deterministic).
        stop_sequences: List of strings that terminate generation.
        requires_knowledge: Whether the skill mandates prior knowledge retrieval.
        allow_code_execution: Whether generated code may be dispatched to a sandbox.
        metadata: Domain-specific policy parameters.
    """
    allowed_tools: List[str] = field(default_factory=list)
    max_context_tokens: int = 512
    temperature: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    requires_knowledge: bool = False
    allow_code_execution: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillPolicy":
        return cls(**data)


@dataclass
class Skill:
    """
    A modular capability definition.
    
    Attributes:
        skill_id: Unique string identifier (e.g., "skill_python_coding_v1").
        name: Human-readable display name.
        version: Semantic version string.
        domain: Capability domain.
        description: Functional description.
        system_prompt_template: Scaffolding system prompt applied during execution.
        policy: Associated execution and safety policy.
        evaluation_suite_id: Identifier of benchmark suite used to validate this skill.
        metadata: Optional skill attributes.
    """
    skill_id: str
    name: str
    version: str
    domain: SkillDomain
    description: str
    system_prompt_template: str = ""
    policy: SkillPolicy = field(default_factory=SkillPolicy)
    evaluation_suite_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["domain"] = self.domain.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        data = dict(data)
        if isinstance(data.get("domain"), str):
            data["domain"] = SkillDomain(data["domain"])
        if isinstance(data.get("policy"), dict):
            data["policy"] = SkillPolicy.from_dict(data["policy"])
        return cls(**data)


@dataclass
class SkillExecutionPlan:
    """
    Resolved execution plan for a specific user request given an active skill.
    
    Attributes:
        skill_id: Identifier of active skill.
        resolved_system_prompt: System prompt with any parameters substituted.
        user_prompt: Raw user query.
        full_prompt: Combined input sequence ready for tokenization.
        policy: Enforced policy.
        active_tools: Tools permitted for this execution.
    """
    skill_id: str
    resolved_system_prompt: str
    user_prompt: str
    full_prompt: str
    policy: SkillPolicy
    active_tools: List[str] = field(default_factory=list)


class SkillRegistry:
    """
    Thread-safe registry for managing, discovering, and resolving skills.
    """

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill, overwrite: bool = False) -> None:
        """Register a new skill into the repository."""
        if skill.skill_id in self._skills and not overwrite:
            raise ValueError(
                f"Skill '{skill.skill_id}' already registered. Use overwrite=True to replace."
            )
        self._skills[skill.skill_id] = skill

    def get(self, skill_id: str) -> Optional[Skill]:
        """Retrieve a skill by identifier."""
        return self._skills.get(skill_id)

    def unregister(self, skill_id: str) -> bool:
        """Unregister a skill. Returns True if removed."""
        if skill_id in self._skills:
            del self._skills[skill_id]
            return True
        return False

    def list_skills(self, domain: Optional[SkillDomain] = None) -> List[Skill]:
        """List all registered skills, optionally filtered by domain."""
        if domain is None:
            return list(self._skills.values())
        return [s for s in self._skills.values() if s.domain == domain]

    def has_skill(self, skill_id: str) -> bool:
        """Check if a skill is registered."""
        return skill_id in self._skills

    def count(self) -> int:
        """Return total number of registered skills."""
        return len(self._skills)

    def clear(self) -> None:
        """Clear all registered skills."""
        self._skills.clear()


@dataclass
class SkillProfile:
    """
    An active composition of skills bound to a ChakrView instance.
    
    Attributes:
        profile_id: Unique profile identifier.
        name: Display name.
        primary_skill_id: Primary default skill.
        active_skill_ids: Set of all active skills in this profile.
        description: Profile purpose.
    """
    profile_id: str
    name: str
    primary_skill_id: str
    active_skill_ids: Set[str] = field(default_factory=set)
    description: str = ""

    def __post_init__(self) -> None:
        if self.primary_skill_id:
            self.active_skill_ids.add(self.primary_skill_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "name": self.name,
            "primary_skill_id": self.primary_skill_id,
            "active_skill_ids": list(self.active_skill_ids),
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillProfile":
        data = dict(data)
        data["active_skill_ids"] = set(data.get("active_skill_ids", []))
        return cls(**data)
