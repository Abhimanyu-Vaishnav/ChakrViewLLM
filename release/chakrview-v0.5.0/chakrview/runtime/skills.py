"""
Skill Abstraction & Registry Layer for ChakrView (Step 9).

Enables specialized capability profiles (coding, reasoning, mathematics, enterprise)
without modifying or duplicating base neural core weights:
Base Brain + Skill + Knowledge + Memory + Tools = Specialized ChakrView Instance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
import re
from typing import Dict, List, Optional, Any, Set, Tuple


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


class SkillResolver(ABC):
    """
    Abstract strategy for resolving an appropriate capability Skill for a user query.
    """

    @abstractmethod
    def resolve(
        self,
        query: str,
        registry: SkillRegistry,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Skill:
        """Resolve a Skill given query text and registry."""
        pass


class RuleBasedSkillResolver(SkillResolver):
    """
    Deterministic rule-based skill router.
    
    Inspects queries for domain-specific keywords and patterns to dispatch to:
    - MATHEMATICS
    - CODING
    - REASONING
    - ENTERPRISE
    - WRITING
    - ANALYSIS
    - SYSTEM
    - GENERAL (fallback)
    
    Can be replaced in future steps by a learned / neural classifier without
    altering runtime contracts.
    """

    DOMAIN_PATTERNS = {
        SkillDomain.MATHEMATICS: [
            r"\b(calculate|calc|math|arithmetic|solve|equation|formula|sum|multiply|divide|subtract|add|derivative|integral|modulo|sqrt|percentage)\b",
            r"[\+\-\*\/\^=]\s*\d+",
        ],
        SkillDomain.CODING: [
            r"\b(python|code|def\s+|class\s+|function|bug|syntax|algorithm|variable|refactor|compile|script|import\s+|lambda|return\s+|array|pointer|git|bash|sql|json|api|regex)\b",
            r"```",
        ],
        SkillDomain.REASONING: [
            r"\b(why|prove|reason|logic|deduce|deduction|imply|conclude|premise|fallacy|argument|syllogism|cause\s+of|inference)\b",
        ],
        SkillDomain.ENTERPRISE: [
            r"\b(company|enterprise|policy|internal|org|report|compliance|chakrview|revenue|quarterly|contract|confidential|sop|guideline)\b",
        ],
        SkillDomain.WRITING: [
            r"\b(write|draft|poem|essay|summary|summarize|rewrite|tone|creative|blog|letter|prose|story|paraphrase)\b",
        ],
        SkillDomain.ANALYSIS: [
            r"\b(analyze|analysis|compare|contrast|metric|benchmark|trend|breakdown|evaluation|pros\s+and\s+cons|tradeoff)\b",
        ],
        SkillDomain.SYSTEM: [
            r"\b(cpu|gpu|hardware|memory|ram|thread|cache|operating\s+system|runtime|config|disk|i\/o|throughput|latency)\b",
        ],
    }

    def __init__(self, default_skill_id: str = "skill_general_v1") -> None:
        self.default_skill_id = default_skill_id

    def resolve(
        self,
        query: str,
        registry: SkillRegistry,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Skill:
        """
        Deterministically match query against domain patterns and resolve skill.
        """
        q_lower = query.lower()

        # Check domain patterns in priority order
        resolved_domain = SkillDomain.GENERAL
        for domain, patterns in self.DOMAIN_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, q_lower):
                    resolved_domain = domain
                    break
            if resolved_domain != SkillDomain.GENERAL:
                break

        # Look up skill in registry for resolved domain
        matching_skills = registry.list_skills(domain=resolved_domain)
        if matching_skills:
            return matching_skills[0]

        # Fallback to default skill
        default_skill = registry.get(self.default_skill_id)
        if default_skill is not None:
            return default_skill

        # If registry has any skills at all, return first
        all_skills = registry.list_skills()
        if all_skills:
            return all_skills[0]

        # Fallback: create ad-hoc generic skill
        return Skill(
            skill_id="skill_fallback_general",
            name="General Capability",
            version="1.0.0",
            domain=SkillDomain.GENERAL,
            description="Default generic cognitive policy",
            system_prompt_template="You are ChakrView, an indigenous intelligent AI assistant.",
            policy=SkillPolicy(),
        )


def get_standard_skill_registry() -> SkillRegistry:
    """Factory returning standard pre-populated SkillRegistry covering all 8 domains."""
    reg = SkillRegistry()

    reg.register(
        Skill(
            skill_id="skill_general_v1",
            name="Universal General Intelligence",
            version="1.0.0",
            domain=SkillDomain.GENERAL,
            description="Broad conversational and instructional baseline.",
            system_prompt_template="You are ChakrView, an indigenous intelligent AI assistant.",
            policy=SkillPolicy(temperature=0.7),
        )
    )
    reg.register(
        Skill(
            skill_id="skill_coding_v1",
            name="Software Engineering & Coding",
            version="1.0.0",
            domain=SkillDomain.CODING,
            description="Clean code generation, refactoring, and syntax analysis.",
            system_prompt_template="You are an expert, precise software engineering assistant. Output clean, correct code.",
            policy=SkillPolicy(temperature=0.1, stop_sequences=["```"]),
        )
    )
    reg.register(
        Skill(
            skill_id="skill_mathematics_v1",
            name="Structured Mathematics & Calculation",
            version="1.0.0",
            domain=SkillDomain.MATHEMATICS,
            description="Rigorous numerical calculation and step-by-step problem solving.",
            system_prompt_template="You are a precise mathematical assistant. Solve step by step.",
            policy=SkillPolicy(temperature=0.0, allowed_tools=["calculator"]),
        )
    )
    reg.register(
        Skill(
            skill_id="skill_reasoning_v1",
            name="Rigorous Logical Reasoning",
            version="1.0.0",
            domain=SkillDomain.REASONING,
            description="Premise analysis, causal inference, and deductive reasoning.",
            system_prompt_template="You are a rigorous logical reasoning assistant. Analyze premises systematically.",
            policy=SkillPolicy(temperature=0.2),
        )
    )
    reg.register(
        Skill(
            skill_id="skill_enterprise_v1",
            name="Enterprise Document Intelligence",
            version="1.0.0",
            domain=SkillDomain.ENTERPRISE,
            description="Grounded document search and factual organizational QA.",
            system_prompt_template="You are an enterprise knowledge assistant. Answer strictly based on verified documents.",
            policy=SkillPolicy(temperature=0.1, requires_knowledge=True),
        )
    )
    reg.register(
        Skill(
            skill_id="skill_writing_v1",
            name="Articulate Creative Writing",
            version="1.0.0",
            domain=SkillDomain.WRITING,
            description="Expressive composition, summarization, and prose styling.",
            system_prompt_template="You are a creative and articulate writing assistant.",
            policy=SkillPolicy(temperature=0.8),
        )
    )
    reg.register(
        Skill(
            skill_id="skill_analysis_v1",
            name="Objective Technical Analysis",
            version="1.0.0",
            domain=SkillDomain.ANALYSIS,
            description="Comparative analysis, tradeoffs, and system evaluations.",
            system_prompt_template="You are an objective analytical assistant. Examine data, metrics, and comparisons.",
            policy=SkillPolicy(temperature=0.2),
        )
    )
    reg.register(
        Skill(
            skill_id="skill_system_v1",
            name="Systems & Hardware Awareness",
            version="1.0.0",
            domain=SkillDomain.SYSTEM,
            description="Hardware-aware execution, throughput analysis, and runtime tuning.",
            system_prompt_template="You are a hardware-aware systems and performance assistant.",
            policy=SkillPolicy(temperature=0.0, allowed_tools=["text_utility"]),
        )
    )

    return reg
