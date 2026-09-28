"""
Unit tests for Step 9 Skill Layer abstractions (chakrview.runtime.skills).
"""

import pytest
from chakrview.runtime.skills import (
    SkillDomain,
    SkillPolicy,
    Skill,
    SkillRegistry,
    SkillProfile,
)


def test_skill_policy_lifecycle():
    policy = SkillPolicy(
        allowed_tools=["calculator", "file_reader"],
        max_context_tokens=256,
        temperature=0.0,
        stop_sequences=["```", "\n\n"],
        requires_knowledge=True,
        allow_code_execution=False,
    )
    d = policy.to_dict()
    assert d["max_context_tokens"] == 256
    assert d["requires_knowledge"] is True
    assert "calculator" in d["allowed_tools"]

    restored = SkillPolicy.from_dict(d)
    assert restored.max_context_tokens == 256
    assert restored.allow_code_execution is False


def test_skill_registration_and_discovery():
    registry = SkillRegistry()

    skill_coding = Skill(
        skill_id="skill_python_v1",
        name="Python Programming",
        version="1.0.0",
        domain=SkillDomain.CODING,
        description="Python code generation and refactoring.",
        system_prompt_template="You are a strict, idiomatic Python assistant.",
        policy=SkillPolicy(allowed_tools=["code_linter"], max_context_tokens=512),
    )

    skill_math = Skill(
        skill_id="skill_gsm8k_math_v1",
        name="Step-by-Step Math",
        version="1.0.0",
        domain=SkillDomain.MATHEMATICS,
        description="Structured arithmetic problem solving.",
        system_prompt_template="Solve step by step. Output calculation tags.",
        policy=SkillPolicy(temperature=0.0),
    )

    registry.register(skill_coding)
    registry.register(skill_math)
    assert registry.count() == 2

    # Duplicate registration without overwrite raises error
    with pytest.raises(ValueError):
        registry.register(skill_coding)

    # Retrieval
    retrieved = registry.get("skill_python_v1")
    assert retrieved is not None
    assert retrieved.domain == SkillDomain.CODING

    # Domain filtering
    coding_skills = registry.list_skills(domain=SkillDomain.CODING)
    assert len(coding_skills) == 1
    assert coding_skills[0].skill_id == "skill_python_v1"

    math_skills = registry.list_skills(domain=SkillDomain.MATHEMATICS)
    assert len(math_skills) == 1

    empty_skills = registry.list_skills(domain=SkillDomain.LEGAL if hasattr(SkillDomain, "LEGAL") else SkillDomain.ANALYSIS)
    assert len(empty_skills) == 0

    # Unregister
    assert registry.unregister("skill_python_v1") is True
    assert registry.count() == 1
    assert registry.has_skill("skill_python_v1") is False


def test_skill_profile():
    profile = SkillProfile(
        profile_id="prof_dev_agent",
        name="Developer Profile",
        primary_skill_id="skill_python_v1",
        active_skill_ids={"skill_git_v1", "skill_terminal_v1"},
        description="Engineering automation profile",
    )
    # primary_skill_id is automatically added to active_skill_ids
    assert "skill_python_v1" in profile.active_skill_ids
    assert len(profile.active_skill_ids) == 3

    d = profile.to_dict()
    restored = SkillProfile.from_dict(d)
    assert restored.primary_skill_id == "skill_python_v1"
    assert restored.active_skill_ids == profile.active_skill_ids
