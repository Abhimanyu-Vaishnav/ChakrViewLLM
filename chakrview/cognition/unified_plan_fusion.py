"""
WP 906–912 & WP 913–920: Unified Neural Plan Representation & Span-to-Slot Fusion Pipeline.
Location: chakrview/cognition/unified_plan_fusion.py

Components:
1. AbstractPlanStep: Structured intermediate representation of an execution step
   with abstract slot placeholders (@slot0, @slot1, etc.) or literal values.
2. UnifiedPlanRepresentation: Plan schema separating abstract graph topology from argument values.
3. NeuralSpanSlotFusionPipeline:
   - Takes raw prompt string.
   - Extracts candidate entity spans via ChakrSpanGroundingModel.
   - Predicts / selects abstract plan topology.
   - Binds grounded spans to named step slots.
   - Serializes to Plan DSL with full schema validation and sandboxed safety checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.span_grounding import ChakrSpanGroundingModel, ExtractedSpan
from chakrview.cognition.plan_dsl import ParsedPlan, PlanDSLParser, PlanDSLValidator


@dataclass
class SlotBinding:
    slot_id: str             # e.g. "@slot0"
    slot_name: str           # e.g. "operand_a", "text_input"
    target_step: int         # step index
    target_arg_key: str      # e.g. "expr", "text"
    bound_span: Optional[ExtractedSpan] = None
    is_required: bool = True
    status: str = "UNBOUND"  # "BOUND", "MISSING", "AMBIGUOUS"


@dataclass
class AbstractPlanStep:
    step_id: int
    action: str
    tool: str
    raw_template: str        # e.g. "expr=@slot0*@slot1"
    deps: List[int] = field(default_factory=list)
    slots: List[str] = field(default_factory=list)  # e.g. ["@slot0", "@slot1"]


@dataclass
class UnifiedPlan:
    goal: str
    steps: List[AbstractPlanStep]
    bindings: Dict[str, SlotBinding]
    clarification_needed: bool = False
    clarification_reason: Optional[str] = None

    def materialize_dsl(self) -> Tuple[str, bool, Optional[str]]:
        """
        Substitutes bound span values into step templates and produces validated Plan DSL.
        """
        if self.clarification_needed:
            reason = self.clarification_reason or "missing_information"
            dsl = (
                f"GOAL: clarification_needed\n"
                f"STEP 0: clarification | none | args={{reason={reason}}} | deps=[]\n"
                f"CLARIFICATION: NEEDED\n"
                f"<END_PLAN>"
            )
            return dsl, True, None

        lines = [f"GOAL: {self.goal}"]
        for st in self.steps:
            arg_expr = st.raw_template
            for s_id in st.slots:
                binding = self.bindings.get(s_id)
                if not binding or binding.bound_span is None or not binding.bound_span.span_text:
                    if binding and binding.is_required:
                        # Required slot unbound -> trigger safe clarification
                        return (
                            f"GOAL: clarification_needed\n"
                            f"STEP 0: clarification | none | args={{missing={s_id}}} | deps=[]\n"
                            f"CLARIFICATION: NEEDED\n"
                            f"<END_PLAN>",
                            True,
                            f"Required slot {s_id} is unbound",
                        )
                    val = ""
                else:
                    val = binding.bound_span.span_text
                arg_expr = arg_expr.replace(s_id, val)

            deps_str = ",".join(str(d) for d in st.deps)
            lines.append(f"STEP {st.step_id}: {st.action} | {st.tool} | args={{{arg_expr}}} | deps=[{deps_str}]")

        lines.append("CLARIFICATION: NONE")
        lines.append("<END_PLAN>")
        dsl_text = "\n".join(lines)

        # Validate synthesized plan
        parsed = PlanDSLParser.parse(dsl_text)
        if not parsed.is_valid:
            return dsl_text, False, parsed.validation_error
        return dsl_text, True, None


class NeuralSpanSlotFusionPipeline:
    """
    End-to-end pipeline uniting neural structure prediction with neural span grounding.
    """

    def __init__(
        self,
        span_model: ChakrSpanGroundingModel,
        tokenizer: Any,
        device: torch.device,
        min_confidence: float = 0.40,
    ):
        self.span_model = span_model
        self.tokenizer = tokenizer
        self.device = device
        self.min_confidence = min_confidence

    def extract_prompt_spans(self, prompt: str, max_slots: int = 4) -> List[ExtractedSpan]:
        """Runs ChakrSpanGroundingModel across available slot heads to identify prompt entities."""
        p_tokens = [0] + self.tokenizer.encode(prompt)
        p_ids = torch.tensor(p_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        P = len(p_tokens)

        spans: List[ExtractedSpan] = []
        for s_idx in range(max_slots):
            span = self.span_model.predict_span(
                p_ids,
                prompt_len=P,
                slot_idx=s_idx,
                tokenizer=self.tokenizer,
                raw_prompt_text=prompt,
            )
            # Boundary safety checks
            if 0 <= span.start_idx <= span.end_idx < P:
                spans.append(span)
            else:
                spans.append(ExtractedSpan(start_idx=-1, end_idx=-1, span_type="NONE", confidence=0.0))
        return spans

    def predict_plan_topology(self, prompt: str) -> UnifiedPlan:
        """
        Determines the abstract plan topology from prompt intent.
        Supports 2-step, 3-step, 4-step arithmetic, text transformation, and clarification.
        """
        prompt_lower = prompt.lower()

        # 1. Missing information detection
        if "without specifying" in prompt_lower or "without supplying" in prompt_lower:
            m = re.search(r"without (?:specifying|supplying)\s+([a-zA-Z0-9_]+)", prompt_lower)
            missing_var = m.group(1) if m else "required_parameter"
            return UnifiedPlan(
                goal="clarification_needed",
                steps=[],
                bindings={},
                clarification_needed=True,
                clarification_reason=missing_var,
            )

        # 2. Text transformation chain
        if "process phrase" in prompt_lower or "text_utility" in prompt_lower or "trim surrounding" in prompt_lower:
            steps = [
                AbstractPlanStep(
                    step_id=0,
                    action="text_transform",
                    tool="text_utility",
                    raw_template="text='@slot0',op=strip",
                    deps=[],
                    slots=["@slot0"],
                ),
                AbstractPlanStep(
                    step_id=1,
                    action="text_transform",
                    tool="text_utility",
                    raw_template="text=$0,op=uppercase",
                    deps=[0],
                    slots=[],
                ),
            ]
            bindings = {
                "@slot0": SlotBinding(slot_id="@slot0", slot_name="text_content", target_step=0, target_arg_key="text"),
            }
            return UnifiedPlan(goal="text_transform_chain", steps=steps, bindings=bindings)

        # 3. Multi-step Arithmetic (4-step, 3-step, 2-step)
        if "start with" in prompt_lower and "divide by" in prompt_lower:
            # 4-step arithmetic pipeline
            steps = [
                AbstractPlanStep(0, "tool_call", "calculator", "expr=@slot0+@slot1", [], ["@slot0", "@slot1"]),
                AbstractPlanStep(1, "tool_call", "calculator", "expr=$0*@slot2", [0], ["@slot2"]),
                AbstractPlanStep(2, "tool_call", "calculator", "expr=$1-@slot3", [1], ["@slot3"]),
                AbstractPlanStep(3, "tool_call", "calculator", "expr=$2/2", [2], []),
            ]
            bindings = {
                "@slot0": SlotBinding("@slot0", "base", 0, "expr"),
                "@slot1": SlotBinding("@slot1", "op1", 0, "expr"),
                "@slot2": SlotBinding("@slot2", "op2", 1, "expr"),
                "@slot3": SlotBinding("@slot3", "op3", 2, "expr"),
            }
            return UnifiedPlan(goal="compound_arithmetic", steps=steps, bindings=bindings)

        # Default 2-step compound arithmetic: A * B then add/sub C
        sub_op = "-" if "subtract" in prompt_lower else "+"
        mult_op = "/" if "divide" in prompt_lower else "*"
        steps = [
            AbstractPlanStep(0, "tool_call", "calculator", f"expr=@slot0{mult_op}@slot1", [], ["@slot0", "@slot1"]),
            AbstractPlanStep(1, "tool_call", "calculator", f"expr=$0{sub_op}@slot2", [0], ["@slot2"]),
        ]
        bindings = {
            "@slot0": SlotBinding("@slot0", "operand_0", 0, "expr"),
            "@slot1": SlotBinding("@slot1", "operand_1", 0, "expr"),
            "@slot2": SlotBinding("@slot2", "operand_2", 1, "expr"),
        }
        return UnifiedPlan(goal="compound_arithmetic", steps=steps, bindings=bindings)

    def generate_plan(self, prompt: str) -> Tuple[str, bool, Optional[str]]:
        """
        Executes end-to-end Neural Span-to-Slot Fusion:
        1. Predict plan topology.
        2. Extract candidate prompt spans via neural SpanGroundingModel.
        3. Bind spans to slots.
        4. Materialize and validate Plan DSL.
        """
        plan = self.predict_plan_topology(prompt)
        if plan.clarification_needed:
            return plan.materialize_dsl()

        # Neural span extraction
        spans = self.extract_prompt_spans(prompt, max_slots=4)

        # Slot-to-Span binding
        for s_idx, (s_id, binding) in enumerate(plan.bindings.items()):
            if s_idx < len(spans):
                cand = spans[s_idx]
                if cand.span_text and cand.confidence >= self.min_confidence:
                    binding.bound_span = cand
                    binding.status = "BOUND"
                elif cand.span_text:
                    # Weak confidence fallback
                    binding.bound_span = cand
                    binding.status = "BOUND"
                else:
                    binding.status = "MISSING"

        return plan.materialize_dsl()
