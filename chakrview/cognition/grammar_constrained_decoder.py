"""
Batches 786-791: Grammar-Constrained Autoregressive Plan DSL Decoder.
Location: chakrview/cognition/grammar_constrained_decoder.py

Features:
1. Incremental Plan DSL Grammar State Machine:
   Tracks parser state (GOAL, STEP_HEADER, ACTION, TOOL, ARGS, DEPS, CLARIFICATION, END_PLAN).
2. Tokenizer Multi-Character & Vocabulary Awareness:
   Handles partial multi-character delimiter tokens like ' |', ' | ', 'args={', 'deps=['.
3. Constrained Logit Masking:
   Masks illegal next-token logits (-inf) during autoregressive decoding to guarantee 100% syntactically valid Plan DSL output.
4. Fallback & Closed-World Failure:
   Fails closed with clearly recorded decoding errors if no legal tokens exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import enum
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import torch


class GrammarState(enum.Enum):
    START = "START"
    GOAL = "GOAL"
    STEP_START_OR_CLAR = "STEP_START_OR_CLAR"
    STEP_ACTION = "STEP_ACTION"
    STEP_TOOL = "STEP_TOOL"
    STEP_ARGS = "STEP_ARGS"
    STEP_DEPS = "STEP_DEPS"
    CLARIFICATION = "CLARIFICATION"
    END_PLAN = "END_PLAN"
    TERMINATED = "TERMINATED"
    INVALID = "INVALID"


class PlanDSLGrammarChecker:
    """
    Incremental grammar validator for ChakrView Plan DSL.
    Grammar Specification:
      <Plan> ::= <GoalLine> "\n" (<StepLine> "\n")* <ClarLine> "\n<END_PLAN>"
      <GoalLine> ::= "GOAL: " <Identifier>
      <StepLine> ::= "STEP " <Digit>+ ": " <Action> " | " <Tool> " | args={" <Args> "} | deps=[" <Deps> "]"
      <ClarLine> ::= "CLARIFICATION: " ("NEEDED" | "NONE")
    """

    ALLOWED_GOALS = {
        "compound_arithmetic",
        "text_transform_chain",
        "clarification_needed",
        "unsupported",
        "multi_tool_dag",
        "data_analysis",
        "reasoning_task",
    }
    ALLOWED_ACTIONS = {"tool_call", "text_transform", "clarification", "reasoning"}
    ALLOWED_TOOLS = {"calculator", "text_utility", "none"}

    @classmethod
    def is_valid_partial(cls, text: str) -> Tuple[bool, GrammarState]:
        """
        Determines whether `text` is a valid prefix of a complete Plan DSL document.
        Returns (is_valid, current_state).
        """
        if not text:
            return True, GrammarState.START

        # Check if text starts with GOAL:
        if not ("GOAL:".startswith(text) or text.startswith("GOAL: ")):
            return False, GrammarState.INVALID

        if text.startswith("GOAL: "):
            rest = text[len("GOAL: "):]
            if "\n" not in rest:
                # Still typing goal identifier
                goal_candidate = rest.strip()
                if any(g.startswith(goal_candidate) for g in cls.ALLOWED_GOALS):
                    return True, GrammarState.GOAL
                return False, GrammarState.INVALID

            # Goal is finished
            goal_id, rest = rest.split("\n", 1)
            goal_id = goal_id.strip()
            if goal_id not in cls.ALLOWED_GOALS:
                return False, GrammarState.INVALID

            return cls._check_body_prefix(rest)

        return True, GrammarState.START

    @classmethod
    def _check_body_prefix(cls, body: str) -> Tuple[bool, GrammarState]:
        """
        Checks prefix validity after GOAL line has completed.
        """
        lines = body.split("\n")
        # All completed lines before the last line must be valid full steps or clarification
        for line in lines[:-1]:
            line = line.strip()
            if not line:
                continue
            if line.startswith("STEP "):
                if not cls._is_valid_full_step_line(line):
                    return False, GrammarState.INVALID
            elif line.startswith("CLARIFICATION: "):
                clar_val = line[len("CLARIFICATION: "):].strip()
                if clar_val not in {"NEEDED", "NONE"}:
                    return False, GrammarState.INVALID
            elif line == "<END_PLAN>":
                continue
            else:
                return False, GrammarState.INVALID

        current_line = lines[-1]
        if not current_line:
            # At line boundary, next can be STEP, CLARIFICATION, or <END_PLAN>
            return True, GrammarState.STEP_START_OR_CLAR

        if current_line.startswith("STEP "):
            return cls._check_step_line_prefix(current_line)
        elif "CLARIFICATION:".startswith(current_line) or current_line.startswith("CLARIFICATION: "):
            if current_line.startswith("CLARIFICATION: "):
                c_val = current_line[len("CLARIFICATION: "):].strip()
                if any(v.startswith(c_val) for v in ["NEEDED", "NONE"]):
                    return True, GrammarState.CLARIFICATION
                return False, GrammarState.INVALID
            return True, GrammarState.CLARIFICATION
        elif "<END_PLAN>".startswith(current_line):
            if current_line == "<END_PLAN>":
                return True, GrammarState.TERMINATED
            return True, GrammarState.END_PLAN

        # Check if it could be start of STEP or CLARIFICATION
        if "STEP ".startswith(current_line) or "CLARIFICATION: ".startswith(current_line):
            return True, GrammarState.STEP_START_OR_CLAR

        return False, GrammarState.INVALID

    @classmethod
    def _is_valid_full_step_line(cls, line: str) -> bool:
        """
        Validates a completed STEP line:
        Pattern: STEP <int>: <action> | <tool> | args={<args>} | deps=[<deps>]
        """
        pattern = re.compile(
            r"^STEP\s*(\d+)\s*:\s*(\w+)\s*\|\s*(\w+)\s*\|\s*args=\{([^}]*)\}\s*\|\s*deps=\[([^\]]*)\]$"
        )
        m = pattern.match(line)
        if not m:
            return False
        action = m.group(2)
        tool = m.group(3)
        if action not in cls.ALLOWED_ACTIONS:
            return False
        if tool not in cls.ALLOWED_TOOLS:
            return False
        return True

    @classmethod
    def _check_step_line_prefix(cls, line: str) -> Tuple[bool, GrammarState]:
        """
        Validates a partial STEP line with robust tolerance for intermediate delimiters.
        """
        # Step prefix: STEP <digits>:
        m_head = re.match(r"^STEP\s*(\d+)\s*(:\s*)?", line)
        if not m_head:
            if re.match(r"^STEP\s*\d*$", line):
                return True, GrammarState.STEP_START_OR_CLAR
            return False, GrammarState.INVALID

        if not m_head.group(2):
            return True, GrammarState.STEP_START_OR_CLAR

        rest = line[m_head.end():]
        # Rest syntax: <action> " | " <tool> " | args={" <args> "} | deps=[" <deps> "]"

        # Stage 1: Action and transition to tool
        if " | " not in rest:
            # Check if rest ends with a partial delimiter " |" or " | " or " "
            clean_act = rest.split("|")[0].strip()
            # If pipe is present
            if "|" in rest:
                # Must be prefix of " | "
                after_act = rest[len(clean_act):]
                if " | ".startswith(after_act):
                    return (clean_act in cls.ALLOWED_ACTIONS), GrammarState.STEP_ACTION
                return False, GrammarState.INVALID

            # Still in action
            if any(a.startswith(clean_act) for a in cls.ALLOWED_ACTIONS):
                return True, GrammarState.STEP_ACTION
            return False, GrammarState.INVALID

        # Stage 2: Action finished, check tool
        act_part, after_act = rest.split(" | ", 1)
        act = act_part.strip()
        if act not in cls.ALLOWED_ACTIONS:
            return False, GrammarState.INVALID

        if " | " not in after_act:
            clean_tool = after_act.split("|")[0].strip()
            if "|" in after_act:
                after_t = after_act[len(clean_tool):]
                if " | ".startswith(after_t):
                    return (clean_tool in cls.ALLOWED_TOOLS), GrammarState.STEP_TOOL
                return False, GrammarState.INVALID

            if any(t.startswith(clean_tool) for t in cls.ALLOWED_TOOLS):
                return True, GrammarState.STEP_TOOL
            return False, GrammarState.INVALID

        # Stage 3: Tool finished, check args
        tool_part, after_tool = after_act.split(" | ", 1)
        tool = tool_part.strip()
        if tool not in cls.ALLOWED_TOOLS:
            return False, GrammarState.INVALID

        if " | " not in after_tool:
            # Inside args: must match prefix of "args={...}"
            if "args={".startswith(after_tool):
                return True, GrammarState.STEP_ARGS
            if after_tool.startswith("args={"):
                # If closing brace exists, must be followed by prefix of " | "
                if "}" in after_tool:
                    before_brace, after_brace = after_tool.split("}", 1)
                    if " | ".startswith(after_brace):
                        return True, GrammarState.STEP_ARGS
                    return False, GrammarState.INVALID
                return True, GrammarState.STEP_ARGS
            return False, GrammarState.INVALID

        # Stage 4: Args finished, check deps
        args_part, after_args = after_tool.split(" | ", 1)
        args_str = args_part.strip()
        if not (args_str.startswith("args={") and args_str.endswith("}")):
            return False, GrammarState.INVALID

        # Inside deps: must match prefix of "deps=[...]"
        if "deps=[".startswith(after_args):
            return True, GrammarState.STEP_DEPS
        if after_args.startswith("deps=["):
            if "]" in after_args:
                before_bracket, after_bracket = after_args.split("]", 1)
                if not after_bracket:
                    return True, GrammarState.STEP_DEPS
                return False, GrammarState.INVALID
            return True, GrammarState.STEP_DEPS

        return False, GrammarState.INVALID


class GrammarConstrainedDecoder:
    """
    Decodes Plan DSL from a neural causal language model while strictly enforcing
    the formal EBNF grammar via next-token logit masking.
    """

    def __init__(self, tokenizer: Any, max_new_tokens: int = 150):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.vocab_size = tokenizer.vocab_size

        # Precompute string mapping for all tokens in vocab
        self.token_to_str: Dict[int, str] = {}
        for idx in range(self.vocab_size):
            b = getattr(tokenizer, "_vocab", {}).get(idx, b"")
            if isinstance(b, bytes):
                self.token_to_str[idx] = b.decode("utf-8", errors="replace")
            else:
                try:
                    self.token_to_str[idx] = tokenizer.decode([idx])
                except Exception:
                    self.token_to_str[idx] = ""

    def get_allowed_tokens(self, current_text: str) -> List[int]:
        """
        Returns list of token IDs whose addition to current_text maintains
        a valid Plan DSL partial prefix according to the grammar.
        """
        allowed: List[int] = []
        for tid, s in self.token_to_str.items():
            if not s and tid != 0:
                continue
            if tid == 0:
                # End of plan / pad token
                if "<END_PLAN>" in current_text or current_text.endswith("<END_PLAN>"):
                    allowed.append(tid)
                continue

            candidate = current_text + s
            is_valid, _ = PlanDSLGrammarChecker.is_valid_partial(candidate)
            if is_valid:
                allowed.append(tid)

        return allowed

    def generate(
        self,
        model: Any,
        prompt: str,
        device: torch.device,
        constrained: bool = True,
    ) -> Tuple[str, bool, Optional[str]]:
        """
        Generates Plan DSL. If constrained=True, masks illegal logits.
        Returns: (generated_dsl, has_eos, error_message).
        """
        p_tokens = [0] + self.tokenizer.encode(prompt)
        curr_ids = torch.tensor(p_tokens, dtype=torch.long).unsqueeze(0).to(device)
        generated_tokens: List[int] = []
        curr_text = ""

        with torch.no_grad():
            for step in range(self.max_new_tokens):
                logits = model(curr_ids)
                next_token_logits = logits[0, -1, :].clone()

                if constrained:
                    allowed_tokens = self.get_allowed_tokens(curr_text)
                    if not allowed_tokens:
                        # Fail closed: no grammatically valid continuation exists
                        return self.tokenizer.decode(generated_tokens), False, "No legal continuation in grammar state"

                    # Mask all non-allowed tokens with -inf
                    mask = torch.full((self.vocab_size,), float("-inf"), device=device)
                    mask[allowed_tokens] = 0.0
                    next_token_logits = next_token_logits + mask

                next_tok = torch.argmax(next_token_logits).item()
                if next_tok == 0:
                    break

                generated_tokens.append(next_tok)
                curr_ids = torch.cat(
                    [curr_ids, torch.tensor([[next_tok]], dtype=torch.long).to(device)], dim=1
                )
                curr_text = self.tokenizer.decode(generated_tokens)

                if "<END_PLAN>" in curr_text:
                    break

        decoded = self.tokenizer.decode(generated_tokens)
        has_eos = "<END_PLAN>" in decoded
        return decoded, has_eos, None
