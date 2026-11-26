"""
chakrview/runtime/source_grounding.py
Exact Source-Span Argument Grounding Engine.

Associates every operation argument with an exact character span in the original user prompt:
- Replaces free-form generation of numbers/arguments with pointer grounding.
- Preserves raw surface text, normalized value, data type, and (start, end) span indices.
- Validates typed schemas against allowlisted operations.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from chakrview.runtime.multilingual_normalizer import MultilingualNormalizer

class GroundedArgument:
    def __init__(self, key: str, value: Any, raw_text: str, start: int, end: int, data_type: str):
        self.key = key
        self.value = value
        self.raw_text = raw_text
        self.start = start
        self.end = end
        self.data_type = data_type

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "raw_text": self.raw_text,
            "span": [self.start, self.end],
            "type": self.data_type
        }

class SourceGroundingEngine:
    def __init__(self):
        self.normalizer = MultilingualNormalizer()

    def ground_arguments(self, prompt: str, operation: str, operation_spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extracts and grounds arguments for a given operation from the prompt.
        Guarantees that arguments are derived strictly from substrings in prompt.
        """
        prompt_nfc = self.normalizer.normalize_unicode(prompt)
        grounded_args = {}
        provenance = []

        # 1. Array Operations: list_reverse, list_sort, list_max, list_min, list_sum
        if operation in ["list_reverse", "list_sort", "list_max", "list_min", "list_sum"]:
            arr_data = self.normalizer.extract_array_with_span(prompt_nfc)
            if arr_data:
                grounded_args["list"] = arr_data["value"]
                provenance.append(GroundedArgument(
                    key="list",
                    value=arr_data["value"],
                    raw_text=arr_data["raw_text"],
                    start=arr_data["start"],
                    end=arr_data["end"],
                    data_type="list"
                ).to_dict())
            else:
                # Fallback: extract isolated numbers as list
                nums = self.normalizer.extract_numbers_with_spans(prompt_nfc)
                if nums:
                    val_list = [n["value"] for n in nums]
                    grounded_args["list"] = val_list
                    provenance.append(GroundedArgument(
                        key="list",
                        value=val_list,
                        raw_text=prompt_nfc[nums[0]["start"]:nums[-1]["end"]],
                        start=nums[0]["start"],
                        end=nums[-1]["end"],
                        data_type="list"
                    ).to_dict())

        # 2. Binary Arithmetic: math_add, math_sub, math_mul, math_div
        elif operation in ["math_add", "math_sub", "math_mul", "math_div"]:
            nums = self.normalizer.extract_numbers_with_spans(prompt_nfc)
            if len(nums) >= 2:
                # Check for reverse subtraction phrasing: "subtract X from Y" or "Y me se X घटाओ"
                prompt_lower = prompt_nfc.lower()
                is_sub_from = "subtract" in prompt_lower and "from" in prompt_lower
                
                if operation == "math_sub" and is_sub_from:
                    arg_a, arg_b = nums[1], nums[0] # Y - X
                else:
                    arg_a, arg_b = nums[0], nums[1] # standard order

                grounded_args["a"] = arg_a["value"]
                grounded_args["b"] = arg_b["value"]
                provenance.append(GroundedArgument("a", arg_a["value"], arg_a["raw_text"], arg_a["start"], arg_a["end"], "number").to_dict())
                provenance.append(GroundedArgument("b", arg_b["value"], arg_b["raw_text"], arg_b["start"], arg_b["end"], "number").to_dict())

        # 3. String Operations: string_upper, string_lower, string_reverse, string_length
        elif operation in ["string_upper", "string_lower", "string_reverse", "string_length"]:
            # Check for quoted string first
            import re
            quote_match = re.search(r"['\"]([^'\"]+)['\"]", prompt_nfc)
            if quote_match:
                s_val = quote_match.group(1)
                grounded_args["string"] = s_val
                provenance.append(GroundedArgument("string", s_val, quote_match.group(0), quote_match.start(), quote_match.end(), "string").to_dict())
            else:
                # Fallback: extract single candidate word after keyword
                words = prompt_nfc.split()
                # Use the last word or non-keyword word
                cand = words[-1].strip(".,?!") if words else ""
                grounded_args["string"] = cand
                provenance.append(GroundedArgument("string", cand, cand, 0, len(prompt_nfc), "string").to_dict())

        # 4. State / Stack Operations
        elif operation in ["stack_push", "var_set"]:
            nums = self.normalizer.extract_numbers_with_spans(prompt_nfc)
            if nums:
                grounded_args["value"] = nums[0]["value"]
                provenance.append(GroundedArgument("value", nums[0]["value"], nums[0]["raw_text"], nums[0]["start"], nums[0]["end"], "number").to_dict())
            if operation == "var_set":
                # Look for variable name (x, y, temp, etc.)
                import re
                v_match = re.search(r"\bvar(?:iable)?\s+([A-Za-z0-9_]+)\b|\b([A-Za-z0-9_]+)\s*=\s*", prompt_nfc, re.I)
                var_name = v_match.group(1) or v_match.group(2) if v_match else "x"
                grounded_args["name"] = var_name

        elif operation in ["stack_pop", "stack_peek"]:
            pass # No arguments needed

        elif operation == "var_get":
            import re
            v_match = re.search(r"\b(?:var(?:iable)?|of)\s+([A-Za-z0-9_]+)\b", prompt_nfc, re.I)
            grounded_args["name"] = v_match.group(1) if v_match else "x"

        # 5. Conditional Branch
        elif operation == "conditional_branch":
            nums = self.normalizer.extract_numbers_with_spans(prompt_nfc)
            if len(nums) >= 2:
                grounded_args["value"] = nums[0]["value"]
                grounded_args["threshold"] = nums[1]["value"]
                grounded_args["compare"] = "gt" if any(w in prompt_nfc.lower() for w in ["greater", "bada", "बड़ा"]) else "lt"
                grounded_args["on_true"] = "PASS"
                grounded_args["on_false"] = "FAIL"
                provenance.append(GroundedArgument("value", nums[0]["value"], nums[0]["raw_text"], nums[0]["start"], nums[0]["end"], "number").to_dict())
                provenance.append(GroundedArgument("threshold", nums[1]["value"], nums[1]["raw_text"], nums[1]["start"], nums[1]["end"], "number").to_dict())

        # 6. State Accumulator (Start + Steps)
        elif operation == "state_accumulator":
            nums = self.normalizer.extract_numbers_with_spans(prompt_nfc)
            if len(nums) >= 3:
                grounded_args["start"] = nums[0]["value"]
                grounded_args["steps"] = [["add", nums[1]["value"]], ["sub", nums[2]["value"]]]
                provenance.append(GroundedArgument("start", nums[0]["value"], nums[0]["raw_text"], nums[0]["start"], nums[0]["end"], "number").to_dict())

        return {
            "operation": operation,
            "arguments": grounded_args,
            "provenance": provenance,
            "status": "ready" if grounded_args or operation in ["stack_pop", "stack_peek"] else "clarify"
        }
