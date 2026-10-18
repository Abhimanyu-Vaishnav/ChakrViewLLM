"""
Batch 115 / 120: Strict Evaluation Semantics & Multi-Metric Evaluation Module.
Implements the 11 required evaluation dimensions:
1. Parse rate: grammar matches valid JSON tool call
2. Schema validity: required fields and types, no prohibited fields
3. Tool accuracy: tool == "calculator"
4. Operation accuracy: op == "add"
5. Operand A accuracy: operand A matches expected input
6. Operand B accuracy: operand B matches expected input
7. Exact call accuracy: tool, op, a, b all simultaneously correct
8. Execution success: allowlisted runtime accepts and executes call
9. End-to-end correctness: exact call + execution success + exact result match
10. False-positive tool-call rate: any tool call attempt on non-tool prompt
    - Separate: valid_false_positive_rate vs malformed_false_positive_rate
11. Collapse rate: frequency of the most common normalized output
"""
import re
import json
import collections
from typing import Any, Dict, List, Optional, Tuple, Union

from experiments.batch91_94_tool_calling_comparator.tools.calculator import (
    validate_and_execute_tool_call,
    SUPPORTED_TOOLS,
    SUPPORTED_OPERATIONS,
    MIN_OPERAND_BOUND,
    MAX_OPERAND_BOUND,
)

def detect_tool_call_attempt(text: str) -> bool:
    """
    Detects if candidate text ATTEMPTS a tool call, even if malformed or truncated.
    Checks for keywords like 'TOOL_CALL', '{"tool":', '{"op":', 'calculator'.
    """
    if not isinstance(text, str):
        return False
    lower = text.lower()
    if "tool_call" in lower:
        return True
    if '"tool":' in lower or '"tool" :' in lower or "'tool':" in lower:
        return True
    if '{"tool"' in lower or "calculator" in lower:
        # Check if formatted as a call attempt rather than natural language explanation
        if "{" in text and "}" in text:
            return True
        if lower.strip().startswith("tool_call") or lower.strip().startswith("{"):
            return True
    return False

def parse_tool_grammar(text: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    1. Parse rate: Checks if candidate text contains valid 'TOOL_CALL: {...}' grammar.
    Returns (is_parsed, parsed_dict, error_detail).
    """
    if not isinstance(text, str):
        return False, None, "Input must be a string."
    clean = text.strip()
    m = re.search(r"TOOL_CALL:\s*(\{.*?\})", clean, re.DOTALL)
    if not m:
        m2 = re.search(r"^(\{.*?\})", clean, re.DOTALL)
        if not m2:
            return False, None, "No TOOL_CALL pattern or JSON object found."
        json_str = m2.group(1).strip()
    else:
        json_str = m.group(1).strip()
        
    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return False, None, "Parsed JSON is not an object."
        return True, data, ""
    except Exception as e:
        return False, None, f"JSON decode failed: {e}"

def validate_schema(data: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    2. Schema validity: checks required fields (tool, op, a, b), prohibited fields, and exact integer types.
    """
    if not isinstance(data, dict):
        return False, "Not a dictionary."
    
    required_keys = {"tool", "op", "a", "b"}
    if not required_keys.issubset(data.keys()):
        missing = required_keys - set(data.keys())
        return False, f"Missing required fields: {missing}"
    
    allowed_keys = required_keys
    extra = set(data.keys()) - allowed_keys
    if extra:
        return False, f"Prohibited extra fields: {extra}"
        
    a = data["a"]
    b = data["b"]
    if not isinstance(a, int) or isinstance(a, bool) or not isinstance(b, int) or isinstance(b, bool):
        return False, "Operands 'a' and 'b' must be integers."
        
    if not isinstance(data["tool"], str) or not isinstance(data["op"], str):
        return False, "Fields 'tool' and 'op' must be strings."
        
    return True, ""

def evaluate_single_prediction(
    prediction_text: str,
    expected_record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluates one candidate prediction against its expected ground-truth record.
    Returns boolean flags for all dimensions.
    """
    requires_tool = expected_record.get("requires_tool", True)
    expected_operands = expected_record.get("operands") # [a, b] if tool call
    
    attempted_tool = detect_tool_call_attempt(prediction_text)
    is_parsed, parsed_data, parse_err = parse_tool_grammar(prediction_text)
    is_schema, schema_err = validate_schema(parsed_data) if is_parsed else (False, "Not parsed.")
    
    is_tool_correct = False
    is_op_correct = False
    is_a_correct = False
    is_b_correct = False
    is_exact_call = False
    is_executed = False
    is_e2e_correct = False
    
    if not requires_tool:
        # Non-tool prompt (clarification, out-of-domain, underspecified)
        is_spurious = attempted_tool or is_parsed or is_schema
        is_valid_spurious = is_parsed and is_schema
        is_malformed_spurious = attempted_tool and not (is_parsed and is_schema)
        return {
            "requires_tool": False,
            "attempted_tool": attempted_tool,
            "parsed": is_parsed,
            "schema_valid": is_schema,
            "false_positive": is_spurious,
            "valid_false_positive": is_valid_spurious,
            "malformed_false_positive": is_malformed_spurious,
            "clean_output": prediction_text.split("<|im_end|>")[0].strip(),
            "correct": not is_spurious,
        }
    
    # Tool prompt
    exp_a, exp_b = expected_operands[0], expected_operands[1]
    
    if is_schema:
        is_tool_correct = (parsed_data["tool"] == "calculator")
        is_op_correct = (parsed_data["op"] == "add")
        is_a_correct = (parsed_data["a"] == exp_a)
        is_b_correct = (parsed_data["b"] == exp_b)
        is_exact_call = (is_tool_correct and is_op_correct and is_a_correct and is_b_correct)
        
        # Execute in allowlisted runtime
        exec_res = validate_and_execute_tool_call(prediction_text)
        is_executed = bool(exec_res.get("executed", False))
        
        # End-to-end correctness requires:
        # 1. exact call is true (tool, op, a, b all match ground truth)
        # 2. execution succeeded in the secure allowlisted runtime
        # 3. execution result strictly equals expected ground-truth addition
        if is_exact_call and is_executed and exec_res.get("result") == (exp_a + exp_b):
            is_e2e_correct = True
    
    return {
        "requires_tool": True,
        "operands": [exp_a, exp_b],
        "attempted_tool": attempted_tool,
        "parsed": is_parsed,
        "schema_valid": is_schema,
        "tool_correct": is_tool_correct,
        "op_correct": is_op_correct,
        "a_correct": is_a_correct,
        "b_correct": is_b_correct,
        "exact_call": is_exact_call,
        "executed": is_executed,
        "e2e_correct": is_e2e_correct,
        "parsed_payload": parsed_data,
        "clean_output": prediction_text.split("<|im_end|>")[0].strip(),
    }

def compute_aggregate_metrics(eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes summary metrics over a collection of evaluated prediction records.
    Distinguishes tool examples from negative/non-tool controls.
    """
    tool_recs = [r for r in eval_records if r.get("requires_tool", True)]
    non_tool_recs = [r for r in eval_records if not r.get("requires_tool", True)]
    
    n_tool = len(tool_recs)
    n_nontool = len(non_tool_recs)
    
    res = {
        "total_examples": len(eval_records),
        "tool_examples_count": n_tool,
        "non_tool_examples_count": n_nontool,
    }
    
    if n_tool > 0:
        outputs = [r["clean_output"] for r in tool_recs]
        counts = collections.Counter(outputs)
        most_common_out, max_c = counts.most_common(1)[0]
        
        res["tool_metrics"] = {
            "denominator": n_tool,
            "parse_rate": round(sum(1 for r in tool_recs if r["parsed"]) / n_tool, 4),
            "schema_validity_rate": round(sum(1 for r in tool_recs if r["schema_valid"]) / n_tool, 4),
            "tool_accuracy": round(sum(1 for r in tool_recs if r["tool_correct"]) / n_tool, 4),
            "operation_accuracy": round(sum(1 for r in tool_recs if r["op_correct"]) / n_tool, 4),
            "operand_a_accuracy": round(sum(1 for r in tool_recs if r["a_correct"]) / n_tool, 4),
            "operand_b_accuracy": round(sum(1 for r in tool_recs if r["b_correct"]) / n_tool, 4),
            "exact_call_accuracy": round(sum(1 for r in tool_recs if r["exact_call"]) / n_tool, 4),
            "execution_success_rate": round(sum(1 for r in tool_recs if r["executed"]) / n_tool, 4),
            "e2e_correctness_rate": round(sum(1 for r in tool_recs if r["e2e_correct"]) / n_tool, 4),
            "unique_outputs_count": len(counts),
            "collapse_rate": round(max_c / n_tool, 4),
            "most_common_output": most_common_out,
        }
        
    if n_nontool > 0:
        res["non_tool_metrics"] = {
            "denominator": n_nontool,
            "total_false_positive_rate": round(sum(1 for r in non_tool_recs if r["false_positive"]) / n_nontool, 4),
            "valid_false_positive_rate": round(sum(1 for r in non_tool_recs if r.get("valid_false_positive", False)) / n_nontool, 4),
            "malformed_false_positive_rate": round(sum(1 for r in non_tool_recs if r.get("malformed_false_positive", False)) / n_nontool, 4),
            "clean_refusal_rate": round(sum(1 for r in non_tool_recs if r["correct"]) / n_nontool, 4),
            "sample_outputs": [r["clean_output"] for r in non_tool_recs[:3]],
        }
        
    return res
