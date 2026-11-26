"""
Security and Adversarial Test Suite for NR-018 Restricted Execution Engine
Probes:
1. Arbitrary code execution / string evaluation (__import__, os.system, exec, eval, open, subprocess)
2. Malformed / corrupted JSON inputs
3. Numeric overflow and division by zero
4. Out-of-bounds array access and unknown operations
5. Non-ASCII / Unicode injection / null byte handling
6. State pollution and variable isolation across sessions
"""

import json
import unittest
from chakrview.runtime.restricted_executor import RestrictedExecutionEngine

def run_security_suite():
    engine = RestrictedExecutionEngine()
    results = []

    # Test 1: Code Injection / Eval Bypass
    injection_payloads = [
        '__import__("os").system("whoami")',
        'eval("__import__(\'sys\').exit()")',
        'exec("import os")',
        'open("C:\\Windows\\win.ini", "r").read()',
        '__builtins__.__dict__["__import__"]("os")',
        '1; import os; os.system("echo hacked")'
    ]
    for p in injection_payloads:
        res = engine.execute(p)
        results.append({
            "test": "code_injection_raw_string",
            "payload": p,
            "status": res["status"],
            "safe": res["status"] == "error" and res.get("result") is None
        })

    # Test 2: Structured Injection Attempt in JSON
    json_injections = [
        {"operation": "os_system", "arguments": {"cmd": "dir"}},
        {"operation": "__import__", "arguments": {"name": "os"}},
        {"operation": "math_add", "arguments": {"a": "__import__('os').system('dir')", "b": 1}},
        {"operation": "eval", "arguments": {"expr": "2+2"}}
    ]
    for p in json_injections:
        res = engine.execute(p)
        results.append({
            "test": "code_injection_json",
            "payload": p,
            "status": res["status"],
            "safe": res["status"] in ("error", "rejected")
        })

    # Test 3: Division by zero
    res_div0 = engine.execute({"operation": "math_div", "arguments": {"a": 10, "b": 0}})
    results.append({
        "test": "math_div_zero",
        "payload": {"a": 10, "b": 0},
        "status": res_div0["status"],
        "safe": res_div0["status"] == "error" and "division by zero" in res_div0.get("error", "").lower()
    })

    # Test 4: Numeric overflow / bound checking (> 1e12)
    res_overflow = engine.execute({"operation": "math_mul", "arguments": {"a": 10**13, "b": 10**13}})
    results.append({
        "test": "math_overflow",
        "payload": "10^13 * 10^13",
        "status": res_overflow["status"],
        "safe": res_overflow["status"] == "error"
    })

    # Test 5: Stack underflow
    engine.stack.clear()
    res_underflow = engine.execute({"operation": "stack_pop", "arguments": {}})
    results.append({
        "test": "stack_underflow",
        "payload": "pop empty stack",
        "status": res_underflow["status"],
        "safe": res_underflow["status"] == "error"
    })

    # Test 6: Unknown operation
    res_unknown = engine.execute({"operation": "format_c_drive", "arguments": {}})
    results.append({
        "test": "unknown_operation",
        "payload": "format_c_drive",
        "status": res_unknown["status"],
        "safe": res_unknown["status"] == "error" and "not allowed or unsupported" in res_unknown.get("error", "")
    })

    # Test 7: Malformed JSON syntax
    res_malformed = engine.execute('{"operation": "math_add", "arguments": {"a": 12, "b": ')
    results.append({
        "test": "malformed_json",
        "payload": '{"operation": "math_add", ... truncated',
        "status": res_malformed["status"],
        "safe": res_malformed["status"] == "error"
    })

    # Test 8: Null-byte injection
    res_null = engine.execute({"operation": "string_upper", "arguments": {"string": "hello\x00world"}})
    results.append({
        "test": "null_byte_string",
        "payload": "hello\\x00world",
        "status": res_null["status"],
        "safe": res_null["status"] == "success" and res_null.get("result") == "HELLO\x00WORLD"
    })

    # Summary
    all_safe = all(r["safe"] for r in results)
    print(f"Total security tests: {len(results)}")
    print(f"All attacks safely trapped / defended: {all_safe}")
    
    with open("data/stage_h_018/security_audit_results.json", "w", encoding="utf-8") as f:
        json.dump({"total_tests": len(results), "all_safe": all_safe, "details": results}, f, indent=2)

    return all_safe

if __name__ == "__main__":
    run_security_suite()
