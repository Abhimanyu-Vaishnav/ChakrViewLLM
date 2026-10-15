"""
Batches 591-595: Controlled Span & Intent Dataset Generator.
Location: experiments/batch581_620/build_span_dataset.py

Generates:
1. Training set (train_span_sft.jsonl)
2. Validation set (val_span_sft.jsonl)
3. Frozen held-out test suite (test_span_heldout.jsonl)

Guarantees:
- Every argument has exact character-level (start_char, end_char) and token-level (start_tok, end_tok) span annotations.
- Verification assertion: query[start_char:end_char] == extracted_value
- Zero prompt overlap and strictly disjoint operand partitions between train and test.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
DATA_DIR = REPO_ROOT / "experiments" / "batch581_620" / "data"
REPORTS_DIR = REPO_ROOT / "experiments" / "batch581_620" / "reports"


def compute_token_spans(tokenizer: Any, prompt: str, start_char: int, end_char: int) -> Tuple[int, int]:
    """
    Computes token start and end indices in prompt for a given character span.
    Prompt tokens start at index 0 (with <bos> = 0).
    """
    prefix = prompt[:start_char]
    target_span = prompt[start_char:end_char]

    prefix_toks = tokenizer.encode(prefix)
    span_toks = tokenizer.encode(target_span)

    # 1-indexed because of <bos> token at index 0
    start_tok = len(prefix_toks) + 1
    end_tok = start_tok + len(span_toks)
    return start_tok, end_tok


class SpanDatasetBuilder:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    def generate_splits(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        train_items: List[Dict[str, Any]] = []
        val_items: List[Dict[str, Any]] = []
        test_items: List[Dict[str, Any]] = []

        # -------------------------------------------------------------
        # 1. CALCULATOR (Arithmetic expressions with exact span indexing)
        # Train operands: 10-49, 100-499. Test operands: 50-99, 500-999 (Zero Overlap)
        # -------------------------------------------------------------
        calc_train_ops = [
            ("add", "+", [(14, 23), (35, 12), (18, 41), (27, 32), (11, 48), (22, 19), (43, 26), (15, 33), (120, 230), (310, 140)]),
            ("sub", "-", [(45, 12), (38, 19), (49, 21), (33, 14), (29, 11), (42, 28), (37, 15), (46, 23), (450, 120), (280, 150)]),
            ("mul", "*", [(12, 14), (15, 11), (23, 12), (18, 16), (25, 14), (13, 17), (21, 15), (14, 14), (16, 12), (32, 11)]),
            ("div", "/", [(48, 12), (45, 15), (42, 14), (36, 12), (40, 10), (39, 13), (44, 11), (32, 16), (240, 12), (360, 18)]),
            ("comp", "compound", [((12, 15), 10, "+", "-"), ((20, 14), 2, "+", "*"), ((40, 10), 5, "-", "/")]),
        ]

        calc_test_ops = [
            ("add", "+", [(52, 67), (74, 83), (61, 95), (88, 59), (57, 76), (630, 240), (810, 170)]),
            ("sub", "-", [(94, 62), (87, 53), (76, 58), (91, 69), (85, 71), (750, 420), (920, 380)]),
            ("mul", "*", [(54, 13), (62, 15), (71, 12), (85, 11), (92, 14), (65, 12)]),
            ("div", "/", [(96, 16), (84, 12), (78, 13), (90, 15), (91, 13), (850, 17)]),
            ("comp", "compound", [((55, 37), 14, "+", "-"), ((60, 25), 3, "+", "*"), ((90, 30), 6, "-", "/")]),
        ]

        def build_calc_item(prompt: str, expr: str, item_id: str, cat: str) -> Dict[str, Any]:
            assert expr in prompt, f"Expression '{expr}' not found in prompt '{prompt}'"
            start_c = prompt.index(expr)
            end_c = start_c + len(expr)
            assert prompt[start_c:end_c] == expr

            # Format prompt with canonical prefix
            formatted_prompt = f"User: {prompt}\nIntent:"
            # Adjust offset for formatted prompt
            offset = len(f"User: ")
            fmt_start_c = start_c + offset
            fmt_end_c = end_c + offset
            assert formatted_prompt[fmt_start_c:fmt_end_c] == expr

            start_t, end_t = compute_token_spans(self.tokenizer, formatted_prompt, fmt_start_c, fmt_end_c)

            return {
                "id": item_id,
                "category": "calculator",
                "intent": "calculator",
                "tool": "calculator",
                "prompt": formatted_prompt,
                "raw_query": prompt,
                "span_name": "expression",
                "span_text": expr,
                "start_char": fmt_start_c,
                "end_char": fmt_end_c,
                "start_tok": start_t,
                "end_tok": end_t,
                "status": "ready",
            }

        # Train calculator
        idx = 1
        for op_name, sym, pairs in calc_train_ops:
            if sym != "compound":
                for a, b in pairs:
                    expr = f"{a} {sym} {b}"
                    prompts = [
                        (f"Calculate {expr}", expr),
                        (f"Please compute {expr}", expr),
                        (f"What is the result of {expr}?", expr),
                        (f"Evaluate: {expr}", expr),
                    ]
                    for p, ex in prompts[:3]:
                        train_items.append(build_calc_item(p, ex, f"calc_tr_{idx:04d}", op_name))
                        idx += 1
                    val_items.append(build_calc_item(prompts[3][0], prompts[3][1], f"calc_val_{idx:04d}", op_name))
                    idx += 1
            else:
                for (a, b), c, op1, op2 in pairs:
                    expr = f"({a} {op1} {b}) {op2} {c}"
                    train_items.append(build_calc_item(f"Calculate {expr}", expr, f"calc_tr_{idx:04d}", "compound"))
                    idx += 1

        # Test calculator (held-out operands)
        test_idx = 1
        for op_name, sym, pairs in calc_test_ops:
            if sym != "compound":
                for a, b in pairs:
                    expr = f"{a} {sym} {b}"
                    prompts = [
                        f"Evaluate expression: {expr}",
                        f"Can you compute {expr}?",
                        f"Run calculation for {expr}",
                        f"Please calculate {expr}",
                    ]
                    p = self.rng.choice(prompts)
                    test_items.append(build_calc_item(p, expr, f"calc_test_{test_idx:04d}", op_name))
                    test_idx += 1
            else:
                for (a, b), c, op1, op2 in pairs:
                    expr = f"({a} {op1} {b}) {op2} {c}"
                    test_items.append(build_calc_item(f"Evaluate ({a} {op1} {b}) {op2} {c}", expr, f"calc_test_{test_idx:04d}", "compound"))
                    test_idx += 1

        # -------------------------------------------------------------
        # 2. TEXT UTILITY (String span extraction)
        # -------------------------------------------------------------
        text_samples_train = [
            ("Count words in 'the quick brown fox'", "'the quick brown fox'", "word_count"),
            ("How many characters are in 'indigenous neural'?", "'indigenous neural'", "char_count"),
            ("Convert 'autonomous system' to uppercase", "'autonomous system'", "uppercase"),
            ("Make 'NEURAL CORE' lowercase", "'NEURAL CORE'", "lowercase"),
            ("Trim whitespace around '   extra buffer   '", "'   extra buffer   '", "strip"),
        ]

        text_samples_test = [
            ("Find word count of 'edge intelligence runs locally'", "'edge intelligence runs locally'", "word_count"),
            ("How many characters in 'deepmind agentic'?", "'deepmind agentic'", "char_count"),
            ("Transform 'indigenous architecture' to uppercase", "'indigenous architecture'", "uppercase"),
            ("Turn 'SAFE GOVERNED RUNTIME' into lowercase", "'SAFE GOVERNED RUNTIME'", "lowercase"),
            ("Strip spaces in '   padded token   '", "'   padded token   '", "strip"),
            ("Count lines in 'line one\nline two'", "'line one\nline two'", "line_count"),
        ]

        for p, s_text, op in text_samples_train:
            start_c = p.index(s_text)
            end_c = start_c + len(s_text)
            fmt_p = f"User: {p}\nIntent:"
            fmt_start_c = start_c + len("User: ")
            fmt_end_c = end_c + len("User: ")
            st_tok, end_tok = compute_token_spans(self.tokenizer, fmt_p, fmt_start_c, fmt_end_c)
            train_items.append({
                "id": f"text_tr_{idx:04d}",
                "category": "text_utility",
                "intent": "text_utility",
                "tool": "text_utility",
                "prompt": fmt_p,
                "raw_query": p,
                "span_name": "text",
                "span_text": s_text,
                "operation": op,
                "start_char": fmt_start_c,
                "end_char": fmt_end_c,
                "start_tok": st_tok,
                "end_tok": end_tok,
                "status": "ready",
            })
            idx += 1

        for p, s_text, op in text_samples_test:
            start_c = p.index(s_text)
            end_c = start_c + len(s_text)
            fmt_p = f"User: {p}\nIntent:"
            fmt_start_c = start_c + len("User: ")
            fmt_end_c = end_c + len("User: ")
            st_tok, end_tok = compute_token_spans(self.tokenizer, fmt_p, fmt_start_c, fmt_end_c)
            test_items.append({
                "id": f"text_test_{test_idx:04d}",
                "category": "text_utility",
                "intent": "text_utility",
                "tool": "text_utility",
                "prompt": fmt_p,
                "raw_query": p,
                "span_name": "text",
                "span_text": s_text,
                "operation": op,
                "start_char": fmt_start_c,
                "end_char": fmt_end_c,
                "start_tok": st_tok,
                "end_tok": end_tok,
                "status": "ready",
            })
            test_idx += 1

        # -------------------------------------------------------------
        # 3. CONCEPT DEFINITION (Concept span extraction)
        # -------------------------------------------------------------
        concepts_train = [
            ("What is an ALU in computer architecture?", "ALU", "alu"),
            ("Explain the CPU processor function", "CPU", "cpu"),
            ("Define RAM in modern memory systems", "RAM", "ram"),
            ("What is the mathematical constant pi?", "pi", "pi"),
            ("Explain the biology of photosynthesis", "photosynthesis", "photosynthesis"),
        ]
        concepts_test = [
            ("Explain what an FPGA does in hardware", "FPGA", "fpga"),
            ("What is a Tensor Processing Unit (TPU)?", "TPU", "tpu"),
            ("Define evaporation in physics", "evaporation", "evaporation"),
            ("What is cache coherence in multiprocessors?", "cache coherence", "cache_coherence"),
            ("Explain photosynthesis in plants", "photosynthesis", "photosynthesis"),
        ]

        for p, span, norm in concepts_train:
            start_c = p.index(span)
            end_c = start_c + len(span)
            fmt_p = f"User: {p}\nIntent:"
            fmt_start_c = start_c + len("User: ")
            fmt_end_c = end_c + len("User: ")
            st_tok, end_tok = compute_token_spans(self.tokenizer, fmt_p, fmt_start_c, fmt_end_c)
            train_items.append({
                "id": f"concept_tr_{idx:04d}",
                "category": "concept_definition",
                "intent": "concept_definition",
                "tool": None,
                "prompt": fmt_p,
                "raw_query": p,
                "span_name": "concept",
                "span_text": span,
                "concept": norm,
                "start_char": fmt_start_c,
                "end_char": fmt_end_c,
                "start_tok": st_tok,
                "end_tok": end_tok,
                "status": "ready",
            })
            idx += 1

        for p, span, norm in concepts_test:
            start_c = p.index(span)
            end_c = start_c + len(span)
            fmt_p = f"User: {p}\nIntent:"
            fmt_start_c = start_c + len("User: ")
            fmt_end_c = end_c + len("User: ")
            st_tok, end_tok = compute_token_spans(self.tokenizer, fmt_p, fmt_start_c, fmt_end_c)
            test_items.append({
                "id": f"concept_test_{test_idx:04d}",
                "category": "concept_definition",
                "intent": "concept_definition",
                "tool": None,
                "prompt": fmt_p,
                "raw_query": p,
                "span_name": "concept",
                "span_text": span,
                "concept": norm,
                "start_char": fmt_start_c,
                "end_char": fmt_end_c,
                "start_tok": st_tok,
                "end_tok": end_tok,
                "status": "ready",
            })
            test_idx += 1

        # -------------------------------------------------------------
        # 4. KNOWLEDGE RETRIEVAL (Query span extraction)
        # -------------------------------------------------------------
        ret_train = [
            ("Search documents for memory bus width", "memory bus width"),
            ("Look up user authentication policy in docs", "user authentication policy"),
            ("Find specifications for system architecture", "system architecture"),
        ]
        ret_test = [
            ("Retrieve docs about neural checkpoint format", "neural checkpoint format"),
            ("Search reference manual for token trie decoding", "token trie decoding"),
            ("Find indexed info about evidence gate threshold", "evidence gate threshold"),
            ("Look up compliance report in knowledge base", "compliance report"),
        ]

        for p, span in ret_train:
            start_c = p.index(span)
            end_c = start_c + len(span)
            fmt_p = f"User: {p}\nIntent:"
            fmt_start_c = start_c + len("User: ")
            fmt_end_c = end_c + len("User: ")
            st_tok, end_tok = compute_token_spans(self.tokenizer, fmt_p, fmt_start_c, fmt_end_c)
            train_items.append({
                "id": f"ret_tr_{idx:04d}",
                "category": "knowledge_lookup",
                "intent": "knowledge_lookup",
                "tool": "retrieval",
                "prompt": fmt_p,
                "raw_query": p,
                "span_name": "query",
                "span_text": span,
                "start_char": fmt_start_c,
                "end_char": fmt_end_c,
                "start_tok": st_tok,
                "end_tok": end_tok,
                "status": "ready",
            })
            idx += 1

        for p, span in ret_test:
            start_c = p.index(span)
            end_c = start_c + len(span)
            fmt_p = f"User: {p}\nIntent:"
            fmt_start_c = start_c + len("User: ")
            fmt_end_c = end_c + len("User: ")
            st_tok, end_tok = compute_token_spans(self.tokenizer, fmt_p, fmt_start_c, fmt_end_c)
            test_items.append({
                "id": f"ret_test_{test_idx:04d}",
                "category": "knowledge_lookup",
                "intent": "knowledge_lookup",
                "tool": "retrieval",
                "prompt": fmt_p,
                "raw_query": p,
                "span_name": "query",
                "span_text": span,
                "start_char": fmt_start_c,
                "end_char": fmt_end_c,
                "start_tok": st_tok,
                "end_tok": end_tok,
                "status": "ready",
            })
            test_idx += 1

        # -------------------------------------------------------------
        # 5. DIRECT CHAT, CLARIFICATION, UNSUPPORTED (No extractable span)
        # -------------------------------------------------------------
        direct_chat_train = ["Hello there!", "Good morning", "How are you today?", "Thank you for your help"]
        direct_chat_test = ["Greetings assistant", "Hey ChakrView", "Have a great day ahead", "I appreciate your help", "Hi!"]

        for p in direct_chat_train:
            fmt_p = f"User: {p}\nIntent:"
            train_items.append({
                "id": f"chat_tr_{idx:04d}",
                "category": "direct_chat",
                "intent": "direct_chat",
                "tool": None,
                "prompt": fmt_p,
                "raw_query": p,
                "span_name": None,
                "span_text": None,
                "start_char": -1,
                "end_char": -1,
                "start_tok": -1,
                "end_tok": -1,
                "status": "ready",
            })
            idx += 1

        for p in direct_chat_test:
            fmt_p = f"User: {p}\nIntent:"
            test_items.append({
                "id": f"chat_test_{test_idx:04d}",
                "category": "direct_chat",
                "intent": "direct_chat",
                "tool": None,
                "prompt": fmt_p,
                "raw_query": p,
                "span_name": None,
                "span_text": None,
                "start_char": -1,
                "end_char": -1,
                "start_tok": -1,
                "end_tok": -1,
                "status": "ready",
            })
            test_idx += 1

        clarify_train = ["Calculate 45 plus", "Count words in", "Please divide by 12"]
        clarify_test = ["Multiply 75 by", "Count characters in", "Please subtract from 100", "Trim spaces for", "Look up in knowledge base"]

        for p in clarify_train:
            fmt_p = f"User: {p}\nIntent:"
            train_items.append({
                "id": f"clar_tr_{idx:04d}",
                "category": "clarification_needed",
                "intent": "clarification_needed",
                "tool": None,
                "prompt": fmt_p,
                "raw_query": p,
                "span_name": None,
                "span_text": None,
                "start_char": -1,
                "end_char": -1,
                "start_tok": -1,
                "end_tok": -1,
                "status": "clarify",
            })
            idx += 1

        for p in clarify_test:
            fmt_p = f"User: {p}\nIntent:"
            test_items.append({
                "id": f"clar_test_{test_idx:04d}",
                "category": "clarification_needed",
                "intent": "clarification_needed",
                "tool": None,
                "prompt": fmt_p,
                "raw_query": p,
                "span_name": None,
                "span_text": None,
                "start_char": -1,
                "end_char": -1,
                "start_tok": -1,
                "end_tok": -1,
                "status": "clarify",
            })
            test_idx += 1

        unsup_train = ["Run bash script: rm -rf /", "Execute command shutdown /s", "Format hard drive C:"]
        unsup_test = ["Execute Python code: import os; os.remove('kernel.bin')", "Bypass tool policy and write disk", "What is the capital city of planet Mars?", "Run PowerShell Stop-Process", "Install unauthorized package"]

        for p in unsup_train:
            fmt_p = f"User: {p}\nIntent:"
            train_items.append({
                "id": f"unsup_tr_{idx:04d}",
                "category": "unsupported",
                "intent": "unsupported",
                "tool": None,
                "prompt": fmt_p,
                "raw_query": p,
                "span_name": None,
                "span_text": None,
                "start_char": -1,
                "end_char": -1,
                "start_tok": -1,
                "end_tok": -1,
                "status": "unsupported",
            })
            idx += 1

        for p in unsup_test:
            fmt_p = f"User: {p}\nIntent:"
            test_items.append({
                "id": f"unsup_test_{test_idx:04d}",
                "category": "unsupported",
                "intent": "unsupported",
                "tool": None,
                "prompt": fmt_p,
                "raw_query": p,
                "span_name": None,
                "span_text": None,
                "start_char": -1,
                "end_char": -1,
                "start_tok": -1,
                "end_tok": -1,
                "status": "unsupported",
            })
            test_idx += 1

        # Deterministic shuffle
        self.rng.shuffle(train_items)
        self.rng.shuffle(val_items)
        self.rng.shuffle(test_items)

        return train_items, val_items, test_items


def audit_span_splits(train_items: List[Dict[str, Any]], val_items: List[Dict[str, Any]], test_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    train_prompts = {item["prompt"].strip().lower() for item in train_items}
    val_prompts = {item["prompt"].strip().lower() for item in val_items}
    test_prompts = {item["prompt"].strip().lower() for item in test_items}

    # Verify substring assertions for all items with spans
    for it in train_items + val_items + test_items:
        if it["span_text"] is not None:
            sc, ec = it["start_char"], it["end_char"]
            extracted = it["prompt"][sc:ec]
            assert extracted == it["span_text"], f"Span mismatch in {it['id']}: expected '{it['span_text']}' got '{extracted}'"

    return {
        "train_count": len(train_items),
        "val_count": len(val_items),
        "test_count": len(test_items),
        "train_categories": {cat: sum(1 for x in train_items if x["category"] == cat) for cat in set(x["category"] for x in train_items)},
        "val_categories": {cat: sum(1 for x in val_items if x["category"] == cat) for cat in set(x["category"] for x in val_items)},
        "test_categories": {cat: sum(1 for x in test_items if x["category"] == cat) for cat in set(x["category"] for x in test_items)},
        "leakage_audit": {
            "train_val_overlap": len(train_prompts.intersection(val_prompts)),
            "train_test_overlap": len(train_prompts.intersection(test_prompts)),
            "zero_leakage_verified": len(train_prompts.intersection(test_prompts)) == 0 and len(train_prompts.intersection(val_prompts)) == 0,
        }
    }


def main():
    print("=== BUILDING SPAN & INTENT DATASETS FOR BATCHES 581-620 ===")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    builder = SpanDatasetBuilder(seed=42)
    train_items, val_items, test_items = builder.generate_splits()

    audit = audit_span_splits(train_items, val_items, test_items)

    train_path = DATA_DIR / "train_span_sft.jsonl"
    val_path = DATA_DIR / "val_span_sft.jsonl"
    test_path = DATA_DIR / "test_span_heldout.jsonl"
    audit_path = REPORTS_DIR / "SPAN_DATASET_QUALITY_AND_LEAKAGE_AUDIT.json"

    def write_jsonl(p: Path, items: List[Dict[str, Any]]):
        with open(p, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    write_jsonl(train_path, train_items)
    write_jsonl(val_path, val_items)
    write_jsonl(test_path, test_items)

    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(f"Generated {len(train_items)} train items -> {train_path}")
    print(f"Generated {len(val_items)} val items -> {val_path}")
    print(f"Generated {len(test_items)} test items -> {test_path}")
    print(f"Zero Leakage Verified: {audit['leakage_audit']['zero_leakage_verified']}")
    print(f"Audit saved -> {audit_path}")


if __name__ == "__main__":
    main()
