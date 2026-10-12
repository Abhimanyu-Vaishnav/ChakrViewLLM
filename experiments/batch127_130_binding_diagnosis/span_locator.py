"""
Deterministic Token-Span Alignment and Diagnostic Module.
Uses tokenizer vocabulary and token-level decoded boundaries to find exact token positions of operands.
Never hardcodes token IDs or assumes fixed token lengths.
"""
from typing import List, Tuple, Optional, Dict, Any

def locate_char_span(text: str, substring: str) -> Optional[Tuple[int, int]]:
    idx = text.find(substring)
    if idx == -1:
        return None
    return idx, idx + len(substring)

def map_char_span_to_tokens(text: str, char_start: int, char_end: int, tokenizer) -> Tuple[List[int], List[int]]:
    """
    Returns (token_indices, token_ids) for all tokens that overlap with [char_start, char_end).
    """
    token_ids = tokenizer.encode(text)
    cum_chars = 0
    token_spans = []
    for i, t in enumerate(token_ids):
        s = tokenizer.decode([t])
        sc = cum_chars
        ec = cum_chars + len(s)
        cum_chars = ec
        token_spans.append((sc, ec))
        
    overlapping_indices = [
        i for i, (sc, ec) in enumerate(token_spans)
        if not (ec <= char_start or sc >= char_end)
    ]
    return overlapping_indices, [token_ids[i] for i in overlapping_indices]

def extract_prompt_operand_token_spans(prompt_text: str, a: int, b: int, tokenizer) -> Dict[str, Any]:
    """
    Deterministically locates operand A and operand B token spans in the prompt.
    """
    str_a = str(a)
    str_b = str(b)
    
    # Locate in prompt
    span_a_char = locate_char_span(prompt_text, str_a)
    span_b_char = locate_char_span(prompt_text, str_b)
    
    res = {
        "operand_a": a,
        "operand_b": b,
        "char_span_a": span_a_char,
        "char_span_b": span_b_char,
    }
    
    if span_a_char:
        tok_idx_a, tok_ids_a = map_char_span_to_tokens(prompt_text, span_a_char[0], span_a_char[1], tokenizer)
        res["token_indices_a"] = tok_idx_a
        res["token_ids_a"] = tok_ids_a
        res["token_count_a"] = len(tok_ids_a)
        
    if span_b_char:
        tok_idx_b, tok_ids_b = map_char_span_to_tokens(prompt_text, span_b_char[0], span_b_char[1], tokenizer)
        res["token_indices_b"] = tok_idx_b
        res["token_ids_b"] = tok_ids_b
        res["token_count_b"] = len(tok_ids_b)
        
    return res
