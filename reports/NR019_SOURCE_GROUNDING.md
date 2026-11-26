# NR-019: Exact Source-Span Grounding & Provenance Architecture

## 1. Principles of Source-Span Grounding
In previous missions, the neural model was tasked with both:
1. *Semantic interpretation*: Understanding user intent and selecting the operation.
2. *Generative copying*: Generating token representations of digits, lists, and strings.

Generative copying failed catastrophically (86.9% of all failures in NR-018) because small autoregressive language models suffer from token frequency bias, numerical hallucination, and Devanagari multi-byte splitting.

**The Solution in NR-019**:
- **Operation Routing**: Neural model acts strictly as an intent classifier emitting only the operation identifier (e.g. `{"operation": "math_mul"}`) or short status (`"unsupported"`, `"clarify"`).
- **Exact Argument Grounding**: Arguments are bound to exact character spans `[start, end]` in the user's original input string.
- **Deterministic Type Coercion**: A sandboxed multilingual extractor converts grounded spans into validated Python data types (integers, floats, arrays, strings) with guaranteed provenance.

---

## 2. Grounding Engine Verification
Verified via `chakrview/runtime/source_grounding.py` and `chakrview/runtime/multilingual_normalizer.py`:
- **ASCII Numerals**: `"Multiply 15 by 9."` $\to$ `{"a": 15, "b": 9}` (provenance: `[9, 11]`, `[15, 16]`).
- **Devanagari Numerals**: `"१५ को ९ से गुणा करो।"` $\to$ `{"a": 15, "b": 9}` (provenance: `[0, 2]`, `[6, 7]`).
- **Hinglish Number Words**: `"pandrah ko nau se multiply karo."` $\to$ `{"a": 15, "b": 9}`.
- **Array literals**: `"[21, 22, 16, 12] list ko reverse kardo."` $\to$ `{"list": [21, 22, 16, 12]}`.
- **Reverse Phrasing**: `"Subtract 11 from 33."` $\to$ correctly binds `a=33` and `b=11` (resolving subtraction order).
- **Quoted Strings**: `"Reverse the word 'neural'."` $\to$ `{"string": "neural"}`.

---

## 3. Grounded Argument Schema
Every extracted parameter carries complete provenance:
```json
{
  "key": "a",
  "value": 15,
  "raw_text": "१५",
  "span": [0, 2],
  "type": "numeric_literal"
}
```
If an input is ambiguous or missing required arguments, the grounding engine emits status `"clarify"` rather than inventing values.
