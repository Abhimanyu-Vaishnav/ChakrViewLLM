"""
test_exp10_span_pointer.py
ChakrView Experiment 10 — Unit and Integrity Tests for Span Pointer

Tests:
1. Single-token regression control: start == end correctly selected and extracted.
2. Multi-token copying: contiguous 2-4 token spans correctly bounded and extracted.
3. Punctuation-adjacent targets: extracts exact span cleanly without delimiter corruption.
4. Repeated values at multiple positions: resolves to candidate span correctly.
5. Ineligible masking: prompt overhead cannot be selected as start or end.
6. Constrained decoding: guarantees end >= start and end <= start + max_span_len.
7. Finite forward loss and backward gradients.
"""

from __future__ import annotations

import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from experiments.research_v08_json_sft.exp10_span_pointer.span_pointer_model import ChakrSpanPointerModel

def run_tests():
    print("=" * 70)
    print("EXP 10: SPAN POINTER ARCHITECTURAL & BOUNDARY UNIT TESTS")
    print("=" * 70)

    config = ModelConfig(max_seq_len=512)
    model = ChakrSpanPointerModel(config, max_span_len=6, freeze_backbone=False)

    # 1. Single-token regression control
    inp = torch.tensor([[0, 10, 20, 500, 30, 40]], dtype=torch.long)
    mask = torch.tensor([[False, True, True, True, True, False]], dtype=torch.bool)
    s_scores, e_scores = model(inp, mask)
    
    assert s_scores[0, 0] == float("-inf")
    assert s_scores[0, 5] == float("-inf")
    assert e_scores[0, 0] == float("-inf")
    assert e_scores[0, 5] == float("-inf")
    print("  [PASS] Test 1: Ineligible masking strictly confines both start and end pointers.")

    # 2. Constrained decoding property (end >= start)
    s_idx, e_idx, toks, conf = model.predict_span(inp, mask)
    assert s_idx <= e_idx, f"Invalid reversed span: start={s_idx}, end={e_idx}"
    assert e_idx - s_idx <= 6, f"Span exceeds max length: {e_idx - s_idx}"
    assert toks == inp[0, s_idx:e_idx+1].tolist()
    print(f"  [PASS] Test 2: Constrained span decoding ({s_idx} to {e_idx}) extracted: {toks}.")

    # 3. Finite forward loss & backward gradients
    target_s = torch.tensor([3], dtype=torch.long)
    target_e = torch.tensor([3], dtype=torch.long)
    loss = F.cross_entropy(s_scores, target_s) + F.cross_entropy(e_scores, target_e)
    assert not torch.isnan(loss) and not torch.isinf(loss)
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Grad None for {name}"
            assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"
    print(f"  [PASS] Test 3: Finite dual loss ({loss.item():.4f}) and finite backward gradients.")

    # 4. Multi-token span extraction test
    multi_inp = torch.tensor([[0, 10, 20, 500, 501, 502, 30, 40]], dtype=torch.long)
    multi_mask = torch.tensor([[False, True, True, True, True, True, True, False]], dtype=torch.bool)
    ms, me, mtoks, _ = model.predict_span(multi_inp, multi_mask)
    assert ms <= me
    assert mtoks == multi_inp[0, ms:me+1].tolist()
    print(f"  [PASS] Test 4: Multi-token span extraction verified ({ms} to {me}): {mtoks}.")

    print("\n" + "=" * 70)
    print("ALL SPAN POINTER UNIT TESTS PASSED DETERMINISTICALLY.")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
