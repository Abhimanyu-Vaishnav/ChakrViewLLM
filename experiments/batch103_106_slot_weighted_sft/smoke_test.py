"""Smoke test for slot_weighted_training.py module functions."""
import sys
sys.path.insert(0, ".")

from experiments.batch103_106_slot_weighted_sft.slot_weighted_training import (
    build_training_sequence, find_arg_value_positions, build_weight_mask,
    parse_tool_call, load_jsonl, A_VALUE_TOKENS, B_VALUE_TOKENS,
    SLOT_WEIGHT, IGNORE_INDEX, BOS_ID
)
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import load_tokenizer_artifacts
from pathlib import Path

tokenizer, _ = load_tokenizer_artifacts("release/chakrview-v0.5.0/data/experiments/vocab_4096")
records = load_jsonl(Path("experiments/batch95_98_slot_binding_diagnostics/data/train.jsonl"))

# Test 1: slot detection on all training records
missing = 0
for rec in records:
    inp_ids, tgt_ids = build_training_sequence(rec, tokenizer)
    slot_pos = find_arg_value_positions(inp_ids, rec["operands"])
    if len(slot_pos) == 0:
        missing += 1
        print(f"  MISSING SLOTS: {rec['operands']}")
print(f"Slot detection: {len(records)-missing}/{len(records)} OK")
assert missing == 0

# Test 2: weight mask correctness
rec = records[4]
inp_ids, tgt_ids = build_training_sequence(rec, tokenizer)
slot_pos = find_arg_value_positions(inp_ids, rec["operands"])
mask = build_weight_mask(len(inp_ids), slot_pos, tgt_ids)
for p in slot_pos:
    assert mask[p].item() == SLOT_WEIGHT, f"pos {p} should be {SLOT_WEIGHT}"
non_slot = [mask[i].item() for i in range(len(inp_ids)) if i not in slot_pos]
assert all(w == 1.0 for w in non_slot), "Non-slot weights should all be 1.0"
print(f"Weight mask test OK: slot_pos={slot_pos}, slot_weights={[mask[p].item() for p in slot_pos]}")

# Test 3: parse_tool_call
good = 'TOOL_CALL: {"tool": "calculator", "op": "add", "a": 3, "b": 0}'
parsed = parse_tool_call(good)
assert parsed is not None and parsed["a"] == 3 and parsed["b"] == 0
assert parse_tool_call("garbage") is None
print("parse_tool_call: OK")

# Test 4: sequence length equality
for rec in records[:10]:
    inp_ids, tgt_ids = build_training_sequence(rec, tokenizer)
    assert len(inp_ids) == len(tgt_ids), f"Length mismatch for {rec['operands']}"
print("Sequence lengths: OK")

# Test 5: slot positions are within target portion (not in prompt mask)
for rec in records[:20]:
    inp_ids, tgt_ids = build_training_sequence(rec, tokenizer)
    slot_pos = find_arg_value_positions(inp_ids, rec["operands"])
    for p in slot_pos:
        assert tgt_ids[p] != IGNORE_INDEX, (
            f"Slot position {p} is in prompt mask region for operands={rec['operands']}"
        )
print("Slot positions within target portion: OK")

# Test 6: spot-check token values at slot positions
for rec in records[:30]:
    inp_ids, tgt_ids = build_training_sequence(rec, tokenizer)
    slot_pos = find_arg_value_positions(inp_ids, rec["operands"])
    a_val = int(rec["operands"][0])
    b_val = int(rec["operands"][1])
    toks_at_slots = [inp_ids[p] for p in slot_pos]
    for tok_id in A_VALUE_TOKENS[a_val]:
        assert tok_id in toks_at_slots, f"a={a_val}: expected tok {tok_id} in {toks_at_slots}"
    for tok_id in B_VALUE_TOKENS[b_val]:
        assert tok_id in toks_at_slots, f"b={b_val}: expected tok {tok_id} in {toks_at_slots}"
print("Spot-check token values at slot positions: OK")

print("\nAll smoke tests PASSED.")
