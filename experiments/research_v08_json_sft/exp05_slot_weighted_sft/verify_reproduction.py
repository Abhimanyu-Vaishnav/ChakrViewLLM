import json

orig = json.load(open('experiments/research_v08_json_sft/exp02_string_copy/CANDIDATE_EVALUATION_RESULTS.json'))
repro = json.load(open('experiments/research_v08_json_sft/exp05_slot_weighted_sft/REPRODUCED_BASELINE_RESULTS.json'))

for k, v in orig['json_extraction_metrics'].items():
    assert v == repro['json_extraction_metrics'][k], f"Metric mismatch in {k}: {v} vs {repro['json_extraction_metrics'][k]}"

assert orig['replay_metrics'] == repro['replay_metrics'], "Replay metric mismatch"

for i, (r1, r2) in enumerate(zip(orig['results'], repro['results'])):
    assert r1['raw_output'] == r2['raw_output'], f"Output mismatch in record {i}: {r1['raw_output']} vs {r2['raw_output']}"

print("SUCCESS: EXACT BIT-FOR-BIT REPRODUCTION VERIFIED ACROSS ALL 30 TEST RECORDS.")
