import json

with open("experiments/cognition/cr_exp_04_validation/independent_conversation_suite.json", encoding="utf-8") as f:
    data = json.load(f)

target_ids = ["A01", "A04", "B01", "B07", "B08", "C01", "D01", "E01", "E04", "F01", "G01"]

for x in data:
    if x["id"] in target_ids:
        print("=" * 60)
        print(f"ID: {x['id']} | Category: {x['category']}")
        print(f"PROMPT:\n{x['prompt']}")
        print(f"PARENT   : {repr(x['parent_response'])}")
        print(f"CANDIDATE: {repr(x['candidate_response'])}")

