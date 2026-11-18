import json

d = json.load(open('scripts/stage_c/sources.json'))
approved = [s for s in d['sources'] if s.get('acquisition_status') == 'APPROVED']
rejected = [s for s in d['sources'] if s.get('acquisition_status') == 'REJECTED']
conditional = [s for s in d['sources'] if s.get('acquisition_status') == 'CONDITIONAL']

print(f"Total Sources registered: {len(d['sources'])}")
print(f"\n--- APPROVED ({len(approved)}) ---")
for s in approved:
    print(f"  {s['source_id']}: {s['name']} | License: {s['license']} | Expected Tokens: {s.get('expected_token_range')}")

print(f"\n--- REJECTED ({len(rejected)}) ---")
for s in rejected:
    print(f"  {s['source_id']}: {s['name']} | License: {s['license']} | Reason: {s.get('redistribution_notes')}")

print(f"\n--- CONDITIONAL ({len(conditional)}) ---")
for s in conditional:
    print(f"  {s['source_id']}: {s['name']} | License: {s['license']} | Reason: {s.get('redistribution_notes')}")
