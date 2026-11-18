import json

with open('experiments/ucip10e_r2/datasets/sealed/benchmark_r2_sealed.json', encoding='utf-8') as f:
    bench_data = json.load(f)
bench_dict = {p['id']: p for p in bench_data}

pass_ids = ['SEA_LOG_002', 'SEA_LOG_007', 'SEA_LOG_012', 'SEA_LOG_017', 'SEA_LOG_022',
            'SEA_LOG_027', 'SEA_LOG_032', 'SEA_LOG_037', 'SEA_LOG_042', 'SEA_LOG_047']

print("--- Passed Logic Probes ---")
for pid in pass_ids:
    p = bench_dict[pid]
    print(f"ID: {pid} | Prompt: {p['prompt']} | EvalType: {p.get('evaluation_type')} | Expected: {p.get('expected_keywords', p.get('metadata'))}")

print("\n--- Failed Logic Probes (sample of 10) ---")
logic_probes = [p for p in bench_data if p['domain'] == 'logic' and p['id'] not in pass_ids]
for p in logic_probes[:10]:
    print(f"ID: {p['id']} | Prompt: {p['prompt']} | EvalType: {p.get('evaluation_type')} | Expected: {p.get('expected_keywords', p.get('metadata'))}")
