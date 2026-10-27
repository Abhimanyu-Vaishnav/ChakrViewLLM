import json
import math
import os
from collections import Counter

EXP25_PRED = "experiments/overnight_runs/exp25_vocabulary_generalization_diagnostic_20261024_232014/CANDIDATE_PER_RECORD_PREDICTIONS.jsonl"
EXP25_REPORT = "experiments/overnight_runs/exp25_vocabulary_generalization_diagnostic_20261024_232014/EVALUATION_REPORT.json"

EXP26_PRED = "experiments/overnight_runs/mission_10h_exp26_15m_readiness_20261024_232621/exp26_15m_scaling_run/CANDIDATE_PER_RECORD_PREDICTIONS.jsonl"
EXP26_REPORT = "experiments/overnight_runs/mission_10h_exp26_15m_readiness_20261024_232621/exp26_15m_scaling_run/EVALUATION_REPORT.json"

def audit_file(pred_path, orig_report_path, name):
    records = [json.loads(line) for line in open(pred_path, encoding='utf-8')]
    orig_rep = json.load(open(orig_report_path, encoding='utf-8'))

    by_split = {}
    for r in records:
        by_split.setdefault(r['split'], []).append(r)

    out = {
        "experiment": name,
        "total_records": len(records),
        "splits": {},
        "discrepancies": []
    }

    for split_name, recs in by_split.items():
        n = len(recs)
        # Independently calculate EM from raw generated_text and target
        em_count = sum(1 for r in recs if r.get('generated_text') == r.get('target'))
        em_rate = em_count / n if n > 0 else 0.0

        # Independently calculate Pair Success (Counterfactual consistency)
        cf_differ = all(r.get('cf_target') != r.get('target') for r in recs)
        pla_count = sum(1 for r in recs if (r.get('generated_text') == r.get('target') and r.get('cf_generated_text') == r.get('cf_target')))
        pla_rate = pla_count / n if n > 0 else 0.0

        # EOS rate: token 1 appears in generated_ids
        eos_count = sum(1 for r in recs if 1 in r.get('generated_ids', []))
        eos_rate = eos_count / n if n > 0 else 0.0

        # Modal token and modal share
        outputs = [r.get('generated_text', '') for r in recs]
        c = Counter(outputs)
        modal_token, modal_count = c.most_common(1)[0]
        modal_share = modal_count / n if n > 0 else 0.0

        # Latency
        lats = [r['latency_ms'] for r in recs if 'latency_ms' in r]
        avg_lat = sum(lats)/len(lats) if lats else None

        # Target tokens
        targets = sorted(list(set(r.get('target') for r in recs)))
        per_token = {}
        for t in targets:
            sub = [r for r in recs if r.get('target') == t]
            per_token[t] = {
                "count": len(sub),
                "em": sum(1 for r in sub if r.get('generated_text') == r.get('target')),
                "accuracy": sum(1 for r in sub if r.get('generated_text') == r.get('target')) / len(sub)
            }

        orig_split_rep = orig_rep.get('splits', {}).get(split_name, {})
        orig_em = orig_split_rep.get('em_micro')
        orig_pla = orig_split_rep.get('pla_micro')
        orig_modal = orig_split_rep.get('modal_token')
        orig_share = orig_split_rep.get('modal_share')

        # Check discrepancies
        em_diff = abs(em_rate - orig_em) if orig_em is not None else None
        pla_diff = abs(pla_rate - orig_pla) if orig_pla is not None else None

        split_summary = {
            "n_records": n,
            "recomputed_em": em_rate,
            "orig_em": orig_em,
            "em_difference": em_diff,
            "recomputed_pla": pla_rate,
            "orig_pla": orig_pla,
            "pla_difference": pla_diff,
            "cf_targets_valid": cf_differ,
            "eos_rate": eos_rate,
            "recomputed_modal_token": modal_token,
            "orig_modal_token": orig_modal,
            "recomputed_modal_share": modal_share,
            "orig_modal_share": orig_share,
            "avg_latency_ms": avg_lat,
            "per_token_accuracy": per_token,
            "target_rank_verifiable": "NOT INDEPENDENTLY VERIFIABLE (per-step full logits not saved in jsonl)"
        }
        out["splits"][split_name] = split_summary

        if em_diff is not None and em_diff > 1e-6:
            out["discrepancies"].append(f"{split_name} EM discrepancy: orig {orig_em} vs recomputed {em_rate}")
        if pla_diff is not None and pla_diff > 1e-6:
            out["discrepancies"].append(f"{split_name} PLA discrepancy: orig {orig_pla} vs recomputed {pla_rate}")

    return out

if __name__ == '__main__':
    r25 = audit_file(EXP25_PRED, EXP25_REPORT, "Exp25")
    r26 = audit_file(EXP26_PRED, EXP26_REPORT, "Exp26")
    combined = {
        "exp25": r25,
        "exp26": r26,
        "status": "COMPLETED_WITHOUT_DISCREPANCIES" if not (r25['discrepancies'] or r26['discrepancies']) else "DISCREPANCIES_FOUND"
    }
    out_path = "experiments/overnight_runs/exp27_forensic_pointer_research_20261025_203725/METRIC_RECOMPUTATION.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2)
    print("Saved recomputation to", out_path)
    print("Exp25 discrepancies:", r25['discrepancies'])
    print("Exp26 discrepancies:", r26['discrepancies'])
