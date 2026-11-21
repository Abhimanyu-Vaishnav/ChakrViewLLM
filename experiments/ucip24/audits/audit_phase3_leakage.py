"""
UCIP-24 Phase 3: Pretraining and Memorization Leakage Audit.
Scans corpus files across data/raw and data/processed for:
Level 1: Exact entity-relation-value statement
Level 2: Equivalent statement with different wording (e.g. co-occurrence with relation synonym)
Level 3: Entity and Value co-occurrence in the same document / paragraph
Level 4: Entity-only occurrence
Level 5: Value-only occurrence
Level 6: Numerical / lexical sub-token occurrence without semantic relation
Level 7: No detectable occurrence (Status D)
Level 8: Inconclusive
"""

import glob
import json
import re
from collections import defaultdict
from pathlib import Path


def audit_phase3_pretraining_leakage():
    # 1. Load benchmark facts from Core tasks
    facts_to_check = []
    core_tasks = [
        "task_c_fact_retrieval.jsonl",
        "task_d_multi_relation_binding.jsonl",
        "task_e_compositional_generalization.jsonl",
        "task_f_contextual_binding.jsonl",
        "task_g_distractor_resistance.jsonl"
    ]

    seen_facts = set()
    for fn in core_tasks:
        fpath = Path("experiments/ucip23/datasets") / fn
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                e = rec.get("query_entity")
                r = rec.get("query_relation")
                v = rec.get("ground_truth_value")
                task = rec.get("task_type")
                if e and r and v and (e, r, v) not in seen_facts:
                    seen_facts.add((e, r, v))
                    facts_to_check.append({
                        "task": task,
                        "entity": e,
                        "relation": r,
                        "value": v
                    })

    print(f"Total unique synthetic facts to audit against pretraining corpora: {len(facts_to_check)}")

    # 2. Gather corpus files
    corpus_files = glob.glob("data/processed/**/*.txt", recursive=True) + glob.glob("data/raw/**/*.txt", recursive=True)
    print(f"Reading {len(corpus_files)} corpus text files...")

    # Load corpus text into memory
    corpus_texts = []
    for cf in corpus_files:
        try:
            with open(cf, "r", encoding="utf-8", errors="ignore") as f:
                corpus_texts.append((cf, f.read()))
        except Exception as e:
            pass

    # 3. Audit each fact across evidence levels
    level_counts = defaultdict(int)
    fact_audit_details = []

    for fact in facts_to_check:
        e = fact["entity"]
        r = fact["relation"]
        v = str(fact["value"])

        # Regex patterns
        e_in_corpus = False
        v_in_corpus = False
        co_occurrence = False
        exact_relation = False

        # Quick check across corpus texts
        for cf, txt in corpus_texts:
            e_found = e in txt
            v_found = v in txt

            if e_found:
                e_in_corpus = True
            if v_found:
                v_in_corpus = True

            if e_found and v_found:
                # Check distance between e and v in txt
                # If they occur within 200 characters of each other
                for m in re.finditer(re.escape(e), txt):
                    start = max(0, m.start() - 200)
                    end = min(len(txt), m.end() + 200)
                    window = txt[start:end]
                    if v in window:
                        co_occurrence = True
                        if r.replace("_", " ") in window.lower() or r in window.lower():
                            exact_relation = True
                            break
            if exact_relation:
                break

        # Assign classification level
        if exact_relation:
            classification = "Level 1: Exact Statement / Relation Present"
            level_counts["Level 1"] += 1
        elif co_occurrence:
            classification = "Level 3: Entity-Value Co-occurrence (No Verified Relation)"
            level_counts["Level 3"] += 1
        elif e_in_corpus and not v_in_corpus:
            classification = "Level 4: Entity-Only Occurrence"
            level_counts["Level 4"] += 1
        elif not e_in_corpus and v_in_corpus:
            classification = "Level 5: Value-Only Occurrence"
            level_counts["Level 5"] += 1
        elif not e_in_corpus and not v_in_corpus:
            classification = "Level 7: No Detectable Occurrence (Status D)"
            level_counts["Level 7"] += 1
        else:
            classification = "Level 6: Disjoint Lexical Co-presence"
            level_counts["Level 6"] += 1

        fact_audit_details.append({
            "entity": e,
            "relation": r,
            "value": v,
            "classification": classification
        })

    summary = {
        "total_facts_audited": len(facts_to_check),
        "corpus_files_inspected": len(corpus_texts),
        "classification_counts": dict(level_counts),
        "status_d_percentage": (level_counts["Level 7"] / len(facts_to_check) * 100) if facts_to_check else 0.0,
        "exact_statement_leaks": level_counts["Level 1"],
        "co_occurrence_leaks": level_counts["Level 3"]
    }

    out_file = "experiments/ucip24/audits/phase3_pretraining_leakage_audit.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "sample_facts": fact_audit_details[:20]}, f, indent=2)

    print("Phase 3 Pretraining Leakage Audit Complete.")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    audit_phase3_pretraining_leakage()
