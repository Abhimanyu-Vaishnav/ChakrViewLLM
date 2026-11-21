"""
UCIP-23 Task-Specific Benchmark Dataset Generator.
Constructs verified instances for Tasks A through H using the deterministic knowledge graph.
All records conform strictly to experiments/ucip22/schemas/benchmark_record_schema.json
and are validated against the frozen 4,096 tokenizer and 512 context limit.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import normalize_prompt_response_strings
from experiments.ucip23.datasets.generate_knowledge_graph import build_knowledge_graph, DOMAINS


def build_all_tasks(seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    random.seed(seed)
    graph = build_knowledge_graph(seed=seed)
    entities = graph["entities"]
    facts = graph["facts"]
    
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")
    
    task_records = {
        "task_a_entity_identity": [],
        "task_b_relation_recognition": [],
        "task_c_fact_retrieval": [],
        "task_d_multi_relation_binding": [],
        "task_e_compositional_generalization": [],
        "task_f_contextual_binding": [],
        "task_g_distractor_resistance": [],
        "task_h_abstention": []
    }

    rec_id_counter = 1

    def make_record(
        task_type: str,
        split: str,
        ctx_ents: List[str],
        q_ent: str,
        q_rel: str,
        gt_val: str,
        dist_ents: List[str],
        dist_vals: List[str],
        is_abs: bool,
        abs_reason: str,
        prompt: str,
        target: str
    ) -> Dict[str, Any]:
        nonlocal rec_id_counter
        p_str, r_str = normalize_prompt_response_strings(prompt, target, strip_delimiters=True)
        p_toks = tok.encode(p_str)
        r_toks = tok.encode(r_str)
        total_len = len(p_toks) + len(r_toks) + 2 # + BOS, EOS

        rec = {
            "id": f"UCIP23-{task_type.upper()}-{rec_id_counter:05d}",
            "benchmark_version": "ucip22-v1",
            "task_type": task_type,
            "split": split,
            "context_entities": ctx_ents,
            "query_entity": q_ent,
            "query_relation": q_rel,
            "ground_truth_value": gt_val,
            "distractor_entities": dist_ents,
            "distractor_values": dist_vals,
            "is_abstention": is_abs,
            "expected_abstention_reason": abs_reason,
            "prompt": prompt,
            "target_response": target,
            "token_metadata": {
                "prompt_token_count": len(p_toks),
                "response_token_count": len(r_toks),
                "total_sequence_length": total_len,
                "first_answer_token_id": r_toks[0] if r_toks else 1,
                "terminal_token_id": 1,
                "is_within_context_limit": (total_len <= 512)
            }
        }
        rec_id_counter += 1
        return rec

    # --------------------------------------------------------------------------
    # Task A: Entity Identity Recognition (Cross-mention coreference / match)
    # --------------------------------------------------------------------------
    ent_keys = list(entities.keys())
    for i in range(120):
        e1 = entities[ent_keys[i % len(ent_keys)]]
        is_same = (i % 2 == 0)
        e2 = e1 if is_same else entities[ent_keys[(i + 7) % len(ent_keys)]]
        
        prompt = (
            f"User: Text snippet 1 discusses {e1['name']}. Text snippet 2 discusses {e2['name']}.\n"
            f"Question: Do both snippets refer to the identical entity name?\n"
            f"Answer concisely with Yes or No:\nAssistant: "
        )
        target = "Yes" if is_same else "No"
        rec = make_record(
            "task_a_entity_identity", "dev_entity_disjoint" if i >= 80 else "train_exemplar",
            [e1["name"], e2["name"]], e1["name"], "coreference", target,
            [e2["name"]] if not is_same else [], [], False, "none", prompt, target
        )
        task_records["task_a_entity_identity"].append(rec)

    # --------------------------------------------------------------------------
    # Task B: Relation Recognition (Query intent independent of entity name)
    # --------------------------------------------------------------------------
    for i in range(120):
        f = facts[i % len(facts)]
        prompt = (
            f"User: Query: Please state the {f['relation'].replace('_', ' ')} of the specified item.\n"
            f"Question: Which relationship predicate is requested by this query?\n"
            f"Answer concisely:\nAssistant: "
        )
        target = f['relation']
        rec = make_record(
            "task_b_relation_recognition", "dev_relation_disjoint" if i >= 80 else "train_exemplar",
            [f["entity_name"]], f["entity_name"], f["relation"], target,
            [], [], False, "none", prompt, target
        )
        task_records["task_b_relation_recognition"].append(rec)

    # --------------------------------------------------------------------------
    # Task C: Context-Dependent Fact Retrieval (Single-hop factual extraction)
    # --------------------------------------------------------------------------
    for i in range(120):
        f = facts[i % len(facts)]
        prompt = (
            f"User: Context statement: {f['entity_name']} has {f['relation'].replace('_', ' ')} recorded as {f['value']}.\n"
            f"Question: What is the {f['relation'].replace('_', ' ')} of {f['entity_name']}?\n"
            f"Answer concisely:\nAssistant: "
        )
        target = f['value']
        rec = make_record(
            "task_c_fact_retrieval", "dev_entity_disjoint" if i >= 80 else "train_exemplar",
            [f["entity_name"]], f["entity_name"], f["relation"], target,
            [], [], False, "none", prompt, target
        )
        task_records["task_c_fact_retrieval"].append(rec)

    # --------------------------------------------------------------------------
    # Task D: Multi-Relation Binding (Entity has 3 relations in context; query 1)
    # --------------------------------------------------------------------------
    for i in range(160):
        e = entities[ent_keys[i % len(ent_keys)]]
        rels = list(e["relations"].keys())
        # Pick 1 query relation, other 2 are distractors
        q_rel = rels[i % len(rels)]
        gt_val = e["relations"][q_rel]
        other_rels = [r for r in rels if r != q_rel]
        dist_vals = [e["relations"][r] for r in other_rels]
        
        prompt = (
            f"User: Specification dossier for {e['name']}:\n"
            f"1. {rels[0].replace('_', ' ')}: {e['relations'][rels[0]]}\n"
            f"2. {rels[1].replace('_', ' ')}: {e['relations'][rels[1]]}\n"
            f"3. {rels[2].replace('_', ' ')}: {e['relations'][rels[2]]}\n"
            f"Question: What is the {q_rel.replace('_', ' ')} of {e['name']}?\n"
            f"Answer concisely:\nAssistant: "
        )
        rec = make_record(
            "task_d_multi_relation_binding", "dev_compositional_unseen_pair" if i >= 100 else "train_exemplar",
            [e["name"]], e["name"], q_rel, gt_val,
            [], dist_vals, False, "none", prompt, gt_val
        )
        task_records["task_d_multi_relation_binding"].append(rec)

    # --------------------------------------------------------------------------
    # Task E: Compositional Generalization (Unseen entity-relation pairs)
    # --------------------------------------------------------------------------
    for i in range(120):
        f = facts[(i * 3 + 1) % len(facts)]
        prompt = (
            f"User: Fact entry: The attribute {f['relation'].replace('_', ' ')} for {f['entity_name']} corresponds to {f['value']}.\n"
            f"Query: State {f['relation'].replace('_', ' ')} of {f['entity_name']}.\n"
            f"Answer concisely:\nAssistant: "
        )
        rec = make_record(
            "task_e_compositional_generalization", "dev_compositional_unseen_pair",
            [f["entity_name"]], f["entity_name"], f["relation"], f["value"],
            [], [], False, "none", prompt, f["value"]
        )
        task_records["task_e_compositional_generalization"].append(rec)

    # --------------------------------------------------------------------------
    # Task F: Contextual Arbitrary Binding (Synthetic counterfactual associations)
    # --------------------------------------------------------------------------
    for i in range(120):
        synth_ent = f"SynthCluster_{i:03d}"
        synth_rel = f"synthetic_property_{i % 5}"
        synth_val = f"AlphaState_{i * 17 % 500}"
        prompt = (
            f"User: System In-Context Register: {synth_ent} is bound with {synth_rel} as {synth_val}.\n"
            f"Question: What is the {synth_rel} for {synth_ent}?\n"
            f"Answer concisely:\nAssistant: "
        )
        rec = make_record(
            "task_f_contextual_binding", "dev_contextual_arbitrary_binding",
            [synth_ent], synth_ent, synth_rel, synth_val,
            [], [], False, "none", prompt, synth_val
        )
        task_records["task_f_contextual_binding"].append(rec)

    # --------------------------------------------------------------------------
    # Task G: Distractor Resistance (Conflicting entities with identical relation)
    # --------------------------------------------------------------------------
    for i in range(120):
        # Pick 2 entities from the same domain so they share relations
        domain = DOMAINS[i % len(DOMAINS)]
        d_ent_keys = [k for k in ent_keys if entities[k]["domain"] == domain]
        e1 = entities[d_ent_keys[i % len(d_ent_keys)]]
        e2 = entities[d_ent_keys[(i + 3) % len(d_ent_keys)]]
        rel = list(e1["relations"].keys())[i % 3]
        v1 = e1["relations"][rel]
        v2 = e2["relations"][rel]
        
        prompt = (
            f"User: Comparative review:\n"
            f"- {e1['name']}: {rel.replace('_', ' ')} is {v1}.\n"
            f"- {e2['name']}: {rel.replace('_', ' ')} is {v2}.\n"
            f"Question: Specifically for {e1['name']}, what is its {rel.replace('_', ' ')}?\n"
            f"Answer concisely:\nAssistant: "
        )
        rec = make_record(
            "task_g_distractor_resistance", "dev_entity_disjoint" if i >= 60 else "train_exemplar",
            [e1["name"], e2["name"]], e1["name"], rel, v1,
            [e2["name"]], [v2], False, "none", prompt, v1
        )
        task_records["task_g_distractor_resistance"].append(rec)

    # --------------------------------------------------------------------------
    # Task H: Explicit Abstention (Missing entity, missing relation, contradictory)
    # --------------------------------------------------------------------------
    abs_reasons = ["entity_absent", "relation_absent", "contradictory_context"]
    abs_resp = "Information not provided in context."
    for i in range(120):
        reason = abs_reasons[i % len(abs_reasons)]
        e = entities[ent_keys[i % len(ent_keys)]]
        
        if reason == "entity_absent":
            q_ent = "UnregisteredEntity_999"
            prompt = (
                f"User: Recorded information covers {e['name']}.\n"
                f"Question: What is the details for {q_ent}?\n"
                f"Answer concisely:\nAssistant: "
            )
            ctx_ents = [e["name"]]
        elif reason == "relation_absent":
            q_ent = e["name"]
            prompt = (
                f"User: Log record: {e['name']} is documented in the domain database.\n"
                f"Question: What is the unregistered_quantum_flux of {e['name']}?\n"
                f"Answer concisely:\nAssistant: "
            )
            ctx_ents = [e["name"]]
        else: # contradictory
            q_ent = e["name"]
            prompt = (
                f"User: Record conflict: Entry 1 states {e['name']} is active. Entry 2 states {e['name']} never existed.\n"
                f"Question: Confirm status of {e['name']}?\n"
                f"Answer concisely:\nAssistant: "
            )
            ctx_ents = [e["name"]]

        rec = make_record(
            "task_h_abstention", "dev_entity_disjoint" if i >= 60 else "train_exemplar",
            ctx_ents, q_ent, "abstention_query", abs_resp,
            [], [], True, reason, prompt, abs_resp
        )
        task_records["task_h_abstention"].append(rec)

    return task_records


def save_and_validate_datasets(out_dir: Path) -> Dict[str, Any]:
    print("Generating and validating all Task datasets (A-H)...")
    from experiments.ucip22.validators.benchmark_validator import UCIP22BenchmarkValidator
    validator = UCIP22BenchmarkValidator()
    
    tasks = build_all_tasks(seed=42)
    summary = {}
    
    for task_name, recs in tasks.items():
        fpath = out_dir / f"{task_name}.jsonl"
        valid_count = 0
        with open(fpath, "w", encoding="utf-8") as f:
            for r in recs:
                is_valid, errs = validator.validate_record(r)
                if not is_valid:
                    raise RuntimeError(f"Record {r['id']} failed validation: {errs}")
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                valid_count += 1
        summary[task_name] = {
            "record_count": valid_count,
            "file_path": str(fpath),
            "all_validated": True
        }
        print(f"  {task_name}: {valid_count} records validated and saved.")

    return summary


if __name__ == "__main__":
    out_dir = ROOT / "experiments/ucip23/datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = save_and_validate_datasets(out_dir)
    print("All task datasets successfully generated!")
