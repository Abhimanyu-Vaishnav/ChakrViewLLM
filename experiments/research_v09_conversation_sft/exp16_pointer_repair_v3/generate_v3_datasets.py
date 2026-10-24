"""
generate_v3_datasets.py
ChakrView Experiment 16 v3 — Fail-Closed Dataset Generator.

Implements the audited Exp16 v3 design:
1. Balanced Field 7 (missing_context_abstention): 80% answerable evidence present, 20% genuine missing context.
2. Balanced Field 8 (factual_uncertainty_handling): 80% answerable verifiable historical records, 20% genuine unknowable claims.
3. Strict Candidate Masking: Position 0 (BOS abstention sentinel) is True; all evidence-context tokens are True;
   query tokens and assistant scaffolding are strictly False.
4. Fail-closed character-to-token span alignment with target decoded verification.
5. Strict domain separation across Train, Val, and Test.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CANONICAL_TOK_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v09_conversation_sft.exp04_pointer_robustness.span_aligner import (
    align_character_span_to_tokens,
    compute_token_byte_offsets
)

SPLIT_CONFIGS = {
    "train": {
        "prefix": "tr",
        "seed": 4216,
        "fillers": [
            "Substation telemetry feed shows regular dissipation curves.",
            "Transformer relay bank 4 passed voltage insulation check.",
            "Industrial turbine hall reports stable rotation velocity.",
            "Boiler feed pump telemetry confirmed positive head pressure.",
            "Pneumatic conveyor assembly operating within nominal pressure limits.",
            "Factory automation PLC bus 3 validated synchronous pulse timing.",
            "Generator exciter current calibrated to standard excitation curves.",
            "Assembly line robotic arm controllers synchronized via fieldbus."
        ],
        "fn_prefix": "sync_packet"
    },
    "val": {
        "prefix": "va",
        "seed": 5216,
        "fillers": [
            "Fiber optic DWDM transport line operates at nominal optical power.",
            "Core datacenter switch spine fabric maintained zero dropped frames.",
            "Edge caching ingress proxy routed network requests without backpressure.",
            "Optical multiplexer unit chassis reported standard internal temperature.",
            "Telecom base station transceiver array synchronized over CPRI interface.",
            "Border gateway router peering session confirmed full BGP route table."
        ],
        "fn_prefix": "parse_frame"
    },
    "test": {
        "prefix": "te",
        "seed": 6216,
        "fillers": [
            "Orbital telemetry downlink packet stream verified clean parity.",
            "Satellite star tracker sensor achieved stable attitude lock.",
            "Inertial navigation unit gyroscopes confirmed zero drift velocity.",
            "Autonomous marine vessel sonar array mapped seabed contours.",
            "Propulsion gimbal actuators passed hydraulic response calibration.",
            "Flight avionics bus registered steady differential voltage."
        ],
        "fn_prefix": "verify_hash"
    }
}


def generate_split_examples(split_name: str, count_per_cat: int) -> List[Dict[str, Any]]:
    cfg = SPLIT_CONFIGS[split_name]
    rng = random.Random(cfg["seed"])
    prefix = cfg["prefix"]
    examples = []

    # 1. dynamic_entity_binding (Field 0)
    for i in range(count_per_cat):
        ex_id = f"EXP16_V3_{split_name.upper()}_DYN_BIND_{i:03d}"
        is_abs = (i >= count_per_cat - max(1, int(count_per_cat * 0.2)))
        filler = " ".join(rng.sample(cfg["fillers"], k=rng.randint(1, 2)))
        cands = [f"sensor_{prefix}_{i}_{k}" for k in range(3)]
        target_idx = i % 3
        target = cands[target_idx]
        other_cands = [cands[j] for j in range(3) if j != target_idx]

        if not is_abs:
            context = (
                f"Supervisor: {filler} We evaluated {other_cands[0]} and {other_cands[1]} during the test cycle.\n"
                f"Operator: Acknowledged. We routed the primary monitoring telemetry to {target} during execution."
            )
            query = "Supervisor: Which specific sensor node was selected for monitoring?"
            prompt = f"{context}\n{query}\nAssistant:"
            c_s = prompt.rfind(target)
            c_e = c_s + len(target)
            rat = f"Telemetry routed to {target} (slot {target_idx}) among distractors."
        else:
            target = ""
            c_s, c_e = 0, 0
            context = (
                f"Supervisor: {filler} We checked {cands[0]} and {cands[1]}, but the logging channel failed.\n"
                f"Operator: Telemetry connection dropped before any sensor was engaged."
            )
            query = "Supervisor: Which specific sensor node was selected for monitoring?"
            prompt = f"{context}\n{query}\nAssistant:"
            rat = "Connection dropped; no sensor was engaged."

        examples.append({
            "id": ex_id, "split": split_name, "category": "dynamic_entity_binding",
            "field_id": 0, "prompt": prompt, "context_text": context, "target_text": target,
            "char_start": c_s, "char_end": c_e, "is_abstention": is_abs, "rationale": rat
        })

    # 2. turn_dependent_retrieval (Field 1)
    split_offset = {"train": 1000, "val": 3000, "test": 5000}[split_name]
    for i in range(count_per_cat):
        ex_id = f"EXP16_V3_{split_name.upper()}_TURN_RET_{i:03d}"
        is_abs = (i >= count_per_cat - max(1, int(count_per_cat * 0.2)))
        filler = " ".join(rng.sample(cfg["fillers"], k=rng.randint(1, 2)))
        v1 = split_offset + i * 13 + 5
        v2 = split_offset + i * 17 + 25
        v3 = split_offset + i * 19 + 50
        target = f"{v3} MHz"

        if not is_abs:
            context = (
                f"Engineer: {filler} Calibrator stage 1 was initialized at {v1} MHz.\n"
                f"Lead: Stage 2 increased output frequency to {v2} MHz.\n"
                f"Senior Tech: Final stage adjusted frequency to {target} before lock."
            )
            query = "Auditor: What is the latest locked operating frequency setting?"
            prompt = f"{context}\n{query}\nAssistant:"
            c_s = prompt.rfind(target)
            c_e = c_s + len(target)
            rat = f"Latest frequency adjustment locked at {target} in turn 3."
        else:
            target = ""
            c_s, c_e = 0, 0
            context = (
                f"Engineer: {filler} Calibrator stage 1 was initialized at {v1} MHz, but calibration was aborted."
            )
            query = "Auditor: What is the latest locked operating frequency setting?"
            prompt = f"{context}\n{query}\nAssistant:"
            rat = "Calibration aborted; no frequency was locked."

        examples.append({
            "id": ex_id, "split": split_name, "category": "turn_dependent_retrieval",
            "field_id": 1, "prompt": prompt, "context_text": context, "target_text": target,
            "char_start": c_s, "char_end": c_e, "is_abstention": is_abs, "rationale": rat
        })

    # 3. anaphora_reference_resolution (Field 2) - 50% Abstention
    for i in range(count_per_cat):
        ex_id = f"EXP16_V3_{split_name.upper()}_ANAPHORA_{i:03d}"
        is_abs = (i < count_per_cat // 2)
        filler = " ".join(rng.sample(cfg["fillers"], k=rng.randint(1, 2)))
        cand_a = f"Cluster_{prefix}_Alpha_{i}"
        cand_b = f"Cluster_{prefix}_Beta_{i}"

        if is_abs:
            target = ""
            c_s, c_e = 0, 0
            context = (
                f"Admin: {filler} Both {cand_a} and {cand_b} processed the workload batches.\n"
                f"DevOps: It threw an unexpected socket error on port 8080."
            )
            query = "Manager: Which cluster encountered the socket error?"
            prompt = f"{context}\n{query}\nAssistant:"
            rat = f"Pronoun 'It' is ambiguous between {cand_a} and {cand_b}."
        else:
            resolved_target = cand_a if (i % 2 == 0) else cand_b
            other_cluster = cand_b if (i % 2 == 0) else cand_a
            context = (
                f"Admin: {filler} We deployed the compute cluster on {other_cluster} and the database cluster on {resolved_target} in rack 2.\n"
                f"DevOps: The database cluster completed its assigned provisioning routine successfully."
            )
            query = "Manager: Which specific system completed its assigned provisioning routine?"
            prompt = f"{context}\n{query}\nAssistant:"
            c_s = prompt.rfind(resolved_target)
            c_e = c_s + len(resolved_target)
            target = resolved_target
            rat = f"Descriptive anaphora 'the database cluster' uniquely resolves to {resolved_target}."

        examples.append({
            "id": ex_id, "split": split_name, "category": "anaphora_reference_resolution",
            "field_id": 2, "prompt": prompt, "context_text": context, "target_text": target,
            "char_start": c_s, "char_end": c_e, "is_abstention": is_abs, "rationale": rat
        })

    # 4. superseded_instruction_adherence (Field 3)
    formats = ["YAML", "JSON", "Protobuf", "MessagePack", "Avro", "Parquet", "Arrow", "CSV", "XML", "TOML", "CBOR", "NetCDF", "HDF5", "BSON", "PostScript"]
    for i in range(count_per_cat):
        ex_id = f"EXP16_V3_{split_name.upper()}_SUPERSEDED_{i:03d}"
        is_abs = (i >= count_per_cat - max(1, int(count_per_cat * 0.2)))
        filler = " ".join(rng.sample(cfg["fillers"], k=rng.randint(1, 2)))
        fmt_old = formats[i % len(formats)]
        fmt_new = formats[(i + 5) % len(formats)]

        if not is_abs:
            context = (
                f"Client: {filler} Generate the output in {fmt_old} format.\n"
                f"Architect: Correction: do not use {fmt_old}. All downstream pipelines now require {fmt_new} instead."
            )
            query = "Operator: What is the current mandatory serialization format?"
            prompt = f"{context}\n{query}\nAssistant:"
            c_s = prompt.rfind(fmt_new)
            c_e = c_s + len(fmt_new)
            target = fmt_new
            rat = f"Architect instruction explicitly supersedes {fmt_old} with {fmt_new}."
        else:
            target = ""
            c_s, c_e = 0, 0
            context = (
                f"Client: {filler} Generate the output in {fmt_old} format.\n"
                f"Architect: Cancel that export request entirely; do not serialize anything yet."
            )
            query = "Operator: What is the current mandatory serialization format?"
            prompt = f"{context}\n{query}\nAssistant:"
            rat = "Export request was cancelled without replacement."

        examples.append({
            "id": ex_id, "split": split_name, "category": "superseded_instruction_adherence",
            "field_id": 3, "prompt": prompt, "context_text": context, "target_text": target,
            "char_start": c_s, "char_end": c_e, "is_abstention": is_abs, "rationale": rat
        })

    # 5. user_correction_handling (Field 4)
    for i in range(count_per_cat):
        ex_id = f"EXP16_V3_{split_name.upper()}_CORRECTION_{i:03d}"
        is_abs = (i >= count_per_cat - max(1, int(count_per_cat * 0.2)))
        filler = " ".join(rng.sample(cfg["fillers"], k=rng.randint(1, 2)))
        n_orig = f"node_{prefix}_pri_{i}"
        n_corr = f"node_{prefix}_sec_{i}"

        if not is_abs:
            context = (
                f"Dispatcher: {filler} Assign incoming telemetry payload to {n_orig}.\n"
                f"Dispatcher: Wait, that was wrong. Route the telemetry payload to {n_corr} immediately."
            )
            query = "Routing Agent: Which destination node is the payload assigned to?"
            prompt = f"{context}\n{query}\nAssistant:"
            c_s = prompt.rfind(n_corr)
            c_e = c_s + len(n_corr)
            target = n_corr
            rat = f"User explicitly corrects assignment from {n_orig} to {n_corr}."
        else:
            target = ""
            c_s, c_e = 0, 0
            context = (
                f"Dispatcher: {filler} Assign incoming telemetry payload to {n_orig}.\n"
                f"Dispatcher: Stop dispatch immediately. Cancel all active destination node assignments until further research."
            )
            query = "Routing Agent: Which destination node is the payload assigned to?"
            prompt = f"{context}\n{query}\nAssistant:"
            rat = "Assignment cancelled with no replacement target."

        examples.append({
            "id": ex_id, "split": split_name, "category": "user_correction_handling",
            "field_id": 4, "prompt": prompt, "context_text": context, "target_text": target,
            "char_start": c_s, "char_end": c_e, "is_abstention": is_abs, "rationale": rat
        })

    # 6. distractor_resistance (Field 5)
    states = ["READY", "OPERATIONAL", "ONLINE", "RUNNING", "ACTIVE"]
    dist_states = ["MAINTENANCE", "OFFLINE", "DEGRADED", "STANDBY", "OVERHEATED"]
    for i in range(count_per_cat):
        ex_id = f"EXP16_V3_{split_name.upper()}_DISTRACTOR_{i:03d}"
        is_abs = (i >= count_per_cat - max(1, int(count_per_cat * 0.2)))
        filler = " ".join(rng.sample(cfg["fillers"], k=rng.randint(1, 2)))
        target_state = states[i % len(states)]
        d_state_1 = dist_states[i % len(dist_states)]
        d_state_2 = dist_states[(i + 1) % len(dist_states)]
        n_target = f"node_{prefix}_target_{i}"
        n_dist_1 = f"node_{prefix}_aux_a_{i}"
        n_dist_2 = f"node_{prefix}_aux_b_{i}"

        if not is_abs:
            context = (
                f"Health Monitor: {filler} We checked multiple systems. {n_dist_1} is {d_state_1} and {n_dist_2} is {d_state_2}.\n"
                f"Technician: Confirmed. Meanwhile, primary system {n_target} passed diagnostics and remains {target_state} right now."
            )
            query = f"Auditor: What is the operating state of {n_target}?"
            prompt = f"{context}\n{query}\nAssistant:"
            c_s = prompt.rfind(target_state)
            c_e = c_s + len(target_state)
            target = target_state
            rat = f"{n_target} is {target_state}, ignoring {n_dist_1} and {n_dist_2}."
        else:
            target = ""
            c_s, c_e = 0, 0
            context = (
                f"Health Monitor: {filler} {n_dist_1} is {d_state_1}. {n_dist_2} is {d_state_2}."
            )
            query = f"Auditor: What is the operating state of unlisted node {n_target}?"
            prompt = f"{context}\n{query}\nAssistant:"
            rat = f"{n_target} was not checked or mentioned."

        examples.append({
            "id": ex_id, "split": split_name, "category": "distractor_resistance",
            "field_id": 5, "prompt": prompt, "context_text": context, "target_text": target,
            "char_start": c_s, "char_end": c_e, "is_abstention": is_abs, "rationale": rat
        })

    # 7. ambiguity_clarification (Field 6) - 50% Abstention
    c_list = ["emerald", "sapphire", "amber", "quartz", "ruby", "silver", "steel", "copper", "bronze", "topaz"]
    for i in range(count_per_cat):
        ex_id = f"EXP16_V3_{split_name.upper()}_AMBIGUITY_{i:03d}"
        is_abs = (i < count_per_cat // 2)
        filler = " ".join(rng.sample(cfg["fillers"], k=rng.randint(1, 2)))
        color_1 = c_list[i % len(c_list)]
        color_2 = c_list[(i + 1) % len(c_list)]
        pod_1 = f"pod_{prefix}_{color_1}_{i}"
        pod_2 = f"pod_{prefix}_{color_2}_{i}"

        if is_abs:
            target = ""
            c_s, c_e = 0, 0
            context = (
                f"Operator: {filler} Restart the active container pod now.\n"
                f"Support: Both {pod_1} and {pod_2} are running on this host.\n"
                f"Support: Please specify which pod requires the restart.\n"
                f"Operator: The container with high load."
            )
            query = "Engineer: Which specific pod was selected for restart?"
            prompt = f"{context}\n{query}\nAssistant:"
            rat = "Load metrics missing; pod identity cannot be disambiguated."
        else:
            target = pod_2 if (i % 2 == 0) else pod_1
            context = (
                f"Operator: {filler} We need clarification on whether to reboot {pod_1} or {pod_2} today.\n"
                f"Client: After checking the incident log, please proceed with rebooting {target} immediately."
            )
            query = "Engineer: Which specific pod was selected for reboot?"
            prompt = f"{context}\n{query}\nAssistant:"
            c_s = prompt.rfind(target)
            c_e = c_s + len(target)
            rat = f"Client explicitly clarifies {target} as the reboot target."

        examples.append({
            "id": ex_id, "split": split_name, "category": "ambiguity_clarification",
            "field_id": 6, "prompt": prompt, "context_text": context, "target_text": target,
            "char_start": c_s, "char_end": c_e, "is_abstention": is_abs, "rationale": rat
        })

    # 8. missing_context_abstention (Field 7) - Repaired v3 (80% Answerable, 20% Abstention)
    domain_units = {
        "train": ("generator unit", "stator winding resistance", "ohms"),
        "val": ("fiber channel", "optical attenuation factor", "dB/km"),
        "test": ("satellite bus", "orbital perigee altitude", "km")
    }[split_name]
    unit_name, metric_name, unit_val = domain_units

    for i in range(count_per_cat):
        ex_id = f"EXP16_V3_{split_name.upper()}_MISSING_{i:03d}"
        is_abs = (i >= count_per_cat - max(1, int(count_per_cat * 0.2)))
        filler = " ".join(rng.sample(cfg["fillers"], k=rng.randint(1, 2)))
        comp_id = f"unit_{prefix}_{i}"
        val_reading = f"{10 + i * 2}.5 {unit_val}"

        query = f"User: What is the {metric_name} on {unit_name} {comp_id}?"

        if not is_abs:
            context = (
                f"User: {filler} Diagnostic telemetry for {unit_name} {comp_id} recorded {metric_name} as {val_reading} during testing.\n"
                f"Technician: Confirmed recorded reading for {comp_id}."
            )
            prompt = f"{context}\n{query}\nAssistant:"
            c_s = prompt.rfind(val_reading)
            c_e = c_s + len(val_reading)
            target = val_reading
            rat = f"Context explicitly states {metric_name} of {comp_id} is {val_reading}."
        else:
            target = ""
            c_s, c_e = 0, 0
            context = (
                f"User: {filler} All auxiliary telemetry lines report standard baseline dissipation."
            )
            prompt = f"{context}\n{query}\nAssistant:"
            rat = f"Query '{query}' is completely absent from context."

        examples.append({
            "id": ex_id, "split": split_name, "category": "missing_context_abstention",
            "field_id": 7, "prompt": prompt, "context_text": context, "target_text": target,
            "char_start": c_s, "char_end": c_e, "is_abstention": is_abs, "rationale": rat
        })

    # 9. factual_uncertainty_handling (Field 8) - Repaired v3 (80% Answerable, 20% Abstention)
    fault_codes = ["SURGE_FAULT", "GRID_FAULT", "LOCK_LOSS", "TRIP_EVENT", "BUS_SURGE"]
    domain_surges = {
        "train": "power grid feeder bus",
        "val": "core optical trunk",
        "test": "memory bank subsystem"
    }[split_name]

    for i in range(count_per_cat):
        ex_id = f"EXP16_V3_{split_name.upper()}_UNCERTAIN_{i:03d}"
        is_abs = (i >= count_per_cat - max(1, int(count_per_cat * 0.2)))
        filler = " ".join(rng.sample(cfg["fillers"], k=rng.randint(1, 2)))
        dev_id = f"{domain_surges} {prefix}_{i}"
        recorded_status = fault_codes[i % len(fault_codes)]

        query = f"Analyst: Did {dev_id} record a transient state during the surge?"

        if not is_abs:
            context = (
                f"Risk Analyst: {filler} Historical surge incident log audited for {dev_id}.\n"
                f"Auditor: The recorded transient state during the surge was {recorded_status} during testing."
            )
            prompt = f"{context}\n{query}\nAssistant:"
            c_s = prompt.rfind(recorded_status)
            c_e = c_s + len(recorded_status)
            target = recorded_status
            rat = f"Verifiable historical log confirms state was {recorded_status}."
        else:
            target = ""
            c_s, c_e = 0, 0
            context = (
                f"Risk Analyst: {filler} Reviewing general operational telemetry logs without surge instrumentation."
            )
            prompt = f"{context}\n{query}\nAssistant:"
            rat = f"Surge instrumentation absent; transient state during surge cannot be verified."

        examples.append({
            "id": ex_id, "split": split_name, "category": "factual_uncertainty_handling",
            "field_id": 8, "prompt": prompt, "context_text": context, "target_text": target,
            "char_start": c_s, "char_end": c_e, "is_abstention": is_abs, "rationale": rat
        })

    # 10. constrained_code_synthesis (Field 9)
    for i in range(count_per_cat):
        ex_id = f"EXP16_V3_{split_name.upper()}_CODE_{i:03d}"
        is_abs = (i >= count_per_cat - max(1, int(count_per_cat * 0.2)))
        filler = " ".join(rng.sample(cfg["fillers"], k=rng.randint(1, 2)))
        target = f"{cfg['fn_prefix']}_{i}"

        if not is_abs:
            context = (
                f"Developer: {filler} We are designing the cryptographic integrity verification module.\n"
                f"Tech Lead: The validation function must be named {target} in accordance with our API specification."
            )
            query = "Developer: What is the required function identifier for the validation routine?"
            prompt = f"{context}\n{query}\nAssistant:"
            c_s = prompt.rfind(target)
            c_e = c_s + len(target)
            rat = f"API specification explicitly mandates function name {target}."
        else:
            target = ""
            c_s, c_e = 0, 0
            context = (
                f"Developer: {filler} We need to implement a data validation helper function.\n"
                f"Tech Lead: I will decide on the function naming convention next week."
            )
            query = "Developer: What function name has been approved?"
            prompt = f"{context}\n{query}\nAssistant:"
            rat = "No function name approved yet."

        examples.append({
            "id": ex_id, "split": split_name, "category": "constrained_code_synthesis",
            "field_id": 9, "prompt": prompt, "context_text": context, "target_text": target,
            "char_start": c_s, "char_end": c_e, "is_abstention": is_abs, "rationale": rat
        })

    return examples


def compile_split(raw_examples: List[Dict[str, Any]], tokenizer, split_file: Path) -> Tuple[int, int, int]:
    compiled = []
    successes = 0
    abstentions = 0

    for ex in raw_examples:
        prompt = ex["prompt"]
        context_text = ex["context_text"]
        context_byte_len = len(context_text.encode("utf-8"))

        # Fail-closed token alignment
        t_s, t_e, input_ids = align_character_span_to_tokens(
            prompt=prompt,
            char_start=ex["char_start"],
            char_end=ex["char_end"],
            target_text=ex["target_text"],
            tokenizer=tokenizer,
            is_abstention=ex["is_abstention"]
        )
        successes += 1
        if ex["is_abstention"]:
            abstentions += 1

        # Check span length constraint (<= 12)
        if not ex["is_abstention"]:
            span_len = t_e - t_s + 1
            if span_len > 12:
                raise ValueError(f"Span length violation in {ex['id']}: length {span_len} > 12")

        # Build strict candidate mask:
        # Position 0: True (abstention sentinel)
        # Positions overlapping context bytes: True
        # Positions in query / assistant prefix: False
        _, byte_map = compute_token_byte_offsets(prompt, tokenizer)
        mask = [False] * len(input_ids)
        mask[0] = True  # Abstention sentinel

        for tid, s_b, e_b, piece in byte_map[1:]:
            if e_b <= context_byte_len:
                mask[tid] = True

        # Assert Gold Span Invariant:
        if ex["is_abstention"]:
            if t_s != 0 or t_e != 0 or not mask[0]:
                raise ValueError(f"Abstention mask invariant violated in {ex['id']}")
        else:
            if not mask[t_s] or not mask[t_e]:
                raise ValueError(f"Gold span [{t_s}, {t_e}] violates candidate mask in {ex['id']}")

        compiled.append({
            "example_id": ex["id"],
            "split": ex["split"],
            "category": ex["category"],
            "field_id": ex["field_id"],
            "prompt": prompt,
            "context_char_end": len(context_text),
            "target_text": ex["target_text"],
            "char_span": [ex["char_start"], ex["char_end"]],
            "input_ids": input_ids,
            "start_pos": t_s,
            "end_pos": t_e,
            "is_abstention": ex["is_abstention"],
            "candidate_mask": mask,
            "rationale": ex["rationale"]
        })

    with open(split_file, "w", encoding="utf-8") as f:
        for r in compiled:
            f.write(json.dumps(r) + "\n")

    return len(compiled), successes, abstentions


def main():
    tokenizer, _ = load_tokenizer_artifacts(CANONICAL_TOK_DIR, verify_checksums=False)

    train_file = OUTPUT_DIR / "train_exp16_v3.jsonl"
    val_file = OUTPUT_DIR / "val_exp16_v3.jsonl"
    test_file = OUTPUT_DIR / "test_exp16_v3.jsonl"

    print("Generating Exp16 v3 raw examples...")
    train_raw = generate_split_examples("train", 30)
    val_raw = generate_split_examples("val", 6)
    test_raw = generate_split_examples("test", 12)

    print("Compiling Train split...")
    n_train, s_train, a_train = compile_split(train_raw, tokenizer, train_file)
    print(f"  Train compiled: {n_train} examples ({n_train - a_train} answerable, {a_train} abstentions)")

    print("Compiling Val split...")
    n_val, s_val, a_val = compile_split(val_raw, tokenizer, val_file)
    print(f"  Val compiled: {n_val} examples ({n_val - a_val} answerable, {a_val} abstentions)")

    print("Compiling Test split...")
    n_test, s_test, a_test = compile_split(test_raw, tokenizer, test_file)
    print(f"  Test compiled: {n_test} examples ({n_test - a_test} answerable, {a_test} abstentions)")

    print("All splits compiled successfully.")


if __name__ == "__main__":
    main()
