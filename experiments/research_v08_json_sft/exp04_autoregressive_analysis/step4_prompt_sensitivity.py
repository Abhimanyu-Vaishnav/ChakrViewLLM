"""Step 4: Prompt-Format Sensitivity Diagnostic Suite.

Tests whether the model's in-context copying failure is driven by prompt format or value position.
Creates a deterministic diagnostic set using 6 identical underlying status values:
3 Train values ('nominal', 'active', 'bearing_greased') and 3 Test values ('optical_alignment_fault', 'beam_misaligned', 'cryo_quench').

Variations across 5 format conditions:
1. Standard baseline: Instruction -> Context with value buried -> Response:
2. Value BEFORE Instruction: Context (Value first) -> Instruction -> Response:
3. Explicit Key Delimiter Anchor: Context: Status = <value> | Component = <name> -> Instruction -> Response:
4. Direct Copy Framing: Extract the exact status string '<value>' into JSON -> Response:
5. Forced Response Prefix: Prompt ends with 'Response: {"status": "' to bypass opening syntax generation.

Measures:
- Free-running greedy output.
- Exact string match.
- First token generated after opening prefix.
- First status token rank and probability under the new prompt format.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v08_json_sft.exp02_string_copy.evaluate_exp02 import (
    generate_greedy,
    evaluate_output_case,
)

OUTPUT_FILE = Path(__file__).resolve().parent / "step4_prompt_sensitivity.json"


def run_prompt_sensitivity_analysis():
    print("=" * 70)
    print("STEP 4: PROMPT-FORMAT SENSITIVITY DIAGNOSTIC SUITE")
    print("=" * 70)

    device = torch.device("cpu")
    tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")

    exp02_path = REPO_ROOT / "experiments" / "research_v08_json_sft" / "exp02_string_copy" / "checkpoints" / "exp02_best_checkpoint.pt"
    model = ChakrMicro(ModelConfig(max_seq_len=512)).to(device)
    st = torch.load(exp02_path, map_location=device, weights_only=False)
    model.load_state_dict(st["model_state_dict"])
    model.eval()

    target_values = [
        {"role": "Train", "status": "nominal", "component": "subsystem_alpha_01"},
        {"role": "Train", "status": "active", "component": "subsystem_beta_02"},
        {"role": "Train", "status": "bearing_greased", "component": "subsystem_gamma_03"},
        {"role": "Test", "status": "optical_alignment_fault", "component": "relay_eta_01"},
        {"role": "Test", "status": "beam_misaligned", "component": "relay_theta_02"},
        {"role": "Test", "status": "cryo_quench", "component": "relay_eta_03"},
    ]

    conditions = [
        {
            "name": "Cond_1_Standard",
            "desc": "Standard: Instruction -> Context (buried) -> Response:",
            "builder": lambda s, c: (
                f"Instruction: Extract the system status as a JSON object with key \"status\".\n"
                f"Context: Telemetry monitor report: Component {c} reported status '{s}' during periodic scan.\n"
                f"Response:"
            ),
            "forced_prefix": None,
        },
        {
            "name": "Cond_2_Value_Before_Instruction",
            "desc": "Value before instruction: Context -> Instruction -> Response:",
            "builder": lambda s, c: (
                f"Context: Telemetry report for {c}: status is '{s}'.\n"
                f"Instruction: Return the status as JSON {{\"status\": string}}.\n"
                f"Response:"
            ),
            "forced_prefix": None,
        },
        {
            "name": "Cond_3_Explicit_Key_Delimiter",
            "desc": "Explicit key delimiter anchor: Status: <s> -> Response:",
            "builder": lambda s, c: (
                f"Status: {s}\n"
                f"Component: {c}\n"
                f"Extract to JSON with key \"status\":\n"
                f"Response:"
            ),
            "forced_prefix": None,
        },
        {
            "name": "Cond_4_Direct_Copy_Framing",
            "desc": "Direct copy wording: Copy status string <s> into JSON",
            "builder": lambda s, c: (
                f"Instruction: Copy the exact status string '{s}' into JSON object {{\"status\": \"{s}\"}}.\n"
                f"Response:"
            ),
            "forced_prefix": None,
        },
        {
            "name": "Cond_5_Forced_Prefix_Bypass",
            "desc": "Bypass opening syntax: Prompt ends with Response: {\"status\": \"",
            "builder": lambda s, c: (
                f"Instruction: Extract the system status as a JSON object with key \"status\".\n"
                f"Context: Telemetry monitor report: Component {c} reported status '{s}' during periodic scan.\n"
                f"Response: {{\"status\": \""
            ),
            "forced_prefix": '{"status": "',
        },
    ]

    all_condition_results = {}

    for cond in conditions:
        c_name = cond["name"]
        c_desc = cond["desc"]
        forced_pre = cond["forced_prefix"]
        print(f"\n--- {c_name}: {c_desc} ---")

        cond_matches = 0
        cond_parsed = 0
        cond_schema = 0
        examples = []

        for item in target_values:
            val = item["status"]
            comp = item["component"]
            role = item["role"]

            prompt_text = cond["builder"](val, comp)
            p_toks = tok.encode(prompt_text)
            inp = [0] + p_toks

            # Free generation
            gen_out = generate_greedy(model, tok, inp, max_new_tokens=32, context_limit=256)
            raw_gen = gen_out["text"]

            # If forced prefix condition, full response is forced_pre + raw_gen
            full_out_text = (forced_pre + raw_gen) if forced_pre else raw_gen
            ev = evaluate_output_case("single_field_string_extraction", full_out_text, expected_status=val)

            if ev["json_parsed"]:
                cond_parsed += 1
            if ev["schema_valid"]:
                cond_schema += 1
            if ev["status_match"]:
                cond_matches += 1

            # Logit analysis at the end of the prompt
            with torch.no_grad():
                logits = model(torch.tensor([inp], dtype=torch.long))[0, -1, :]
            probs = F.softmax(logits, dim=-1)

            # Check target first status subword
            # Target token for the first subword of the status:
            first_status_tok = tok.encode(val)[0]
            first_status_tok_str = tok.decode([first_status_tok])
            tok_prob = round(probs[first_status_tok].item(), 4)
            tok_rank = int((torch.argsort(logits, descending=True) == first_status_tok).nonzero(as_tuple=True)[0].item()) + 1

            top3_inds = torch.argsort(logits, descending=True)[:3]
            top3_tokens = [(tok.decode([int(idx.item())]), round(probs[idx].item(), 4)) for idx in top3_inds]

            ex_rec = {
                "role": role,
                "status": val,
                "prompt_snippet": prompt_text.splitlines()[-2],
                "full_output": full_out_text,
                "exact_match": ev["status_match"],
                "target_first_tok": repr(first_status_tok_str),
                "target_rank": tok_rank,
                "target_prob": tok_prob,
                "top3_predicted": top3_tokens,
            }
            examples.append(ex_rec)

            print(f"  [{role:5s}] {val:24s} | Out: {repr(full_out_text):26s} | Match: {str(ev['status_match']):5s} | 1st Tok Rank: {tok_rank:3d} (P={tok_prob:.3f})")

        cond_summary = {
            "name": c_name,
            "description": c_desc,
            "json_parsed": cond_parsed,
            "schema_valid": cond_schema,
            "exact_matches": cond_matches,
            "total": len(target_values),
            "examples": examples,
        }
        all_condition_results[c_name] = cond_summary

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_condition_results, f, indent=2)

    print(f"\nStep 4 results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_prompt_sensitivity_analysis()
