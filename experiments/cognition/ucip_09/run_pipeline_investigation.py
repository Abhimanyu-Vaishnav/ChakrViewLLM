import sys
import json
import time
import math
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path(r"D:\Project\ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.runtime.sampling import apply_repetition_penalty, apply_no_repeat_ngram

PROBES_FILE = REPO_ROOT / "experiments" / "cognition" / "ucip_08" / "sealed_owner_reality_100.json"
CHECKPOINTS = {
    "ucip07_medium": REPO_ROOT / "releases" / "ucip_07_candidates" / "chakrview_ucip07_medium_model.pt",
    "ucip08_medium": REPO_ROOT / "releases" / "ucip_08_candidates" / "chakrview_ucip08_medium_model.pt"
}

ARMS = [
    {"name": "Arm_A_Raw_Greedy", "rep_penalty": 1.0, "no_repeat_ngram": 0},
    {"name": "Arm_B_RepPen_Only", "rep_penalty": 1.15, "no_repeat_ngram": 0},
    {"name": "Arm_C_NoRepeat3_Only", "rep_penalty": 1.0, "no_repeat_ngram": 3},
    {"name": "Arm_D_Chat_CLI_Default", "rep_penalty": 1.15, "no_repeat_ngram": 3},
    {"name": "Arm_E_Conservative", "rep_penalty": 1.05, "no_repeat_ngram": 0},
]

def run_diagnostics():
    probes_data = json.load(open(PROBES_FILE, encoding="utf-8"))["probes"]
    print(f"Loaded {len(probes_data)} probes from {PROBES_FILE.name}")

    results = {}

    for ckpt_key, ckpt_path in CHECKPOINTS.items():
        print(f"\n=======================================================")
        print(f"Initializing DirectNeuralRunner for {ckpt_key}...")
        runner = DirectNeuralRunner(ckpt_path)
        tokenizer = runner.tokenizer
        model = runner.model
        model.eval()

        ckpt_results = {"arms": {arm["name"]: [] for arm in ARMS}, "token_level_traces": []}

        # Select 5 representative probes for deep token-level trace
        trace_indices = [0, 15, 25, 65, 75] # Greeting, Explanation, Hindi, Copying, Arithmetic

        for idx, probe in enumerate(probes_data):
            raw_prompt = probe["prompt"]
            # IMPORTANT: format prompt according to training convention
            formatted_prompt = f"User: {raw_prompt}\nAssistant:"
            prompt_tokens = tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)

            is_trace_probe = idx in trace_indices

            for arm in ARMS:
                rep_pen = arm["rep_penalty"]
                nr_ngram = arm["no_repeat_ngram"]

                curr_tokens = list(prompt_tokens)
                gen_token_ids = []
                step_diagnostics = []

                t0 = time.perf_counter()
                with torch.no_grad():
                    for step in range(48):
                        seq_tensor = torch.tensor([curr_tokens], dtype=torch.long)
                        if seq_tensor.shape[1] > model.config.max_seq_len:
                            seq_tensor = seq_tensor[:, -model.config.max_seq_len:]

                        logits = model(seq_tensor)[:, -1, :].squeeze(0)

                        if is_trace_probe and arm["name"] == "Arm_A_Raw_Greedy" and step < 5:
                            probs = F.softmax(logits, dim=-1)
                            entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
                            top5_probs, top5_ids = torch.topk(probs, 5)
                            step_diagnostics.append({
                                "step": step,
                                "entropy": entropy,
                                "top5": [{"token_id": int(tid), "prob": float(p), "text": tokenizer.decode([int(tid)], errors="replace")} for p, tid in zip(top5_probs, top5_ids)]
                            })

                        # Apply repetition penalty
                        if rep_pen != 1.0 and gen_token_ids:
                            logits = apply_repetition_penalty(logits, gen_token_ids, rep_pen)

                        # Apply no-repeat n-gram
                        if nr_ngram > 0 and len(curr_tokens) >= nr_ngram:
                            logits = apply_no_repeat_ngram(logits, curr_tokens, nr_ngram)

                        next_token_id = int(torch.argmax(logits).item())
                        if next_token_id == 1: # EOS
                            gen_token_ids.append(next_token_id)
                            break

                        gen_token_ids.append(next_token_id)
                        curr_tokens.append(next_token_id)

                duration_ms = (time.perf_counter() - t0) * 1000
                tokens_to_decode = gen_token_ids[:-1] if gen_token_ids and gen_token_ids[-1] == 1 else gen_token_ids
                decoded_text = tokenizer.decode(tokens_to_decode, errors="replace")

                ckpt_results["arms"][arm["name"]].append({
                    "id": probe["id"],
                    "category": probe["category"],
                    "prompt": raw_prompt,
                    "generated_text": decoded_text,
                    "gen_token_count": len(gen_token_ids),
                    "ended_with_eos": bool(gen_token_ids and gen_token_ids[-1] == 1),
                    "duration_ms": duration_ms
                })

                if is_trace_probe and arm["name"] == "Arm_A_Raw_Greedy":
                    ckpt_results["token_level_traces"].append({
                        "probe_id": probe["id"],
                        "category": probe["category"],
                        "prompt": raw_prompt,
                        "step_diagnostics": step_diagnostics
                    })

        results[ckpt_key] = ckpt_results

    out_file = REPO_ROOT / "experiments" / "cognition" / "ucip_09" / "pipeline_diagnostics_raw.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved raw diagnostics to {out_file}")

if __name__ == "__main__":
    run_diagnostics()
