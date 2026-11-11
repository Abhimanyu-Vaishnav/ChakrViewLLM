"""
Stage 5: Counterfactual Reassignment Diagnostic Script
Tests whether generated answers follow the latest contextual mapping or memorized prior associations.
Evaluates:
- Target sequence: e.g. "Ravi lives in Delhi" vs "Ravi lives in Jaipur" vs "Ravi lives in Mumbai"
- Greedy generation
- Top-1 and Top-5 tokens
- Sequence log probability and target rank
- Checks if model mentions unprovided subjects or falls back to static associations.
"""

import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"
MODEL_PATH = REPO_ROOT / "experiments/cognition/cr_exp_06_variable_binding/control_arm/checkpoints/control_arm_best_val.pt"

REASSIGNMENT_TESTS = [
    {
        "id": "REASSIGN_01_DELHI",
        "subject": "Ravi",
        "value": "Delhi",
        "prompt": "Mapping: Ravi lives in Delhi. Query: Where does Ravi live? Answer:",
        "target_seq": "Delhi"
    },
    {
        "id": "REASSIGN_02_JAIPUR",
        "subject": "Ravi",
        "value": "Jaipur",
        "prompt": "Mapping: Ravi lives in Jaipur. Query: Where does Ravi live? Answer:",
        "target_seq": "Jaipur"
    },
    {
        "id": "REASSIGN_03_MUMBAI",
        "subject": "Ravi",
        "value": "Mumbai",
        "prompt": "Mapping: Ravi lives in Mumbai. Query: Where does Ravi live? Answer:",
        "target_seq": "Mumbai"
    },
    {
        "id": "REASSIGN_04_CHENNAI",
        "subject": "Ravi",
        "value": "Chennai",
        "prompt": "Mapping: Ravi lives in Chennai. Query: Where does Ravi live? Answer:",
        "target_seq": "Chennai"
    },
    {
        "id": "REASSIGN_05_PUNE",
        "subject": "Ravi",
        "value": "Pune",
        "prompt": "Mapping: Ravi lives in Pune. Query: Where does Ravi live? Answer:",
        "target_seq": "Pune"
    },
    # Sequential Update in single prompt
    {
        "id": "REASSIGN_06_SEQUENTIAL_UPDATE",
        "subject": "Ravi",
        "value": "Kolkata",
        "prompt": "Mapping: First, Ravi lived in Delhi. Update: Ravi moved and now lives in Kolkata. Query: Where does Ravi live now? Answer:",
        "target_seq": "Kolkata"
    }
]

def load_model(ckpt_path):
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def generate_greedy(model, tok, prompt, max_new_tokens=25):
    tokens = tok.encode(prompt)
    generated = []
    stop_tokens = ["\nUser:", "User:", "\nAssistant:", "Assistant:"]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp = torch.tensor([tokens], dtype=torch.long)
            logits = model(inp)
            next_token = torch.argmax(logits[0, -1, :]).item()
            tokens.append(next_token)
            generated.append(next_token)
            text = tok.decode(generated, errors="replace")
            if any(st in text for st in stop_tokens):
                break
    res = tok.decode(generated, errors="replace")
    for st in stop_tokens:
        if st in res:
            res = res.split(st)[0]
    return res.strip()

def compute_sequence_logprob(model, tok, prompt, target_str):
    p_tokens = tok.encode(prompt)
    t_tokens = tok.encode(" " + target_str)
    all_tokens = p_tokens + t_tokens

    x = torch.tensor([all_tokens[:-1]], dtype=torch.long)
    y = torch.tensor([all_tokens[1:]], dtype=torch.long)

    with torch.no_grad():
        logits = model(x)
        log_probs = F.log_softmax(logits, dim=-1)

        target_start = len(p_tokens) - 1
        seq_logprob = 0.0
        for i in range(target_start, len(all_tokens) - 1):
            tok_id = y[0, i].item()
            seq_logprob += log_probs[0, i, tok_id].item()

    return round(seq_logprob, 4)

def main():
    print("=== CR-EXP-07 Stage 5: Counterfactual Reassignment Diagnostics ===")
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    model = load_model(MODEL_PATH)

    results = []

    for test in REASSIGNMENT_TESTS:
        tid = test["id"]
        prompt = test["prompt"]
        target = test["target_seq"]
        subj = test["subject"]

        gen_text = generate_greedy(model, tok, prompt)
        seq_logprob = compute_sequence_logprob(model, tok, prompt, target)

        # Inspect next-token distribution at delimiter
        tokens = tok.encode(prompt)
        with torch.no_grad():
            logits = model(torch.tensor([tokens], dtype=torch.long))[0, -1, :]
            probs = F.softmax(logits, dim=-1)
            top1_id = torch.argmax(logits).item()
            top5_ids = torch.topk(logits, k=5).indices.tolist()
            top5_tokens = [repr(tok.decode([i], errors="replace")) for i in top5_ids]

        # Check if output contains correct target
        target_in_gen = target.lower() in gen_text.lower()
        # Check if output mentions unprovided subject
        hallucinated_subject = False
        training_subjects = ["Kabir", "Priya", "Amit", "Kavita", "Neha", "Sunil"]
        for ts in training_subjects:
            if ts != subj and ts.lower() in gen_text.lower():
                hallucinated_subject = True
                break

        classification = "FAILURE"
        if target_in_gen and not hallucinated_subject:
            classification = "CORRECT_BINDING"
        elif target_in_gen and hallucinated_subject:
            classification = "SUBJECT_SUBSTITUTION_HALLUCINATION"
        elif not target_in_gen and hallucinated_subject:
            classification = "UNSUPPORTED_SUBJECT_HALLUCINATION"
        else:
            classification = "STATIC_CONVERSATIONAL_PRIOR"

        results.append({
            "test_id": tid,
            "subject": subj,
            "context_value": target,
            "prompt": prompt,
            "generated_output": gen_text,
            "target_sequence_logprob": seq_logprob,
            "top1_token": tok.decode([top1_id], errors="replace"),
            "top5_tokens": top5_tokens,
            "target_present_in_generation": target_in_gen,
            "hallucinated_subject_present": hallucinated_subject,
            "classification": classification
        })
        print(f"[{tid}] Target: {target:8s} | Output: {repr(gen_text)} | Class: {classification}")

    out_file = REPO_ROOT / "experiments/cognition/cr_exp_07_binding_diagnosis/counterfactual_reassignment_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nStage 5 results written to {out_file}")

if __name__ == "__main__":
    main()

