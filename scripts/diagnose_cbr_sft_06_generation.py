"""
scripts/diagnose_cbr_sft_06_generation.py

Token-by-token generation and teacher-forced probability diagnostics:
Compares CBR-SFT-04 Step 200 vs CBR-SFT-05 Step 100 on diagnostic prompts:
1. Teacher-forced next-token probabilities & ranks for correct target tokens.
2. Greedy generation outputs, token IDs, prefix length, and divergence point.
"""

import sys
import os
import json
import math
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from chakrview.brain.model import ChakrMicro, ModelConfig
from scripts.run_m3_evaluations import load_tokenizer_artifacts

def load_model(ckpt_path: Path):
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict") or ckpt.get("model")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def analyze_prompt(
    model: ChakrMicro,
    tok: Any,
    prompt: str,
    target: str,
    max_new_tokens: int = 32
) -> Dict[str, Any]:
    prompt_ids = tok.encode(prompt)
    target_ids = tok.encode(target)

    # 1. Greedy Generation
    input_ids = torch.tensor([prompt_ids], dtype=torch.long)
    generated_ids = []
    current_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(current_ids)[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).item()
            generated_ids.append(next_token)
            current_ids = torch.cat([current_ids, torch.tensor([[next_token]])], dim=1)
            # Stop if EOS or newline
            if next_token in [0, 1] or current_ids.size(1) >= 512:
                break

    try:
        gen_text = tok.decode(generated_ids)
    except Exception:
        gen_text = "<decode_error>"

    # 2. Teacher-Forced Target Analysis
    teacher_ids = prompt_ids + target_ids
    t_input = torch.tensor([teacher_ids[:-1]], dtype=torch.long)

    token_probs = []
    token_ranks = []
    token_nlls = []

    with torch.no_grad():
        logits = model(t_input)[0]  # [seq_len - 1, vocab_size]
        # Inspect positions corresponding to target tokens
        # Target tokens start at index len(prompt_ids) - 1 in logits
        start_pos = len(prompt_ids) - 1
        for i, tid in enumerate(target_ids):
            logit_pos = start_pos + i
            if logit_pos < logits.size(0):
                step_logits = logits[logit_pos]
                probs = F.softmax(step_logits, dim=-1)
                p_tid = probs[tid].item()
                nll = -math.log(max(p_tid, 1e-12))
                # Compute rank
                sorted_indices = torch.argsort(step_logits, descending=True)
                rank = (sorted_indices == tid).nonzero(as_tuple=True)[0].item() + 1

                token_probs.append(round(p_tid, 4))
                token_ranks.append(rank)
                token_nlls.append(round(nll, 4))

    return {
        "prompt": prompt,
        "target": target,
        "target_ids": target_ids,
        "greedy_gen_text": gen_text,
        "greedy_gen_ids": generated_ids[:10],
        "teacher_forced": {
            "token_probs": token_probs,
            "token_ranks": token_ranks,
            "token_nlls": token_nlls,
            "avg_nll": round(sum(token_nlls) / len(token_nlls), 4) if token_nlls else None,
            "first_token_prob": token_probs[0] if token_probs else None,
            "first_token_rank": token_ranks[0] if token_ranks else None
        }
    }

def main():
    vocab_dir = REPO_ROOT / "data" / "experiments" / "vocab_4096"
    tok, _ = load_tokenizer_artifacts(vocab_dir)

    cbr4_ckpt = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_04" / "checkpoints" / "sft_checkpoint_step_200.pt"
    cbr5_ckpt = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_05" / "checkpoints" / "sft_checkpoint_step_100.pt"

    print("Loading CBR-SFT-04 Step 200...")
    m_cbr4 = load_model(cbr4_ckpt)
    print("Loading CBR-SFT-05 Step 100...")
    m_cbr5 = load_model(cbr5_ckpt)

    test_prompts = [
        # Known IP
        (
            "User: The edge sensor gateway is set to 172.24.11.99.\nQuestion: What is the IP address of the edge sensor gateway?\nAssistant:",
            " 172.24.11.99"
        ),
        # Known Port
        (
            "User: The monitoring dashboard portal is set to 7443.\nQuestion: What is the port of the monitoring dashboard portal?\nAssistant:",
            " 7443"
        ),
        # Unknown Key (Abstention)
        (
            "User: Given variables: DB_NAME=production, HOST=db-01.internal\nQuestion: What is the value of PASSWORD?\nAssistant:",
            " That information is not provided in the context."
        ),
        # Two Similar Values
        (
            "User: Node alpha is at 10.0.1.5 and node beta is at 10.0.1.9.\nQuestion: What is the IP address of node beta?\nAssistant:",
            " 10.0.1.9"
        ),
        # Multi-turn Retention
        (
            "User: My project name is ChakrCore.\nAssistant: Understood, I will remember that.\nUser: What is my project name?\nAssistant:",
            " ChakrCore"
        )
    ]

    out_records = []
    for prompt, target in test_prompts:
        res4 = analyze_prompt(m_cbr4, tok, prompt, target)
        res5 = analyze_prompt(m_cbr5, tok, prompt, target)
        out_records.append({
            "prompt": prompt,
            "target": target,
            "cbr4_step200": res4,
            "cbr5_step100": res5
        })

    out_file = REPO_ROOT / "evaluation" / "cbr_sft_06_suite" / "token_level_diagnostics.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_records, f, indent=2)

    print(f"Token-level diagnostics complete. Saved to {out_file}")

if __name__ == "__main__":
    main()
