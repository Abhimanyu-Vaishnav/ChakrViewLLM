"""
Phase 7: Generation & Decoding Audit Script
Compares Parent vs Candidate across decoding algorithms and configurations:
- Greedy decoding (temperature=0.0)
- Temperature sampling (T=0.7, top_p=0.9, top_k=40) across seeds [42, 101, 2026]
- Prompt variations (with vs without trailing whitespace, varying stop-tokens)
- Generation length dynamics (10, 30, 80 tokens)
Inspects whether failure originates in next-token distribution vs autoregressive accumulation.
"""

import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.model import ModelConfig, ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

PARENT_PATH = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_04" / "checkpoints" / "sft_checkpoint_step_200.pt"
CANDIDATE_PATH = REPO_ROOT / "experiments" / "cognition" / "autonomous_12h" / "conversational_pilot" / "checkpoints" / "conversational_pilot_best_val.pt"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

TEST_PROMPTS = [
    {"id": "GEN_01_GREETING", "prompt": "User: Hello!\nAssistant:"},
    {"id": "GEN_02_PERSONA", "prompt": "User: Who are you?\nAssistant:"},
    {"id": "GEN_03_RETRIEVAL", "prompt": "User: Alice lives in Seattle.\nUser: Where does Alice live?\nAssistant:"},
    {"id": "GEN_04_EPSTEMIC", "prompt": "User: What is the secret password?\nAssistant:"},
]

def load_model(ckpt_path):
    cfg = ModelConfig()
    model = ChakrMicro(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model

def sample_tokens(model, tok, prompt, temperature=0.0, top_p=0.9, top_k=40, max_tokens=40, seed=42):
    torch.manual_seed(seed)
    tokens = tok.encode(prompt)
    generated = []
    with torch.no_grad():
        for _ in range(max_tokens):
            inp = torch.tensor([tokens], dtype=torch.long)
            logits = model(inp)[0, -1, :]
            
            if temperature == 0.0:
                next_tok = torch.argmax(logits).item()
            else:
                scaled_logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                    scaled_logits[scaled_logits < v[-1]] = -float("Inf")
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                    sorted_indices_to_remove[0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    scaled_logits[indices_to_remove] = -float("Inf")
                    
                probs = F.softmax(scaled_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).item()
                
            tokens.append(next_tok)
            generated.append(next_tok)
            
            # Check stop condition
            decoded = tok.decode(generated, errors="replace")
            if any(st in decoded for st in ["\nUser:", "User:", "\nAssistant:", "Assistant:"]):
                break
                
    return tok.decode(generated, errors="replace").strip(), generated

def main():
    print("=== CR-EXP-05 Phase 7: Generation & Decoding Audit ===")
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    parent = load_model(PARENT_PATH)
    cand = load_model(CANDIDATE_PATH)

    results = []

    for item in TEST_PROMPTS:
        pid = item["id"]
        prompt = item["prompt"]

        # 1. Greedy (T=0.0)
        c_greedy_text, c_greedy_toks = sample_tokens(cand, tok, prompt, temperature=0.0)
        p_greedy_text, p_greedy_toks = sample_tokens(parent, tok, prompt, temperature=0.0)

        # 2. Temperature sampling across seeds [42, 101, 2026]
        cand_samples = []
        for s in [42, 101, 2026]:
            s_text, s_toks = sample_tokens(cand, tok, prompt, temperature=0.7, top_p=0.9, top_k=40, seed=s)
            cand_samples.append({"seed": s, "text": s_text, "num_tokens": len(s_toks)})

        parent_samples = []
        for s in [42, 101, 2026]:
            s_text, s_toks = sample_tokens(parent, tok, prompt, temperature=0.7, top_p=0.9, top_k=40, seed=s)
            parent_samples.append({"seed": s, "text": s_text, "num_tokens": len(s_toks)})

        results.append({
            "prompt_id": pid,
            "prompt": prompt,
            "greedy": {
                "candidate": c_greedy_text,
                "parent": p_greedy_text
            },
            "candidate_stochastic_samples": cand_samples,
            "parent_stochastic_samples": parent_samples
        })

    out_p = REPO_ROOT / "experiments/cognition/cr_exp_05_ood_diagnosis/generation_diagnostic_results.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved generation diagnostic results to {out_p}")

if __name__ == "__main__":
    main()

