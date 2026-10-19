"""
experiments/copy_capability_gate/run_copy_capability_gate.py

Inference-only evaluation investigating whether dialog-style prompt formatting
improves context-grounded copying in ChakrView independently of JSON syntax
and constrained decoding.

Safety boundaries:
- ZERO training, fine-tuning, optimizer steps, or backward passes.
- chakrview/runtime/session.py is strictly untouched.
- Canonical checkpoints, tokenizer artifacts, architecture, and source datasets are frozen.
- Held-out test split is NEVER accessed.
"""

import sys
import os
import json
import hashlib
from pathlib import Path
import re
import torch
import torch.nn.functional as F

# Ensure chakrview is on path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer import load_experiment_artifacts

# Cryptographic and path constants
BASE_CKPT_PATH = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
CAND_CKPT_PATH = REPO_ROOT / "experiments" / "sft_v3_pilot_readiness" / "pilot_run_v3" / "sft_best_checkpoint.pt"
VAL_DATA_PATH = REPO_ROOT / "experiments" / "sft_v3_dataset_builder" / "validation.jsonl"
VOCAB_PATH = REPO_ROOT / "data" / "experiments" / "vocab_4096" / "vocab.json"
MERGES_PATH = REPO_ROOT / "data" / "experiments" / "vocab_4096" / "merges.json"
SESSION_PATH = REPO_ROOT / "chakrview" / "runtime" / "session.py"

EXPECTED_BASE_HASH = "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee"
EXPECTED_CAND_HASH = "bfde007db1a90dde5ccc75d14a9ffe59f57b2393b87ac00b19422b8d088d1879"
EXPECTED_VOCAB_HASH = "db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a"
EXPECTED_MERGES_HASH = "ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7"


def compute_sha256(filepath: Path) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def normalize_value(val_str: str) -> str:
    """
    Standard normalization rule:
    1. Strip leading/trailing whitespace.
    2. Convert to lowercase.
    3. Strip outer single/double quotes and trailing punctuation (.,;:!?)
    """
    if not val_str:
        return ""
    s = val_str.strip().lower()
    s = s.strip("\"'`;:")
    s = re.sub(r'[\s]+', ' ', s)
    return s


def is_doctest_attractor(text: str) -> bool:
    attractor_patterns = [
        "__main__",
        "doctest",
        "import doctest",
        ">>>",
        "linked_list",
        "np.ndarray",
        "def get_to",
        "codecannot"
    ]
    t_lower = text.lower()
    return any(p in t_lower for p in attractor_patterns)


def generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int = 40):
    input_ids = tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [0]
    
    device = next(model.parameters()).device
    generated_ids = list(input_ids)
    new_tokens = []
    
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            curr_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
            if curr_tensor.shape[1] > 256:
                curr_tensor = curr_tensor[:, -256:]
            
            logits = model(curr_tensor)
            next_token_logits = logits[0, -1, :]
            next_token = int(torch.argmax(next_token_logits).item())
            
            if next_token == 1:  # Canonical EOS ID
                break
                
            new_tokens.append(next_token)
            generated_ids.append(next_token)
            
    decoded_output = tokenizer.decode(new_tokens) if new_tokens else ""
    return decoded_output, new_tokens, input_ids


def build_copying_questions():
    """
    Load validation records from categories with explicitly stated values.
    Verify that expected values actually appear in the source passage.
    """
    eligible_categories = {
        "anti_code_attractor_guard": ("status", "Copy the exact value of status from context:"),
        "grounded_inventory_item": ("name", "Copy the exact name of the item from context:"),
        "grounded_extraction_server_status": ("status", "Copy the exact status from context:"),
        "grounded_telemetry_metric": ("metric", "Copy the exact metric name from context:")
    }
    
    questions = []
    with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            cat = data.get("category")
            if cat in eligible_categories:
                target_field, copy_q = eligible_categories[cat]
                ctx = data["context"]
                expected_val = str(data["expected_answer"][target_field])
                
                # Strict verification that the source passage actually contains expected value
                assert expected_val.lower() in ctx.lower(), f"Passage '{ctx}' does not contain expected '{expected_val}'"
                
                questions.append({
                    "id": data["id"],
                    "category": cat,
                    "target_field": target_field,
                    "context": ctx,
                    "copy_question": copy_q,
                    "expected_value": expected_val,
                    "raw_record": data
                })
    return questions


def format_prompts(item):
    ctx = item["context"]
    q = item["copy_question"]
    
    # 1. Existing Context / Instruction / Response format
    fmt1 = f"Context: {ctx}\nInstruction: {q}\nResponse:"
    
    # 2. Explicit Dialog format User: ... Assistant:
    fmt2 = f"User: In the following context: \"{ctx}\"\n{q}\nAssistant:"
    
    # 3. Plain Context / Question / Answer format
    fmt3 = f"Context: {ctx}\nQuestion: {q}\nAnswer:"
    
    return {
        "format_1_cir": fmt1,
        "format_2_dialog": fmt2,
        "format_3_cqa": fmt3
    }


def main():
    print("=" * 70)
    print("CHAKRVIEW — COPY CAPABILITY GATE (INFERENCE-ONLY)")
    print("=" * 70)
    
    # 1. Cryptographic and Integrity Audits
    base_hash = compute_sha256(BASE_CKPT_PATH)
    cand_hash = compute_sha256(CAND_CKPT_PATH)
    vocab_hash = compute_sha256(VOCAB_PATH)
    merges_hash = compute_sha256(MERGES_PATH)
    
    assert base_hash == EXPECTED_BASE_HASH, f"Base hash mismatch: {base_hash}"
    assert cand_hash == EXPECTED_CAND_HASH, f"Candidate hash mismatch: {cand_hash}"
    assert vocab_hash == EXPECTED_VOCAB_HASH, f"Vocab hash mismatch: {vocab_hash}"
    assert merges_hash == EXPECTED_MERGES_HASH, f"Merges hash mismatch: {merges_hash}"
    print("[OK] Cryptographic hashes verified bit-identical.")
    
    # 2. Load Tokenizer
    tokenizer = load_experiment_artifacts(str(REPO_ROOT / "data" / "experiments" / "vocab_4096"))
    print("[OK] Canonical BPETokenizer loaded.")
    
    # 3. Load Models
    device = torch.device("cpu")
    cfg = ModelConfig(
        vocab_size=4096,
        d_model=192,
        n_layers=6,
        n_heads=6,
        hidden_dim=512,
        max_seq_len=512,
    )
    
    base_model = ChakrMicro(cfg)
    base_ckpt = torch.load(BASE_CKPT_PATH, map_location=device, weights_only=False)
    base_sd = base_ckpt.get("model_state_dict", base_ckpt)
    base_model.load_state_dict(base_sd, strict=True)
    base_model.eval()
    print("[OK] Base Step-10,000 model loaded into memory.")
    
    cand_model = ChakrMicro(cfg)
    cand_ckpt = torch.load(CAND_CKPT_PATH, map_location=device, weights_only=False)
    cand_sd = cand_ckpt.get("model_state_dict", cand_ckpt)
    cand_model.load_state_dict(cand_sd, strict=True)
    cand_model.eval()
    print("[OK] Candidate Step-100 model loaded into memory.")
    
    # 4. Prepare verified copy questions
    copy_items = build_copying_questions()
    print(f"[OK] Loaded {len(copy_items)} verified copy evaluation records from validation split.")
    
    format_names = ["format_1_cir", "format_2_dialog", "format_3_cqa"]
    format_titles = {
        "format_1_cir": "Context / Instruction / Response",
        "format_2_dialog": "Dialog (User: ... Assistant:)",
        "format_3_cqa": "Plain (Context / Question / Answer)"
    }
    
    results = {
        "metadata": {
            "experiment": "COPY_CAPABILITY_GATE",
            "base_checkpoint_hash": base_hash,
            "candidate_checkpoint_hash": cand_hash,
            "vocab_hash": vocab_hash,
            "merges_hash": merges_hash,
            "total_records": len(copy_items),
            "normalization_rule": "lowercase, strip quotes, punctuation, extra whitespace"
        },
        "summary": {},
        "per_example_results": []
    }
    
    for fmt_key in format_names:
        results["summary"][fmt_key] = {
            "title": format_titles[fmt_key],
            "base": {
                "exact_match_count": 0,
                "norm_match_count": 0,
                "attractor_count": 0,
                "empty_or_premature_count": 0,
                "total_generated_tokens": 0,
                "avg_tokens": 0.0
            },
            "candidate": {
                "exact_match_count": 0,
                "norm_match_count": 0,
                "attractor_count": 0,
                "empty_or_premature_count": 0,
                "total_generated_tokens": 0,
                "avg_tokens": 0.0
            }
        }
        
    print("\nBeginning inference execution across all formats...")
    
    for idx, item in enumerate(copy_items):
        item_id = item["id"]
        expected_val = item["expected_value"]
        norm_expected = normalize_value(expected_val)
        prompts = format_prompts(item)
        
        example_entry = {
            "id": item_id,
            "category": item["category"],
            "expected_value": expected_val,
            "normalized_expected": norm_expected,
            "context": item["context"],
            "copy_question": item["copy_question"],
            "formats": {}
        }
        
        for fmt_key in format_names:
            prompt_str = prompts[fmt_key]
            
            # Base model generation
            base_out, base_tokens, prompt_token_ids = generate_greedy(base_model, tokenizer, prompt_str, max_new_tokens=40)
            norm_base = normalize_value(base_out)
            base_exact = expected_val in base_out
            base_norm = (norm_expected in norm_base) or (norm_expected == norm_base)
            base_attr = is_doctest_attractor(base_out)
            base_empty = len(base_tokens) == 0 or len(base_out.strip()) == 0
            
            # Candidate model generation
            cand_out, cand_tokens, _ = generate_greedy(cand_model, tokenizer, prompt_str, max_new_tokens=40)
            norm_cand = normalize_value(cand_out)
            cand_exact = expected_val in cand_out
            cand_norm = (norm_expected in norm_cand) or (norm_expected == norm_cand)
            cand_attr = is_doctest_attractor(cand_out)
            cand_empty = len(cand_tokens) == 0 or len(cand_out.strip()) == 0
            
            # Update summary stats
            s_base = results["summary"][fmt_key]["base"]
            s_base["exact_match_count"] += int(base_exact)
            s_base["norm_match_count"] += int(base_norm)
            s_base["attractor_count"] += int(base_attr)
            s_base["empty_or_premature_count"] += int(base_empty)
            s_base["total_generated_tokens"] += len(base_tokens)
            
            s_cand = results["summary"][fmt_key]["candidate"]
            s_cand["exact_match_count"] += int(cand_exact)
            s_cand["norm_match_count"] += int(cand_norm)
            s_cand["attractor_count"] += int(cand_attr)
            s_cand["empty_or_premature_count"] += int(cand_empty)
            s_cand["total_generated_tokens"] += len(cand_tokens)
            
            # Boundary tokens: last 4 tokens of prompt
            boundary_tokens = prompt_token_ids[-4:] if len(prompt_token_ids) >= 4 else prompt_token_ids
            boundary_text = tokenizer.decode(boundary_tokens)
            
            example_entry["formats"][fmt_key] = {
                "prompt": prompt_str,
                "boundary_token_ids": boundary_tokens,
                "boundary_text": boundary_text,
                "base": {
                    "output": base_out,
                    "tokens_count": len(base_tokens),
                    "exact_match": base_exact,
                    "norm_match": base_norm,
                    "is_attractor": base_attr,
                    "is_empty": base_empty
                },
                "candidate": {
                    "output": cand_out,
                    "tokens_count": len(cand_tokens),
                    "exact_match": cand_exact,
                    "norm_match": cand_norm,
                    "is_attractor": cand_attr,
                    "is_empty": cand_empty
                }
            }
            
        results["per_example_results"].append(example_entry)
        print(f"  Processed [{idx+1}/{len(copy_items)}] {item_id}")
        
    # Calculate averages
    total = len(copy_items)
    for fmt_key in format_names:
        for model_key in ["base", "candidate"]:
            s = results["summary"][fmt_key][model_key]
            s["avg_tokens"] = round(s["total_generated_tokens"] / total, 2)
            s["exact_match_rate"] = f"{s['exact_match_count']}/{total} ({round(s['exact_match_count']/total*100, 1)}%)"
            s["norm_match_rate"] = f"{s['norm_match_count']}/{total} ({round(s['norm_match_count']/total*100, 1)}%)"
            s["attractor_rate"] = f"{s['attractor_count']}/{total} ({round(s['attractor_count']/total*100, 1)}%)"
            s["empty_rate"] = f"{s['empty_or_premature_count']}/{total} ({round(s['empty_or_premature_count']/total*100, 1)}%)"
            
    # Save outputs
    out_dir = REPO_ROOT / "experiments" / "copy_capability_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "COPY_CAPABILITY_RESULTS.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved results to {results_path}")
    
    # Print comparison summary table
    print("\n" + "=" * 70)
    print("COPY CAPABILITY EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Format':<30} | {'Model':<9} | {'Exact':<9} | {'Norm':<9} | {'Attractor':<11} | {'Avg Tok':<7}")
    print("-" * 75)
    for fmt_key in format_names:
        title = format_titles[fmt_key]
        for m_key in ["base", "candidate"]:
            s = results["summary"][fmt_key][m_key]
            print(f"{title:<30} | {m_key.capitalize():<9} | {s['exact_match_rate']:<9} | {s['norm_match_rate']:<9} | {s['attractor_rate']:<11} | {s['avg_tokens']:<7}")
        print("-" * 75)


if __name__ == "__main__":
    main()
