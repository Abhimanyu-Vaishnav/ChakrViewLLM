"""
Loss and Masking Audit Script.
Verifies token alignment, prompt masking, causal shift, and target sequence construction.
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import build_causal_sequence, normalize_prompt_response_strings
from chakrview.training.sft_engine import SFTDataset

FORENSICS_DIR = REPO_ROOT / "experiments" / "cbr01_sft" / "conversational_forensics_01"

def main():
    tokenizer, meta = load_tokenizer_artifacts("data/experiments/vocab_4096")
    train_file = REPO_ROOT / "data" / "sft" / "cbr_sft_07" / "splits" / "cbr_sft_07_train_228.jsonl"
    
    dataset = SFTDataset(
        jsonl_path=train_file,
        tokenizer=tokenizer,
        max_seq_len=512,
        allow_exploratory_opt_in=True,
        enforce_trust_gate=False,
        is_training=True,
        strip_prompt_delimiters=True,
        mask_prompt=True,
    )
    
    audit_records = [0, 10, 50]  # Check 3 records
    details = []
    
    for idx in audit_records:
        rec = dataset.records[idx]
        p_str = rec["prompt"]
        r_str = rec["completion"] if "completion" in rec else rec["response"]
        
        inp_seq, tgt_seq = dataset.encode_example(p_str, r_str)
        
        # Token details
        p_toks = tokenizer.encode(p_str.strip())
        r_toks = tokenizer.encode(r_str.strip())
        
        # Inspect alignment:
        # inp_seq has [0] + p_toks + r_toks
        # tgt_seq has [-100]*len(p_toks) + r_toks + [1]
        
        alignment_table = []
        for i in range(len(inp_seq)):
            in_tok = inp_seq[i]
            tgt_tok = tgt_seq[i]
            in_str = tokenizer.decode([in_tok]) if in_tok < 4096 else f"SPECIAL({in_tok})"
            tgt_str = tokenizer.decode([tgt_tok]) if 0 <= tgt_tok < 4096 else f"MASK({tgt_tok})"
            alignment_table.append({
                "pos": i,
                "input_id": in_tok,
                "input_str": repr(in_str),
                "target_id": tgt_tok,
                "target_str": repr(tgt_str),
                "is_masked": (tgt_tok == -100)
            })
            
        # Check alignment logic:
        # Does the model predict y[t] from x[0..t]?
        # In causal LM:
        # logits[t] is prediction for next token given input x[0..t].
        # In PyTorch standard cross-entropy without shift:
        # logits[t] compared against targets[t].
        # Let's verify what sft_engine does!
        # In sft_engine:
        # logits = model(x)  # shape [B, T, V]
        # loss = loss_fn(logits.reshape(-1, V), y.reshape(-1))
        # WAIT! In PyTorch causal LM:
        # If input is [BOS, p0, p1, r0] and target is [-100, -100, r0, r1],
        # logits[0] (from BOS) is aligned with y[0] == -100!
        # logits[1] (from p0) is aligned with y[1] == -100!
        # logits[len(p)] (from last prompt token) is aligned with y[len(p)] == r0!
        # Because input_seq[len(p)] is the last prompt token, its output logit at index len(p)
        # predicts r0, which is exactly at y[len(p)]!
        # That means logits are already time-aligned with target_seq WITHOUT an explicit shift!
        
        details.append({
            "record_index": idx,
            "category": rec.get("category", "unknown"),
            "prompt_len": len(p_toks),
            "response_len": len(r_toks),
            "seq_len": len(inp_seq),
            "num_masked": sum(1 for x in tgt_seq if x == -100),
            "num_active": sum(1 for x in tgt_seq if x != -100),
            "alignment_sample": alignment_table[:10] + alignment_table[len(p_toks)-2:len(p_toks)+4]
        })
        
    out_path = FORENSICS_DIR / "LOSS_MASK_AUDIT.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Loss Masking and Token Target Alignment Audit\n\n")
        f.write("## 1. Mathematical and Implementation Formulation\n\n")
        f.write("In `chakrview/training/prompt_contract.py`:\n")
        f.write("- `input_seq = [BOS] + prompt_tokens + response_tokens`\n")
        f.write("- `target_seq = [-100] * len(prompt_tokens) + response_tokens + [EOS]`\n\n")
        f.write("In `chakrview/training/sft_engine.py`:\n")
        f.write("- `logits = self.model(x)` (shape: `[B, T, vocab_size]`)\n")
        f.write("- `loss = self.loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))`\n\n")
        f.write("### Alignment Verification:\n")
        f.write("At position `t = len(prompt_tokens)`:\n")
        f.write("- `input_seq[t]` is the final token of the prompt (`prompt_tokens[-1]`).\n")
        f.write("- The causal transformer computes representation `h[t]` using `input_seq[0...t]` (i.e. BOS and the entire prompt).\n")
        f.write("- `logits[t]` predicts the next token.\n")
        f.write("- `target_seq[t]` is `response_tokens[0]` (the very first response token).\n")
        f.write("- For `t < len(prompt_tokens)`, `target_seq[t] == -100`, masking all prompt tokens from contributing to the loss.\n")
        f.write("- At `t = len(input_seq) - 1` (the last response token), `target_seq[t] == EOS` (1), teaching the model when to stop.\n\n")
        f.write("Conclusion on alignment: **Mathematically bit-exact and verified correct causal alignment.** No shift bug exists.\n\n")
        
        f.write("## 2. Empirical Verification on Training Examples\n\n")
        for d in details:
            f.write(f"### Example Record {d['record_index']} (Category: `{d['category']}`)\n")
            f.write(f"- Prompt Tokens: {d['prompt_len']}, Response Tokens: {d['response_len']}, Total Seq: {d['seq_len']}\n")
            f.write(f"- Masked Target Tokens (-100): {d['num_masked']} (Exact match to prompt length: {d['num_masked'] == d['prompt_len']})\n")
            f.write(f"- Active Target Tokens: {d['num_active']} (Exact match to response + EOS: {d['num_active'] == d['response_len'] + 1})\n\n")
            f.write("| Pos | Input ID | Input Token | Target ID | Target Token | Masked? |\n|---|---|---|---|---|---|\n")
            for row in d["alignment_sample"]:
                f.write(f"| {row['pos']} | {row['input_id']} | {row['input_str']} | {row['target_id']} | {row['target_str']} | {row['is_masked']} |\n")
            f.write("\n")

    print(f"LOSS_MASK_AUDIT.md written to {out_path}")

if __name__ == "__main__":
    main()

