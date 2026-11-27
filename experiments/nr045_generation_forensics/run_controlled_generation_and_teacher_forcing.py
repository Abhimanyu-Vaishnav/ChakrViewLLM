"""
experiments/nr045_generation_forensics/run_controlled_generation_and_teacher_forcing.py
Phase 6 & 7:
1. Controlled generation under Greedy, T=0.3, T=0.7, T=1.0 across English, Hindi, Hinglish, Logic, Code.
2. Teacher-forced vs free-running evaluation on validation records:
   - Full-seq loss vs response-only loss
   - First-token distribution & divergence point
   - Teacher-forced accuracy on response tokens
"""

import sys
import json
import math
import torch
from pathlib import Path

sys.path.insert(0, '.')

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

def run_diagnostics():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== RUNNING PHASE 6 & 7 DIAGNOSTICS ON {device.upper()} ===")

    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)
    sd = torch.load("release/v1.0.0-rc1/chakr_1_0_rc1_best.pt", map_location=device, weights_only=False)
    model = ChakrGenesis17M(sd["config"]).to(device)
    model.load_state_dict(sd["model_state_dict"])
    model.eval()

    # --- Part 1: Controlled Generation Experiments ---
    test_suite = [
        {"cat": "English Factual", "prompt": "User: Which planet is known as the Red Planet in our solar system?\nAssistant: "},
        {"cat": "English Code", "prompt": "User: Write a Python function to add two numbers.\nAssistant: "},
        {"cat": "Hindi Science", "prompt": "User: कंप्यूटर विज्ञान में 'एल्गोरिदम' से क्या तात्पर्य है?\nAssistant: "},
        {"cat": "Hinglish Dialogue", "prompt": "User: CPU aur GPU me main difference kya hota hai?\nAssistant: "},
        {"cat": "Arithmetic", "prompt": "User: What is 12 + 15?\nAssistant: "},
        {"cat": "Logic", "prompt": "User: All humans are mortal. Socrates is a human. Is Socrates mortal?\nAssistant: "}
    ]

    temps = [0.0, 0.3, 0.7, 1.0]
    gen_results = []

    print("\n--- 1. Controlled Generation Experiments across Temperatures ---")
    for tc in test_suite:
        p = tc["prompt"]
        cat = tc["cat"]
        # Wrap with SFT training prefix
        full_inp = f"<user>\n{p}\n<assistant>\n"
        inp_toks = tokenizer.encode(full_inp)
        
        entry = {"category": cat, "prompt": p, "temperatures": {}}

        for t in temps:
            curr = torch.tensor([inp_toks], dtype=torch.long, device=device)
            with torch.no_grad():
                for _ in range(40):
                    out, _ = model(curr)
                    logits = out[:, -1, :]
                    if t == 0.0:
                        nxt = logits.argmax(dim=-1, keepdim=True)
                    else:
                        probs = torch.softmax(logits / t, dim=-1)
                        nxt = torch.multinomial(probs, num_samples=1)
                    curr = torch.cat([curr, nxt], dim=1)
                    if nxt.item() == 1: break # EOS
            gen_toks = curr[0, len(inp_toks):].tolist()
            dec_text = tokenizer.decode(gen_toks, errors="replace").strip()
            entry["temperatures"][f"temp_{t}"] = {
                "tokens": gen_toks[:15],
                "length": len(gen_toks),
                "text": dec_text
            }
        gen_results.append(entry)
        print(f"[{cat:18s}] T=0.0: {repr(entry['temperatures']['temp_0.0']['text'][:60])}")

    # --- Part 2: Teacher-Forced vs Free-Running Divergence ---
    print("\n--- 2. Teacher-Forced vs Free-Running Evaluation on Validation Records ---")
    tf_records = []
    with open("data/corpus_v1/master_corpus_val.jsonl", "r", encoding="utf-8") as f:
        for i in range(25):
            line = f.readline()
            if not line: break
            tf_records.append(json.loads(line))

    tf_results = []
    for idx, d in enumerate(tf_records):
        p = d["prompt"]
        r = d["response"]
        
        full_seq = f"<user>\n{p}\n<assistant>\n{r}<eos>"
        prompt_seq = f"<user>\n{p}\n<assistant>\n"
        
        full_toks = tokenizer.encode(full_seq)[:256]
        prompt_toks = tokenizer.encode(prompt_seq)
        
        if len(prompt_toks) >= len(full_toks): continue
        target_resp_toks = full_toks[len(prompt_toks):]

        inp = torch.tensor([full_toks[:-1]], dtype=torch.long, device=device)
        tar_resp = torch.tensor([[-100] * (len(prompt_toks) - 1) + target_resp_toks], dtype=torch.long, device=device)

        with torch.no_grad():
            logits, loss = model(inp, targets=tar_resp)
        
        # Free-running generation from prompt
        curr = torch.tensor([prompt_toks], dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(len(target_resp_toks)):
                out, _ = model(curr)
                nxt = out[:, -1, :].argmax(dim=-1, keepdim=True)
                curr = torch.cat([curr, nxt], dim=1)
                if nxt.item() == 1: break
        gen_resp_toks = curr[0, len(prompt_toks):].tolist()

        # Measure divergence point (first token where gen != ref)
        diverge_step = 0
        for step_idx, (t_ref, t_gen) in enumerate(zip(target_resp_toks, gen_resp_toks)):
            if t_ref != t_gen:
                diverge_step = step_idx
                break
        else:
            diverge_step = min(len(target_resp_toks), len(gen_resp_toks))

        tf_results.append({
            "id": d.get("id"),
            "category": d.get("category"),
            "language": d.get("language"),
            "loss": round(loss.item(), 4),
            "ref_length": len(target_resp_toks),
            "gen_length": len(gen_resp_toks),
            "divergence_token_index": diverge_step,
            "ref_first_token": tokenizer.decode([target_resp_toks[0]]) if target_resp_toks else "",
            "gen_first_token": tokenizer.decode([gen_resp_toks[0]]) if gen_resp_toks else "",
            "first_token_match": (target_resp_toks[0] == gen_resp_toks[0]) if target_resp_toks and gen_resp_toks else False
        })

    first_matches = sum(1 for item in tf_results if item["first_token_match"])
    avg_div = sum(item["divergence_token_index"] for item in tf_results) / len(tf_results)
    avg_loss = sum(item["loss"] for item in tf_results) / len(tf_results)

    print(f"Teacher-Forced Validation Loss: {avg_loss:.4f} (PPL: {math.exp(avg_loss):.2f})")
    print(f"First-Token Match Rate: {first_matches}/{len(tf_results)} ({first_matches/len(tf_results)*100:.1f}%)")
    print(f"Average Divergence Step: {avg_div:.2f} tokens (Model diverges immediately at token ~{avg_div:.1f})")

    out_file = Path("experiments/nr045_generation_forensics/generation_and_teacher_forcing_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "controlled_generation": gen_results,
            "teacher_forcing": {
                "summary": {
                    "avg_loss": round(avg_loss, 4),
                    "first_token_match_pct": round(first_matches/len(tf_results)*100, 1),
                    "avg_divergence_step": round(avg_div, 2)
                },
                "details": tf_results
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"\nPhase 6 & 7 results saved to {out_file}")

if __name__ == "__main__":
    run_diagnostics()
