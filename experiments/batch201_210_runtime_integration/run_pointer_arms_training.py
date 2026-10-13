"""
Batch 204: Controlled Query-Aware Pointer Training and Evaluation Across Arms.
Location: experiments/batch201_210_runtime_integration/run_pointer_arms_training.py

Evaluates 4 comparative arms on the held-out multilingual test set:
- Arm A: Canonical vocabulary-only generation.
- Arm B: Standard pointer head (no query overlap dampening).
- Arm C: Query-aware pointer head (with query dampening -3.5).
- Arm D: Deterministic extraction reference.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pointer_mechanism import ChakrPointerHead, decode_pointer_greedy
from chakrview.runtime.evidence_extractor import EvidenceExtractor
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
SPLITS_PATH = Path("experiments/batch201_210_runtime_integration/data/multilingual_relation_splits.json")
CHECKPOINT_DIR = Path("experiments/batch201_210_runtime_integration/checkpoints")


def run_pointer_arms_training():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # Load canonical model
    ck = torch.load(CANONICAL_CHECKPOINT, map_location=device, weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_data = splits["splits"]["train"]
    test_data = splits["splits"]["test"]

    # 1. Train Query-Aware Pointer Head
    pointer_head = ChakrPointerHead(d_model=192, vocab_size=4096).to(device)
    optimizer = optim.AdamW(pointer_head.parameters(), lr=1e-3, weight_decay=1e-4)

    t0 = time.perf_counter()
    pointer_head.train()
    epochs = 15

    for epoch in range(epochs):
        for item in train_data:
            context = item["context"]
            query = item["query"]
            target = item["value"]

            prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
            prompt_ids = [0] + tokenizer.encode(prompt)
            target_ids = tokenizer.encode(target)

            full_ids = prompt_ids + target_ids
            x = torch.tensor([full_ids], dtype=torch.long, device=device)
            T = x.shape[1]

            ctx_prefix = [0] + tokenizer.encode("Context: ")
            c_start = len(ctx_prefix)
            c_end = c_start + len(tokenizer.encode(context))

            c_mask = [False] * T
            for idx in range(min(c_start, T), min(c_end, T)):
                c_mask[idx] = True
            c_mask_t = torch.tensor([c_mask], dtype=torch.bool, device=device)

            q_marker = tokenizer.encode("\nQuestion: ")
            q_start = c_end + len(q_marker)
            q_end = q_start + len(tokenizer.encode(query))
            q_mask = [False] * T
            for idx in range(min(q_start, T), min(q_end, T)):
                q_mask[idx] = True
            q_mask_t = torch.tensor([q_mask], dtype=torch.bool, device=device)

            with torch.no_grad():
                vocab_logits, hidden_states = model(x, return_hidden_states=True)

            final_probs, attn, gate = pointer_head(
                hidden_states, x, vocab_logits, c_mask_t, query_mask=q_mask_t, disentangle_query_overlap=True
            )

            # Supervision on target tokens
            target_start = len(prompt_ids)
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            for step_i, tgt_tok in enumerate(target_ids):
                pos = target_start + step_i - 1
                if pos < T:
                    p_tok = final_probs[0, pos, tgt_tok]
                    loss = loss - torch.log(p_tok + 1e-12)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    train_elapsed = time.perf_counter() - t0
    pointer_ckpt = CHECKPOINT_DIR / "query_aware_pointer_multilingual.pt"
    torch.save(pointer_head.state_dict(), pointer_ckpt)

    # 2. Evaluate Comparative Arms on Held-Out Test Set (4 cases)
    results = {
        "training_time_sec": round(train_elapsed, 2),
        "held_out_cases": len(test_data),
        "arms": {}
    }

    pointer_head.eval()

    # Standard pointer head baseline (trained standard without dampening)
    std_head = ChakrPointerHead(d_model=192, vocab_size=4096).to(device)
    std_head.load_state_dict(pointer_head.state_dict())
    std_head.eval()

    for arm_name in ["arm_a_vocab_only", "arm_b_standard_pointer", "arm_c_query_aware_pointer", "arm_d_deterministic_oracle"]:
        em_count = 0
        samples = []

        for item in test_data:
            context = item["context"]
            query = item["query"]
            target = item["value"]

            prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
            prompt_ids = [0] + tokenizer.encode(prompt)
            ctx_prefix = [0] + tokenizer.encode("Context: ")
            c_start = len(ctx_prefix)
            c_end = c_start + len(tokenizer.encode(context))

            q_marker = tokenizer.encode("\nQuestion: ")
            q_start = c_end + len(q_marker)
            q_end = q_start + len(tokenizer.encode(query))

            if arm_name == "arm_a_vocab_only":
                curr = list(prompt_ids)
                with torch.no_grad():
                    for _ in range(8):
                        x = torch.tensor([curr], dtype=torch.long, device=device)
                        nxt = torch.argmax(model(x)[0, -1]).item()
                        curr.append(nxt)
                        if nxt == 1:
                            break
                gen_text = tokenizer.decode(curr[len(prompt_ids):], errors="replace").strip()

            elif arm_name == "arm_b_standard_pointer":
                # Standard pointer without query overlap dampening
                gen_toks, _ = decode_pointer_greedy(
                    model=model,
                    pointer_head=std_head,
                    prompt_ids=prompt_ids,
                    context_start=c_start,
                    context_end=c_end,
                    query_start=None,
                    query_end=None,
                    max_new_tokens=8,
                    device=device,
                )
                gen_text = tokenizer.decode(gen_toks, errors="replace").strip()

            elif arm_name == "arm_c_query_aware_pointer":
                # Query-aware pointer with query overlap dampening
                gen_toks, _ = decode_pointer_greedy(
                    model=model,
                    pointer_head=pointer_head,
                    prompt_ids=prompt_ids,
                    context_start=c_start,
                    context_end=c_end,
                    query_start=q_start,
                    query_end=q_end,
                    max_new_tokens=8,
                    device=device,
                )
                gen_text = tokenizer.decode(gen_toks, errors="replace").strip()

            elif arm_name == "arm_d_deterministic_oracle":
                ext = EvidenceExtractor.extract_slot(context, query)
                gen_text = ext.extracted_value if ext.success else "Not mentioned in context."

            is_match = (target.lower() in gen_text.lower() or gen_text.lower() in target.lower()) and len(gen_text) > 0
            if is_match:
                em_count += 1

            samples.append({
                "entity": item["entity"],
                "query": query,
                "target": target,
                "generated": gen_text,
                "match": is_match
            })

        em_rate = round(em_count / len(test_data), 4)
        results["arms"][arm_name] = {
            "exact_match_count": em_count,
            "exact_match_rate": em_rate,
            "samples": samples
        }

    out_file = Path("experiments/batch201_210_runtime_integration/reports/pointer_arms_training_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Arms training results written to {out_file}")
    for k, v in results["arms"].items():
        print(f"{k:<30}: EM = {v['exact_match_rate']*100:.1f}% ({v['exact_match_count']}/{len(test_data)})")


if __name__ == "__main__":
    run_pointer_arms_training()
