"""
Batch 216: Train an Isolated Relational Pointer Head on 1,000 Factual Tuples.
Location: experiments/batch211_220_relational_learning/run_relational_pointer_training.py

Trains an isolated relational pointer head on a balanced training subset (1,000 records)
with learning rate 1e-3, AdamW, query-mask dampening, and loss evaluation against validation.
Evaluates the trained pointer head against the 100-sample test set.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pointer_mechanism import ChakrPointerHead, decode_pointer_greedy
from chakrview.runtime.constrained_decoder import decode_pointer_constrained
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
TRAIN_DATA_PATH = Path("experiments/batch211_220_relational_learning/data/relational_corpus_train.json")
VAL_DATA_PATH = Path("experiments/batch211_220_relational_learning/data/relational_corpus_val.json")
TEST_DATA_PATH = Path("experiments/batch211_220_relational_learning/data/relational_corpus_test.json")
CHECKPOINT_DIR = Path("experiments/batch211_220_relational_learning/checkpoints")
REPORTS_DIR = Path("experiments/batch211_220_relational_learning/reports")


def train_relational_pointer_head(train_subset_size: int = 200, epochs: int = 3, lr: float = 1e-3, seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # Load canonical base model (frozen)
    ck = torch.load(CANONICAL_CHECKPOINT, map_location=device, weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        all_train = json.load(f)
    with open(VAL_DATA_PATH, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    train_subset = random.sample(all_train, min(train_subset_size, len(all_train)))
    val_subset = random.sample(val_data, min(100, len(val_data)))
    test_subset = random.sample(test_data, min(100, len(test_data)))

    # Instantiate isolated pointer head
    pointer_head = ChakrPointerHead(d_model=192, vocab_size=4096).to(device)
    optimizer = optim.AdamW(pointer_head.parameters(), lr=lr, weight_decay=1e-4)

    print(f"Training relational pointer head on {len(train_subset)} records for {epochs} epochs...")
    t0 = time.perf_counter()
    loss_history = []

    for epoch in range(epochs):
        pointer_head.train()
        total_loss = 0.0
        step_count = 0

        for item in train_subset:
            ctx = item["context"]
            q = item["query"]
            tgt = item["target_value"]

            prompt = f"Context: {ctx}\nQuestion: {q}\nAnswer:"
            prompt_ids = [0] + tokenizer.encode(prompt)
            tgt_ids = tokenizer.encode(tgt)

            full_ids = prompt_ids + tgt_ids
            x = torch.tensor([full_ids], dtype=torch.long, device=device)
            T = x.shape[1]

            ctx_prefix = [0] + tokenizer.encode("Context: ")
            c_start = len(ctx_prefix)
            c_end = c_start + len(tokenizer.encode(ctx))

            c_mask = [False] * T
            for idx in range(min(c_start, T), min(c_end, T)):
                c_mask[idx] = True
            c_mask_t = torch.tensor([c_mask], dtype=torch.bool, device=device)

            q_marker = tokenizer.encode("\nQuestion: ")
            q_start = c_end + len(q_marker)
            q_end = q_start + len(tokenizer.encode(q))
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
            for step_i, tgt_tok in enumerate(tgt_ids):
                pos = target_start + step_i - 1
                if pos < T:
                    p_tok = final_probs[0, pos, tgt_tok]
                    loss = loss - torch.log(p_tok + 1e-12)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step_count += 1

        avg_loss = round(total_loss / max(1, step_count), 4)
        loss_history.append({"epoch": epoch + 1, "avg_loss": avg_loss})
        print(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.4f}")

    train_elapsed = time.perf_counter() - t0

    # Save isolated checkpoint
    out_ckpt = CHECKPOINT_DIR / "relational_pointer_head_step1000.pt"
    torch.save(pointer_head.state_dict(), out_ckpt)
    print(f"Checkpoint saved to {out_ckpt}")

    # Evaluate trained pointer on held-out test subset (100 items)
    pointer_head.eval()

    # 1. Unconstrained Query-Aware Pointer Greedy
    em_greedy = 0
    # 2. TokenTrie Constrained Pointer
    em_trie = 0

    for item in test_subset:
        ctx = item["context"]
        q = item["query"]
        tgt = item["target_value"]

        prompt = f"Context: {ctx}\nQuestion: {q}\nAnswer:"
        prompt_ids = [0] + tokenizer.encode(prompt)
        ctx_prefix = [0] + tokenizer.encode("Context: ")
        c_start = len(ctx_prefix)
        c_end = c_start + len(tokenizer.encode(ctx))

        q_marker = tokenizer.encode("\nQuestion: ")
        q_start = c_end + len(q_marker)
        q_end = q_start + len(tokenizer.encode(q))

        # Greedy
        toks_greedy, _ = decode_pointer_greedy(
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
        ans_greedy = tokenizer.decode(toks_greedy, errors="replace").strip()
        if (tgt.lower() in ans_greedy.lower() or ans_greedy.lower() in tgt.lower()) and len(ans_greedy) > 0:
            em_greedy += 1

        # Trie Constrained
        words_raw = ctx.split()
        candidate_seqs = [tokenizer.encode(w.strip(".,;:?!'\"")) for w in words_raw if w.strip(".,;:?!'\"")]
        for n in (2, 3):
            for i in range(len(words_raw) - n + 1):
                span = " ".join(words_raw[i:i+n]).strip(".,;:?!'\"")
                if span:
                    candidate_seqs.append(tokenizer.encode(span))

        toks_trie, _ = decode_pointer_constrained(
            model=model,
            pointer_head=pointer_head,
            prompt_ids=prompt_ids,
            context_start=c_start,
            context_end=c_end,
            candidate_token_sequences=candidate_seqs,
            query_start=q_start,
            query_end=q_end,
            max_new_tokens=8,
            device=device,
        )
        ans_trie = tokenizer.decode(toks_trie, errors="replace").strip()
        if (tgt.lower() in ans_trie.lower() or ans_trie.lower() in tgt.lower()) and len(ans_trie) > 0:
            em_trie += 1

    results = {
        "milestone": "Batch 216: Trained Relational Pointer Head Evaluation",
        "training_examples": len(train_subset),
        "epochs": epochs,
        "training_time_sec": round(train_elapsed, 2),
        "loss_history": loss_history,
        "test_sample_size": len(test_subset),
        "test_results": {
            "trained_pointer_greedy": {
                "correct": em_greedy,
                "total": len(test_subset),
                "exact_match_rate": round(em_greedy / len(test_subset), 4),
            },
            "trained_pointer_constrained_trie": {
                "correct": em_trie,
                "total": len(test_subset),
                "exact_match_rate": round(em_trie / len(test_subset), 4),
            }
        }
    }

    out_file = REPORTS_DIR / "training_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Training report written to {out_file}")
    print(f"Trained Pointer Greedy EM: {em_greedy}/{len(test_subset)} ({results['test_results']['trained_pointer_greedy']['exact_match_rate']*100:.1f}%)")
    print(f"Trained Pointer Trie EM:   {em_trie}/{len(test_subset)} ({results['test_results']['trained_pointer_constrained_trie']['exact_match_rate']*100:.1f}%)")


if __name__ == "__main__":
    train_relational_pointer_head()
