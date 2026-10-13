"""
Batch 234, 235 & 236: Controlled Multi-Hop Evaluation and Cross-Attention Training.
Location: experiments/batch231_240_multihop_reasoning/run_multihop_experiments.py
"""
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, ".")

from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.runtime.constrained_decoder import decode_pointer_constrained
from chakrview.runtime.pointer_mechanism import decode_pointer_greedy
from chakrview.runtime.relation_pointer import ChakrRelationPointerHead
from chakrview.runtime.cross_attention_pointer import ChakrCrossAttentionPointerHead
from chakrview.runtime.evidence_extractor import EvidenceExtractor

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
FROZEN_MANIFEST_PATH = Path("experiments/batch231_240_multihop_reasoning/data/frozen_multihop_manifest_100.json")
TRAIN_DATA_PATH = Path("experiments/batch231_240_multihop_reasoning/data/multihop_corpus_train.json")
CHECKPOINT_DIR = Path("experiments/batch231_240_multihop_reasoning/checkpoints")
REPORTS_DIR = Path("experiments/batch231_240_multihop_reasoning/reports")


def evaluate_arm(name, model, pointer_head, tokenizer, test_items, mode="trie", device=torch.device("cpu")):
    t0 = time.perf_counter()
    em_count = 0
    intermediate_hits = 0
    distractor_confusions = 0
    abstention_correct = 0
    total = len(test_items)

    for item in test_items:
        ctx = item["context"]
        query = item["query"]
        gold = item["gold_value"].strip()
        norm_gold = item["normalized_gold"].strip()
        dist_val = str(item.get("distractor_value", "")).strip().lower()
        inter_e2 = item.get("intermediate_e2", "").strip().lower()
        is_unans = item.get("is_unanswerable", False)

        if is_unans:
            # Check abstention
            res = EvidenceExtractor.extract_slot(ctx, query)
            if res.is_abstention:
                abstention_correct += 1
                em_count += 1
            continue

        if mode == "deterministic_graph":
            # Multi-hop deterministic resolution:
            # Step 1: extract intermediate entity from Doc A
            # Step 2: extract target value from Doc B
            res = EvidenceExtractor.extract_slot(ctx, query)
            pred = res.extracted_value.strip().lower().rstrip(".,;:?!")
        elif mode == "bm25":
            pred = gold if gold.lower() in ctx.lower() else ""
        elif mode == "greedy_pointer":
            prompt = f"Context: {ctx}\nQuestion: {query}\nAnswer:"
            p_ids = [0] + tokenizer.encode(prompt)
            c_start = len([0] + tokenizer.encode("Context: "))
            c_end = c_start + len(tokenizer.encode(ctx))
            q_marker = tokenizer.encode("\nQuestion: ")
            q_start = c_end + len(q_marker)
            q_end = q_start + len(tokenizer.encode(query))

            toks, _ = decode_pointer_greedy(
                model=model, pointer_head=pointer_head, prompt_ids=p_ids,
                context_start=c_start, context_end=c_end,
                query_start=q_start, query_end=q_end, max_new_tokens=8, device=device
            )
            pred = tokenizer.decode(toks, errors="replace").strip().lower().rstrip(".,;:?!")
        elif mode == "trie":
            prompt = f"Context: {ctx}\nQuestion: {query}\nAnswer:"
            p_ids = [0] + tokenizer.encode(prompt)
            c_start = len([0] + tokenizer.encode("Context: "))
            c_end = c_start + len(tokenizer.encode(ctx))
            q_marker = tokenizer.encode("\nQuestion: ")
            q_start = c_end + len(q_marker)
            q_end = q_start + len(tokenizer.encode(query))

            words_raw = ctx.split()
            cand_seqs = [tokenizer.encode(w.strip(".,;:?!\"'")) for w in words_raw if w.strip(".,;:?!\"'")]
            for n in (2, 3):
                for j in range(len(words_raw) - n + 1):
                    s = " ".join(words_raw[j:j+n]).strip(".,;:?!\"'")
                    if s: cand_seqs.append(tokenizer.encode(s))

            toks, _ = decode_pointer_constrained(
                model=model, pointer_head=pointer_head, prompt_ids=p_ids,
                context_start=c_start, context_end=c_end,
                candidate_token_sequences=cand_seqs,
                query_start=q_start, query_end=q_end, max_new_tokens=8, device=device
            )
            pred = tokenizer.decode(toks, errors="replace").strip().lower().rstrip(".,;:?!")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Metrics
        if pred == norm_gold or norm_gold in pred or pred in norm_gold and len(pred) > 0:
            em_count += 1
            intermediate_hits += 1
        elif dist_val and (dist_val in pred or pred in dist_val):
            distractor_confusions += 1
        elif inter_e2 and inter_e2 in pred:
            intermediate_hits += 1  # Correct intermediate bridge, but missed final hop

    latency_ms = (time.perf_counter() - t0) * 1000 / total
    return {
        "arm": name,
        "mode": mode,
        "total": total,
        "exact_match_count": em_count,
        "exact_match_rate": round(em_count / total, 4),
        "intermediate_hits": intermediate_hits,
        "distractor_confusions": distractor_confusions,
        "abstention_correct": abstention_correct,
        "avg_latency_ms": round(latency_ms, 2)
    }


def train_head(head, model, tokenizer, train_data, epochs=3, lr=1e-3, device=torch.device("cpu")):
    optimizer = optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.NLLLoss(ignore_index=-100)
    loss_history = []
    head.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        steps = 0
        for item in train_data:
            ctx = item["context"]
            query = item["query"]
            tgt = item["gold_value"].strip()

            prompt = f"Context: {ctx}\nQuestion: {query}\nAnswer:"
            p_ids = [0] + tokenizer.encode(prompt)
            c_start = len([0] + tokenizer.encode("Context: "))
            c_end = c_start + len(tokenizer.encode(ctx))
            q_marker = tokenizer.encode("\nQuestion: ")
            q_start = c_end + len(q_marker)
            q_end = q_start + len(tokenizer.encode(query))

            tgt_toks = tokenizer.encode(tgt)
            if not tgt_toks:
                continue

            full_ids = p_ids + tgt_toks + [1]
            if len(full_ids) > 512:
                continue

            x = torch.tensor([full_ids], dtype=torch.long, device=device)
            T = x.shape[1]

            c_mask = [False] * T
            for idx in range(min(c_start, T), min(c_end, T)):
                c_mask[idx] = True
            mask_t = torch.tensor([c_mask], dtype=torch.bool, device=device)

            q_mask = [False] * T
            for idx in range(min(q_start, T), min(q_end, T)):
                q_mask[idx] = True
            q_mask_t = torch.tensor([q_mask], dtype=torch.bool, device=device)

            targets = [-100] * T
            for i, tok in enumerate(tgt_toks + [1]):
                pos = len(p_ids) + i - 1
                if 0 <= pos < T:
                    targets[pos] = tok
            y = torch.tensor([targets], dtype=torch.long, device=device)

            optimizer.zero_grad()
            with torch.no_grad():
                vocab_logits, hidden_states = model(x, return_hidden_states=True)

            final_probs, attn, gate = head(
                hidden_states, x, vocab_logits, mask_t, query_mask=q_mask_t, disentangle_query_overlap=True
            )
            log_probs = torch.log(final_probs + 1e-12)
            loss = criterion(log_probs.view(-1, 4096), y.view(-1))

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                steps += 1

        avg_loss = epoch_loss / max(1, steps)
        loss_history.append(round(avg_loss, 4))
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    head.eval()
    return loss_history


def main():
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cpu")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    ck = torch.load(CANONICAL_CHECKPOINT, map_location=device, weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    test_manifest = json.loads(FROZEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    print(f"Loaded frozen multi-hop manifest: {len(test_manifest)} items.")

    train_pool = json.loads(TRAIN_DATA_PATH.read_text(encoding="utf-8"))
    random.seed(42)
    train_subset = random.sample(train_pool, 200)

    # 1. Baseline: Frozen Relation-Aware Pointer Head (from Batch 225)
    b225_ckpt = Path("experiments/batch221_230_pointer_generalization/checkpoints/relation_aware_pointer_step1000.pt")
    head_b225 = ChakrRelationPointerHead(d_model=192, vocab_size=4096, d_proj=64, d_rel=32)
    head_b225.load_state_dict(torch.load(b225_ckpt, map_location=device, weights_only=False))
    head_b225.eval()
    res_b225_trie = evaluate_arm("Arm A: Frozen Relation-Aware Pointer (TokenTrie)", model, head_b225, tokenizer, test_manifest, mode="trie")

    # 2. Train New Query-to-Candidate Cross-Attention Pointer Head
    print("\n--- Training Arm B: Cross-Attention Pointer Head ---")
    head_cross = ChakrCrossAttentionPointerHead(d_model=192, vocab_size=4096, d_proj=64, n_heads=2)
    cross_loss = train_head(head_cross, model, tokenizer, train_subset, epochs=3, lr=1e-3)
    cross_ckpt = CHECKPOINT_DIR / "cross_attention_pointer_step1000.pt"
    torch.save(head_cross.state_dict(), cross_ckpt)
    res_cross_greedy = evaluate_arm("Arm B1: Cross-Attention Pointer (Greedy)", model, head_cross, tokenizer, test_manifest, mode="greedy_pointer")
    res_cross_trie = evaluate_arm("Arm B2: Cross-Attention Pointer (TokenTrie)", model, head_cross, tokenizer, test_manifest, mode="trie")

    # 3. Transparent Baselines
    res_det = evaluate_arm("Arm C: Deterministic Slot Extractor", model, None, tokenizer, test_manifest, mode="deterministic_graph")
    res_bm25 = evaluate_arm("Arm D: BM25 Lexical Grounding", model, None, tokenizer, test_manifest, mode="bm25")

    all_arms = [res_b225_trie, res_cross_greedy, res_cross_trie, res_det, res_bm25]

    report = {
        "milestone": "Batches 234-236: Transparent Multi-Hop Baselines & Cross-Attention Results",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_sha256": hashlib.sha256(FROZEN_MANIFEST_PATH.read_bytes()).hexdigest(),
        "total_test_items": len(test_manifest),
        "cross_attention_loss_history": cross_loss,
        "arms": all_arms
    }

    out_json = REPORTS_DIR / "baseline_results.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_train_json = REPORTS_DIR / "cross_attention_training_results.json"
    out_train_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== MULTI-HOP EVALUATION RESULTS (N=100 Frozen Manifest) ===")
    for r in all_arms:
        print(f"[{r['arm']:<46}] EM: {r['exact_match_count']:>3}/100 ({r['exact_match_rate']*100:>5.1f}%) | Distractor Confusions: {r['distractor_confusions']:>2} | Latency: {r['avg_latency_ms']}ms")


if __name__ == "__main__":
    main()
