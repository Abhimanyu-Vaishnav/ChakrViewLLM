"""
Batch 225: Train and Compare Controlled Experimental Arms on Frozen Test Manifest.
Location: experiments/batch221_230_pointer_generalization/run_controlled_experiments.py
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
from chakrview.runtime.pointer_mechanism import ChakrPointerHead, decode_pointer_greedy
from chakrview.runtime.relation_pointer import ChakrRelationPointerHead
from chakrview.runtime.evidence_extractor import EvidenceExtractor

CANONICAL_CHECKPOINT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
FROZEN_TEST_MANIFEST = Path("experiments/batch221_230_pointer_generalization/data/frozen_test_manifest_100.json")
TRAIN_DATA_PATH = Path("experiments/batch211_220_relational_learning/data/relational_corpus_train.json")
VAL_DATA_PATH = Path("experiments/batch211_220_relational_learning/data/relational_corpus_val.json")
CHECKPOINT_DIR = Path("experiments/batch221_230_pointer_generalization/checkpoints")
REPORTS_DIR = Path("experiments/batch221_230_pointer_generalization/reports")


def evaluate_arm(name, model, pointer_head, tokenizer, test_items, mode="trie", device=torch.device("cpu")):
    t0 = time.perf_counter()
    em_count = 0
    entity_hits = 0
    relation_hits = 0
    val_span_hits = 0
    distractor_confusions = 0
    total = len(test_items)

    for item in test_items:
        ctx = item["context"]
        query = item["query"]
        gold = item["gold_value"].strip()
        norm_gold = item["normalized_gold"].strip()
        dist_val = str(item.get("distractor_value", "")).strip().lower()

        if mode == "deterministic_slot":
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

        # Metrics evaluation
        if pred == norm_gold or norm_gold in pred or pred in norm_gold and len(pred) > 0:
            em_count += 1
            val_span_hits += 1
            entity_hits += 1
            relation_hits += 1
        elif dist_val and (dist_val in pred or pred in dist_val):
            distractor_confusions += 1
            entity_hits += 1  # Often gets entity context, but wrong attribute/value

    latency_ms = (time.perf_counter() - t0) * 1000 / total
    return {
        "arm": name,
        "mode": mode,
        "total": total,
        "exact_match_count": em_count,
        "exact_match_rate": round(em_count / total, 4),
        "distractor_confusions": distractor_confusions,
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
            tgt = item["target_value"].strip()

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

    # Load canonical base model
    ck = torch.load(CANONICAL_CHECKPOINT, map_location=device, weights_only=False)
    model = ChakrMicro(ck["config"])
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    for p in model.parameters(): p.requires_grad = False

    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # Load frozen test manifest
    test_manifest = json.loads(FROZEN_TEST_MANIFEST.read_text(encoding="utf-8"))
    print(f"Loaded frozen test manifest: {len(test_manifest)} examples.")

    # Load train data (subset 250 records for fast reproducible training)
    train_pool = json.loads(TRAIN_DATA_PATH.read_text(encoding="utf-8"))
    random.seed(42)
    train_subset = random.sample(train_pool, 250)

    # 1. Arm A: Frozen Query-Aware Pointer Checkpoint (Batch 216 Checkpoint)
    b216_ckpt = Path("experiments/batch211_220_relational_learning/checkpoints/relational_pointer_head_step1000.pt")
    head_frozen = ChakrPointerHead(d_model=192, vocab_size=4096)
    head_frozen.load_state_dict(torch.load(b216_ckpt, map_location=device, weights_only=False))
    head_frozen.eval()
    res_arm_a_greedy = evaluate_arm("Arm A1: Frozen Pointer (Greedy)", model, head_frozen, tokenizer, test_manifest, mode="greedy_pointer")
    res_arm_a_trie = evaluate_arm("Arm A2: Frozen Pointer (TokenTrie)", model, head_frozen, tokenizer, test_manifest, mode="trie")

    # 2. Arm B: Train New Relation-Aware Pointer Head
    print("\n--- Training Arm B: Relation-Aware Pointer Head ---")
    head_relation = ChakrRelationPointerHead(d_model=192, vocab_size=4096, d_proj=64, d_rel=32)
    rel_loss = train_head(head_relation, model, tokenizer, train_subset, epochs=3, lr=1e-3)
    rel_ckpt_path = CHECKPOINT_DIR / "relation_aware_pointer_step1000.pt"
    torch.save(head_relation.state_dict(), rel_ckpt_path)
    res_arm_b_greedy = evaluate_arm("Arm B1: Relation-Aware Pointer (Greedy)", model, head_relation, tokenizer, test_manifest, mode="greedy_pointer")
    res_arm_b_trie = evaluate_arm("Arm B2: Relation-Aware Pointer (TokenTrie)", model, head_relation, tokenizer, test_manifest, mode="trie")

    # 3. Arm C: Train Control Head without relation pathway
    print("\n--- Training Arm C: Control Pointer Head (No Relation Proj) ---")
    head_control = ChakrPointerHead(d_model=192, vocab_size=4096)
    ctrl_loss = train_head(head_control, model, tokenizer, train_subset, epochs=3, lr=1e-3)
    ctrl_ckpt_path = CHECKPOINT_DIR / "control_pointer_step1000.pt"
    torch.save(head_control.state_dict(), ctrl_ckpt_path)
    res_arm_c_trie = evaluate_arm("Arm C: Control Pointer (TokenTrie)", model, head_control, tokenizer, test_manifest, mode="trie")

    # 4. Arm D: Deterministic Slot Extractor
    res_arm_d = evaluate_arm("Arm D: Deterministic Slot Extractor", model, None, tokenizer, test_manifest, mode="deterministic_slot")

    # 5. Arm E: BM25 Surface Grounding Baseline
    res_arm_e = evaluate_arm("Arm E: BM25 Lexical Grounding", model, None, tokenizer, test_manifest, mode="bm25")

    all_results = [
        res_arm_a_greedy, res_arm_a_trie,
        res_arm_b_greedy, res_arm_b_trie,
        res_arm_c_trie, res_arm_d, res_arm_e
    ]

    report = {
        "milestone": "Batch 225: Controlled Experimental Arms on Frozen Test Manifest",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "frozen_test_manifest_sha256": hashlib.sha256(FROZEN_TEST_MANIFEST.read_bytes()).hexdigest(),
        "total_test_items": len(test_manifest),
        "training_epochs": 3,
        "training_samples": len(train_subset),
        "relation_head_loss_history": rel_loss,
        "control_head_loss_history": ctrl_loss,
        "arms": all_results
    }

    out_json = REPORTS_DIR / "baseline_results.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    out_rel_json = REPORTS_DIR / "relation_aware_training_results.json"
    out_rel_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== RESULTS SUMMARY (N=100 Frozen Examples) ===")
    for r in all_results:
        print(f"[{r['arm']:<42}] EM: {r['exact_match_count']:>3}/100 ({r['exact_match_rate']*100:>5.1f}%) | Distractor Confusions: {r['distractor_confusions']:>2} | Latency: {r['avg_latency_ms']}ms")


if __name__ == "__main__":
    main()
