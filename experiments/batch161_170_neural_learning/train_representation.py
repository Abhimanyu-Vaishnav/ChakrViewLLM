"""
Batch 164: Train and evaluate ChakrBiEncoder using contrastive triplet loss.
Compares:
1. Lexical BM25 baseline
2. Deterministic Hash Embedding baseline
3. Learned Representation model (ChakrBiEncoder)
"""
import json
import time
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.runtime.learned_embedding import ChakrBiEncoder, LearnedEmbeddingProvider
from chakrview.runtime.retrieval import DeterministicHashEmbeddingProvider, EmbeddingProvider
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

REPO_ROOT = Path(".")
DATA_PATH = Path("experiments/batch161_170_neural_learning/data/split_manifest.json")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
CKPT_DIR = Path("experiments/batch161_170_neural_learning/checkpoints")
REPORTS_DIR = Path("experiments/batch161_170_neural_learning/reports")


def tokenize_batch(texts, tokenizer, max_len=128):
    batch_ids = []
    batch_masks = []
    for t in texts:
        tokens = tokenizer.encode(t, add_bos=True, add_eos=False)[:max_len]
        batch_ids.append(tokens)
    
    max_b_len = max(len(t) for t in batch_ids)
    padded_ids = []
    masks = []
    for t in batch_ids:
        pad_len = max_b_len - len(t)
        padded_ids.append(t + [0] * pad_len)
        masks.append([1] * len(t) + [0] * pad_len)
        
    return torch.tensor(padded_ids, dtype=torch.long), torch.tensor(masks, dtype=torch.long)


def train_representation_model(epochs=100, lr=1e-3):
    device = torch.device("cpu")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    train_data = manifest["splits"]["train"]
    val_data = manifest["splits"]["val"]
    test_data = manifest["splits"]["test"]
    
    model = ChakrBiEncoder(vocab_size=4096, d_model=128, d_embed=64, n_layers=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.TripletMarginLoss(margin=0.2, p=2)
    
    queries = [item["query"] for item in train_data]
    pos_docs = [item["doc"] for item in train_data]
    neg_docs = [item["neg_doc"] for item in train_data]
    
    q_ids, q_masks = tokenize_batch(queries, tokenizer)
    p_ids, p_masks = tokenize_batch(pos_docs, tokenizer)
    n_ids, n_masks = tokenize_batch(neg_docs, tokenizer)
    
    t0 = time.perf_counter()
    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        q_emb = model(q_ids, q_masks)
        p_emb = model(p_ids, p_masks)
        n_emb = model(n_ids, n_masks)
        loss = criterion(q_emb, p_emb, n_emb)
        loss.backward()
        optimizer.step()
        
    train_time = time.perf_counter() - t0
    print(f"Training completed in {train_time:.2f}s | Final Train Triplet Loss: {loss.item():.4f}")
    
    # Save checkpoint
    ckpt_path = CKPT_DIR / "learned_bi_encoder.pt"
    learned_provider = LearnedEmbeddingProvider(model=model, tokenizer=tokenizer, device=device)
    learned_provider.save_checkpoint(ckpt_path)
    
    # Evaluation on Held-Out Test Split
    hash_provider = DeterministicHashEmbeddingProvider(dimension=64)
    
    def evaluate_provider(provider, eval_split):
        pos_sims = []
        neg_sims = []
        correct_rank = 0
        for item in eval_split:
            q_vec = provider.embed_text(item["query"])
            p_vec = provider.embed_text(item["doc"])
            n_vec = provider.embed_text(item["neg_doc"])
            
            s_pos = EmbeddingProvider.cosine_similarity(q_vec, p_vec)
            s_neg = EmbeddingProvider.cosine_similarity(q_vec, n_vec)
            pos_sims.append(s_pos)
            neg_sims.append(s_neg)
            if s_pos > s_neg:
                correct_rank += 1
                
        acc = correct_rank / len(eval_split) if eval_split else 0.0
        avg_pos = sum(pos_sims) / len(pos_sims)
        avg_neg = sum(neg_sims) / len(neg_sims)
        separation = avg_pos - avg_neg
        return {
            "accuracy": round(acc, 4),
            "avg_pos_similarity": round(avg_pos, 4),
            "avg_neg_similarity": round(avg_neg, 4),
            "margin_separation": round(separation, 4),
        }
        
    hash_test_res = evaluate_provider(hash_provider, test_data)
    learned_test_res = evaluate_provider(learned_provider, test_data)
    
    results = {
        "milestone": "Batch 163-164 Learned Representation vs Baselines",
        "training_epochs": epochs,
        "training_time_sec": round(train_time, 2),
        "final_train_loss": round(loss.item(), 4),
        "test_split_size": len(test_data),
        "baselines": {
            "deterministic_hash_baseline": hash_test_res,
            "learned_bi_encoder": learned_test_res,
        }
    }
    
    report_file = REPORTS_DIR / "representation_training_results.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {report_file}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    train_representation_model()
