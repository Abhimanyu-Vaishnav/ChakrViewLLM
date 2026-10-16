"""
Domain-Adaptive Contrastive Training Pipeline for ChakrBiEncoder.
Location: experiments/batch1181_1220/training/train_domain_adaptive.py

Invariants:
- 100% indigenous architecture (ChakrBiEncoder) and indigenous 4096-vocab BPE tokenizer.
- Zero external libraries/weights (No Hugging Face, No pretrained models, No cloud APIs).
- CPU-only execution with bounded resources and deterministic seed.
- Checkpoint selection based strictly on development loss/MRR; held-out test split is evaluated only once at completion.
- Saves model to experiments/batch1181_1220/checkpoints/domain_adaptive_bi_encoder.pt.
"""
import hashlib
import json
import random
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.runtime.learned_embedding import ChakrBiEncoder, LearnedEmbeddingProvider
from chakrview.runtime.retrieval import EmbeddingProvider
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

DATA_DIR = Path("experiments/batch1181_1220/data")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
CKPT_DIR = Path("experiments/batch1181_1220/checkpoints")
RESULTS_DIR = Path("experiments/batch1181_1220/results")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def tokenize_batch(texts, tokenizer, max_len=128):
    batch_ids = []
    for t in texts:
        tokens = tokenizer.encode(t, add_bos=True, add_eos=False)[:max_len]
        batch_ids.append(tokens if tokens else [0])
    
    max_b_len = max(len(t) for t in batch_ids)
    padded_ids = []
    masks = []
    for t in batch_ids:
        pad_len = max_b_len - len(t)
        padded_ids.append(t + [0] * pad_len)
        masks.append([1] * len(t) + [0] * pad_len)
        
    return torch.tensor(padded_ids, dtype=torch.long), torch.tensor(masks, dtype=torch.long)

def compute_eval_metrics(model, data, tokenizer, device):
    model.eval()
    queries = [item["query"] for item in data]
    pos_docs = [item["pos_passage"] for item in data]
    neg_docs = [item["neg_passage"] for item in data]
    
    q_ids, q_masks = tokenize_batch(queries, tokenizer)
    p_ids, p_masks = tokenize_batch(pos_docs, tokenizer)
    n_ids, n_masks = tokenize_batch(neg_docs, tokenizer)
    
    with torch.no_grad():
        q_emb = model(q_ids.to(device), q_masks.to(device))
        p_emb = model(p_ids.to(device), p_masks.to(device))
        n_emb = model(n_ids.to(device), n_masks.to(device))
        
        pos_sim = torch.sum(q_emb * p_emb, dim=-1)
        neg_sim = torch.sum(q_emb * n_emb, dim=-1)
        
        # Accuracy: positive similarity > negative similarity
        correct = (pos_sim > neg_sim).float().sum().item()
        margin = (pos_sim - neg_sim).mean().item()
        loss = torch.clamp(0.2 - (pos_sim - neg_sim), min=0.0).mean().item()
        
    return {
        "accuracy": round(correct / len(data), 4),
        "mean_margin": round(margin, 4),
        "triplet_loss": round(loss, 4),
        "avg_pos_sim": round(pos_sim.mean().item(), 4),
        "avg_neg_sim": round(neg_sim.mean().item(), 4)
    }

def train_pipeline(epochs=60, lr=1e-3, batch_size=32):
    device = torch.device("cpu")
    print(f"Loading indigenous tokenizer from {TOKENIZER_DIR}...")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    train_data = json.loads((DATA_DIR / "train_split.json").read_text(encoding="utf-8"))
    dev_data = json.loads((DATA_DIR / "dev_split.json").read_text(encoding="utf-8"))
    test_data = json.loads((DATA_DIR / "test_split.json").read_text(encoding="utf-8"))
    
    print(f"Dataset Loaded: Train={len(train_data)}, Dev={len(dev_data)}, Test={len(test_data)}")
    
    # Initialize fresh ChakrBiEncoder
    model = ChakrBiEncoder(
        vocab_size=4096,
        d_model=128,
        d_embed=64,
        max_seq_len=256,
        n_layers=2
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.TripletMarginLoss(margin=0.2, p=2)
    
    best_dev_acc = 0.0
    best_dev_loss = float("inf")
    best_model_state = None
    
    loss_history = []
    
    t0 = time.perf_counter()
    print("Starting CPU-bound domain-adaptive contrastive training...")
    for ep in range(1, epochs + 1):
        model.train()
        random.shuffle(train_data)
        
        ep_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]
            q_ids, q_masks = tokenize_batch([b["query"] for b in batch], tokenizer)
            p_ids, p_masks = tokenize_batch([b["pos_passage"] for b in batch], tokenizer)
            n_ids, n_masks = tokenize_batch([b["neg_passage"] for b in batch], tokenizer)
            
            optimizer.zero_grad()
            q_emb = model(q_ids.to(device), q_masks.to(device))
            p_emb = model(p_ids.to(device), p_masks.to(device))
            n_emb = model(n_ids.to(device), n_masks.to(device))
            
            loss = criterion(q_emb, p_emb, n_emb)
            loss.backward()
            optimizer.step()
            
            ep_loss += loss.item()
            num_batches += 1
            
        avg_ep_loss = ep_loss / num_batches
        loss_history.append(avg_ep_loss)
        
        # Periodic evaluation on Dev split
        if ep % 5 == 0 or ep == epochs:
            dev_metrics = compute_eval_metrics(model, dev_data, tokenizer, device)
            print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {avg_ep_loss:.4f} | Dev Acc: {dev_metrics['accuracy']:.4f} | Dev Margin: {dev_metrics['mean_margin']:.4f}")
            
            if dev_metrics["accuracy"] >= best_dev_acc and dev_metrics["triplet_loss"] < best_dev_loss:
                best_dev_acc = dev_metrics["accuracy"]
                best_dev_loss = dev_metrics["triplet_loss"]
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
    train_time = time.perf_counter() - t0
    print(f"Training completed in {train_time:.2f}s. Best Dev Accuracy: {best_dev_acc:.4f}")
    
    # Load best checkpoint
    model.load_state_dict(best_model_state)
    
    # Save checkpoint
    out_ckpt_path = CKPT_DIR / "domain_adaptive_bi_encoder.pt"
    learned_provider = LearnedEmbeddingProvider(
        model=model,
        tokenizer=tokenizer,
        device=device,
        provider_id="domain_adaptive_bi_encoder_v1"
    )
    learned_provider.save_checkpoint(out_ckpt_path)
    ckpt_hash = hashlib.sha256(out_ckpt_path.read_bytes()).hexdigest()
    print(f"Saved trained checkpoint -> {out_ckpt_path} (SHA-256: {ckpt_hash})")
    
    # Final one-shot evaluation on held-out test split
    test_metrics = compute_eval_metrics(model, test_data, tokenizer, device)
    print(f"Held-Out Test Results: Accuracy={test_metrics['accuracy']:.4f}, Margin={test_metrics['mean_margin']:.4f}")
    
    train_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "training_time_sec": round(train_time, 2),
        "checkpoint_path": str(out_ckpt_path),
        "checkpoint_sha256": ckpt_hash,
        "best_dev_accuracy": best_dev_acc,
        "best_dev_loss": best_dev_loss,
        "test_split_metrics": test_metrics,
        "loss_history": [round(x, 4) for x in loss_history]
    }
    
    summary_path = RESULTS_DIR / "TRAINING_SUMMARY.json"
    summary_path.write_text(json.dumps(train_summary, indent=2), encoding="utf-8")
    print(f"Wrote training summary to {summary_path}")

if __name__ == "__main__":
    train_pipeline()
