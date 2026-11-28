"""
Mission NR-058B Stage 1: Domain-Specific Evaluation and Comparative Analysis.
Evaluates final 17M and 38M checkpoints across the domain validation subsets using doc['domain']:
- code_python_stdlib
- wiki_en_stem_humanities
- wiki_hi_devanagari
- local_hindi
- local_sanskrit
- local_english
- corpus_v4_curated
Note: Mathematics domain is absent in validation set (0 tokens, assigned to train/test).

Generates comparative analysis JSON and metrics comparison.
"""

import os
import sys
import json
import math
import hashlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.brain.genesis.chakr_genesis_38m import ChakrGenesis38M, Genesis38MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

class DomainChunkDataset(Dataset):
    def __init__(self, domain_docs, tokenizer, max_len=256):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        for doc in domain_docs:
            text = doc.get("text", "")
            tokens = tokenizer.encode(text)
            if len(tokens) == 0:
                continue
            tokens.append(2)  # EOS
            for i in range(0, len(tokens), max_len):
                chunk = tokens[i:i + max_len]
                if len(chunk) >= 16:
                    self.samples.append(chunk)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        chunk = self.samples[idx]
        if len(chunk) < self.max_len:
            pad_len = self.max_len - len(chunk)
            input_ids = chunk + [0] * pad_len
        else:
            input_ids = chunk[:self.max_len]
        return torch.tensor(input_ids, dtype=torch.long)

def eval_loader(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            inp = batch[:, :-1].contiguous()
            tgt = batch[:, 1:].contiguous()
            mask = (tgt != 0)
            tgt_loss = tgt.clone()
            tgt_loss[~mask] = -100

            with torch.amp.autocast('cuda', dtype=torch.float16):
                out = model(inp)
                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                loss = criterion(logits.reshape(-1, 4096), tgt_loss.reshape(-1))

            n_active = mask.sum().item()
            total_loss += loss.item() * n_active
            total_tokens += n_active

            preds = logits.argmax(dim=-1)
            correct_tokens += ((preds == tgt) & mask).sum().item()

    if total_tokens == 0:
        return 0.0, 0.0, 0.0, 0
    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(20.0, avg_loss))
    acc = (correct_tokens / total_tokens) * 100.0
    return avg_loss, ppl, acc, total_tokens

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Domain Evaluation on {device}...")

    tok_dir = "data/experiments/vocab_4096"
    tokenizer, _ = load_tokenizer_artifacts(tok_dir)

    val_path = "data/corpus_v6/splits/corpus_v6_val.jsonl"
    domain_docs = {}
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            dom = doc.get("domain", "unknown")
            if dom not in domain_docs:
                domain_docs[dom] = []
            domain_docs[dom].append(doc)

    print("Domain distribution in validation set:")
    domain_loaders = {}
    for dom, docs in domain_docs.items():
        ds = DomainChunkDataset(docs, tokenizer, max_len=256)
        domain_loaders[dom] = DataLoader(ds, batch_size=16, shuffle=False)
        print(f"  {dom:25s}: {len(docs)} documents | {len(ds)} chunks")

    # Models to evaluate
    models_info = [
        {
            "name": "ChakrGenesis17M",
            "ckpt_path": "experiments/stage1_nr058b/chakr_chakrgenesis17m_stage1_final.pt",
            "cls": ChakrGenesis17M,
            "config_cls": Genesis17MConfig
        },
        {
            "name": "ChakrGenesis38M",
            "ckpt_path": "experiments/stage1_nr058b/chakr_chakrgenesis38m_stage1_final.pt",
            "cls": ChakrGenesis38M,
            "config_cls": Genesis38MConfig
        }
    ]

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    domain_results = {}

    for m_info in models_info:
        m_name = m_info["name"]
        print(f"\nEvaluating {m_name} across domains...")
        ckpt = torch.load(m_info["ckpt_path"], map_location=device, weights_only=False)
        model = m_info["cls"](ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        domain_results[m_name] = {}
        for dom, loader in domain_loaders.items():
            if len(loader.dataset) == 0:
                continue
            loss, ppl, acc, n_toks = eval_loader(model, loader, criterion, device)
            print(f"  [{m_name}] {dom:25s} | Loss: {loss:.4f} | PPL: {ppl:.2f} | Acc: {acc:.2f}% | Tokens: {n_toks:,}")
            domain_results[m_name][dom] = {
                "loss": round(loss, 4),
                "perplexity": round(ppl, 2),
                "accuracy_pct": round(acc, 2),
                "active_tokens": n_toks
            }

    # Load 17M and 38M summaries
    with open("experiments/stage1_nr058b/chakrgenesis17m_stage1_summary.json", "r") as f:
        sum_17m = json.load(f)
    with open("experiments/stage1_nr058b/chakrgenesis38m_stage1_summary.json", "r") as f:
        sum_38m = json.load(f)

    # Comparative analysis compilation
    comparative = {
        "mission": "NR-058B Stage 1 Controlled Pretraining",
        "description": "Comparative 1-epoch pretraining analysis of ChakrGenesis17M vs ChakrGenesis38M on Corpus V6",
        "dataset": {
            "name": "Corpus V6 Train",
            "train_tokens_manifest": 4163555,
            "active_training_tokens": 4170667,
            "total_chunks": 20865,
            "sequence_length": 256,
            "batch_size": 16,
            "total_steps_per_epoch": 1305
        },
        "models": {
            "ChakrGenesis17M": {
                "parameters": sum_17m["parameters"],
                "peak_learning_rate": sum_17m["peak_learning_rate"],
                "final_learning_rate": sum_17m["final_learning_rate"],
                "initial_train_loss": sum_17m["initial_train_loss"],
                "final_train_loss": sum_17m["final_train_loss"],
                "initial_val_loss": sum_17m["initial_val_loss"],
                "final_val_loss": sum_17m["final_val_loss"],
                "initial_val_ppl": sum_17m["initial_val_ppl"],
                "final_val_ppl": sum_17m["final_val_ppl"],
                "final_val_acc_pct": sum_17m["final_val_acc_pct"],
                "elapsed_seconds": sum_17m["elapsed_seconds"],
                "tokens_per_sec": sum_17m["tokens_per_sec"],
                "steps_per_sec": sum_17m["steps_per_sec"],
                "peak_vram_gb": sum_17m["peak_vram_gb"],
                "checkpoint_sha256": sum_17m["final_checkpoint_sha256"],
                "domain_validation": domain_results["ChakrGenesis17M"]
            },
            "ChakrGenesis38M": {
                "parameters": sum_38m["parameters"],
                "peak_learning_rate": sum_38m["peak_learning_rate"],
                "final_learning_rate": sum_38m["final_learning_rate"],
                "initial_train_loss": sum_38m["initial_train_loss"],
                "final_train_loss": sum_38m["final_train_loss"],
                "initial_val_loss": sum_38m["initial_val_loss"],
                "final_val_loss": sum_38m["final_val_loss"],
                "initial_val_ppl": sum_38m["initial_val_ppl"],
                "final_val_ppl": sum_38m["final_val_ppl"],
                "final_val_acc_pct": sum_38m["final_val_acc_pct"],
                "elapsed_seconds": sum_38m["elapsed_seconds"],
                "tokens_per_sec": sum_38m["tokens_per_sec"],
                "steps_per_sec": sum_38m["steps_per_sec"],
                "peak_vram_gb": sum_38m["peak_vram_gb"],
                "checkpoint_sha256": sum_38m["final_checkpoint_sha256"],
                "domain_validation": domain_results["ChakrGenesis38M"]
            }
        },
        "head_to_head_deltas": {
            "validation_loss_delta_38m_vs_17m": round(sum_38m["final_val_loss"] - sum_17m["final_val_loss"], 4),
            "validation_ppl_delta_38m_vs_17m": round(sum_38m["final_val_ppl"] - sum_17m["final_val_ppl"], 2),
            "validation_accuracy_delta_38m_vs_17m_pct": round(sum_38m["final_val_acc_pct"] - sum_17m["final_val_acc_pct"], 2),
            "training_time_ratio_38m_vs_17m": round(sum_38m["elapsed_seconds"] / sum_17m["elapsed_seconds"], 2),
            "throughput_ratio_38m_vs_17m": round(sum_38m["tokens_per_sec"] / sum_17m["tokens_per_sec"], 2),
            "vram_delta_gb": round(sum_38m["peak_vram_gb"] - sum_17m["peak_vram_gb"], 3),
            "superior_candidate_overall": "ChakrGenesis38M" if sum_38m["final_val_loss"] < sum_17m["final_val_loss"] else "ChakrGenesis17M"
        },
        "mathematics_domain_note": "Corpus V6 Validation split contains 0 mathematics documents/tokens because all math documents were allocated to train and held-out test splits. Reported validation metrics reflect code_python_stdlib, wiki_en_stem_humanities, wiki_hi_devanagari, local_hindi, local_sanskrit, local_english, and corpus_v4_curated."
    }

    out_comp = "experiments/stage1_nr058b/comparative_stage1_analysis.json"
    with open(out_comp, "w", encoding="utf-8") as f:
        json.dump(comparative, f, indent=2)

    print(f"\nComparative analysis written to {out_comp}")
    print(f"Summary: 38M achieved val loss {sum_38m['final_val_loss']} vs 17M {sum_17m['final_val_loss']} (delta: {comparative['head_to_head_deltas']['validation_loss_delta_38m_vs_17m']})")
    print(f"Summary: 38M achieved val PPL {sum_38m['final_val_ppl']} vs 17M {sum_17m['final_val_ppl']} (delta: {comparative['head_to_head_deltas']['validation_ppl_delta_38m_vs_17m']})")

if __name__ == "__main__":
    main()
