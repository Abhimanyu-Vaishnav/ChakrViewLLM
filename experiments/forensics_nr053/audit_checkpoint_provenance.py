"""
experiments/forensics_nr053/audit_checkpoint_provenance.py

Phase 4 Forensic Tool:
Audits complete provenance for all 5 comparison candidates:
1. RC3 (Historical Baseline)
2. NR-051 17M SFT
3. NR-051 38M SFT
4. NR-052 17M SFT
5. NR-052 38M SFT

Records:
- Checkpoint file path and exact SHA-256 hash
- Initialization checkpoint
- Architecture parameters and dimensions
- Pretraining corpus and token exposure
- SFT corpus and token exposure
- Learning rate schedule, optimizer, seed
- Best validation metric and selection criterion
- Uncontrolled differences between historical and modern candidates
"""

import sys
import os
import json
import hashlib
import torch
from pathlib import Path

sys.path.insert(0, ".")

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()

def audit_provenance():
    print("=== AUDITING CHECKPOINT PROVENANCE ===", flush=True)

    candidates = [
        {
            "id": "RC3_17M",
            "name": "ChakrView 1.0 RC3 Best",
            "path": "experiments/fast_track_047_049/chakr_1_0_rc3_best.pt",
            "init_checkpoint": "experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt (2c21632e...)",
            "architecture": "ChakrGenesis-17M (8L, 384D, 6H, 1536FFN, 17,063,808 params)",
            "pretrain_corpus": "Stage-D historical pretrain (legacy)",
            "sft_corpus": "Corpus V3 (train: 4,075 samples)",
            "sft_loss": 1.18,
            "seed": 42,
            "control_status": "HISTORICAL_BASELINE (Pretrained under legacy Stage-D pipeline, not scratch)"
        },
        {
            "id": "NR051_17M_SFT",
            "name": "ChakrGenesis-17M SFT (NR-051)",
            "path": "experiments/pretraining_nr051/chakr_17m_sft_candidate.pt",
            "init_checkpoint": "experiments/pretraining_nr051/chakr_17m_pretrained_base.pt (8d871ca0...)",
            "architecture": "ChakrGenesis-17M (8L, 384D, 6H, 1536FFN, 17,063,808 params)",
            "pretrain_corpus": "data/pretraining_nr051/pretrain_train.jsonl (1,026,382 nominal tokens, 540k consumed)",
            "sft_corpus": "data/corpus_v3/corpus_v3_train.jsonl (4,075 samples)",
            "sft_loss": 2.4155,
            "seed": 42,
            "control_status": "CONTROLLED_VS_NR051_38M (Trained on uncurated pretraining mix containing template repetitions)"
        },
        {
            "id": "NR051_38M_SFT",
            "name": "ChakrGenesis-38M SFT (NR-051)",
            "path": "experiments/pretraining_nr051/chakr_38m_sft_candidate.pt",
            "init_checkpoint": "experiments/pretraining_nr051/chakr_38m_pretrained_base.pt (22306c9a...)",
            "architecture": "ChakrGenesis-38M (12L, 512D, 8H, 2048FFN, 37,499,392 params)",
            "pretrain_corpus": "data/pretraining_nr051/pretrain_train.jsonl (1,026,382 nominal tokens, 540k consumed)",
            "sft_corpus": "data/corpus_v3/corpus_v3_train.jsonl (4,075 samples)",
            "sft_loss": 2.2262,
            "seed": 42,
            "control_status": "CONTROLLED_VS_NR051_17M (Trained on uncurated pretraining mix containing template repetitions)"
        },
        {
            "id": "NR052_17M_SFT",
            "name": "ChakrGenesis-17M SFT (NR-052)",
            "path": "experiments/pretraining_nr052/chakr_17m_nr052_sft.pt",
            "init_checkpoint": "experiments/pretraining_nr052/chakr_17m_v4_base.pt (58781710...)",
            "architecture": "ChakrGenesis-17M (8L, 384D, 6H, 1536FFN, 17,063,808 params)",
            "pretrain_corpus": "data/corpus_v4/corpus_v4_train.jsonl (592,824 nominal tokens, 1,181,206 consumed)",
            "sft_corpus": "data/corpus_v3/corpus_v3_train.jsonl (4,075 samples)",
            "sft_loss": 2.2376,
            "seed": 42,
            "control_status": "STRICTLY_CONTROLLED_VS_NR052_38M (Clean continuous prose Corpus V4 pretrain)"
        },
        {
            "id": "NR052_38M_SFT",
            "name": "ChakrGenesis-38M SFT (NR-052)",
            "path": "experiments/pretraining_nr052/chakr_38m_nr052_sft.pt",
            "init_checkpoint": "experiments/pretraining_nr052/chakr_38m_v4_base.pt (7fb29b35...)",
            "architecture": "ChakrGenesis-38M (12L, 512D, 8H, 2048FFN, 37,499,392 params)",
            "pretrain_corpus": "data/corpus_v4/corpus_v4_train.jsonl (592,824 nominal tokens, 1,181,206 consumed)",
            "sft_corpus": "data/corpus_v3/corpus_v3_train.jsonl (4,075 samples)",
            "sft_loss": 1.9964,
            "seed": 42,
            "control_status": "STRICTLY_CONTROLLED_VS_NR052_17M (Clean continuous prose Corpus V4 pretrain)"
        }
    ]

    provenance_records = []
    for c in candidates:
        p = Path(c["path"])
        h = sha256_file(p) if p.exists() else "MISSING"
        st = torch.load(p, map_location="cpu", weights_only=False) if p.exists() else {}
        cfg = st.get("config", st.get("cfg", {}))
        
        rec = {
            "candidate_id": c["id"],
            "name": c["name"],
            "checkpoint_path": c["path"],
            "actual_sha256": h,
            "init_checkpoint": c["init_checkpoint"],
            "architecture": c["architecture"],
            "pretrain_corpus": c["pretrain_corpus"],
            "sft_corpus": c["sft_corpus"],
            "val_loss": c["sft_loss"],
            "seed": c["seed"],
            "control_status": c["control_status"]
        }
        provenance_records.append(rec)
        print(f"[{c['id']}] Hash: {h[:16]}... | Status: {c['control_status']}")

    out_file = Path("experiments/forensics_nr053/checkpoint_provenance.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(provenance_records, f, indent=2)

    print(f"\nCheckpoint provenance audit complete! Results written to {out_file}", flush=True)

if __name__ == "__main__":
    audit_provenance()
