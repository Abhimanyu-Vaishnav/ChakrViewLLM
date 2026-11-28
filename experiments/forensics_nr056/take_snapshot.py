import os
import sys
import json
import hashlib
import subprocess

def get_sha256(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

snapshot = {
    "mission": "NR-056",
    "timestamp": "2026-11-28T08:18:00+05:30",
    "git": {
        "branch": "exp/cbr-01-conversational-breakthrough",
        "remote": "https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git",
        "head_commit": "7429bf764f61f9dfa0ae58c89c8942b826b1c85d",
        "head_commit_msg": "feat(nr-055): architecture forensics, model loader validation, and Benchmark V2 evaluation"
    },
    "key_artifacts": {}
}

artifacts = {
    "tokenizer_vocab": "data/experiments/vocab_4096/vocab.json",
    "tokenizer_merges": "data/experiments/vocab_4096/merges.txt",
    "benchmark_v2_json": "data/evaluation/benchmark_v2/benchmark_v2_100.json",
    "benchmark_v2_manifest": "data/manifests/benchmark_v2_manifest.json",
    "benchmark_v2_results": "experiments/forensics_nr055/benchmark_v2_results.json",
    "benchmark_v2_blind_records": "experiments/forensics_nr055/blind_evaluation_records.json",
    "robust_evaluator_script": "experiments/forensics_nr055/robust_evaluator.py",
    "checkpoint_staged_best": "experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt",
    "checkpoint_rc1": "release/v1.0.0-rc1/chakr_1_0_rc1_best.pt",
    "checkpoint_rc2": "experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt",
    "checkpoint_rc3": "experiments/fast_track_047_049/chakr_1_0_rc3_best.pt",
    "checkpoint_nr051_17m_base": "experiments/pretraining_nr051/chakr_17m_pretrained_base.pt",
    "checkpoint_nr051_38m_base": "experiments/pretraining_nr051/chakr_38m_pretrained_base.pt",
    "checkpoint_nr051_17m_sft": "experiments/pretraining_nr051/chakr_17m_sft_candidate.pt",
    "checkpoint_nr051_38m_sft": "experiments/pretraining_nr051/chakr_38m_sft_candidate.pt",
    "checkpoint_nr052_17m_base": "experiments/pretraining_nr052/chakr_17m_v4_base.pt",
    "checkpoint_nr052_38m_base": "experiments/pretraining_nr052/chakr_38m_v4_base.pt",
    "checkpoint_nr052_17m_sft": "experiments/pretraining_nr052/chakr_17m_nr052_sft.pt",
    "checkpoint_nr052_38m_sft": "experiments/pretraining_nr052/chakr_38m_nr052_sft.pt"
}

for name, path in artifacts.items():
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    sha = get_sha256(path) if exists else None
    snapshot["key_artifacts"][name] = {
        "path": path,
        "exists": exists,
        "size_bytes": size,
        "sha256": sha
    }

with open("experiments/forensics_nr056/repository_snapshot.json", "w", encoding="utf-8") as f:
    json.dump(snapshot, f, indent=2)

print("Repository snapshot saved successfully. Artifacts verified:", len(snapshot["key_artifacts"]))
