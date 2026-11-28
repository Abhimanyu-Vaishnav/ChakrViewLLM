import os
import json
import hashlib
import torch
import psutil

def get_sha256(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

baseline = {
    "mission": "NR-057",
    "timestamp": "2026-11-28T08:50:00+05:30",
    "git": {
        "branch": "exp/cbr-01-conversational-breakthrough",
        "remote": "https://github.com/Abhimanyu-Vaishnav/ChakrViewLLM.git",
        "head_commit": "a00d7543888ae185b37651a25fe38fa74780df6d",
        "clean": True
    },
    "environment": {
        "os": "Windows",
        "python": "3.11",
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0,
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "disk_d_free_gb": round(psutil.disk_usage("D:\\").free / 1e9, 2),
        "disk_c_free_gb": round(psutil.disk_usage("C:\\").free / 1e9, 2)
    },
    "canonical_checkpoints": {
        "genesis_17m_staged_best": {
            "path": "experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt",
            "sha256": get_sha256("experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt")
        },
        "chakr_1_0_rc1_best": {
            "path": "release/v1.0.0-rc1/chakr_1_0_rc1_best.pt",
            "sha256": get_sha256("release/v1.0.0-rc1/chakr_1_0_rc1_best.pt")
        },
        "chakr_1_0_rc2_best": {
            "path": "experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt",
            "sha256": get_sha256("experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt")
        },
        "chakr_1_0_rc3_best": {
            "path": "experiments/fast_track_047_049/chakr_1_0_rc3_best.pt",
            "sha256": get_sha256("experiments/fast_track_047_049/chakr_1_0_rc3_best.pt")
        },
        "chakr_17m_v4_base": {
            "path": "experiments/pretraining_nr052/chakr_17m_v4_base.pt",
            "sha256": get_sha256("experiments/pretraining_nr052/chakr_17m_v4_base.pt")
        },
        "chakr_38m_v4_base": {
            "path": "experiments/pretraining_nr052/chakr_38m_v4_base.pt",
            "sha256": get_sha256("experiments/pretraining_nr052/chakr_38m_v4_base.pt")
        },
        "chakr_17m_nr052_sft": {
            "path": "experiments/pretraining_nr052/chakr_17m_nr052_sft.pt",
            "sha256": get_sha256("experiments/pretraining_nr052/chakr_17m_nr052_sft.pt")
        },
        "chakr_38m_nr052_sft": {
            "path": "experiments/pretraining_nr052/chakr_38m_nr052_sft.pt",
            "sha256": get_sha256("experiments/pretraining_nr052/chakr_38m_nr052_sft.pt")
        }
    },
    "canonical_tokenizer": {
        "vocab_path": "data/experiments/vocab_4096/vocab.json",
        "vocab_sha256": get_sha256("data/experiments/vocab_4096/vocab.json"),
        "merges_path": "data/experiments/vocab_4096/merges.txt",
        "merges_sha256": get_sha256("data/experiments/vocab_4096/merges.txt")
    }
}

with open("experiments/corpus_nr057/repository_baseline.json", "w") as f:
    json.dump(baseline, f, indent=2)

print("Baseline recorded successfully.")
