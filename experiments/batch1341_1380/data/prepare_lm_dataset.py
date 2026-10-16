"""
Split generator and provenance manifest builder for Batch 1341–1380.
Location: experiments/batch1341_1380/data/prepare_lm_dataset.py
"""
import hashlib
import json
from pathlib import Path

CORPUS_DIR = Path("experiments/batch1141_1180/data/corpus")
OUT_DIR = Path("experiments/batch1341_1380/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()

def main():
    files = sorted(CORPUS_DIR.glob("*.md"))
    print(f"Total corpus files: {len(files)}")
    
    # Stratified or alternating split by category to ensure balanced representation
    # Categories: billing, compute, network, security, sop, storage
    by_category = {}
    for f in files:
        cat = f.name.split("_")[0]
        by_category.setdefault(cat, []).append(f)
        
    train_files = []
    test_files = []
    
    for cat, doc_list in sorted(by_category.items()):
        # Split 50/50 per category
        mid = len(doc_list) // 2
        train_files.extend(doc_list[:mid])
        test_files.extend(doc_list[mid:])
        
    print(f"Train/Calibration set: {len(train_files)} files")
    print(f"Held-Out Test set: {len(test_files)} files")
    
    manifest = {
        "description": "ChakrMicro Language Modeling Evaluation Corpus Split",
        "total_documents": len(files),
        "train_split": [
            {"filename": f.name, "sha256": compute_sha256(f), "size_bytes": f.stat().st_size}
            for f in train_files
        ],
        "test_split": [
            {"filename": f.name, "sha256": compute_sha256(f), "size_bytes": f.stat().st_size}
            for f in test_files
        ]
    }
    
    out_path = OUT_DIR / "corpus_lm_split_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to: {out_path}")

if __name__ == "__main__":
    main()
