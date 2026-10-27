"""
Generate SHA-256 artifact manifest for Experiment 32.
"""
import hashlib
import json
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    manifest = {}
    for p in sorted(EXP_DIR.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            rel_path = p.relative_to(EXP_DIR).as_posix()
            manifest[rel_path] = {
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p)
            }

    out_path = EXP_DIR / "MANIFEST.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Exp32 artifact manifest written to {out_path} ({len(manifest)} files).")

if __name__ == "__main__":
    main()
