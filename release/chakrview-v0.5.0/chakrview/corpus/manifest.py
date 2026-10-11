"""
ChakrView Corpus Pipeline: Manifest Module.

Generates reproducible metadata manifests for all corpus files with:
- SHA-256 hashes
- File byte sizes and line counts
- Category and linguistic metadata
- Licensing status
- Pipeline timestamps
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def generate_file_manifest(file_path: Union[str, Path], category: str) -> Dict[str, Any]:
    """Generate metadata and SHA-256 hash for a single corpus file."""
    p = Path(file_path)
    raw_b = p.read_bytes()
    text = raw_b.decode("utf-8", errors="replace")

    return {
        "filename": p.name,
        "relative_path": f"{category}/{p.name}",
        "category": category,
        "byte_count": len(raw_b),
        "line_count": len(text.splitlines()),
        "sha256": hashlib.sha256(raw_b).hexdigest(),
        "source": "Project-authored, indigenous synthesis & public domain classical texts",
        "license": "CC0-1.0 / MIT Compatible",
        "preprocessing_status": "Clean UTF-8 raw text without destructive normalization",
    }


def generate_corpus_manifest(
    root_dir: Union[str, Path],
    title: str = "ChakrView Research Corpus Manifest (Step 3)",
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Generate complete manifest across all categories in root_dir.
    """
    p = Path(root_dir)
    manifest: Dict[str, Any] = {
        "title": title,
        "project": "ChakrView",
        "version": "0.1.0",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_files": 0,
        "total_bytes": 0,
        "categories": {},
    }

    total_files = 0
    total_bytes = 0

    for cat_dir in sorted(p.iterdir()):
        if cat_dir.is_dir() and not cat_dir.name.startswith("."):
            cat = cat_dir.name
            manifest["categories"][cat] = {"files": [], "category_bytes": 0}
            cat_b = 0
            for txt_file in sorted(cat_dir.glob("*.txt")):
                f_meta = generate_file_manifest(txt_file, cat)
                manifest["categories"][cat]["files"].append(f_meta)
                cat_b += f_meta["byte_count"]
                total_bytes += f_meta["byte_count"]
                total_files += 1
            manifest["categories"][cat]["category_bytes"] = cat_b

    manifest["total_files"] = total_files
    manifest["total_bytes"] = total_bytes

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return manifest
