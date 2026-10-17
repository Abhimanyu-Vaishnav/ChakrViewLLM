"""
Deterministic Corpus Ingestion and Splitting Runner for ChakrView Batches 2021-2060.
Recreates and verifies Corpus v2 and Split v2 manifests with complete provenance and deduplication.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json
import hashlib
from collections import Counter
from typing import Dict, List, Any
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def build_corpus_v2(
    provenance_path: Path = Path("experiments/batch2021_2060/SOURCE_PROVENANCE_MANIFEST.json"),
    vocab_dir: Path = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096"),
    output_corpus_path: Path = Path("experiments/batch2021_2060/CORPUS_V2_MANIFEST.json"),
    output_split_path: Path = Path("experiments/batch2021_2060/SPLIT_V2_MANIFEST.json"),
    max_loney_docs: int = 150
) -> Dict[str, Any]:
    tokenizer, _ = load_tokenizer_artifacts(vocab_dir)

    with open(provenance_path, encoding="utf-8") as f:
        prov = json.load(f)

    split_assignment = {
        "gutenberg_tagore_gitanjali": "train",
        "ietf_rfc793_tcp": "train",
        "cpython_algorithms_heapq": "train",
        "wikipedia_hi_premchand": "train",
        "wikipedia_en_turing_machine": "train",
        "wikipedia_hi_chemistry": "train",
        "cpython_algorithms_bisect": "validation",
        "wikipedia_hi_ganit": "validation",
        "google_hinglish_top_dialogues": "validation",
        "gutenberg_loney_coordinate_geometry": "test",
        "wikipedia_hi_sanskrit_sahitya": "test",
        "wikipedia_sa_bhagavad_gita": "test",
        "wikipedia_en_information_theory": "test"
    }

    all_docs: List[Dict[str, Any]] = []
    doc_counter = 1
    seen_hashes = set()

    for s in prov["sources"]:
        sid = s["source_id"]
        target_split = split_assignment[sid]
        domain = s["domain"]
        lang = s["language"]

        fname = s.get("file_name", sid + ".txt")
        p1 = Path("experiments/batch1941_1980/data/raw_sources") / fname
        p2 = Path("experiments/batch2021_2060/data/raw_sources") / fname
        raw_path = p1 if p1.exists() else p2
        
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw source file missing: {raw_path}")

        text = raw_path.read_text(encoding="utf-8")
        if "gutenberg" in sid:
            s_idx = text.find("*** START OF")
            if s_idx != -1:
                text = text[text.find("\n", s_idx) + 1 :]
            e_idx = text.find("*** END OF")
            if e_idx != -1:
                text = text[:e_idx]

        lines = [l.strip() for l in text.split("\n") if l.strip()]
        curr: List[str] = []
        curr_len = 0
        source_docs: List[Dict[str, Any]] = []

        for l in lines:
            curr.append(l)
            curr_len += len(l) + 1
            if curr_len >= 350:
                doc_str = "\n".join(curr)
                h = hashlib.sha256(doc_str.encode("utf-8")).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    toks = tokenizer.encode(doc_str)
                    source_docs.append({
                        "document_id": f"NATURAL_V2_{target_split.upper()}_{doc_counter:05d}",
                        "source_id": sid,
                        "source_title": s["title"],
                        "author": s["author"],
                        "domain": domain,
                        "language": lang,
                        "license": s["license"],
                        "split": target_split,
                        "character_count": len(doc_str),
                        "token_count": len(toks),
                        "sha256": h,
                        "text": doc_str
                    })
                    doc_counter += 1
                if sid == "gutenberg_loney_coordinate_geometry" and max_loney_docs > 0 and len(source_docs) >= max_loney_docs:
                    break
                curr = []
                curr_len = 0

        if curr and curr_len >= 120 and not (sid == "gutenberg_loney_coordinate_geometry" and max_loney_docs > 0 and len(source_docs) >= max_loney_docs):
            doc_str = "\n".join(curr)
            h = hashlib.sha256(doc_str.encode("utf-8")).hexdigest()
            if h not in seen_hashes:
                seen_hashes.add(h)
                toks = tokenizer.encode(doc_str)
                source_docs.append({
                    "document_id": f"NATURAL_V2_{target_split.upper()}_{doc_counter:05d}",
                    "source_id": sid,
                    "source_title": s["title"],
                    "author": s["author"],
                    "domain": domain,
                    "language": lang,
                    "license": s["license"],
                    "split": target_split,
                    "character_count": len(doc_str),
                    "token_count": len(toks),
                    "sha256": h,
                    "text": doc_str
                })
                doc_counter += 1

        all_docs.extend(source_docs)

    corpus_manifest = {
        "manifest_version": "2.0.0",
        "milestone": "batch2021_2060",
        "total_documents": len(all_docs),
        "total_tokens": sum(d["token_count"] for d in all_docs),
        "total_characters": sum(d["character_count"] for d in all_docs),
        "documents": all_docs
    }

    with open(output_corpus_path, "w", encoding="utf-8") as f:
        json.dump(corpus_manifest, f, indent=2, ensure_ascii=False)

    split_manifest = {
        "manifest_version": "2.0.0",
        "milestone": "batch2021_2060",
        "splits": {
            "train": [d["document_id"] for d in all_docs if d["split"] == "train"],
            "validation": [d["document_id"] for d in all_docs if d["split"] == "validation"],
            "test": [d["document_id"] for d in all_docs if d["split"] == "test"]
        },
        "split_summary": {
            "train": {
                "document_count": len([d for d in all_docs if d["split"] == "train"]),
                "token_count": sum(d["token_count"] for d in all_docs if d["split"] == "train"),
                "source_families": sorted(list(set(d["source_id"] for d in all_docs if d["split"] == "train")))
            },
            "validation": {
                "document_count": len([d for d in all_docs if d["split"] == "validation"]),
                "token_count": sum(d["token_count"] for d in all_docs if d["split"] == "validation"),
                "source_families": sorted(list(set(d["source_id"] for d in all_docs if d["split"] == "validation")))
            },
            "test": {
                "document_count": len([d for d in all_docs if d["split"] == "test"]),
                "token_count": sum(d["token_count"] for d in all_docs if d["split"] == "test"),
                "source_families": sorted(list(set(d["source_id"] for d in all_docs if d["split"] == "test")))
            }
        }
    }

    with open(output_split_path, "w", encoding="utf-8") as f:
        json.dump(split_manifest, f, indent=2, ensure_ascii=False)

    return corpus_manifest


if __name__ == "__main__":
    c = build_corpus_v2()
    print(f"Successfully processed {c['total_documents']} documents and {c['total_tokens']} tokens.")
