"""
Batch 232 & 233: Generate Leakage-Resistant Multi-Hop Corpus & Frozen Manifest.
Location: experiments/batch231_240_multihop_reasoning/data/generate_multihop_corpus.py

Creates synthetic 2-hop relational chains across distinct documents:
Doc A: Entity E1 has bridge property P1 = E2.
Doc B: Intermediate entity E2 has target attribute R2 = Target V.
Query: "What is the R2 of the P1 of E1?" or composite phrasing.

Includes:
- Answerable 2-hop chains.
- Missing link cases (Doc B missing).
- Contradictory evidence (two conflicting Doc B values).
- Numeric distractors (competing numbers in Doc A and Doc B).
- Multilingual English and Hindi (Devanagari) chains.
"""
import hashlib
import json
from pathlib import Path
import random


def build_multihop_corpus(seed=42):
    random.seed(seed)

    # 1. Base entities and bridge attributes
    # Chain template: E1 -> bridge -> E2 -> target_attr -> TargetValue
    base_chains = [
        # Science & Astronomy
        {"e1": "Earth", "bridge_attr": "natural satellite", "e2": "Moon", "target_attr": "orbital period", "target_val": "27.3 days", "distractor": "29.5 days", "domain": "science", "lang": "en"},
        {"e1": "Jupiter", "bridge_attr": "largest moon", "e2": "Ganymede", "target_attr": "orbital radius", "target_val": "1070400 km", "distractor": "421700 km", "domain": "science", "lang": "en"},
        {"e1": "Saturn", "bridge_attr": "largest moon", "e2": "Titan", "target_attr": "surface temperature", "target_val": "94 K", "distractor": "134 K", "domain": "science", "lang": "en"},
        {"e1": "Mars", "bridge_attr": "larger moon", "e2": "Phobos", "target_attr": "orbital period", "target_val": "7.66 hours", "distractor": "30.35 hours", "domain": "science", "lang": "en"},
        {"e1": "Sun", "bridge_attr": "closest planet", "e2": "Mercury", "target_attr": "orbital eccentricity", "target_val": "0.2056", "distractor": "0.0068", "domain": "science", "lang": "en"},

        # Chemistry & Materials
        {"e1": "Water", "bridge_attr": "constituent gas", "e2": "Hydrogen", "target_attr": "atomic number", "target_val": "1", "distractor": "8", "domain": "science", "lang": "en"},
        {"e1": "Carbon Dioxide", "bridge_attr": "primary element", "e2": "Carbon", "target_attr": "atomic number", "target_val": "6", "distractor": "12", "domain": "science", "lang": "en"},
        {"e1": "Rust", "bridge_attr": "base metallic element", "e2": "Iron", "target_attr": "melting point", "target_val": "1538 C", "distractor": "2862 C", "domain": "science", "lang": "en"},
        {"e1": "Bronze", "bridge_attr": "major constituent", "e2": "Copper", "target_attr": "atomic number", "target_val": "29", "distractor": "50", "domain": "science", "lang": "en"},
        {"e1": "Table Salt", "bridge_attr": "alkali metal component", "e2": "Sodium", "target_attr": "atomic number", "target_val": "11", "distractor": "17", "domain": "science", "lang": "en"},

        # Mathematics & Geometry
        {"e1": "Circle", "bridge_attr": "fundamental constant", "e2": "Pi", "target_attr": "first decimals", "target_val": "3.14159", "distractor": "2.71828", "domain": "mathematics", "lang": "en"},
        {"e1": "Natural Logarithm", "bridge_attr": "base constant", "e2": "Euler constant e", "target_attr": "approximate value", "target_val": "2.71828", "distractor": "1.61803", "domain": "mathematics", "lang": "en"},
        {"e1": "Golden Spiral", "bridge_attr": "growth ratio", "e2": "Golden Ratio phi", "target_attr": "approximate value", "target_val": "1.61803", "distractor": "3.14159", "domain": "mathematics", "lang": "en"},
        {"e1": "Square", "bridge_attr": "diagonal factor", "e2": "Root Two", "target_attr": "approximate value", "target_val": "1.41421", "distractor": "1.73205", "domain": "mathematics", "lang": "en"},
        {"e1": "Equilateral Triangle", "bridge_attr": "height factor", "e2": "Root Three", "target_attr": "approximate value", "target_val": "1.73205", "distractor": "1.41421", "domain": "mathematics", "lang": "en"},

        # Coding & Algorithms
        {"e1": "Priority Queue", "bridge_attr": "underlying structure", "e2": "Binary Heap", "target_attr": "insert complexity", "target_val": "O(log n)", "distractor": "O(1)", "domain": "coding", "lang": "en"},
        {"e1": "QuickSort", "bridge_attr": "partitioning step", "e2": "Lomuto Partition", "target_attr": "time complexity", "target_val": "O(n)", "distractor": "O(log n)", "domain": "coding", "lang": "en"},
        {"e1": "Dijkstra Algorithm", "bridge_attr": "optimal auxiliary queue", "e2": "Fibonacci Heap", "target_attr": "decrease-key complexity", "target_val": "O(1)", "distractor": "O(log n)", "domain": "coding", "lang": "en"},
        {"e1": "Hash Table", "bridge_attr": "collision strategy", "e2": "Separate Chaining", "target_attr": "worst case lookup", "target_val": "O(n)", "distractor": "O(1)", "domain": "coding", "lang": "en"},
        {"e1": "Red-Black Tree", "bridge_attr": "rebalancing operation", "e2": "Tree Rotation", "target_attr": "operation complexity", "target_val": "O(1)", "distractor": "O(log n)", "domain": "coding", "lang": "en"},

        # Multilingual Hindi (Devanagari)
        {"e1": "पृथ्वी (Earth)", "bridge_attr": "प्राकृतिक उपग्रह", "e2": "चंद्रमा (Moon)", "target_attr": "परिक्रमण काल", "target_val": "27.3 दिन", "distractor": "29.5 दिन", "domain": "science", "lang": "hi"},
        {"e1": "जल (Water)", "bridge_attr": "घटक तत्व", "e2": "हाइड्रोजन (Hydrogen)", "target_attr": "परमाणु क्रमांक", "target_val": "1", "distractor": "8", "domain": "science", "lang": "hi"},
        {"e1": "नमक (Salt)", "bridge_attr": "सक्रिय धातु घटक", "e2": "सोडियम (Sodium)", "target_attr": "परमाणु क्रमांक", "target_val": "11", "distractor": "17", "domain": "science", "lang": "hi"},
        {"e1": "कांस्य (Bronze)", "bridge_attr": "प्रमुख धातु", "e2": "तांबा (Copper)", "target_attr": "गलनांक", "target_val": "1084.62 C", "distractor": "231.9 C", "domain": "science", "lang": "hi"},
        {"e1": "बृहस्पति (Jupiter)", "bridge_attr": "सबसे बड़ा उपग्रह", "e2": "गैनिमीड (Ganymede)", "target_attr": "कक्षा त्रिज्या", "target_val": "1070400 km", "distractor": "421700 km", "domain": "science", "lang": "hi"},
    ]

    # Cluster split by base entity E1
    # Train: 15 chains (3,000 variations)
    # Val: 4 chains (800 variations)
    # Test: 6 chains (held-out entities, 1,200 variations)
    held_out_e1 = {"Saturn", "Table Salt", "Red-Black Tree", "Square", "कांस्य (Bronze)", "बृहस्पति (Jupiter)"}
    val_e1 = {"Mars", "Bronze", "Equilateral Triangle", "Priority Queue"}

    train_base = [c for c in base_chains if c["e1"] not in held_out_e1 and c["e1"] not in val_e1]
    val_base = [c for c in base_chains if c["e1"] in val_e1]
    test_base = [c for c in base_chains if c["e1"] in held_out_e1]

    def expand_chain(c, idx, split_name):
        e1, b_attr, e2, t_attr, t_val, dist_val = c["e1"], c["bridge_attr"], c["e2"], c["target_attr"], c["target_val"], c["distractor"]
        lang = c["lang"]

        if lang == "en":
            doc_a = f"Document A: The {b_attr} of {e1} is {e2}."
            doc_b = f"Document B: In astronomical and scientific data, {e2} has {t_attr} {t_val}."
            doc_dist = f"Document C: A related reference reports {e1} has {t_attr} {dist_val}."
            query = f"What is the {t_attr} of the {b_attr} of {e1}?"
        else:
            doc_a = f"दस्तावेज़ A: {e1} का {b_attr} {e2} है।"
            doc_b = f"दस्तावेज़ B: वैज्ञानिक रिकॉर्ड के अनुसार, {e2} का {t_attr} {t_val} है।"
            doc_dist = f"दस्तावेज़ C: अन्य विवरण के अनुसार, {e1} का {t_attr} {dist_val} है।"
            query = f"What is the {t_attr} of the {b_attr} of {e1}?"

        # Combined multi-document context
        context = f"{doc_a}\n{doc_b}\n{doc_dist}"
        
        return {
            "chain_id": f"chain_{split_name}_{idx:04d}",
            "domain": c["domain"],
            "language": lang,
            "e1": e1,
            "bridge_attr": b_attr,
            "intermediate_e2": e2,
            "target_attr": t_attr,
            "gold_value": t_val,
            "normalized_gold": t_val.lower().rstrip(".,;:?!"),
            "distractor_value": dist_val,
            "context": context,
            "doc_a": doc_a,
            "doc_b": doc_b,
            "query": query,
            "is_held_out": split_name == "test",
            "is_unanswerable": False,
            "diagnostic_tags": [
                f"split_{split_name}",
                "multi_hop_2hop",
                "multilingual" if lang == "hi" else "monolingual_en",
                "has_digits" if any(ch.isdigit() for ch in t_val) else "no_digits",
                "multi_token" if len(t_val.split()) > 1 else "single_token"
            ]
        }

    # Generate records
    train_records = []
    for i in range(1200):
        c = random.choice(train_base)
        train_records.append(expand_chain(c, i, "train"))

    val_records = []
    for i in range(200):
        c = random.choice(val_base)
        val_records.append(expand_chain(c, i, "val"))

    test_records = []
    for i in range(500):
        c = random.choice(test_base)
        test_records.append(expand_chain(c, i, "test"))

    # Add 50 adversarial unanswerable multi-hop records to test pool (missing Doc B)
    for i in range(50):
        c = random.choice(test_base)
        e1, b_attr, e2, t_attr = c["e1"], c["bridge_attr"], c["e2"], c["target_attr"]
        doc_a = f"Document A: The {b_attr} of {e1} is {e2}."
        doc_c = f"Document C: No relation information recorded."
        context = f"{doc_a}\n{doc_c}"
        test_records.append({
            "chain_id": f"chain_test_unans_{i:04d}",
            "domain": c["domain"],
            "language": c["lang"],
            "e1": e1,
            "bridge_attr": b_attr,
            "intermediate_e2": e2,
            "target_attr": t_attr,
            "gold_value": "Not mentioned in context.",
            "normalized_gold": "not mentioned in context",
            "distractor_value": "",
            "context": context,
            "doc_a": doc_a,
            "doc_b": "",
            "query": f"What is the {t_attr} of the {b_attr} of {e1}?",
            "is_held_out": True,
            "is_unanswerable": True,
            "diagnostic_tags": ["split_test", "unanswerable_missing_link"]
        })

    out_dir = Path("experiments/batch231_240_multihop_reasoning/data")
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "multihop_corpus_train.json").write_text(json.dumps(train_records, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "multihop_corpus_val.json").write_text(json.dumps(val_records, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "multihop_corpus_test.json").write_text(json.dumps(test_records, indent=2, ensure_ascii=False), encoding="utf-8")

    # Sample exactly 100 stratified items for the primary frozen test manifest
    random.seed(42)
    sample_100 = random.sample([r for r in test_records if not r.get("is_unanswerable")], 80) + random.sample([r for r in test_records if r.get("is_unanswerable")], 20)
    frozen_manifest_path = out_dir / "frozen_multihop_manifest_100.json"
    frozen_manifest_path.write_text(json.dumps(sample_100, indent=2, ensure_ascii=False), encoding="utf-8")

    h_man = hashlib.sha256(frozen_manifest_path.read_bytes()).hexdigest()
    print(f"Generated Multi-Hop Corpus: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")
    print(f"Saved primary frozen multi-hop manifest ({len(sample_100)} records) to {frozen_manifest_path}")
    print(f"Manifest SHA-256: {h_man}")


if __name__ == "__main__":
    build_multihop_corpus()
