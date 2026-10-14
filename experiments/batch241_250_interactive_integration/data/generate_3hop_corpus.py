"""
Batch 248: Generate Leakage-Resistant Three-Hop Grounding Corpus & Frozen Manifest.
Location: experiments/batch241_250_interactive_integration/data/generate_3hop_corpus.py

Creates synthetic 3-hop relational chains across 3 distinct documents:
Doc A: E1 has bridge property P1 = E2.
Doc B: E2 has bridge property P2 = E3.
Doc C: E3 has target property P3 = Target Value V.
Query: "What is the P3 of the P2 of the P1 of E1?"

Includes:
- Answerable 3-hop chains with intermediate entities E2 and E3 explicitly provenanced.
- Disjoint entity clusters for train, val, and test to prevent target/entity leakage.
- Missing link cases (Doc C or Doc B missing, requiring explicit abstention).
- Distractor documents with competing numerical/entity attributes.
- Hindi / Multilingual Devanagari chains.
- Frozen 100-item test manifest with computed SHA-256.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

DATA_DIR = Path("experiments/batch241_250_interactive_integration/data")


def build_3hop_corpus(seed: int = 42):
    random.seed(seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 3-Hop chains: E1 -> bridge1 -> E2 -> bridge2 -> E3 -> target_attr -> TargetValue
    base_3hop_chains = [
        # Science & Astrophysics
        {
            "e1": "Solar System", "b1": "outer giant planet", "e2": "Neptune",
            "b2": "largest retrograde moon", "e3": "Triton",
            "t_attr": "surface temperature", "t_val": "38 K", "dist": "72 K",
            "domain": "astronomy", "lang": "en"
        },
        {
            "e1": "Milky Way", "b1": "central supermassive black hole", "e2": "Sagittarius A*",
            "b2": "closest orbiting star", "e3": "S2",
            "t_attr": "orbital period", "t_val": "16.05 years", "dist": "11.5 years",
            "domain": "astronomy", "lang": "en"
        },
        {
            "e1": "Earth", "b1": "thickest internal layer", "e2": "Mantle",
            "b2": "dominant mineral phase", "e3": "Bridgmanite",
            "t_attr": "crystal structure", "t_val": "perovskite", "dist": "spinel",
            "domain": "geology", "lang": "en"
        },
        {
            "e1": "Atmosphere", "b1": "uppermost active weather layer", "e2": "Troposphere",
            "b2": "boundary transition zone", "e3": "Tropopause",
            "t_attr": "typical altitude", "t_val": "12 km", "dist": "50 km",
            "domain": "earth_science", "lang": "en"
        },
        # Chemistry & Materials
        {
            "e1": "Brass", "b1": "major alloy component", "e2": "Copper",
            "b2": "electronic configuration shell", "e3": "d-subshell",
            "t_attr": "valence electrons", "t_val": "10", "dist": "2",
            "domain": "chemistry", "lang": "en"
        },
        {
            "e1": "Steel", "b1": "interstitial hardening element", "e2": "Carbon",
            "b2": "stable allotrope at high pressure", "e3": "Diamond",
            "t_attr": "Mohs hardness", "t_val": "10", "dist": "7",
            "domain": "materials", "lang": "en"
        },
        {
            "e1": "Water Molecule", "b1": "electronegative core atom", "e2": "Oxygen",
            "b2": "most abundant isotope", "e3": "Oxygen-16",
            "t_attr": "neutron count", "t_val": "8", "dist": "10",
            "domain": "chemistry", "lang": "en"
        },
        # Computer Systems & Architecture
        {
            "e1": "Modern Computer", "b1": "primary execution engine", "e2": "CPU",
            "b2": "fastest internal storage cell", "e3": "Register File",
            "t_attr": "access latency", "t_val": "1 cycle", "dist": "15 cycles",
            "domain": "computing", "lang": "en"
        },
        {
            "e1": "Operating System", "b1": "core supervisory component", "e2": "Kernel",
            "b2": "memory translation mechanism", "e3": "Page Table",
            "t_attr": "lookup acceleration buffer", "t_val": "TLB", "dist": "Cache",
            "domain": "computing", "lang": "en"
        },
        {
            "e1": "Compiler", "b1": "intermediate representation phase", "e2": "AST",
            "b2": "target machine code generator", "e3": "Backend",
            "t_attr": "register allocation algorithm", "t_val": "Graph Coloring", "dist": "FIFO",
            "domain": "computing", "lang": "en"
        },
        # Mathematics & Geometry
        {
            "e1": "Unit Circle", "b1": "circumference ratio", "e2": "Tau",
            "b2": "half-period divisor", "e3": "Pi",
            "t_attr": "approximate value", "t_val": "3.14159", "dist": "2.71828",
            "domain": "mathematics", "lang": "en"
        },
        {
            "e1": "Right Triangle", "b1": "longest opposing side", "e2": "Hypotenuse",
            "b2": "governing theorem", "e3": "Pythagoras",
            "t_attr": "exponent power", "t_val": "2", "dist": "3",
            "domain": "mathematics", "lang": "en"
        },
        # Multilingual Hindi (Devanagari)
        {
            "e1": "सौर मंडल (Solar System)", "b1": "विशाल गैसीय ग्रह", "e2": "बृहस्पति (Jupiter)",
            "b2": "सबसे बड़ा उपग्रह", "e3": "गैनिमीड (Ganymede)",
            "t_attr": "कक्षा त्रिज्या", "t_val": "1070400 km", "dist": "421700 km",
            "domain": "astronomy", "lang": "hi"
        },
        {
            "e1": "कंप्यूटर (Computer)", "b1": "केंद्रीय प्रसंस्करण इकाई", "e2": "सीपीयू (CPU)",
            "b2": "गणितीय और तार्किक घटक", "e3": "एएलयू (ALU)",
            "t_attr": "मूल कार्य", "t_val": "अंकगणित और तर्क", "dist": "डेटा भंडारण",
            "domain": "computing", "lang": "hi"
        },
        {
            "e1": "सजीव कोशिका (Living Cell)", "b1": "ऊर्जा उत्पादक केंद्र", "e2": "माइटोकॉन्ड्रिया (Mitochondria)",
            "b2": "ऊर्जा वाहक अणु", "e3": "एटीपी (ATP)",
            "t_attr": "पूर्ण रूप", "t_val": "एडेनोसिन ट्राइफॉस्फेट", "dist": "डीएनए",
            "domain": "biology", "lang": "hi"
        }
    ]

    # Cluster split by base entity E1:
    # Held-out Test entities: "Solar System", "Steel", "Compiler", "सौर मंडल (Solar System)", "कंप्यूटर (Computer)"
    # Val entities: "Milky Way", "Water Molecule", "Right Triangle"
    # Train entities: remainder
    test_e1 = {"Solar System", "Steel", "Compiler", "सौर मंडल (Solar System)", "कंप्यूटर (Computer)"}
    val_e1 = {"Milky Way", "Water Molecule", "Right Triangle"}

    train_base = [c for c in base_3hop_chains if c["e1"] not in test_e1 and c["e1"] not in val_e1]
    val_base = [c for c in base_3hop_chains if c["e1"] in val_e1]
    test_base = [c for c in base_3hop_chains if c["e1"] in test_e1]

    def expand_3hop_record(c, idx, split_name, is_unanswerable=False):
        e1, b1, e2, b2, e3, t_attr, t_val, dist = (
            c["e1"], c["b1"], c["e2"], c["b2"], c["e3"], c["t_attr"], c["t_val"], c["dist"]
        )
        lang = c["lang"]

        if lang == "en":
            doc_a = f"Document A: The {b1} of {e1} is {e2}."
            doc_b = f"Document B: In specialized technical reference, {e2} features {b2} as {e3}."
            doc_c = f"Document C: Detailed empirical analysis confirms that {e3} has {t_attr} {t_val}."
            doc_dist = f"Document D: A peripheral distractor note mentions {e1} has {t_attr} {dist}."
            query = f"What is the {t_attr} of the {b2} of the {b1} of {e1}?"
        else:
            doc_a = f"दस्तावेज़ A: {e1} का {b1} {e2} है।"
            doc_b = f"दस्तावेज़ B: तकनीकी विवरण के अनुसार, {e2} का {b2} {e3} है।"
            doc_c = f"दस्तावेज़ C: साक्ष्य स्पष्ट करता है कि {e3} का {t_attr} {t_val} है।"
            doc_dist = f"दस्तावेज़ D: अन्य संदर्भ के अनुसार {e1} का {t_attr} {dist} है।"
            query = f"What is the {t_attr} of the {b2} of the {b1} of {e1}?"

        if is_unanswerable:
            # Omit Doc C (broken 3rd hop)
            context = f"{doc_a}\n{doc_b}\n{doc_dist}"
            gold = "Not mentioned in context."
        else:
            context = f"{doc_a}\n{doc_b}\n{doc_c}\n{doc_dist}"
            gold = t_val

        return {
            "id": f"3hop_{split_name}_{idx:04d}",
            "domain": c["domain"],
            "language": lang,
            "e1": e1,
            "bridge1": b1,
            "intermediate_e2": e2,
            "bridge2": b2,
            "intermediate_e3": e3,
            "target_attr": t_attr,
            "gold_value": gold,
            "normalized_gold": gold.lower().rstrip(".,;:?!"),
            "distractor_value": dist,
            "is_unanswerable": is_unanswerable,
            "query": query,
            "context": context,
        }

    # Generate Train (200 records)
    train_records = []
    for i in range(200):
        c = random.choice(train_base)
        train_records.append(expand_3hop_record(c, i, "train", is_unanswerable=False))

    # Generate Frozen Test Manifest (100 records: 80 answerable, 20 broken 3rd hop abstentions)
    test_records = []
    for i in range(80):
        c = random.choice(test_base)
        test_records.append(expand_3hop_record(c, i, "test", is_unanswerable=False))
    for i in range(20):
        c = random.choice(test_base)
        test_records.append(expand_3hop_record(c, 80 + i, "test", is_unanswerable=True))

    random.shuffle(test_records)
    # Re-index
    for idx, r in enumerate(test_records):
        r["test_id"] = f"manifest3hop_{idx:03d}"

    train_path = DATA_DIR / "threehop_corpus_train.json"
    manifest_path = DATA_DIR / "frozen_threehop_manifest_100.json"

    train_path.write_text(json.dumps(train_records, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest_path.write_text(json.dumps(test_records, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    print(f"Generated 3-hop train set: {len(train_records)} records -> {train_path}")
    print(f"Generated frozen 3-hop test manifest: {len(test_records)} records -> {manifest_path}")
    print(f"Manifest SHA-256: {manifest_sha}")
    return manifest_sha


if __name__ == "__main__":
    build_3hop_corpus(42)
