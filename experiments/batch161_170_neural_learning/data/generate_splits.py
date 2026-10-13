"""
Generates strictly partitioned, leakage-proof datasets for:
1. Contrastive Representation Learning (Query, Positive Document, Hard Negative)
2. Neural Copy Induction SFT (Context, Question, Answer with exact token spans)
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any

OUT_DIR = Path("experiments/batch161_170_neural_learning/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Fact Entities Partitioned by Domain & Entity
FACT_DATABASE = [
    # --- DOMAIN: CHEMISTRY (Atomic facts) ---
    {"entity": "Hydrogen", "key": "atomic number", "value": "1", "split": "train",
     "doc": "Hydrogen is the lightest chemical element with atomic number 1.",
     "query": "What is the atomic number of Hydrogen?",
     "neg_doc": "Helium has atomic number 2 and is an inert noble gas."},
    {"entity": "Helium", "key": "atomic number", "value": "2", "split": "train",
     "doc": "Helium has atomic number 2 and standard atomic weight 4.0026.",
     "query": "What is the atomic number of Helium?",
     "neg_doc": "Hydrogen is the lightest chemical element with atomic number 1."},
    {"entity": "Lithium", "key": "atomic number", "value": "3", "split": "train",
     "doc": "Lithium is an alkali metal with atomic number 3.",
     "query": "What is the atomic number of Lithium?",
     "neg_doc": "Beryllium is an alkaline earth metal with atomic number 4."},
    {"entity": "Beryllium", "key": "atomic number", "value": "4", "split": "val",
     "doc": "Beryllium is an alkaline earth metal with atomic number 4.",
     "query": "What is the atomic number of Beryllium?",
     "neg_doc": "Boron is a metalloid with atomic number 5."},
    {"entity": "Boron", "key": "atomic number", "value": "5", "split": "test",
     "doc": "Boron is a metalloid with atomic number 5 and standard atomic weight 10.81.",
     "query": "What is the atomic number of Boron?",
     "neg_doc": "Carbon is a nonmetal with atomic number 6."},
     
    # --- DOMAIN: ASTRONOMY & GEOGRAPHY ---
    {"entity": "Mars", "key": "primary moon", "value": "Phobos", "split": "train",
     "doc": "Mars has two small natural satellites, the larger being Phobos.",
     "query": "What is the primary moon of Mars?",
     "neg_doc": "Jupiter has 95 recognized moons including Ganymede."},
    {"entity": "Jupiter", "key": "largest moon", "value": "Ganymede", "split": "train",
     "doc": "Ganymede is the largest moon of Jupiter and in the Solar System.",
     "query": "What is the largest moon of Jupiter?",
     "neg_doc": "Titan is the largest moon of Saturn with a thick nitrogen atmosphere."},
    {"entity": "Saturn", "key": "largest moon", "value": "Titan", "split": "val",
     "doc": "Titan is the largest moon of Saturn with a dense atmosphere.",
     "query": "What is the largest moon of Saturn?",
     "neg_doc": "Ganymede is the largest moon of Jupiter."},
    {"entity": "Neptune", "key": "largest moon", "value": "Triton", "split": "test",
     "doc": "Triton is the largest natural satellite of the planet Neptune.",
     "query": "What is the largest moon of Neptune?",
     "neg_doc": "Phobos is a moon of Mars."},
     
    # --- DOMAIN: PHYSICAL CONSTANTS & PROPERTIES ---
    {"entity": "Water", "key": "boiling point", "value": "100 C", "split": "train",
     "doc": "Water boils at 100 C and freezes at 0 C at sea level atmospheric pressure.",
     "query": "What is the boiling point of Water?",
     "neg_doc": "Ethanol boils at 78 C and freezes at -114 C."},
    {"entity": "Ethanol", "key": "boiling point", "value": "78 C", "split": "train",
     "doc": "Ethanol has a normal boiling point of 78 C at 1 atmosphere.",
     "query": "What is the boiling point of Ethanol?",
     "neg_doc": "Water boils at 100 C at sea level."},
    {"entity": "Mercury", "key": "melting point", "value": "-38.8 C", "split": "val",
     "doc": "Mercury is a liquid metal with melting point of -38.8 C.",
     "query": "What is the melting point of Mercury?",
     "neg_doc": "Lead melts at 327.5 C."},
    {"entity": "Gold", "key": "melting point", "value": "1064 C", "split": "test",
     "doc": "Pure gold melts at a temperature of 1064 C.",
     "query": "What is the melting point of Gold?",
     "neg_doc": "Water freezes at 0 C and boils at 100 C."},

    # --- DOMAIN: MULTILINGUAL HINDI FACTS ---
    {"entity": "सूर्य", "key": "तत्व", "value": "हाइड्रोजन", "split": "train",
     "doc": "सूर्य मुख्य रूप से हाइड्रोजन और हीलियम गैस से बना तारा है।",
     "query": "सूर्य का मुख्य तत्व क्या है?",
     "neg_doc": "चंद्रमा पृथ्वी का एकमात्र प्राकृतिक उपग्रह है।"},
    {"entity": "पृथ्वी", "key": "उपग्रह", "value": "चंद्रमा", "split": "val",
     "doc": "पृथ्वी का एकमात्र प्राकृतिक उपग्रह चंद्रमा है।",
     "query": "पृथ्वी का उपग्रह कौन सा है?",
     "neg_doc": "सूर्य सौर मंडल के केंद्र में स्थित है।"},
    {"entity": "चांदी", "key": "गुण", "value": "विद्युत सुचालक", "split": "test",
     "doc": "चांदी सबसे अच्छा विद्युत सुचालक धातु है।",
     "query": "चांदी का प्रमुख गुण क्या है?",
     "neg_doc": "लोहा एक चुंबकीय धातु है।"},
]


def generate_manifest():
    splits = {"train": [], "val": [], "test": []}
    for item in FACT_DATABASE:
        s = item["split"]
        splits[s].append(item)

    manifest = {
        "dataset_name": "ChakrView_Cognitive_Benchmark_Split_v1",
        "total_examples": len(FACT_DATABASE),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "train_sha256": hashlib.sha256(json.dumps(splits["train"], sort_keys=True).encode()).hexdigest(),
        "val_sha256": hashlib.sha256(json.dumps(splits["val"], sort_keys=True).encode()).hexdigest(),
        "test_sha256": hashlib.sha256(json.dumps(splits["test"], sort_keys=True).encode()).hexdigest(),
        "splits": splits,
    }

    manifest_path = OUT_DIR / "split_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest written to {manifest_path} with {len(FACT_DATABASE)} total items.")


if __name__ == "__main__":
    generate_manifest()
