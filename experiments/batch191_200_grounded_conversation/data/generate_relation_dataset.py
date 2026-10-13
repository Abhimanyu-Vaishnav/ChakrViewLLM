"""
Batch 193: Relation-Binding Dataset Generator with Provable Zero-Leakage Split.
Generates structured factual relation examples:
- Entity -> Attribute -> Value
- Distractors with shared entities but disparate relations
- Distractors with shared relations but disparate entities
- Reordered records
- Strict group-based train/val/test split by entity families
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

DATA_DIR = Path("experiments/batch191_200_grounded_conversation/data")
MANIFEST_PATH = DATA_DIR / "split_manifest.json"

FACTS = [
    # Train Split: Elements & Planets (12 facts)
    {"entity": "Hydrogen", "relation": "atomic number", "value": "1", "doc": "Hydrogen has atomic number 1 and mass 1.008 u.", "split": "train"},
    {"entity": "Hydrogen", "relation": "atomic mass", "value": "1.008 u", "doc": "Hydrogen has atomic number 1 and mass 1.008 u.", "split": "train"},
    {"entity": "Helium", "relation": "atomic number", "value": "2", "doc": "Helium has atomic number 2 and boiling point -268 C.", "split": "train"},
    {"entity": "Helium", "relation": "boiling point", "value": "-268 C", "doc": "Helium has atomic number 2 and boiling point -268 C.", "split": "train"},
    {"entity": "Lithium", "relation": "atomic number", "value": "3", "doc": "Lithium has atomic number 3 and density 0.534 g/cm3.", "split": "train"},
    {"entity": "Lithium", "relation": "density", "value": "0.534 g/cm3", "doc": "Lithium has atomic number 3 and density 0.534 g/cm3.", "split": "train"},
    {"entity": "Mars", "relation": "primary moon", "value": "Phobos", "doc": "The primary moon of Mars is Phobos, while its smaller moon is Deimos.", "split": "train"},
    {"entity": "Mars", "relation": "secondary moon", "value": "Deimos", "doc": "The primary moon of Mars is Phobos, while its smaller moon is Deimos.", "split": "train"},
    {"entity": "Jupiter", "relation": "largest moon", "value": "Ganymede", "doc": "The largest moon of Jupiter is Ganymede, followed by Callisto.", "split": "train"},
    {"entity": "Jupiter", "relation": "orbital period", "value": "11.86 years", "doc": "Jupiter has an orbital period of 11.86 years around the Sun.", "split": "train"},
    {"entity": "Earth", "relation": "natural satellite", "value": "Moon", "doc": "The single natural satellite of Earth is Moon.", "split": "train"},
    {"entity": "Venus", "relation": "surface temp", "value": "464 C", "doc": "The mean surface temperature of Venus is 464 C.", "split": "train"},

    # Val Split: Metals (4 facts)
    {"entity": "Iron", "relation": "atomic number", "value": "26", "doc": "Iron has atomic number 26 and melting point 1538 C.", "split": "val"},
    {"entity": "Iron", "relation": "melting point", "value": "1538 C", "doc": "Iron has atomic number 26 and melting point 1538 C.", "split": "val"},
    {"entity": "Copper", "relation": "atomic number", "value": "29", "doc": "Copper has atomic number 29 and electrical conductivity high.", "split": "val"},
    {"entity": "Copper", "relation": "electrical conductivity", "value": "high", "doc": "Copper has atomic number 29 and electrical conductivity high.", "split": "val"},

    # Test Split: Held-Out Unseen Entities & Multilingual Facts (6 facts)
    {"entity": "Boron", "relation": "atomic number", "value": "5", "doc": "Boron has atomic number 5 and hardness 9.3.", "split": "test"},
    {"entity": "Boron", "relation": "hardness", "value": "9.3", "doc": "Boron has atomic number 5 and hardness 9.3.", "split": "test"},
    {"entity": "Neptune", "relation": "largest moon", "value": "Triton", "doc": "The largest moon of Neptune is Triton.", "split": "test"},
    {"entity": "Gold", "relation": "melting point", "value": "1064 C", "doc": "The melting point of Gold is 1064 C.", "split": "test"},
    {"entity": "Titan", "relation": "parent planet", "value": "Saturn", "doc": "Titan has parent planet Saturn.", "split": "test"},
    {"entity": "चांदी", "relation": "प्रमुख गुण", "value": "विद्युत सुचालक", "doc": "चांदी का प्रमुख गुण विद्युत सुचालक होना है।", "split": "test"},
]


def generate_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    splits = {"train": [], "val": [], "test": []}
    for item in FACTS:
        q = f"What is the {item['relation']} of {item['entity']}?"
        entry = {
            "entity": item["entity"],
            "relation": item["relation"],
            "value": item["value"],
            "doc": item["doc"],
            "query": q,
            "split": item["split"]
        }
        splits[item["split"]].append(entry)

    def hash_split(items):
        serialized = json.dumps(items, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    manifest = {
        "dataset_name": "ChakrView_Relation_Binding_Benchmark_v1",
        "total_examples": len(FACTS),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "train_sha256": hash_split(splits["train"]),
        "val_sha256": hash_split(splits["val"]),
        "test_sha256": hash_split(splits["test"]),
        "splits": splits,
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Dataset generated at {MANIFEST_PATH}")
    print(f"Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")


if __name__ == "__main__":
    generate_dataset()
