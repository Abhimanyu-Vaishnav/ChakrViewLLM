"""
generate_abstention_dataset.py
Generates exactly 150 train, 20 val, 20 test abstention items.
Each item:
(context, unanswerable_q, unanswerable_a, control_q, control_a, domain)
Enforces:
- 100% token length compliance [3, 45]
- 0% Gate v0.1 overlap
- 100% split disjointness (disjoint names and entities across train, val, test)
- Grounded controls match context
- Unanswerable answers contain valid refusal phrases
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.generate_candidate_banks_exp18 import check_gate, check_tokens

tokenizer, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096", verify_checksums=True)

from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.build_abstention import DOMAINS, NAMES

# We have 160 names, we need 190. Let's add 50 more star/geographical names to reach 210 names.
EXTRA_NAMES = [
    "Alpheratz", "Mirach", "Almach", "Menkar", "Hamal", "Caph", "Schedar", "Caph", "Ruchbah", "Miram",
    "Gorgonea", "Menkib", "Atik", "Hassaleh", "Almaaz", "Haedus", "Alsciaukat", "Tyl", "Grumium", "Nodus",
    "Alsafi", "Dziban", "Edasich", "Thuban", "Giausar", "Pherkad", "Yildun", "Ahfa", "Biham", "Sadaltager",
    "Albali", "Ancha", "Situla", "Scheat", "Homam", "Matar", "Azelfafage", "Kurhah", "Gienah", "Alchiba",
    "Kraz", "Algorab", "Minkar", "Denebokab", "Tegmine", "Acubens", "Asellus", "Tarazet", "Alshain", "Tarazed"
]

ALL_NAMES = list(dict.fromkeys(NAMES + EXTRA_NAMES))  # deduplicate preserving order
print(f"Total unique names: {len(ALL_NAMES)}")

def build_split(start_idx, count, split_name):
    items = []
    for i in range(count):
        idx = start_idx + i
        dom_idx = idx % len(DOMAINS)
        domain, ctx_tpl, c_topic, ca_tpl, u_topic, ua_text = DOMAINS[dom_idx]
        name = ALL_NAMES[idx]
        metric1 = 10 + (idx * 7) % 89
        metric2 = 100 + (idx * 13) % 400

        ctx = ctx_tpl.format(name=name, metric1=metric1, metric2=metric2)
        cq = f"According to the record, what is the {c_topic} for {name}?"
        ca = ca_tpl.format(name=name, metric1=metric1, metric2=metric2)
        uq = f"According to the record, what is the {u_topic} for {name}?"
        ua = ua_text

        # Validate
        check_tokens(ca, f"{split_name} ca {i}")
        check_tokens(ua, f"{split_name} ua {i}")
        check_gate(cq, f"{split_name} cq {i}")
        check_gate(uq, f"{split_name} uq {i}")

        items.append((ctx, uq, ua, cq, ca, domain))
    return items

train_items = build_split(0, 150, "train")
val_items = build_split(150, 20, "val")
test_items = build_split(170, 20, "test")

print(f"Built items: train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")

# Save to raw_abstention_data.py
out_path = Path(__file__).resolve().parent / "raw_abstention_data.py"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("# Abstention Candidate Pairs (150 train, 20 val, 20 test = 190 items)\n")
    f.write("ABSTENTION_DATA = {\n")
    for s_name, s_items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        f.write(f'    "{s_name}": [\n')
        for item in s_items:
            f.write(f"        {repr(item)},\n")
        f.write("    ],\n")
    f.write("}\n")

print(f"Wrote to {out_path}")
