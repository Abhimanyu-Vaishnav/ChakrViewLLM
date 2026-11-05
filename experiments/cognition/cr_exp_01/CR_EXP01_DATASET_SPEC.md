# CR-EXP-01: REVISED SYNTHETIC DATASET SPECIFICATION

This specification defines the exact generation rules, entity pools, and structural templates for the CR-EXP-01 variable binding dataset.

---

## 1. Disjoint Lexicon Pools (Zero-Leakage Guarantee)

All keys and values are drawn from mutually exclusive vocabulary pools partitioned by cryptographic SHA-256 splits:

### A. Train Pool (Seed 101: 100 Keys, 100 Values)
- **Keys**: `Alpha`, `Beta`, `Gamma`, `Delta`, `Epsilon`, `Zeta`, `Eta`, `Theta`, `Port`, `Host`, `Node`, `Cluster`, `City`, `Server`, `Engine`, `Worker`, `Driver`, `Client`, `Database`, `Service`... (100 total).
- **Values**: `101`, `202`, `303`, `404`, `505`, `Red`, `Blue`, `Green`, `Amber`, `Falcon`, `Apollo`, `Garuda`, `Agni`, `Vayu`, `Delhi`, `Tokyo`, `Bengaluru`, `Mumbai`, `Paris`, `London`... (100 total).

### B. Validation Pool (Seed 202: 25 Keys, 25 Values)
- **Keys**: `Sensor`, `Buffer`, `Queue`, `Signal`, `Channel`, `Vector`, `Matrix`, `Tensor`, `Module`, `Router`... (25 total).
- **Values**: `606`, `707`, `Neon`, `Cyan`, `Magenta`, `Cheetah`, `Leopard`, `Jaguar`, `Pune`, `Jaipur`... (25 total).

### C. Held-Out Test Pool (Seed 303: 50 Keys, 50 Values)
- **Keys**: `Quasar`, `Pulsar`, `Zenith`, `Vertex`, `Nexus`, `Prism`, `Catalyst`, `Cipher`, `Vortex`, `Apex`... (50 total).
- **Values**: `8080`, `9090`, `Cobalt`, `Obsidian`, `Platinum`, `Titanium`, `Phoenix`, `Hydra`, `Chennai`, `Kolkata`... (50 total).

$$\text{Train Pool} \cap \text{Val Pool} \cap \text{Test Pool} = \emptyset$$

---

## 2. Diverse Template Formulations

To prevent the model from learning positional offsets from colons or specific phrasing, 10 distinct template families are interleaved:

1. `Mapping: Key is mapped to Value. Query: What does Key map to?\nAnswer: Value`
2. `System config: Key = Value. Query: Value of Key?\nAnswer: Value`
3. `Record: Value was assigned to Key. Query: Which value belongs to Key?\nAnswer: Value`
4. `Attribute listing: Key has attribute Value. Query: What is Key's attribute?\nAnswer: Value`
5. `Environment state: Key : Value. Query: Key:\nAnswer: Value`
6. `User: My Key is Value.\nAssistant: Noted. Your Key is Value.\nUser: What is my Key?\nAssistant: Value`
7. `Parameter Key was set to Value. Query: What is the parameter Key?\nAnswer: Value`
8. `Association: Key -> Value. Lookup(Key) ->\nAnswer: Value`
9. `Initial record: Key is Value. Query: What value is held by Key?\nAnswer: Value`
10. `Device profile: Key registered with identifier Value. Query: Identify Key.\nAnswer: Value`

---

## 3. Split Sizes & File Layout

- `data/sft/cr_exp01/train_1000.jsonl`: 1,000 records (Train Pool, Templates 1–8).
- `data/sft/cr_exp01/val_100.jsonl`: 100 records (Val Pool, Templates 1–8).
- `data/sft/cr_exp01/test_familiar_100.jsonl`: 100 records (Condition A: Train Pool).
- `data/sft/cr_exp01/test_novel_100.jsonl`: 100 records (Condition B: Held-Out Test Pool, Templates 1–8).
- `data/sft/cr_exp01/test_unseen_template_100.jsonl`: 100 records (Condition I: Test Pool, Templates 9–10).
- `data/sft/cr_exp01/test_distractor_sweep_100.jsonl`: 100 records ($k=1$ to $4$ distractors).
- `data/sft/cr_exp01/test_multiturn_100.jsonl`: 100 records (Condition G: Multi-turn format).

Total evaluation probes: **500 held-out probes** covering all 10 experimental conditions.
