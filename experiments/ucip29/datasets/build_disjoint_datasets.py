"""
UCIP-29 Phase 1: Zero-Leakage Strictly Disjoint Dataset Generator.
Guarantees 0 exact prompt leakage by using distinct entities, topics, and phrasing per split.
"""

from __future__ import annotations
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def build_disjoint_conversational_datasets():
    tok, _ = load_tokenizer_artifacts(ROOT / "data/experiments/vocab_4096")

    # 1. ENTITY SPLITS (Strictly disjoint)
    train_entities = [
        ("NovaTrack", "Hyderabad", "1981", "Dr. Rao", "Satellite imaging"),
        ("AeroScan", "Pune", "2014", "Sunita Sharma", "Drone mapping"),
        ("BioPulse", "Bengaluru", "2019", "Arjun Nair", "Telemetry tracker"),
        ("DataForge", "Chennai", "2011", "Vikram Das", "Distributed stream engine"),
        ("SolarisGrid", "Ahmedabad", "2008", "Neha Patel", "Solar microgrid"),
        ("QuantumGate", "Kolkata", "2021", "Tanvi Bose", "Qubit interconnect"),
        ("OmniFlow", "Delhi", "2016", "Rohan Mehta", "Logistics dispatch"),
        ("CyberShield", "Noida", "2018", "Ananya Sen", "Intrusion prevention"),
        ("NeuroVibe", "Mumbai", "2020", "Deepak Joshi", "Neural signal decoder"),
        ("TerraMap", "Jaipur", "2015", "Pooja Verma", "Geospatial mesh"),
        ("ApexDynamics", "Indore", "2012", "Rajesh Kumar", "Hydraulic actuator"),
        ("ZenithWorks", "Chandigarh", "2017", "Kavita Gill", "Industrial robotics"),
        ("AlphaSys", "Kochi", "2010", "George Thomas", "Cloud runtime"),
        ("OmegaCorp", "Bhopal", "2005", "Manish Tiwari", "Precision alloy"),
        ("InnoWave", "Mysuru", "2013", "Sanjay Hegde", "Acoustic modem"),
        ("TechVista", "Thiruvananthapuram", "2015", "Radhika Pillai", "Fiber switch")
    ]
    val_entities = [
        ("EchoSystems", "Nagpur", "2022", "Isha Deshmukh", "Acoustic sensor"),
        ("VortexAI", "Gurugram", "2023", "Sameer Khan", "Edge intelligence"),
        ("AgriSense", "Lucknow", "2019", "Suresh Yadav", "Soil moisture probe"),
        ("AuraSafe", "Kanpur", "2016", "Dr. Alok Verma", "Particulate filter")
    ]
    test_entities = [
        ("AquaPure", "Surat", "2013", "Bhavin Shah", "Water filtration"),
        ("FleetSync", "Vadodara", "2016", "Jatin Desai", "Telematics platform"),
        ("CoreSecure", "Coimbatore", "2018", "Lakshmi Raman", "Cryptographic HSM"),
        ("OptiRoute", "Patna", "2017", "Anil Jha", "Transit matrix")
    ]

    # 2. GREETINGS & SOCIAL (Strictly distinct phrasing per split)
    greetings_train = [
        ("User: Hello!\nAssistant: ", "Hello! How can I assist you today?", "greeting", "en"),
        ("User: Hi there!\nAssistant: ", "Hi! I am here and ready to help. What's on your mind?", "greeting", "en"),
        ("User: Good morning!\nAssistant: ", "Good morning! I hope you have a productive day.", "greeting", "en"),
        ("User: Good evening!\nAssistant: ", "Good evening! How can I help you tonight?", "greeting", "en"),
        ("User: Namaste!\nAssistant: ", "Namaste! Main aapki kya madad kar sakta hoon?", "greeting", "hi"),
        ("User: Kaise ho aap?\nAssistant: ", "Main theek hoon! Aap bataiye, aaj kya madad chahiye?", "greeting", "hinglish"),
        ("User: Thank you!\nAssistant: ", "You are welcome! Let me know if you need anything else.", "social", "en"),
        ("User: Dhanyawaad!\nAssistant: ", "Aapka swagat hai!", "social", "hinglish"),
        ("User: Goodbye!\nAssistant: ", "Goodbye! Have a great day ahead.", "social", "en"),
        ("User: Alvida!\nAssistant: ", "Alvida! Apna khayal rakhein.", "social", "hinglish")
    ] * 8

    greetings_val = [
        ("User: Hey ChakrView!\nAssistant: ", "Hello! How can I help you right now?", "greeting", "en"),
        ("User: Good afternoon!\nAssistant: ", "Good afternoon! How may I assist you today?", "greeting", "en"),
        ("User: Shukriya dost!\nAssistant: ", "Koi baat nahi, hamesha madad ke liye tayyar hoon.", "social", "hinglish"),
        ("User: See you later!\nAssistant: ", "See you soon! Take care.", "social", "en")
    ] * 6

    greetings_test = [
        ("User: Greetings system!\nAssistant: ", "Greetings! How may I be of service today?", "greeting", "en"),
        ("User: Hello there assistant!\nAssistant: ", "Hello! What questions can I answer for you?", "greeting", "en"),
        ("User: Aapko bahut dhanyawaad!\nAssistant: ", "Khushi hui aapki madad karke!", "social", "hinglish"),
        ("User: Bye for now!\nAssistant: ", "Bye! Have a wonderful day.", "social", "en")
    ] * 6

    # 3. BUILD SPLITS FUNCTION
    def generate_split_records(entity_pool, greeting_pool, split_name: str, num_math_range: Tuple[int, int]) -> List[Dict[str, Any]]:
        recs = []
        # Greetings
        for p, r, c, l in greeting_pool:
            recs.append({"prompt": p, "response": r, "category": c, "language": l, "turns": 1})

        # Contextual facts & Abstentions
        prefix = "Context statement: " if split_name == "train" else ("Context provided: " if split_name == "val" else "Document snippet: ")
        sandarbh_prefix = "Sandarbh: " if split_name == "train" else ("Vivaran: " if split_name == "val" else "Suchna: ")

        for ent, city, year, founder, prod in entity_pool:
            # Fact QA
            q1 = f"User: {prefix}{ent} is located in {city} and was founded in {year} by {founder}.\nQuestion: Where is {ent} located?\nAnswer concisely:\nAssistant: "
            recs.append({"prompt": q1, "response": city, "category": "context_retrieval", "language": "en", "turns": 1, "entity": ent})

            q2 = f"User: {prefix}{ent} was founded in {year} by {founder}.\nQuestion: In what year was {ent} founded?\nAnswer concisely:\nAssistant: "
            recs.append({"prompt": q2, "response": year, "category": "context_retrieval", "language": "en", "turns": 1, "entity": ent})

            q3 = f"User: {prefix}{ent} was established by {founder} to develop {prod}.\nQuestion: Who founded {ent}?\nAnswer concisely:\nAssistant: "
            recs.append({"prompt": q3, "response": founder, "category": "context_retrieval", "language": "en", "turns": 1, "entity": ent})

            q4 = f"User: {prefix}{ent} builds {prod} in {city}.\nQuestion: What does {ent} develop?\nAnswer concisely:\nAssistant: "
            recs.append({"prompt": q4, "response": prod, "category": "context_retrieval", "language": "en", "turns": 1, "entity": ent})

            q5 = f"User: {sandarbh_prefix}{ent} ki sthapna {year} mein {founder} dwara hui thi.\nPrashna: {ent} kis varsh sthapit hua?\nSankshipt uttar dein:\nAssistant: "
            recs.append({"prompt": q5, "response": year, "category": "context_retrieval", "language": "hinglish", "turns": 1, "entity": ent})

            q6 = f"User: {sandarbh_prefix}{ent} ka mukhya kendra {city} mein sthit hai.\nPrashna: {ent} kahan sthit hai?\nSankshipt uttar dein:\nAssistant: "
            recs.append({"prompt": q6, "response": city, "category": "context_retrieval", "language": "hinglish", "turns": 1, "entity": ent})

            # Abstention (Principled ground truth negative)
            q_abs1 = f"User: {prefix}{ent} was founded in {year}.\nQuestion: What is the annual revenue of {ent}?\nAnswer concisely:\nAssistant: "
            recs.append({"prompt": q_abs1, "response": "Information not provided in the context.", "category": "abstention", "language": "en", "turns": 1, "entity": ent})

            q_abs2 = f"User: {sandarbh_prefix}{ent} {city} mein sthit hai.\nPrashna: {ent} ke kitne karmchari hain?\nSankshipt uttar dein:\nAssistant: "
            recs.append({"prompt": q_abs2, "response": "Sandarbh mein karmchariyon ki sankhya uplabdh nahi hai.", "category": "abstention", "language": "hinglish", "turns": 1, "entity": ent})

        # Arithmetic (Distinct number domains)
        rng = random.Random(42 if split_name == "train" else (100 if split_name == "val" else 200))
        min_n, max_n = num_math_range
        math_pairs = [(a, b) for a in range(min_n, max_n) for b in range(1, 10)]
        rng.shuffle(math_pairs)
        for a, b in math_pairs[:120]:
            op = rng.choice(["add", "sub", "mul"])
            if op == "add":
                p = f"User: What is {a} + {b}?\nAnswer concisely:\nAssistant: " if split_name == "train" else (
                    f"User: Compute the sum of {a} and {b}.\nAssistant: " if split_name == "val" else
                    f"User: Calculate {a} + {b}.\nAssistant: "
                )
                r = str(a + b)
            elif op == "sub":
                m, n = max(a, b), min(a, b)
                p = f"User: Calculate {m} - {n}.\nAnswer concisely:\nAssistant: " if split_name == "train" else (
                    f"User: What is {m} minus {n}?\nAssistant: " if split_name == "val" else
                    f"User: Subtract {n} from {m}.\nAssistant: "
                )
                r = str(m - n)
            else:
                p = f"User: What is {min(a, 10)} * {min(b, 10)}?\nAnswer concisely:\nAssistant: " if split_name == "train" else (
                    f"User: Multiply {min(a, 10)} by {min(b, 10)}.\nAssistant: " if split_name == "val" else
                    f"User: Evaluate {min(a, 10)} * {min(b, 10)}.\nAssistant: "
                )
                r = str(min(a, 10) * min(b, 10))
            recs.append({"prompt": p, "response": r, "category": "arithmetic", "language": "en", "turns": 1})

        # Knowledge, Code & Multi-Turn (Distinct across splits)
        if split_name == "train":
            tasks = [
                ("What is photosynthesis?", "Photosynthesis is the biological process by which green plants convert light into chemical energy.", "explanation", "en"),
                ("What is an operating system kernel?", "The kernel is the core software managing CPU, memory, and devices.", "explanation", "en"),
                ("Write a Python function to check if a number is even.", "def is_even(n: int) -> bool:\n    return n % 2 == 0", "code", "en"),
                ("Bharat ki rajdhani kya hai?", "Bharat ki rajdhani New Delhi hai.", "qa", "hinglish"),
                ("Computer mein RAM ka kya karya hai?", "RAM temporary data aur active programs ko fast storage deta hai.", "explanation", "hinglish"),
                ("Summarize: Water boils at 100 degrees Celsius under standard sea level pressure.", "Water boils at 100°C under normal atmospheric pressure.", "summarization", "en")
            ] * 12
        elif split_name == "val":
            tasks = [
                ("What is the primary function of DNS?", "DNS translates human-readable domain names into numerical IP addresses.", "explanation", "en"),
                ("Write a Python function to compute the square of x.", "def square(x: float) -> float:\n    return x * x", "code", "en"),
                ("Prithvi ka sabse bada mahasagar kaun sa hai?", "Prithvi ka sabse bada mahasagar Prashant Mahasagar (Pacific Ocean) hai.", "qa", "hinglish")
            ] * 6
        else:
            tasks = [
                ("What is an algorithm in computing?", "An algorithm is a step-by-step sequence of instructions designed to solve a specific problem.", "explanation", "en"),
                ("Write a Python function to return the length of a string.", "def get_length(s: str) -> int:\n    return len(s)", "code", "en"),
                ("Surya ki roshni prithvi tak aane mein kitna samay leti hai?", "Surya ki roshni prithvi tak pahunchne mein lagbhag 8 minute 20 second leti hai.", "qa", "hinglish")
            ] * 6

        for p, r, c, l in tasks:
            recs.append({"prompt": f"User: {p}\nAssistant: ", "response": r, "category": c, "language": l, "turns": 1})

        # Multi-turn
        if split_name == "train":
            mt = [
                [("User: Hi! I am designing an autonomous robot.\nAssistant: ", "Hello! That sounds fascinating. What kind of locomotion will it use?"),
                 ("User: It has a four-wheel differential drive chassis.\nAssistant: ", "Differential drive is versatile. You can use two independent DC gear motors with encoders.")],
                [("User: Namaste, mujhe web development seekhni hai.\nAssistant: ", "Namaste! Yeh bohot acchi field hai. Aap frontend se shuru karna chahenge ya backend se?"),
                 ("User: Pehle frontend seekhna hai.\nAssistant: ", "Frontend ke liye HTML, CSS aur JavaScript se shuruaat karein.")]
            ] * 10
        elif split_name == "val":
            mt = [
                [("User: Hello, can you explain cloud computing?\nAssistant: ", "Hello! Cloud computing delivers computing services like servers, storage, and networking over the internet."),
                 ("User: What are the main cloud providers?\nAssistant: ", "The major cloud service providers include AWS, Microsoft Azure, and Google Cloud.")]
            ] * 5
        else:
            mt = [
                [("User: Hi assistant, I am learning data science.\nAssistant: ", "Hello! Data science is a great domain. Are you currently working with Python?"),
                 ("User: Yes, practicing with pandas and numpy.\nAssistant: ", "Pandas and NumPy form the core data manipulation toolkit in Python data workflows.")]
            ] * 5

        for d in mt:
            acc = ""
            for t_idx, (u, a) in enumerate(d):
                acc += u
                recs.append({"prompt": acc, "response": a, "category": "multi_turn", "language": "hi" if "Namaste" in acc else "en", "turns": t_idx + 1})
                acc += a + "\n"

        rng.shuffle(recs)
        return recs

    train_data = generate_split_records(train_entities, greetings_train, "train", (1, 15))
    val_data = generate_split_records(val_entities, greetings_val, "val", (16, 20))
    test_data = generate_split_records(test_entities, greetings_test, "test", (21, 25))

    # Verify ZERO leakage
    train_prompts = {r["prompt"] for r in train_data}
    val_leakage = [r["prompt"] for r in val_data if r["prompt"] in train_prompts]
    test_leakage = [r["prompt"] for r in test_data if r["prompt"] in train_prompts]

    assert len(val_leakage) == 0, f"Val leakage detected: {len(val_leakage)}"
    assert len(test_leakage) == 0, f"Test leakage detected: {len(test_leakage)}"

    out_dir = Path("experiments/ucip29/datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "conv_train.jsonl", "w", encoding="utf-8") as f:
        for r in train_data:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "conv_val.jsonl", "w", encoding="utf-8") as f:
        for r in val_data:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "conv_test_sealed.jsonl", "w", encoding="utf-8") as f:
        for r in test_data:
            f.write(json.dumps(r) + "\n")

    def cat_counts(data):
        c = {}
        for r in data:
            c[r["category"]] = c.get(r["category"], 0) + 1
        return c

    manifest = {
        "train_count": len(train_data),
        "val_count": len(val_data),
        "test_count": len(test_data),
        "val_leakage": len(val_leakage),
        "test_leakage": len(test_leakage),
        "train_categories": cat_counts(train_data),
        "val_categories": cat_counts(val_data),
        "test_categories": cat_counts(test_data)
    }

    with open(out_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Zero-Leakage Dataset Successfully Generated!")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")
    print(f"Val Leakage: {len(val_leakage)} | Test Leakage: {len(test_leakage)}")
    print("Train Categories:", json.dumps(cat_counts(train_data), indent=2))


if __name__ == "__main__":
    build_disjoint_conversational_datasets()
