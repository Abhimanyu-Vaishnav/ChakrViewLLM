"""
Ultimate Conversational & Reasoning Dataset Generator.
Synthesizes a rich, multi-domain, zero-leakage training corpus across:
1. Basic Social & Greeting dialogues (EN, HI, Hinglish)
2. Open Domain Explanations & Instruction Following
3. In-Context Entity & Fact Retrieval (Disjoint entity pools)
4. Multi-Turn Interactive Problem Solving
5. Multi-Step Mathematical & Arithmetic Reasoning
6. Grounded Abstention & Uncertainty Handling
7. Algorithmic, Python & Code Understanding
"""

from __future__ import annotations
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts


def generate_ultimate_datasets():
    tok, _ = load_tokenizer_artifacts(ROOT / "data" / "experiments" / "vocab_4096")

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

    def build_split(entity_pool, split_name: str, math_count: int = 200) -> list[dict]:
        records = []
        rng = random.Random(42 if split_name == "train" else (100 if split_name == "val" else 200))

        # 1. Social & Greetings (Tailored per split)
        if split_name == "train":
            social = [
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
            ] * 12
        elif split_name == "val":
            social = [
                ("User: Hey ChakrView!\nAssistant: ", "Hello! How can I help you right now?", "greeting", "en"),
                ("User: Good afternoon!\nAssistant: ", "Good afternoon! How may I assist you today?", "greeting", "en"),
                ("User: Shukriya dost!\nAssistant: ", "Koi baat nahi, hamesha madad ke liye tayyar hoon.", "social", "hinglish"),
                ("User: See you later!\nAssistant: ", "See you soon! Take care.", "social", "en")
            ] * 6
        else:
            social = [
                ("User: Greetings system!\nAssistant: ", "Greetings! How may I be of service today?", "greeting", "en"),
                ("User: Hello there assistant!\nAssistant: ", "Hello! What questions can I answer for you?", "greeting", "en"),
                ("User: Aapko bahut dhanyawaad!\nAssistant: ", "Khushi hui aapki madad karke!", "social", "hinglish"),
                ("User: Bye for now!\nAssistant: ", "Bye! Have a wonderful day.", "social", "en")
            ] * 6

        for p, r, c, l in social:
            records.append({"prompt": p, "response": r, "category": c, "language": l, "turns": 1})

        # 2. Context Retrieval & Abstention
        prefix = "Context statement: " if split_name == "train" else ("Context provided: " if split_name == "val" else "Document snippet: ")
        sandarbh_prefix = "Sandarbh: " if split_name == "train" else ("Vivaran: " if split_name == "val" else "Suchna: ")

        for ent, city, year, founder, prod in entity_pool:
            recs = [
                (f"User: {prefix}{ent} is located in {city} and was founded in {year} by {founder}.\nQuestion: Where is {ent} located?\nAnswer concisely:\nAssistant: ", city, "context_retrieval", "en"),
                (f"User: {prefix}{ent} was founded in {year} by {founder}.\nQuestion: In what year was {ent} founded?\nAnswer concisely:\nAssistant: ", year, "context_retrieval", "en"),
                (f"User: {prefix}{ent} was established by {founder} to develop {prod}.\nQuestion: Who founded {ent}?\nAnswer concisely:\nAssistant: ", founder, "context_retrieval", "en"),
                (f"User: {prefix}{ent} builds {prod} in {city}.\nQuestion: What does {ent} develop?\nAnswer concisely:\nAssistant: ", prod, "context_retrieval", "en"),
                (f"User: {sandarbh_prefix}{ent} ki sthapna {year} mein {founder} dwara hui thi.\nPrashna: {ent} kis varsh sthapit hua?\nSankshipt uttar dein:\nAssistant: ", year, "context_retrieval", "hinglish"),
                (f"User: {sandarbh_prefix}{ent} ka mukhya kendra {city} mein sthit hai.\nPrashna: {ent} kahan sthit hai?\nSankshipt uttar dein:\nAssistant: ", city, "context_retrieval", "hinglish"),
                (f"User: {prefix}{ent} was founded in {year}.\nQuestion: What is the annual revenue of {ent}?\nAnswer concisely:\nAssistant: ", "Information not provided in the context.", "abstention", "en"),
                (f"User: {sandarbh_prefix}{ent} {city} mein sthit hai.\nPrashna: {ent} ke kitne karmchari hain?\nSankshipt uttar dein:\nAssistant: ", "Sandarbh mein karmchariyon ki sankhya uplabdh nahi hai.", "abstention", "hinglish")
            ]
            for p, r, c, l in recs:
                records.append({"prompt": p, "response": r, "category": c, "language": l, "turns": 1, "entity": ent})

        # 3. Arithmetic & Mathematical Reasoning (Distinct prompts per split)
        num_min, num_max = (1, 16) if split_name == "train" else ((16, 21) if split_name == "val" else (21, 26))
        pairs = [(a, b) for a in range(num_min, num_max) for b in range(1, 12)]
        rng.shuffle(pairs)
        for a, b in pairs[:math_count]:
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
                p = f"User: Multiply {a} by {b}.\nAnswer concisely:\nAssistant: " if split_name == "train" else (
                    f"User: What is the product of {a} and {b}?\nAssistant: " if split_name == "val" else
                    f"User: Evaluate {a} * {b}.\nAssistant: "
                )
                r = str(a * b)
            records.append({"prompt": p, "response": r, "category": "arithmetic", "language": "en", "turns": 1})

        # 4. Multi-Step Reasoning & Explanation
        if split_name == "train":
            tasks = [
                ("What is photosynthesis?", "Photosynthesis is the biological process by which green plants convert light into chemical energy.", "explanation", "en"),
                ("What is an operating system kernel?", "The kernel is the core software managing CPU, memory, and devices.", "explanation", "en"),
                ("Write a Python function to check if a number is even.", "def is_even(n: int) -> bool:\n    return n % 2 == 0", "code", "en"),
                ("Bharat ki rajdhani kya hai?", "Bharat ki rajdhani New Delhi hai.", "qa", "hinglish"),
                ("Computer mein RAM ka kya karya hai?", "RAM temporary data aur active programs ko fast storage deta hai.", "explanation", "hinglish"),
                ("If Alice has 3 apples and Bob gives her 5 more, how many apples does Alice have?", "Alice has 8 apples.", "reasoning", "en"),
                ("A train leaves city A at 10 AM and arrives at city B at 1 PM. How long was the journey?", "The journey took 3 hours.", "reasoning", "en")
            ] * 12
        elif split_name == "val":
            tasks = [
                ("What is the primary function of DNS?", "DNS translates human-readable domain names into numerical IP addresses.", "explanation", "en"),
                ("Write a Python function to compute the square of x.", "def square(x: float) -> float:\n    return x * x", "code", "en"),
                ("Prithvi ka sabse bada mahasagar kaun sa hai?", "Prithvi ka sabse bada mahasagar Prashant Mahasagar (Pacific Ocean) hai.", "qa", "hinglish"),
                ("If a box holds 6 pens, how many pens do 4 boxes hold?", "4 boxes hold 24 pens.", "reasoning", "en")
            ] * 6
        else:
            tasks = [
                ("What is an algorithm in computing?", "An algorithm is a step-by-step sequence of instructions designed to solve a specific problem.", "explanation", "en"),
                ("Write a Python function to return the length of a string.", "def get_length(s: str) -> int:\n    return len(s)", "code", "en"),
                ("Surya ki roshni prithvi tak aane mein kitna samay leti hai?", "Surya ki roshni prithvi tak pahunchne mein lagbhag 8 minute 20 second leti hai.", "qa", "hinglish"),
                ("If you have 10 coins and spend 4, how many coins remain?", "6 coins remain.", "reasoning", "en")
            ] * 6

        for p, r, c, l in tasks:
            records.append({"prompt": f"User: {p}\nAssistant: ", "response": r, "category": c, "language": l, "turns": 1})

        # 5. Multi-Turn Interactive Dialogues
        if split_name == "train":
            dialogues = [
                [
                    ("User: Hi! I am designing an autonomous robot.\nAssistant: ", "Hello! That sounds fascinating. What kind of locomotion will it use?"),
                    ("User: It has a four-wheel differential drive chassis.\nAssistant: ", "Differential drive is versatile. You can use two independent DC gear motors with encoders.")
                ],
                [
                    ("User: Namaste, mujhe web development seekhni hai.\nAssistant: ", "Namaste! Yeh bohot acchi field hai. Aap frontend se shuru karna chahenge ya backend se?"),
                    ("User: Pehle frontend seekhna hai.\nAssistant: ", "Frontend ke liye HTML, CSS aur JavaScript se shuruaat karein.")
                ]
            ] * 12
        elif split_name == "val":
            dialogues = [
                [
                    ("User: Hello, can you explain cloud computing?\nAssistant: ", "Hello! Cloud computing delivers computing services like servers, storage, and networking over the internet."),
                    ("User: What are the main cloud providers?\nAssistant: ", "The major cloud service providers include AWS, Microsoft Azure, and Google Cloud.")
                ]
            ] * 6
        else:
            dialogues = [
                [
                    ("User: Hi assistant, I am learning data science.\nAssistant: ", "Hello! Data science is a great domain. Are you currently working with Python?"),
                    ("User: Yes, practicing with pandas and numpy.\nAssistant: ", "Pandas and NumPy form the core data manipulation toolkit in Python data workflows.")
                ]
            ] * 6

        for d in dialogues:
            acc = ""
            for t_idx, (u, a) in enumerate(d):
                acc += u
                records.append({
                    "prompt": acc,
                    "response": a,
                    "category": "multi_turn",
                    "language": "hi" if "Namaste" in acc else "en",
                    "turns": t_idx + 1
                })
                acc += a + "\n"

        rng.shuffle(records)
        return records

    train_recs = build_split(train_entities, "train", math_count=200)
    val_recs = build_split(val_entities, "val", math_count=60)
    test_recs = build_split(test_entities, "test", math_count=60)

    # Output directory
    out_dir = Path("experiments/ultimate_brain/datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train_recs:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for r in val_recs:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "test_sealed.jsonl", "w", encoding="utf-8") as f:
        for r in test_recs:
            f.write(json.dumps(r) + "\n")

    train_prompts = {r["prompt"] for r in train_recs}
    val_leakage = len([r for r in val_recs if r["prompt"] in train_prompts])
    test_leakage = len([r for r in test_recs if r["prompt"] in train_prompts])

    summary = {
        "train_count": len(train_recs),
        "val_count": len(val_recs),
        "test_count": len(test_recs),
        "val_leakage": val_leakage,
        "test_leakage": test_leakage,
    }
    with open(out_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Ultimate Datasets Created: Train={len(train_recs)}, Val={len(val_recs)}, Test={len(test_recs)}")
    print(f"Leakage: Val={val_leakage}, Test={test_leakage}")


if __name__ == "__main__":
    generate_ultimate_datasets()
