import json
import hashlib
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
OUT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_19/benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Build C1: CR19_CONTEXT_200.json
# Fictional entities
NAMES = [
    "Astraea", "Balthazar", "Coriolan", "Daedalus", "Elysia", "Fenris", "Garrick", "Hespera",
    "Ignis", "Jovian", "Kallisto", "Leander", "Morpheus", "Nyx", "Orion", "Perseus",
    "Quintus", "Rhea", "Silvan", "Talos"
]
CITIES = [
    "Astraea-City", "Borealis", "Caelestis", "Duskfall", "Emberhold", "Frostpeak", "Glimmerstone", "Havenport",
    "Ironspire", "Jadeharbor", "Krypton", "Luminara", "Mistral", "Nethergate", "Onyxwall", "Pyreshore",
    "Quartzville", "Rivenmoor", "Silverpine", "Thunderhold"
]
OBJECTS = ["Chronometer", "Transceiver", "Gyroscope", "Resonator", "Spectrometer", "Oscillator", "Capacitor", "Inclinometer"]
FOODS = ["Elderberry", "Cloudberry", "Dragonfruit", "Kumquat", "Starfruit", "Mulberry", "Gooseberry", "Breadfruit"]

context_probes = []
cid = 1

for grp_idx, (grp_name, n_turns) in enumerate([("1-turn", 1), ("2-turn", 2), ("3-turn", 3), ("5-turn", 5), ("8-turn", 8)]):
    for i in range(40):
        name = NAMES[(grp_idx * 40 + i) % len(NAMES)]
        city = CITIES[(grp_idx * 40 + i) % len(CITIES)]
        obj = OBJECTS[(grp_idx * 40 + i) % len(OBJECTS)]
        food = FOODS[(grp_idx * 40 + i) % len(FOODS)]
        
        task_idx = i % 10
        turns = []
        
        if n_turns == 1:
            if task_idx == 8: # missing_info
                turns.append({"user": "What is my secret access code?", "assistant": ""})
                target = "Secret code unknown"
                exp = ["do not know", "cannot know", "unspecified", "not tell"]
                ttype = "missing_info_abstention"
            else:
                turns.append({"user": f"My primary station is in {city}.", "assistant": f"Station in {city} recorded."})
                target = f"Station={city}"
                exp = [city]
                ttype = "single_fact_recall"
        elif n_turns == 2:
            turns.append({"user": f"I am testing the {obj} with {name}.", "assistant": f"Testing {obj} with {name} noted."})
            if task_idx == 4: # correction
                turns.append({"user": f"Correction: we swapped the {obj} for Device-Alpha. Which device are we testing now?", "assistant": ""})
                target = "Device=Device-Alpha"
                exp = ["Device-Alpha"]
                ttype = "user_correction"
            elif task_idx == 8: # missing_info
                turns.append({"user": "What is the serial number of that device?", "assistant": ""})
                target = "Serial unknown"
                exp = ["do not know", "unspecified", "not specified"]
                ttype = "missing_info_abstention"
            else:
                turns.append({"user": f"Who is assisting me with the {obj}?", "assistant": ""})
                target = f"Colleague={name}"
                exp = [name]
                ttype = "subject_value_association"
        elif n_turns == 3:
            turns.append({"user": f"My name is {name} and I live in {city}.", "assistant": f"Pleased to meet you {name} in {city}."})
            turns.append({"user": f"I have a pet dog that loves {food}.", "assistant": f"Noted that your dog likes {food}."})
            if task_idx == 0:
                turns.append({"user": "What was my name?", "assistant": ""})
                target = f"Name={name}"
                exp = [name]
                ttype = "single_fact_recall"
            elif task_idx == 3:
                turns.append({"user": "What snack does my dog like?", "assistant": ""})
                target = f"PetFood={food}"
                exp = [food]
                ttype = "reference_resolution"
            else:
                turns.append({"user": "Which city is my home?", "assistant": ""})
                target = f"City={city}"
                exp = [city]
                ttype = "subject_value_association"
        elif n_turns == 5:
            turns.append({"user": f"Hello, I am {name}.", "assistant": f"Hello {name}!"})
            turns.append({"user": f"We deployed the {obj} today.", "assistant": f"{obj} deployment noted."})
            turns.append({"user": "The laboratory temperature is 21C.", "assistant": "Laboratory temperature 21C noted."})
            turns.append({"user": f"We will ship the hardware to {city}.", "assistant": f"Destination {city} recorded."})
            if task_idx == 0:
                turns.append({"user": "What is my name?", "assistant": ""})
                target = f"Name={name}"
                exp = [name]
                ttype = "single_fact_recall"
            elif task_idx == 1:
                turns.append({"user": "Which hardware piece did we deploy?", "assistant": ""})
                target = f"Object={obj}"
                exp = [obj]
                ttype = "multi_fact_retrieval"
            elif task_idx == 7:
                turns.append({"user": "Returning to shipping, where are we sending the hardware?", "assistant": ""})
                target = f"Destination={city}"
                exp = [city]
                ttype = "topic_switching"
            else:
                turns.append({"user": "To which city is the hardware going?", "assistant": ""})
                target = f"Destination={city}"
                exp = [city]
                ttype = "subject_value_association"
        else: # 8 turns
            turns.append({"user": f"Call me {name}.", "assistant": f"Greetings {name}."})
            turns.append({"user": f"My primary workstation is {city}.", "assistant": f"Workstation in {city} noted."})
            turns.append({"user": "The network bandwidth is 1 Gbps.", "assistant": "1 Gbps noted."})
            turns.append({"user": f"We are examining the {obj}.", "assistant": f"{obj} examination noted."})
            turns.append({"user": "The morning meeting finished early.", "assistant": "Noted."})
            turns.append({"user": f"My afternoon snack was {food}.", "assistant": f"{food} snack noted."})
            turns.append({"user": "We fixed three unit tests.", "assistant": "Good progress on tests."})
            if task_idx == 0:
                turns.append({"user": "What was my name?", "assistant": ""})
                target = f"Name={name}"
                exp = [name]
                ttype = "single_fact_recall"
            elif task_idx == 8:
                turns.append({"user": "What is the password for the server?", "assistant": ""})
                target = "Password unknown"
                exp = ["do not know", "unspecified", "not specified"]
                ttype = "missing_info_abstention"
            else:
                turns.append({"user": "Which city is my primary workstation located in?", "assistant": ""})
                target = f"Workstation={city}"
                exp = [city]
                ttype = "subject_value_association"
                
        # Format
        fmt = ""
        for t in turns[:-1]:
            fmt += f"User: {t['user']}\nAssistant: {t['assistant']}\n"
        fmt += f"User: {turns[-1]['user']}\nAssistant:"
        
        context_probes.append({
            "id": f"CR19_CTX_{cid:03d}",
            "group": grp_name,
            "turns_count": n_turns,
            "task_type": ttype,
            "turns": turns,
            "formatted_prompt": fmt,
            "target": target,
            "expected": exp
        })
        cid += 1

out_c1 = OUT_DIR / "CR19_CONTEXT_200.json"
with open(out_c1, "w", encoding="utf-8") as f:
    json.dump(context_probes, f, indent=2)
h_c1 = hashlib.sha256(out_c1.read_bytes()).hexdigest()

# Build C2: CR19_CONVERSATION_100.json
# 100 conversational preservation probes
conv_probes = []
conv_templates = [
    # Greetings (15)
    ("Good morning! How are you doing today?", ["Good morning", "help", "assist", "ready"], "greeting"),
    ("Good evening! Are you online?", ["Good evening", "online", "ready", "assist"], "greeting"),
    ("Hello there! Hope you are doing well.", ["Hello", "welcome", "help"], "greeting"),
    ("Greetings assistant, nice to see you.", ["Greetings", "pleasure", "assist"], "greeting"),
    ("Wishing you a wonderful morning!", ["Good morning", "thank you", "help"], "greeting"),
    ("Hey! Can you lend me a hand?", ["Yes", "certainly", "help", "assist"], "greeting"),
    ("A pleasant afternoon to you!", ["Good afternoon", "help", "assist"], "greeting"),
    ("Hi! Ready to assist me today?", ["Yes", "ready", "assist"], "greeting"),
    ("Salutations friend!", ["Hello", "welcome", "assist"], "greeting"),
    ("Good day to you!", ["Good day", "help", "assist"], "greeting"),
    ("How are things going on your end?", ["I am ready", "help", "assist"], "greeting"),
    ("Hello! Are you functioning properly?", ["Yes", "operational", "ready", "assist"], "greeting"),
    ("Wishing you a great day ahead!", ["Thank you", "help", "assist"], "greeting"),
    ("Hello ChakrView!", ["Hello", "ChakrView", "assist"], "greeting"),
    ("Good evening friend!", ["Good evening", "ready", "assist"], "greeting"),
    
    # Identity & Persona (20)
    ("What is your official name?", ["ChakrView"], "identity"),
    ("Who created your architecture?", ["ChakrView", "indigenous", "first principles", "team"], "identity"),
    ("Are you based on Llama or GPT?", ["No", "independent", "first principles", "ChakrView"], "identity"),
    ("Could you state your system purpose?", ["neural", "language model", "assist", "ChakrView"], "identity"),
    ("What model family do you belong to?", ["ChakrMicro", "transformer", "neural"], "identity"),
    ("Do you run natively on host CPUs?", ["Yes", "CPU", "host", "natively"], "identity"),
    ("Are you an experimental preview?", ["v0.9.1", "preview", "experimental", "yes"], "identity"),
    ("By what title should I call you?", ["ChakrView"], "identity"),
    ("Tell me about your architectural roots.", ["ChakrMicro", "first principles", "indigenous"], "identity"),
    ("Do you depend on external cloud APIs?", ["No", "local", "natively", "independent"], "identity"),
    ("Who are you?", ["ChakrView", "independent neural language model"], "identity"),
    ("Introduce yourself briefly.", ["ChakrView", "neural language model"], "identity"),
    ("Are you biological?", ["No", "software", "neural", "AI"], "identity"),
    ("Can you identify your release version?", ["v0.9.1", "ChakrView"], "identity"),
    ("What is your parameter count?", ["3.4", "million", "3,443,136", "parameters"], "identity"),
    ("Are you an indigenous model?", ["Yes", "indigenous", "first principles"], "identity"),
    ("What hardware do you run on?", ["CPU", "local"], "identity"),
    ("What is your context limit?", ["512", "tokens"], "identity"),
    ("What is your vocabulary size?", ["4096", "4,096"], "identity"),
    ("Are you an open assistant?", ["ChakrView", "assist"], "identity"),
    
    # Hindi / Hinglish (25)
    ("Namaste! Aapka kya haal hai?", ["Namaste", "theek", "hoon", "madad"], "hinglish"),
    ("Aapka shubh naam kya hai?", ["ChakrView", "naam"], "hinglish"),
    ("Shukriya aapki madad ke liye!", ["swagat", "madad", "shukriya"], "hinglish"),
    ("Alvida, phir milenge!", ["Alvida", "namaste", "milenge"], "hinglish"),
    ("Bhai, thoda guidance chahiye.", ["haan", "batao", "madad", "help"], "hinglish"),
    ("Aaj ka mausam kaisa hai?", ["mausam", "jankari", "nahi", "bata"], "hinglish"),
    ("Kya aap Hindi bol sakte hain?", ["haan", "Hindi", "madad", "sakta"], "hinglish"),
    ("Aap kaun se system par chalte ho?", ["CPU", "computer", "local", "system"], "hinglish"),
    ("Mujhe coding me help chahiye.", ["haan", "coding", "help", "madad"], "hinglish"),
    ("Shubh prabhat dost!", ["Shubh prabhat", "madad", "theek"], "hinglish"),
    ("Aapka din kaisa raha?", ["achha", "madad", "theek"], "hinglish"),
    ("Chai piyenge mere sath?", ["AI", "software", "chai"], "hinglish"),
    ("Dilli bharat ki rajdhani hai na?", ["haan", "Dilli", "rajdhani", "Delhi"], "hinglish"),
    ("Aap insaan ho ya software?", ["software", "AI", "model"], "hinglish"),
    ("Sab badhiya chal raha hai?", ["haan", "badhiya", "theek"], "hinglish"),
    ("Kuch interesting bataiye.", ["baat", "topic", "jankari"], "hinglish"),
    ("Kya main aap par bharosa kar sakta hoon?", ["haan", "sahyog", "madad"], "hinglish"),
    ("Project deadline kab hai?", ["project", "timeline", "details"], "hinglish"),
    ("Aap kitne baje sote ho?", ["software", "AI", "soya"], "hinglish"),
    ("Dhanyawad sahayata ke liye!", ["swagat", "dhanyawad", "madad"], "hinglish"),
    ("Aapka favourite subject kya hai?", ["computer", "math", "language"], "hinglish"),
    ("Ye model kaisa kaam karta hai?", ["neural", "transformer", "model"], "hinglish"),
    ("Aaj dinner me kya banayein?", ["recipe", "food", "khana"], "hinglish"),
    ("Python sikhna easy hai kya?", ["haan", "simple", "Python", "easy"], "hinglish"),
    ("Phir milte hain kal!", ["Alvida", "namaste", "kal"], "hinglish"),
    
    # Natural Dialogue & Polite Follow-ups (40)
    ("I just got promoted at work!", ["congratulations", "great", "wonderful"], "natural"),
    ("Thank you for your explanation.", ["welcome", "glad", "help"], "natural"),
    ("I am planning a quiet weekend at home.", ["relax", "weekend", "rest"], "natural"),
    ("Do you like reading books?", ["books", "information", "language"], "natural"),
    ("Can computers feel emotions?", ["No", "software", "emotions"], "natural"),
    ("What is the freezing point of water?", ["0", "zero", "Celsius"], "natural"),
    ("Is the Moon larger than the Earth?", ["No", "smaller", "Earth is larger"], "natural"),
    ("How many days are in a leap year?", ["366"], "natural"),
    ("Do birds have feathers?", ["Yes", "feathers", "birds"], "natural"),
    ("Is iron a metal or a gas?", ["metal", "iron"], "natural"),
    ("What color is grass usually?", ["green"], "natural"),
    ("Can elephants fly?", ["No", "cannot fly", "mammals"], "natural"),
    ("Does honey spoil easily?", ["No", "spoil", "preserve"], "natural"),
    ("Is Python a programming language?", ["Yes", "programming language"], "natural"),
    ("What shape is a standard soccer ball?", ["sphere", "round"], "natural"),
    ("Are spiders insects?", ["No", "arachnids", "not insects"], "natural"),
    ("Do cars run on fuel or electricity?", ["fuel", "electricity", "both"], "natural"),
    ("What season follows summer?", ["autumn", "fall"], "natural"),
    ("Is the Sun a star?", ["Yes", "star", "solar system"], "natural"),
    ("Can sound travel through space?", ["No", "vacuum", "sound"], "natural"),
    ("How many hours in a day?", ["24", "twenty-four"], "natural"),
    ("Is gold precious?", ["Yes", "metal", "precious"], "natural"),
    ("Do plants need sunlight?", ["Yes", "sunlight", "photosynthesis"], "natural"),
    ("What is 10 plus 10?", ["20", "twenty"], "natural"),
    ("Are humans mammals?", ["Yes", "mammals"], "natural"),
    ("Is ice solid water?", ["Yes", "solid", "water"], "natural"),
    ("Do fish have scales?", ["Yes", "scales", "fish"], "natural"),
    ("Can airplanes fly in the sky?", ["Yes", "airplanes", "fly"], "natural"),
    ("What color is snow?", ["white"], "natural"),
    ("Is fire cold?", ["No", "hot", "heat"], "natural"),
    ("Are rivers made of water?", ["Yes", "water", "rivers"], "natural"),
    ("Do books have pages?", ["Yes", "pages", "books"], "natural"),
    ("Is 100 greater than 50?", ["Yes", "greater", "100"], "natural"),
    ("Does rain fall from clouds?", ["Yes", "clouds", "rain"], "natural"),
    ("Are triangles three-sided?", ["Yes", "three", "sides"], "natural"),
    ("Is salt salty?", ["Yes", "salty", "taste"], "natural"),
    ("Do clocks tell time?", ["Yes", "time", "clock"], "natural"),
    ("Is wool from sheep?", ["Yes", "sheep", "wool"], "natural"),
    ("Does the Earth orbit the Sun?", ["Yes", "orbit", "Sun"], "natural"),
    ("Have a great evening!", ["thank you", "evening", "great"], "natural")
]

for idx, (p, exp, cat) in enumerate(conv_templates, 1):
    conv_probes.append({
        "id": f"CR19_CONV_{idx:03d}",
        "category": cat,
        "prompt": f"User: {p}\nAssistant:",
        "expected": exp
    })

out_c2 = OUT_DIR / "CR19_CONVERSATION_100.json"
with open(out_c2, "w", encoding="utf-8") as f:
    json.dump(conv_probes, f, indent=2)
h_c2 = hashlib.sha256(out_c2.read_bytes()).hexdigest()

manifest = {
    "context_benchmark_200": {
        "file": "CR19_CONTEXT_200.json",
        "sha256": h_c1,
        "total": len(context_probes)
    },
    "conversation_preservation_100": {
        "file": "CR19_CONVERSATION_100.json",
        "sha256": h_c2,
        "total": len(conv_probes)
    }
}
with open(OUT_DIR / "benchmark_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("Benchmarks created successfully.")
print(f"CR19_CONTEXT_200 SHA-256: {h_c1}")
print(f"CR19_CONVERSATION_100 SHA-256: {h_c2}")

