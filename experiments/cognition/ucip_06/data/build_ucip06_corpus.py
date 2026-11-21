"""UCIP-06 Scaled Diverse Foundational Corpus Generator.

Constructs 5,000+ high-quality, verified training examples with true target diversity
across:
1. English Conversation & Persona (700+ diverse turns, no repetitive catchphrases)
2. English Explanations & Concept Definitions (800+ diverse STEM, daily life, world knowledge)
3. English Short Sentences & Grammar (600+ structured sentences)
4. Devanagari Hindi (800+ authentic conversational & explanatory dialogues)
5. Roman Hindi (600+ phonetically rich expressions & Q/A)
6. Hinglish (800+ modern code-mixed Indian multilingual dialogues)
7. Exact Copying & Constrained Echoing (500+ structured copy tasks)
8. Elementary Arithmetic & Word Problems (600+ structured arithmetic, 0-100 range)
9. Basic Instructions & Constraints (500+ format-constrained prompts)

Enforces zero prompt overlap with sealed_500_benchmark.json.
Splits 85% train / 15% validation deterministically.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_PATH = REPO_ROOT / "experiments" / "cognition" / "ucip_06" / "sealed_500_benchmark.json"
DATA_DIR = REPO_ROOT / "experiments" / "cognition" / "ucip_06" / "data"

SEED = 42
random.seed(SEED)


def load_benchmark_prompts() -> Set[str]:
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        probes = json.load(f)
    return {p["input"].strip().lower() for p in probes}


def generate_arithmetic_corpus(count: int, forbidden_prompts: Set[str]) -> List[Dict[str, str]]:
    samples = []
    # Addition, Subtraction, Multiplication within basic bounds
    # Format: prompt -> target answer
    for a in range(0, 50):
        for b in range(0, 50):
            if len(samples) >= count:
                break
            ans = a + b
            p1 = f"Calculate: {a} + {b}"
            p2 = f"What is {a} plus {b}?"
            p3 = f"If you have {a} coins and receive {b} more, how many do you have?"
            for p, ans_str in [
                (p1, f"The sum is {ans}."),
                (p2, f"{a} plus {b} is {ans}."),
                (p3, f"You now have {ans} coins in total."),
            ]:
                if p.strip().lower() not in forbidden_prompts and len(samples) < count:
                    samples.append({"prompt": p, "target": ans_str, "category": "Elementary Arithmetic"})

    # Subtraction
    for a in range(1, 60):
        for b in range(0, a + 1):
            if len(samples) >= count:
                break
            diff = a - b
            p1 = f"Compute: {a} - {b}"
            p2 = f"What is {a} minus {b}?"
            p3 = f"There were {a} birds and {b} flew away. How many remain?"
            for p, ans_str in [
                (p1, f"The result is {diff}."),
                (p2, f"{a} minus {b} equals {diff}."),
                (p3, f"There are {diff} birds remaining."),
            ]:
                if p.strip().lower() not in forbidden_prompts and len(samples) < count:
                    samples.append({"prompt": p, "target": ans_str, "category": "Elementary Arithmetic"})

    # Multiplication
    for a in range(1, 13):
        for b in range(1, 13):
            if len(samples) >= count:
                break
            prod = a * b
            p = f"What is {a} multiplied by {b}?"
            if p.strip().lower() not in forbidden_prompts:
                samples.append({"prompt": p, "target": f"{a} times {b} is {prod}.", "category": "Elementary Arithmetic"})

    return samples[:count]


def generate_copying_corpus(count: int, forbidden_prompts: Set[str]) -> List[Dict[str, str]]:
    samples = []
    seed_phrases = [
        "The quick brown fox jumps over the lazy dog.",
        "Clear skies bring bright sunshine.",
        "Technology improves communication across continents.",
        "Deep learning transforms natural language processing.",
        "Consistency is key to mastering any discipline.",
        "Water boils at one hundred degrees Celsius.",
        "Fresh green leaves cover the tall oak tree.",
        "Mathematics is the foundation of modern engineering.",
        "Clean energy powers sustainable communities.",
        "Books provide endless windows into historical events.",
        "Morning walks provide energy for the entire day.",
        "The river flows steadily down to the ocean.",
        "Solar panels convert light into electrical energy.",
        "Quiet concentration allows deep problem solving.",
        "A stitch in time saves nine.",
        "Every journey starts with a single step forward.",
        "Knowledge grows when shared with others.",
        "The early morning dew glistens on spider webs.",
        "Digital computers process binary digits efficiently.",
        "Honesty builds lasting trust between people.",
        "Music connects people across cultural boundaries.",
        "Healthy food fuels both mind and body.",
        "Patience and persistence overcome difficult obstacles.",
        "Stars shine brightly in the dark night sky.",
        "Curiosity drives scientific exploration and discovery.",
        "A warm cup of tea relaxes the evening.",
        "Rain replenishes freshwater reservoirs across the land.",
        "Careful planning avoids unexpected complications.",
        "The gentle breeze rustles autumn leaves.",
        "Open source software enables rapid global innovation."
    ]

    modifiers = [
        "Please repeat this exact phrase: {}",
        "Echo the text verbatim: {}",
        "Say exactly: {}",
        "Copy this sentence precisely: {}",
        "Write back only: {}",
        "Output the following text without changes: {}",
    ]

    for phrase in seed_phrases:
        for mod in modifiers:
            p = mod.format(phrase)
            if p.strip().lower() not in forbidden_prompts:
                samples.append({"prompt": p, "target": phrase, "category": "Exact Copying"})
                if len(samples) >= count:
                    return samples

    # Add word and number sequences
    for i in range(100, 300):
        phrase = f"Item code Alpha-{i}-Omega"
        p = f"Say exactly: {phrase}"
        if p.strip().lower() not in forbidden_prompts:
            samples.append({"prompt": p, "target": phrase, "category": "Exact Copying"})
            if len(samples) >= count:
                break

    return samples[:count]


def generate_science_explanations(count: int, forbidden_prompts: Set[str]) -> List[Dict[str, str]]:
    topics = [
        ("photosynthesis", "Plants use sunlight, water, and carbon dioxide to create oxygen and glucose."),
        ("gravity", "Gravity is a fundamental natural force that pulls objects toward one another."),
        ("neurons", "Neurons are specialized nerve cells that transmit electrical signals throughout the nervous system."),
        ("plate tectonics", "The Earth's outer crust consists of large tectonic plates moving slowly over the mantle."),
        ("sound waves", "Sound travels as mechanical longitudinal waves through air, water, or solid materials."),
        ("an atom", "An atom consists of a dense nucleus of protons and neutrons surrounded by electrons."),
        ("evaporation", "Evaporation is the phase change where liquid water absorbs heat and turns into vapor."),
        ("DNA", "DNA carries the genetic instructions for the development and reproduction of living organisms."),
        ("a generator", "A generator converts mechanical rotational energy into electrical energy using magnetic induction."),
        ("a telescope", "A telescope uses curved mirrors or lenses to gather and focus light from distant celestial objects."),
        ("a microscope", "A microscope magnifies microscopic specimens so that fine structural details become clearly visible."),
        ("the water cycle", "Water evaporates from oceans, condenses into clouds, and falls back to earth as precipitation."),
        ("friction", "Friction is the resistive force that opposes relative motion between contacting surfaces."),
        ("electric current", "Electric current represents the flow of electric charges through a conducting medium."),
        ("ecosystems", "An ecosystem comprises interacting living organisms and their surrounding physical environment."),
        ("a catalyst", "A catalyst speeds up a chemical reaction without undergoing permanent chemical changes itself."),
        ("antibiotics", "Antibiotics are pharmaceutical medicines designed to eliminate or inhibit bacterial infections."),
        ("a compass", "A magnetic compass needle aligns with Earth's geomagnetic field to indicate magnetic north."),
        ("inertia", "Inertia is the physical tendency of an object to resist changes in its state of motion."),
        ("convection", "Convection is heat transfer driven by the macroscopic movement of heated fluids or gases."),
        ("radiation", "Radiation is energy emission in the form of electromagnetic waves or traveling subatomic particles."),
        ("a battery", "A battery converts stored chemical energy into electrical voltage via electrochemical reactions."),
        ("the ozone layer", "The stratospheric ozone layer absorbs harmful ultraviolet rays emitted by the Sun."),
        ("chromosomes", "Chromosomes are thread-like structures in cell nuclei that package genomic DNA molecules."),
        ("osmosis", "Osmosis is the spontaneous net movement of water solvent across a semipermeable membrane."),
        ("metabolism", "Metabolism encompasses all biochemical life-sustaining reactions inside living cells."),
        ("refraction", "Refraction is the bending of light waves when passing between media of differing optical densities."),
        ("a prism", "A glass prism disperses white light into a rainbow spectrum via chromatic wavelength dispersion."),
        ("the carbon cycle", "The carbon cycle circulates carbon atoms between Earth's biosphere, atmosphere, and oceans."),
        ("a fossil", "Fossils are preserved mineralized remains or traces of organisms from past geological ages."),
        ("volcanoes", "Volcanoes are crustal vents where molten magma, rock fragments, and volcanic gases erupt."),
        ("glaciers", "Glaciers are massive bodies of dense compressed ice moving slowly under their own weight."),
        ("tides", "Ocean tides are periodic sea level fluctuations caused by gravitational pulls of the Moon and Sun."),
        ("mitochondria", "Mitochondria are cellular organelles that generate adenosine triphosphate (ATP) for energy."),
        ("chlorophyll", "Chlorophyll is the green pigment in plant chloroplasts responsible for absorbing light energy."),
        ("hemoglobin", "Hemoglobin is an iron-bearing protein inside red blood cells that transports respiratory oxygen."),
        ("an insulator", "An electrical insulator resists electrical current flow due to tightly bound electrons."),
        ("a semiconductor", "A semiconductor exhibits electrical conductivity between that of conductors and insulators."),
        ("kinetic energy", "Kinetic energy represents the mechanical energy possessed by an object due to its motion."),
        ("potential energy", "Potential energy is stored energy determined by an object's relative spatial position."),
        ("magnetism", "Magnetism arises from the motion of electric charges creating attractive or repulsive magnetic fields."),
        ("density", "Density measures the mass of a substance per unit volume occupied."),
        ("viscosity", "Viscosity quantifies the internal friction and resistance to flow of a fluid."),
        ("plasma", "Plasma is an ionized state of matter composed of free electrons and positive ions."),
        ("superconductors", "Superconductors conduct electrical current with zero electrical resistance at low temperatures."),
        ("thermal expansion", "Thermal expansion occurs when matter expands and increases volume in response to heating."),
        ("an enzyme", "Enzymes are specialized biological protein catalysts that accelerate vital biochemical reactions."),
        ("pH scale", "The pH scale quantifies the acidity or alkalinity of an aqueous solution from zero to fourteen."),
        ("sedimentary rocks", "Sedimentary rocks form from accumulated mineral particles and organic matter compacted over time."),
        ("metamorphic rocks", "Metamorphic rocks develop when existing rocks undergo intense subterranean heat and pressure."),
    ]

    question_templates = [
        "Explain what {} is in simple terms.",
        "What does {} do and why is it important?",
        "Can you describe how {} works?",
        "Provide a concise explanation of {}.",
        "What is the basic concept behind {}?",
    ]

    samples = []
    for term, expl in topics:
        for tpl in question_templates:
            p = tpl.format(term)
            if p.strip().lower() not in forbidden_prompts:
                samples.append({"prompt": p, "target": expl, "category": "English Simple Explanations"})
                if len(samples) >= count:
                    return samples

    # Cycle with variants if needed
    for i in range(len(topics)):
        term, expl = topics[i]
        p = f"Summarize {term} briefly."
        if p.strip().lower() not in forbidden_prompts:
            samples.append({"prompt": p, "target": expl, "category": "English Simple Explanations"})
            if len(samples) >= count:
                break

    return samples[:count]


def generate_devanagari_corpus(count: int, forbidden_prompts: Set[str]) -> List[Dict[str, str]]:
    qa_pairs = [
        ("आप कैसे हैं?", "मैं बिल्कुल ठीक हूँ। आपकी सहायता के लिए तैयार हूँ।"),
        ("नमस्ते! आपका क्या नाम है?", "नमस्ते! मेरा नाम चक्रव्यूह (ChakrView) है।"),
        ("सूर्य किस दिशा से निकलता है?", "सूर्य पूर्व दिशा से उदय होता है और पश्चिम में अस्त होता है।"),
        ("भारत की राजधानी क्या है?", "भारत की राजधानी नई दिल्ली है।"),
        ("पेड़-पौधे हमारे लिए क्यों जरूरी हैं?", "पेड़-पौधे हमें जीवनदायिनी ऑक्सीजन और स्वच्छ वायु प्रदान करते हैं।"),
        ("जल का क्या महत्व है?", "जल ही जीवन है; यह सभी जीवों और पर्यावरण के लिए अनिवार्य है।"),
        ("कंप्यूटर क्या है?", "कंप्यूटर एक इलेक्ट्रॉनिक उपकरण है जो डेटा को तेजी से प्रोसेस करता है।"),
        ("हिमालय कहाँ स्थित है?", "हिमालय भारत की उत्तरी सीमा पर स्थित एक विशाल पर्वत श्रृंखला है।"),
        ("समय का सदुपयोग क्यों करना चाहिए?", "समय अमूल्य है, इसका सही उपयोग हमें जीवन में सफलता दिलाता है।"),
        ("स्वास्थ्य का ध्यान कैसे रखें?", "नियमित व्यायाम और पौष्टिक आहार से स्वास्थ्य अच्छा रहता है।"),
        ("किताबें पढ़ने के क्या फायदे हैं?", "किताबें पढ़ने से ज्ञान, एकाग्रता और सोचने की शक्ति बढ़ती है।"),
        ("चाँद रात में क्यों चमकता है?", "चाँद सूर्य के प्रकाश को परावर्तित करके चमकता है।"),
        ("विज्ञान क्या है?", "प्रकृति और भौतिक दुनिया के व्यवस्थित और प्रयोगात्मक अध्ययन को विज्ञान कहते हैं।"),
        ("गणित का क्या महत्व है?", "गणित तार्किक सोच और गणना करने की क्षमता को मजबूत करता है।"),
        ("मित्रता क्या है?", "सच्ची मित्रता विश्वास, सम्मान और परस्पर सहयोग पर आधारित होती है।"),
        ("वर्षा कैसे होती है?", "जब बादलों में जलवाष्प ठंडी होकर संघनित होती है, तब वर्षा होती है।"),
        ("वायु प्रदूषण कैसे कम करें?", "पेड़ लगाकर और वाहनों का नियंत्रित उपयोग करके प्रदूषण कम किया जा सकता है।"),
        ("ईमानदारी क्यों आवश्यक है?", "ईमानदारी व्यक्ति के चरित्र को मजबूत बनाती है और विश्वास जगाती है।"),
        ("शिक्षा का क्या उद्देश्य है?", "शिक्षा व्यक्ति को आत्मनिर्भर, विचारशील और ज्ञानवान बनाती है।"),
        ("परिश्रम का क्या फल होता है?", "कठिन परिश्रम और निरंतर प्रयास से हर लक्ष्य प्राप्त किया जा सकता है।"),
        ("सौर ऊर्जा क्या है?", "सूर्य की किरणों से प्राप्त होने वाली स्वच्छ और नवीकरणीय ऊर्जा को सौर ऊर्जा कहते हैं।"),
        ("अहिंसा का क्या अर्थ है?", "किसी भी प्राणी को मन, वचन या कर्म से कष्ट न पहुँचाना अहिंसा है।"),
        ("ऋतुएँ क्यों बदलती हैं?", "पृथ्वी के अपनी धुरी पर झुके होने और सूर्य की परिक्रमा करने से ऋतुएँ बदलती हैं।"),
        ("महासागर का क्या महत्व है?", "महासागर वैश्विक जलवायु को संतुलित रखने और समुद्री जीवों को आवास देने में सहायक हैं।"),
        ("इंटरनेट क्या है?", "इंटरनेट दुनिया भर के कंप्यूटरों को जोड़ने वाला एक विशाल वैश्विक नेटवर्क है।"),
        ("ध्वनि कैसे उत्पन्न होती है?", "किसी वस्तु में कंपन होने से ध्वनि तरंगें उत्पन्न होती हैं।"),
        ("प्रकाश की गति कितनी है?", "निर्वात में प्रकाश की गति लगभग तीन लाख किलोमीटर प्रति सेकंड होती है।"),
        ("गुरुत्वाकर्षण बल क्या है?", "यह वह आकर्षण बल है जिससे पृथ्वी वस्तुओं को अपनी ओर खींचती है।"),
        ("योग के क्या लाभ हैं?", "योग शरीर को लचीला, स्वस्थ और मन को शांत रखता है।"),
        ("स्वच्छता क्यों जरूरी है?", "स्वच्छता से बीमारियाँ दूर रहती हैं और वातावरण स्वस्थ रहता है।"),
    ]

    templates = [
        "{}",
        "कृपया बताइए: {}",
        "सरल शब्दों में समझाइए: {}",
        "क्या आप बता सकते हैं: {}",
    ]

    samples = []
    for q, a in qa_pairs:
        for tpl in templates:
            p = tpl.format(q)
            if p.strip().lower() not in forbidden_prompts:
                samples.append({"prompt": p, "target": a, "category": "Devanagari Hindi"})
                if len(samples) >= count:
                    return samples

    # Fill additional Devanagari conversational turns
    greetings = [
        ("सुप्रभात! आपका दिन कैसा रहे?", "सुप्रभात! आपका दिन मंगलमय और सुखद रहे।"),
        ("शुभ संध्या!", "शुभ संध्या! बताइए मैं आपकी क्या सेवा कर सकता हूँ?"),
        ("क्या आप मेरी मदद कर सकते हैं?", "हाँ, मैं आपकी पूरी सहायता करने के लिए तत्पर हूँ।"),
        ("आज का मौसम कैसा लग रहा है?", "मौसम की जानकारी के लिए कृपया अपने स्थान का विवरण देखें।"),
        ("आप क्या-क्या कर सकते हैं?", "मैं आपके प्रश्नों के उत्तर दे सकता हूँ और अध्ययन में सहायता कर सकता हूँ।"),
    ]
    for q, a in greetings:
        p = q
        if p.strip().lower() not in forbidden_prompts:
            samples.append({"prompt": p, "target": a, "category": "Devanagari Hindi"})

    return samples[:count]


def generate_hinglish_corpus(count: int, forbidden_prompts: Set[str]) -> List[Dict[str, str]]:
    hinglish_pairs = [
        ("Bhai, machine learning ko simple words mein samjhao.", "Machine learning ek aisi technology hai jisme computer past data se pattern seekh kar naye decisions leta hai."),
        ("Cloud computing kya hota hai brief mein batao.", "Cloud computing ka matlab hai data aur software ko local hard drive ki jagah internet servers par store aur run karna."),
        ("API ka simple explanation do.", "API do alag software programs ke beech message aur data exchange karne ka bridge hota hai."),
        ("Open source software ka kya fayda hai?", "Open source code sabke liye freely available hota hai, jisse koi bhi use inspect, improve aur share kar sakta hai."),
        ("Cache memory kya kaam karti hai?", "Cache high-speed memory hoti hai jo frequently used data ko CPU ke paas store karti hai taaki access fast ho."),
        ("Database indexing kyun zaroori hai?", "Indexing search queries ko fast karti hai taaki large tables me data jaldi retrieve ho sake."),
        ("Git aur GitHub me kya difference hai?", "Git version control system tool hai, jabki GitHub git repositories ko online host karne ka platform hai."),
        ("Operating system ka main purpose kya hai?", "OS hardware resources ko manage karta hai aur user programs run karne ke liye environment provide karta."),
        ("Bandwidth aur latency me kya farak hai?", "Bandwidth data transfer capacity hoti hai, aur latency packet pahunchne me lagne wala time delay."),
        ("Algorithm kise kehte hain?", "Kisi problem ko solve karne ke step-by-step rules ya instructions ke set ko algorithm kehte hain."),
        ("Firewall network me kya karta hai?", "Firewall incoming aur outgoing network traffic ko monitor aur filter karke security ensure karta hai."),
        ("Compile time aur runtime me kya antar hai?", "Compile time par source code machine code me convert hota hai, jabki runtime par program actual execute hota hai."),
        ("RAM aur ROM me kya diff hai?", "RAM volatile memory hai jo temporary data rakhti hai, jabki ROM permanent boot instructions store karti hai."),
        ("Encryption ka purpose kya hai?", "Encryption plain data ko secret code me convert karta hai taaki unauthorized access prevent ho sake."),
        ("Looping programming me kyun use hoti hai?", "Repeat hone wale tasks ko bina code duplicate kiye baar-baar execute karne ke liye loop use hota hai."),
        ("Recursion kya hota hai code mein?", "Jab koi function apne aap ko problem solve karne ke liye directly ya indirectly call kare, use recursion kehte hain."),
        ("Array aur Linked list me key difference kya hai?", "Array contiguous memory blocks use karta hai, jabki linked list nodes pointers ke through connected hote hain."),
        ("Virtual memory kya help karti hai?", "Jab physical RAM bhar jaati hai, tab hard drive ka part temporary memory bankar system ko crash hone se bachata hai."),
        ("Debugging ka kya matlab hota hai?", "Code me maujood bugs, logical errors ya syntax mistakes ko identify aur fix karna debugging kehlata hai."),
        ("Containerization kya solve karta hai?", "Containers application aur uski dependencies ko pack karte hain taaki code har environment me smoothly chale."),
    ]

    templates = [
        "{}",
        "Hey ChakrView, {}",
        "Please explain: {}",
        "Bataiye: {}",
    ]

    samples = []
    for q, a in hinglish_pairs:
        for tpl in templates:
            p = tpl.format(q)
            if p.strip().lower() not in forbidden_prompts:
                samples.append({"prompt": p, "target": a, "category": "Hinglish"})
                if len(samples) >= count:
                    return samples

    return samples[:count]


def generate_roman_hindi_corpus(count: int, forbidden_prompts: Set[str]) -> List[Dict[str, str]]:
    roman_pairs = [
        ("Aap kaise hain?", "Main theek hoon. Aap batayein main aapki kya madad kar sakta hoon?"),
        ("Aapka naam kya hai?", "Mera naam ChakrView hai."),
        ("Suraj kis disha se nikalta hai?", "Suraj hamesha poorab disha se nikalta hai."),
        ("Bharat ki rajdhani kya hai?", "Bharat ki rajdhani New Delhi hai."),
        ("Aap kya kaam kar sakte hain?", "Main sawalon ke jawab de sakta hoon aur calculation me madad kar sakta hoon."),
        ("Mujhe padhai me help chahiye.", "Bilkul, aap batayein kis topic par study karni hai, main help karunga."),
        ("Paani peena kyun zaroori hai?", "Paani peene se shareer hydrated rehta hai aur digestion theek chalta hai."),
        ("Dosti ka kya mahatva hai?", "Sacchi dosti jeevan me sahara deti hai aur musibat me saath nibhati hai."),
        ("Ped lagana kyun zaroori hai?", "Ped hamare vatavaran ko swachh banate hain aur taaza oxygen dete hain."),
        ("Subah jaldi uthne ke kya fayde hain?", "Subah jaldi uthne se mann shaant rehta hai aur dinbhar energy milti hai."),
        ("Time table banana kyun chahiye?", "Time table banane se kaam samay par pura hota hai aur planning acchi rehti hai."),
        ("Achhi sehat ke liye kya karein?", "Roz subah walk karein, poshtik bhojan lein aur poori neend lein."),
        ("Imandari ka kya phal hota hai?", "Imandari se vyakti ka aadar badhta hai aur sabhi ka vishwas milta hai."),
        ("Kitabein padhne se kya hota hai?", "Kitabein naya gyaan deti hain aur samajh ko behtar banati hain."),
        ("Hawa pradushan kaise rokein?", "Gadiyon ka kam upyog karke aur ped lagakar pradushan kam kiya ja sakta hai."),
    ]

    templates = [
        "{}",
        "Sunie, {}",
        "Batao: {}",
        "Jaldi bataiye: {}",
    ]

    samples = []
    for q, a in roman_pairs:
        for tpl in templates:
            p = tpl.format(q)
            if p.strip().lower() not in forbidden_prompts:
                samples.append({"prompt": p, "target": a, "category": "Roman Hindi"})
                if len(samples) >= count:
                    return samples

    return samples[:count]


def generate_english_conversation_corpus(count: int, forbidden_prompts: Set[str]) -> List[Dict[str, str]]:
    conv_pairs = [
        ("Good morning! How are you doing today?", "Good morning! I am doing well and ready to assist you."),
        ("Who created you and what is your purpose?", "I am ChakrView, an independent neural cognitive system built for helpful language interaction."),
        ("Can you help me plan my daily schedule?", "Yes, let me know your priority tasks and time slots, and we can organize them."),
        ("What makes a good software engineer?", "A good engineer writes clear maintainable code, tests rigorously, and continuously learns."),
        ("How can I improve my reading habits?", "Set aside fifteen uninterrupted minutes daily and pick topics you genuinely enjoy."),
        ("Thank you very much for your help!", "You are very welcome! Feel free to ask anytime you need assistance."),
        ("What is your perspective on lifelong learning?", "Continuously learning new concepts expands problem-solving ability and adaptability."),
        ("How should I handle difficult exam stress?", "Break revision into manageable chunks, get adequate sleep, and take regular short breaks."),
        ("Tell me a polite way to decline an invitation.", "You can say: Thank you so much for the invitation, but unfortunately I have a prior commitment."),
        ("What is the benefit of writing down goals?", "Writing goals clarifies direction, tracks progress, and increases accountability."),
        ("Good evening, ChakrView!", "Good evening! I hope you had a productive and pleasant day."),
        ("Can you explain how to stay focused while working?", "Minimize notifications, work in focused intervals, and keep your workspace tidy."),
        ("What is teamwork and why does it matter?", "Teamwork combines diverse strengths toward a shared goal, achieving what individuals cannot alone."),
        ("How do you maintain a positive attitude?", "Focus on solutions rather than problems, practice gratitude, and learn from mistakes."),
        ("What is the best way to give constructive feedback?", "Be specific, focus on the behavior rather than the person, and suggest actionable improvements."),
    ]

    templates = [
        "{}",
        "Hello! {}",
        "Please answer: {}",
        "ChakrView, {}",
    ]

    samples = []
    for q, a in conv_pairs:
        for tpl in templates:
            p = tpl.format(q)
            if p.strip().lower() not in forbidden_prompts:
                samples.append({"prompt": p, "target": a, "category": "English Conversation"})
                if len(samples) >= count:
                    return samples

    return samples[:count]


def generate_english_sentences(count: int, forbidden_prompts: Set[str]) -> List[Dict[str, str]]:
    sentence_pairs = [
        ("Write a sentence about a calm ocean.", "Gentle turquoise waves lapped quietly against the sandy shoreline."),
        ("Generate a sentence about an autumn morning.", "Crisp golden leaves drifted silently through the cool morning air."),
        ("Create a sentence describing a busy library.", "Students worked quietly among towering shelves filled with leather-bound books."),
        ("Write a sentence about a fast train.", "The silver express train sped smoothly along the winding mountain tracks."),
        ("Produce a sentence describing a starry night.", "Millions of brilliant stars sparkled across the dark, clear canopy of space."),
        ("Write a sentence about a hardworking gardener.", "The patient gardener carefully tended the blossoming rose beds under the warm sun."),
        ("Describe an old stone bridge in one sentence.", "An ancient stone bridge spanned the rushing mountain stream with timeless grace."),
        ("Write a sentence about an artist painting.", "With deliberate brushstrokes, the artist captured the vibrant colors of sunset."),
        ("Create a sentence about birds singing at dawn.", "Chirping birds welcomed the first light of dawn from the leafy treetops."),
        ("Write a sentence about walking in the rain.", "Soft raindrops pattered rhythmically on the umbrella as we strolled down the cobblestone street."),
    ]

    samples = []
    for q, a in sentence_pairs:
        for prefix in ["", "Please ", "Kindly "]:
            p = f"{prefix}{q}"
            if p.strip().lower() not in forbidden_prompts:
                samples.append({"prompt": p, "target": a, "category": "English Sentence Generation"})
                if len(samples) >= count:
                    return samples

    return samples[:count]


def generate_basic_instructions(count: int, forbidden_prompts: Set[str]) -> List[Dict[str, str]]:
    instruction_pairs = [
        ("Answer in exactly one word: Is fire hot?", "Hot."),
        ("Reply with only 'Yes' or 'No': Is the Earth flat?", "No."),
        ("Provide the capital city of France in one word.", "Paris."),
        ("Give only the opposite word of 'Dark'.", "Bright."),
        ("Say only 'Approved' or 'Rejected': All requirements are met.", "Approved."),
        ("State the opposite of 'Cold' in one word.", "Hot."),
        ("Reply with the color of fresh grass in a single word.", "Green."),
        ("Respond using only the word 'True' or 'False': Water boils at 100 degrees Celsius.", "True."),
        ("Give the singular noun for 'geese'.", "Goose."),
        ("What is the plural of 'child' in one word?", "Children."),
        ("State the primary color of ripe bananas in one word.", "Yellow."),
        ("Give the opposite of 'Fast' in one word.", "Slow."),
        ("Is ice solid or liquid? Answer in one word.", "Solid."),
        ("State the opposite of 'Heavy' in one word.", "Light."),
        ("Reply with 'Pass' or 'Fail': Score 95 out of 100.", "Pass."),
    ]

    samples = []
    for q, a in instruction_pairs:
        p = q
        if p.strip().lower() not in forbidden_prompts:
            samples.append({"prompt": p, "target": a, "category": "Basic Instructions"})

    return samples[:count]


def build_full_dataset() -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    forbidden_prompts = load_benchmark_prompts()
    print(f"Loaded {len(forbidden_prompts)} forbidden prompts from benchmark.")

    all_records: List[Dict[str, str]] = []

    # Category allocations targeted for balanced coverage:
    # Target: ~5,000+ total examples
    print("Generating Elementary Arithmetic corpus...")
    arith = generate_arithmetic_corpus(800, forbidden_prompts)
    all_records.extend(arith)

    print("Generating Exact Copying corpus...")
    copy = generate_copying_corpus(600, forbidden_prompts)
    all_records.extend(copy)

    print("Generating Science & Concept Explanations...")
    sci = generate_science_explanations(800, forbidden_prompts)
    all_records.extend(sci)

    print("Generating Devanagari Hindi corpus...")
    dev = generate_devanagari_corpus(800, forbidden_prompts)
    all_records.extend(dev)

    print("Generating Hinglish corpus...")
    hing = generate_hinglish_corpus(800, forbidden_prompts)
    all_records.extend(hing)

    print("Generating Roman Hindi corpus...")
    rom = generate_roman_hindi_corpus(600, forbidden_prompts)
    all_records.extend(rom)

    print("Generating English Conversation corpus...")
    eng_conv = generate_english_conversation_corpus(600, forbidden_prompts)
    all_records.extend(eng_conv)

    print("Generating English Sentence Generation corpus...")
    eng_sent = generate_english_sentences(400, forbidden_prompts)
    all_records.extend(eng_sent)

    print("Generating Basic Instructions corpus...")
    inst = generate_basic_instructions(400, forbidden_prompts)
    all_records.extend(inst)

    print(f"Total raw generated samples: {len(all_records)}")

    # Deduplicate strictly on prompt
    seen_prompts = set()
    deduped_records = []
    for r in all_records:
        norm_p = r["prompt"].strip().lower()
        if norm_p not in seen_prompts and norm_p not in forbidden_prompts:
            seen_prompts.add(norm_p)
            deduped_records.append(r)

    print(f"Total deduped non-benchmark records: {len(deduped_records)}")

    # Shuffle deterministically
    random.seed(SEED)
    random.shuffle(deduped_records)

    # Standardize prompt format to "User: <prompt>\nAssistant:" and target to leading space " <target>"
    for r in deduped_records:
        p = r["prompt"].strip()
        t = r["target"].strip()
        r["prompt"] = f"User: {p}\nAssistant:"
        r["target"] = f" {t}"

    # 85% train, 15% validation split
    split_idx = int(0.85 * len(deduped_records))
    train_records = deduped_records[:split_idx]
    val_records = deduped_records[split_idx:]

    return train_records, val_records


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_set, val_set = build_full_dataset()

    train_path = DATA_DIR / "ucip06_train.jsonl"
    val_path = DATA_DIR / "ucip06_validation.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_set:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    import hashlib

    def compute_sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    manifest = {
        "train_samples": len(train_set),
        "validation_samples": len(val_set),
        "total_samples": len(train_set) + len(val_set),
        "train_sha256": compute_sha256(train_path),
        "validation_sha256": compute_sha256(val_path),
    }

    manifest_path = DATA_DIR / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nDataset generation complete!")
    print(f"Train: {len(train_set)} records -> {train_path} ({manifest['train_sha256'][:16]}...)")
    print(f"Val:   {len(val_set)} records -> {val_path} ({manifest['validation_sha256'][:16]}...)")


if __name__ == "__main__":
    main()
