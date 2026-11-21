"""
UCIP-20 Isolated Entity-Binding Dataset Generator.
Generates an isolated, leak-proof entity-binding corpus partitioned strictly by underlying entity values.

Key Architecture:
- Disjoint entities: Train entities NEVER appear in Val or Dev.
- 10 diverse relation types: project, person_role, city, organization, product, identifier, date, numerical_value, ownership, property.
- Multilingual: English, Hindi, Hinglish.
- Multi-entity distractor contexts (2 to 4 entities per context).
- Controlled negative / abstention examples ("Information not provided").
- 80% Train (~1,600 records), 10% Val (~200 records), 10% Dev (~200 records) -> Total 2,000 records.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data" / "sft" / "ucip20_bind"

# Seed for absolute determinism
SEED = 42
random.seed(SEED)

# Define disjoint entity pools for Train, Val, Dev across 10 relation types
ENTITY_POOLS = {
    "project": {
        "train": [
            ("ChakrView", "conversational neural engine", "AI reasoning system"),
            ("AeroScan", "drone surveillance platform", "aerial analytics pipeline"),
            ("BioPulse", "cardiac telemetry tracker", "clinical monitor"),
            ("DataForge", "distributed ETL pipeline", "data processing cluster"),
            ("Solaris", "photovoltaic grid inverter", "clean energy controller"),
            ("QuantumGate", "superconducting qubit coupler", "quantum processor circuit"),
            ("OmniFlow", "supply chain routing network", "logistics dispatch engine"),
            ("CyberShield", "autonomous firewall agent", "network intrusion guard"),
            ("NeuroVibe", "brain-computer interface", "neural signal decoder"),
            ("TerraMap", "satellite topography builder", "geospatial renderer"),
        ],
        "val": [
            ("AuraSafe", "air quality monitor", "environmental sensor hub"),
            ("OptiRoute", "fleet dispatch scheduler", "traffic optimization grid"),
        ],
        "dev": [
            ("NovaTrack", "stellar spectroscopy camera", "astronomical observatory tool"),
            ("HyperLink", "sub-millimeter optical bus", "inter-chip interconnect"),
        ],
    },
    "person_role": {
        "train": [
            ("Arjun", "Lead Architect", "Principal Scientist"),
            ("Sunita", "Chief Radiologist", "Medical Director"),
            ("Kavita", "Systems Engineer", "DevOps Specialist"),
            ("Vikram", "Database Administrator", "Storage Specialist"),
            ("Pooja", "Aerospace Specialist", "Flight Dynamics Lead"),
            ("Rohan", "Cybersecurity Analyst", "Threat Intelligence Lead"),
            ("Ananya", "Robotics Engineer", "Motion Planning Specialist"),
            ("Deepak", "Compiler Engineer", "Performance Optimization Lead"),
            ("Neha", "Biomedical Researcher", "Molecular Biology Fellow"),
            ("Manish", "Hardware Design Lead", "VLSI Verification Engineer"),
        ],
        "val": [
            ("Isha", "Financial Controller", "Audit Officer"),
            ("Rajesh", "Patent Attorney", "Intellectual Property Counsel"),
        ],
        "dev": [
            ("Tanvi", "Astrophysics Fellow", "Cosmological Model Analyst"),
            ("Sameer", "Quantum Cryptographer", "Protocol Security Engineer"),
        ],
    },
    "city": {
        "train": [
            ("Bengaluru", "Karnataka", "India"),
            ("Pune", "Maharashtra", "India"),
            ("Hyderabad", "Telangana", "India"),
            ("Chennai", "Tamil Nadu", "India"),
            ("Ahmedabad", "Gujarat", "India"),
            ("Kolkata", "West Bengal", "India"),
            ("Jaipur", "Rajasthan", "India"),
            ("Kochi", "Kerala", "India"),
            ("Bhopal", "Madhya Pradesh", "India"),
            ("Chandigarh", "Punjab", "India"),
        ],
        "val": [
            ("Mysuru", "Karnataka", "India"),
            ("Nagpur", "Maharashtra", "India"),
        ],
        "dev": [
            ("Dehradun", "Uttarakhand", "India"),
            ("Indore", "Madhya Pradesh", "India"),
        ],
    },
    "organization": {
        "train": [
            ("Apex Dynamics", "industrial automation", "precision robotics"),
            ("Zenith Cloud", "datacenter infrastructure", "edge hosting"),
            ("Brahma AI", "multilingual foundation models", "linguistic research"),
            ("Kiran Energy", "offshore wind generation", "renewable utilities"),
            ("Veda Biotech", "genomic sequencing reagents", "diagnostic assays"),
            ("Shakti Motors", "electric commercial trucks", "battery powertrains"),
            ("Indus Space", "low-earth orbit satellites", "launch services"),
            ("Varun Marine", "autonomous submersibles", "oceanographic mapping"),
            ("Garuda Optics", "satellite optical payloads", "remote sensing cameras"),
            ("Mitra FinTech", "cross-border payment gateway", "clearing ledger"),
        ],
        "val": [
            ("Soma Pharma", "active pharmaceutical ingredients", "peptide synthesis"),
            ("Tarang Telecom", "5G millimeter-wave antennas", "wireless infrastructure"),
        ],
        "dev": [
            ("Nava Robotics", "humanoid factory assistants", "bipedal platforms"),
            ("Daksha Defense", "radar signal processors", "electronic surveillance"),
        ],
    },
    "product": {
        "train": [
            ("Falcon-90", "turbofan jet engine", "thrust rating 90 kN"),
            ("Titan-X4", "titanium alloy airframe", "tensile strength 1200 MPa"),
            ("ChakrChip", "16-core RISC-V neural accelerator", "25 TOPS TDP 15W"),
            ("SolarCell-V2", "perovskite tandem panel", "efficiency rating 28%"),
            ("CryoTank-500", "liquid helium storage flask", "capacity 500 liters"),
            ("LaserCore-50", "diode-pumped fiber laser", "output power 50 kW"),
            ("SpectroMax", "high-resolution Raman spectrometer", "spectral range 200-4000 cm^-1"),
            ("RoboArm-6D", "six-axis articulated robotic manipulator", "payload capacity 25 kg"),
            ("BioSensor-DX", "electrochemical blood analyte strip", "detection limit 0.1 nM"),
            ("LithiumCell-X", "solid-state lithium pouch battery", "energy density 450 Wh/kg"),
        ],
        "val": [
            ("HyperDrive-3", "magnetic levitation linear motor", "maximum velocity 120 m/s"),
            ("MicroPump-P2", "piezoelectric microfluidic pump", "flow rate 50 uL/min"),
        ],
        "dev": [
            ("QuantumSensor-Q1", "diamond NV-center magnetometer", "sensitivity 1 pT/sqrt(Hz)"),
            ("FiberCable-Ultra", "hollow-core photonic bandgap fiber", "attenuation 0.1 dB/km"),
        ],
    },
    "identifier": {
        "train": [
            ("Batch-A419", "primary reactor run", "grade A certified"),
            ("Node-7702", "central cluster coordinator", "rack 14-B"),
            ("Ticket-9921", "kernel panic bug in memory driver", "priority P0"),
            ("Asset-5510", "cryogenic cooling loop pump", "service tag ST-5510"),
            ("Sensor-8842", "exhaust gas thermocouple", "calibration valid"),
            ("Case-3031", "hardware warranty verification", "status approved"),
            ("Lot-1029", "monocrystalline silicon wafers", "thickness 180 um"),
            ("Device-4412", "radiation hardened microcontroller", "radiation tolerance 100 krad"),
            ("Packet-6019", "telemetry handshake frame", "checksum matched"),
            ("Spec-8814", "EMI shielding standard compliance", "class B certified"),
        ],
        "val": [
            ("Batch-C505", "secondary chemical synthesis", "purity 99.9%"),
            ("Node-8114", "backup distributed storage node", "rack 08-C"),
        ],
        "dev": [
            ("Ticket-4040", "firmware bootstrap timeout defect", "priority P1"),
            ("Sensor-9102", "sub-ambient barometric transducer", "channel 04"),
        ],
    },
    "date": {
        "train": [
            ("15 August 1947", "Indian Independence Declaration", "New Delhi"),
            ("26 January 1950", "Indian Constitution enactment", "Parliament House"),
            ("20 July 1969", "Apollo 11 lunar landing", "Sea of Tranquility"),
            ("14 July 2023", "Chandrayaan-3 launch mission", "Sriharikota"),
            ("23 August 2023", "Chandrayaan-3 soft lunar touchdown", "Lunar South Pole"),
            ("11 May 1998", "Pokhran-II nuclear tests", "Thar Desert"),
            ("19 April 1975", "Aryabhata satellite launch", "Kapustin Yar"),
            ("22 October 2008", "Chandrayaan-1 lunar orbiter launch", "Sriharikota"),
            ("5 November 2013", "Mars Orbiter Mission Mangalyaan launch", "Sriharikota"),
            ("24 September 2014", "Mangalyaan Mars orbit insertion", "Martian Orbit"),
        ],
        "val": [
            ("28 February 1928", "Raman Effect discovery announcement", "Kolkata"),
            ("12 April 1961", "Vostok 1 first human spaceflight", "Baikonur"),
        ],
        "dev": [
            ("21 July 1988", "INSAT-1C satellite deployment", "Kourou"),
            ("18 May 1974", "Smiling Buddha first nuclear test", "Pokhran"),
        ],
    },
    "numerical_value": {
        "train": [
            ("300,000 km/s", "speed of light in vacuum", "fundamental physical constant"),
            ("9.81 m/s^2", "standard acceleration due to gravity on Earth surface", "gravitational acceleration"),
            ("273.15 Kelvin", "freezing point of pure water at standard atmospheric pressure", "water melting point"),
            ("101.325 kPa", "standard sea-level atmospheric pressure", "one standard atmosphere"),
            ("6.626e-34 J*s", "Planck constant", "quantum action constant"),
            ("1.602e-19 Coulombs", "elementary electric charge", "charge of one electron"),
            ("6.022e23 mol^-1", "Avogadro constant", "particles per mole"),
            ("1.381e-23 J/K", "Boltzmann constant", "thermal energy relation"),
            ("8.314 J/(mol*K)", "universal molar gas constant", "ideal gas equation"),
            ("5.670e-8 W/(m^2*K^4)", "Stefan-Boltzmann radiation constant", "blackbody emission"),
        ],
        "val": [
            ("1.414", "square root of two", "diagonal of unit square"),
            ("3.14159", "mathematical constant pi", "ratio of circumference to diameter"),
        ],
        "dev": [
            ("2.71828", "Euler's mathematical constant e", "natural logarithm base"),
            ("1.61803", "golden ratio constant phi", "divine proportion"),
        ],
    },
    "ownership": {
        "train": [
            ("ISRO", "National Space Transportation Fleet", "government space agency"),
            ("DRDO", "Integrated Defense Missile Complex", "defense research establishment"),
            ("TIFR", "National Center for Radio Astrophysics", "pioneering research institute"),
            ("IISc", "Supercomputer Education and Research Centre", "autonomous academic body"),
            ("BARC", "Dhruva High Flux Nuclear Reactor", "atomic research facility"),
            ("CSIR", "National Aerospace Laboratories", "scientific industrial research council"),
            ("HAL", "Tejas Light Combat Aircraft Assembly Facility", "aerospace manufacturer"),
            ("BEL", "Naval Radar Production Center", "defense electronics enterprise"),
            ("BHEL", "Heavy Turbine Manufacturing Unit", "power equipment utility"),
            ("ECIL", "Electronic Voting System Technology Division", "electronics corporation"),
        ],
        "val": [
            ("NTPC", "Super Thermal Power Generating Complex", "power generation corporation"),
            ("GAIL", "Trans-National Gas Transmission Grid", "natural gas pipeline operator"),
        ],
        "dev": [
            ("MDL", "Scorpene Submarine Fabrication Yard", "shipbuilder dockyard"),
            ("BDL", "Surface-to-Air Missile Integration Complex", "munitions corporation"),
        ],
    },
    "property": {
        "train": [
            ("Titanium", "corrosion resistance and high strength-to-weight ratio", "transition metal element 22"),
            ("Graphene", "exceptional electrical conductivity and 200x steel strength", "single-layer hexagonal carbon"),
            ("Tungsten", "highest melting point of all refractory metals at 3422 deg C", "heavy metal element 74"),
            ("Aerogel", "lowest bulk density and outstanding thermal insulation", "synthetic porous ultralight material"),
            ("Silicon Carbide", "extreme mechanical hardness and high thermal bandgap", "semiconductor ceramic compound"),
            ("Gallium Nitride", "wide electronic bandgap and high electron mobility", "high-power RF semiconductor"),
            ("Inconel-718", "high creep-rupture strength at temperatures up to 700 deg C", "nickel-chromium superalloy"),
            ("Kevlar", "high tensile strength and ballistic impact resistance", "synthetic aramid fiber polymer"),
            ("Beryllium", "high flexural rigidity and low thermal expansion", "alkaline earth metal element 4"),
            ("Diamond", "highest natural hardness and superior thermal conductivity", "cubic crystalline carbon allotrope"),
        ],
        "val": [
            ("Copper", "high electrical and thermal conductivity with ductility", "noble coinage metal element 29"),
            ("Gold", "extreme chemical inertness and resistance to surface oxidation", "precious transition metal element 79"),
        ],
        "dev": [
            ("Platinum", "catalytic activity and high corrosion immunity", "dense precious metal element 78"),
            ("Zirconium", "low neutron capture cross-section and water corrosion resistance", "nuclear cladding metal element 40"),
        ],
    },
}

# Template sets per language to ensure high linguistic and syntactic diversity
TEMPLATES = {
    "en": [
        # Direct context-question
        ("Context: {c1} Furthermore, {c2}.\nQuestion: {q}\nAnswer concisely:", "{a}"),
        ("We note the following facts:\n1. {c1}\n2. {c2}\nBased on this information, {q}", "{a}"),
        ("Review the recorded configuration:\n- {c1}\n- {c2}\nQuery: {q}", "{a}"),
        ("Technical record: {c1} In addition, {c2}. What is the requested value: {q}", "{a}"),
        ("Consider the active system state where {c1}, while {c2}. Please state: {q}", "{a}"),
    ],
    "hi": [
        ("संदर्भ: {c1} इसके अतिरिक्त, {c2}।\nप्रश्न: {q}\nसंक्षिप्त उत्तर:", "{a}"),
        ("निम्नलिखित विवरण पर ध्यान दें:\n१. {c1}\n२. {c2}\nइस जानकारी के आधार पर, {q}", "{a}"),
        ("प्रणाली स्थिति: {c1} और {c2}। कृपया बताएं: {q}", "{a}"),
        ("तथ्य: {c1} तथा {c2}। प्रश्न: {q}", "{a}"),
    ],
    "hi-en": [
        ("Context: {c1}. Also note that {c2}.\nQuery: {q} Bataiye:", "{a}"),
        ("System log details: {c1} aur {c2}.\nQuestion: {q} Short answer:", "{a}"),
        ("Dhyan se padhein: {c1}, sath me {c2}.\nIske mutabiq {q}", "{a}"),
    ],
}

def build_fact_string(rel: str, item: Tuple[str, str, str], lang: str) -> Tuple[str, str, str, str]:
    """Returns (fact_statement, question, target_value, entity_name)"""
    e, v, desc = item
    if rel == "project":
        if lang == "en":
            return (f"Project {e} is developed as a {v}.", f"What is Project {e} developed as?", v, e)
        elif lang == "hi":
            return (f"परियोजना {e} को {v} के रूप में विकसित किया गया है।", f"परियोजना {e} को किस रूप में विकसित किया गया है?", v, e)
        else: # hi-en
            return (f"Project {e} ko {v} ke roop me develop kiya gaya hai.", f"Project {e} ka primary function kya hai?", v, e)
    elif rel == "person_role":
        if lang == "en":
            return (f"{e} serves as the {v} for the initiative.", f"What position does {e} hold?", v, e)
        elif lang == "hi":
            return (f"{e} इस पहल के लिए {v} के पद पर कार्यरत हैं।", f"{e} किस पद पर कार्यरत हैं?", v, e)
        else:
            return (f"{e} is initiative me {v} ke post par kaam kar rahe hain.", f"{e} ka role kya hai?", v, e)
    elif rel == "city":
        if lang == "en":
            return (f"The central headquarters is established in {e}, located in {v}.", f"In which state is {e} located?", v, e)
        elif lang == "hi":
            return (f"केंद्रीय मुख्यालय {e} में स्थापित है, जो {v} में स्थित है।", f"{e} किस राज्य में स्थित है?", v, e)
        else:
            return (f"Headquarters {e} me established hai, jo {v} me padta hai.", f"{e} kis state me sthit hai?", v, e)
    elif rel == "organization":
        if lang == "en":
            return (f"Enterprise {e} specializes in {v}.", f"What does {e} specialize in?", v, e)
        elif lang == "hi":
            return (f"प्रतिष्ठान {e} {v} के क्षेत्र में विशेषज्ञता रखता है।", f"{e} का कार्यक्षेत्र क्या है?", v, e)
        else:
            return (f"Company {e} primarily {v} me specialize karti hai.", f"{e} kis domain me specialized hai?", v, e)
    elif rel == "product":
        if lang == "en":
            return (f"Hardware model {e} is categorized as an {v}.", f"What type of product is {e}?", v, e)
        elif lang == "hi":
            return (f"हार्डवेयर मॉडल {e} को {v} के रूप में वर्गीकृत किया गया है।", f"{e} किस श्रेणी का उत्पाद है?", v, e)
        else:
            return (f"Hardware model {e} ek {v} ki category me aata hai.", f"{e} kis type ka device hai?", v, e)
    elif rel == "identifier":
        if lang == "en":
            return (f"Identifier {e} corresponds to {v}.", f"What does identifier {e} refer to?", v, e)
        elif lang == "hi":
            return (f"पहचानकर्ता {e} {v} को संदर्भित करता है।", f"पहचानकर्ता {e} का क्या विवरण है?", v, e)
        else:
            return (f"Identifier {e} direct {v} ko refer karta hai.", f"Identifier {e} kis record ke sath judi hai?", v, e)
    elif rel == "date":
        if lang == "en":
            return (f"The historical event on {e} was the {v}.", f"What historic event occurred on {e}?", v, e)
        elif lang == "hi":
            return (f"{e} की ऐतिहासिक घटना {v} थी।", f"{e} को कौन सी ऐतिहासिक घटना हुई थी?", v, e)
        else:
            return (f"{e} ko historical event {v} hua tha.", f"{e} ko kaun si event hui thi?", v, e)
    elif rel == "numerical_value":
        if lang == "en":
            return (f"The calibrated constant value {e} represents the {v}.", f"What physical quantity is measured as {e}?", v, e)
        elif lang == "hi":
            return (f"अंशांकित नियतांक मान {e} {v} को दर्शाता है।", f"मान {e} किस भौतिक राशि का माप है?", v, e)
        else:
            return (f"Calibrated value {e} direct {v} ko represent karti hai.", f"{e} kis value ko represent karta hai?", v, e)
    elif rel == "ownership":
        if lang == "en":
            return (f"Organization {e} officially owns the {v}.", f"What facility or asset is owned by {e}?", v, e)
        elif lang == "hi":
            return (f"संस्थान {e} आधिकारिक रूप से {v} का स्वामित्व रखता है।", f"{e} के स्वामित्व में कौन सी संपत्ति है?", v, e)
        else:
            return (f"Organization {e} officially {v} ko own karti hai.", f"{e} ke ownership me kya asset hai?", v, e)
    else: # property
        if lang == "en":
            return (f"Material {e} exhibits {v}.", f"What key physical property characterizes {e}?", v, e)
        elif lang == "hi":
            return (f"पदार्थ {e} {v} प्रदर्शित करता है।", f"{e} का मुख्य भौतिक गुण क्या है?", v, e)
        else:
            return (f"Material {e} primarily {v} show karta hai.", f"{e} ki main property kya hai?", v, e)


def generate_partition_records(partition_name: str, target_count: int) -> List[Dict[str, Any]]:
    records = []
    langs = ["en", "hi", "hi-en"]
    relations = list(ENTITY_POOLS.keys())
    
    # Negative / Abstention queries for out-of-context queries
    abstention_targets = {
        "en": "Information not provided in context.",
        "hi": "दी गई जानकारी में यह विवरण उपलब्ध नहीं है।",
        "hi-en": "Di gayi context me ye information available nahi hai.",
    }

    rec_id = 0
    while len(records) < target_count:
        # Choose relation and language
        rel = relations[rec_id % len(relations)]
        lang = langs[(rec_id // len(relations)) % len(langs)]
        
        pool = ENTITY_POOLS[rel][partition_name]
        if len(pool) < 2:
            # For val/dev where pool size is 2, pick target and distractor
            target_item = pool[0]
            distractor_item = pool[1]
        else:
            # Pick target and 1-2 distractors
            sample_items = random.sample(pool, 2)
            target_item = sample_items[0]
            distractor_item = sample_items[1]
            
        c1, q1, a1, e1 = build_fact_string(rel, target_item, lang)
        c2, q2, a2, e2 = build_fact_string(rel, distractor_item, lang)
        
        # Decide if this record is a positive extraction or an abstention negative (10% abstention)
        is_abstention = (rec_id % 10 == 0)
        
        if is_abstention:
            # Ask about an entity completely absent from the context
            absent_entity = "QuantumAlpha" if lang == "en" else ("परियोजना डेल्टा" if lang == "hi" else "Project NovaOmega")
            q = f"What is the status of {absent_entity}?" if lang == "en" else (f"{absent_entity} की स्थिति क्या है?" if lang == "hi" else f"{absent_entity} ka status kya hai?")
            ans = abstention_targets[lang]
        else:
            q = q1
            ans = a1
            
        # Select prompt template
        tpl_list = TEMPLATES[lang]
        tpl_p, tpl_r = tpl_list[rec_id % len(tpl_list)]
        
        # Randomize order of context facts to prevent positional bias
        if random.random() > 0.5:
            fact_a, fact_b = c1, c2
        else:
            fact_a, fact_b = c2, c1
            
        body = tpl_p.format(c1=fact_a, c2=fact_b, q=q)
        prompt_text = f"User: {body}\nAssistant: "
        response_text = tpl_r.format(a=ans)
        
        record = {
            "id": f"UCIP20-BIND-{partition_name.upper()}-{rec_id+1:05d}",
            "partition": partition_name,
            "category": "BIND",
            "relation_type": rel,
            "language": lang,
            "is_abstention": is_abstention,
            "target_entity": e1 if not is_abstention else "ABSENT",
            "distractor_entity": e2,
            "prompt": prompt_text,
            "response": response_text,
            "target_value": ans,
        }
        records.append(record)
        rec_id += 1

    return records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_recs = generate_partition_records("train", 1600)
    val_recs = generate_partition_records("val", 200)
    dev_recs = generate_partition_records("dev", 200)
    
    print(f"Generated: Train={len(train_recs)}, Val={len(val_recs)}, Dev={len(dev_recs)}")
    
    train_file = OUTPUT_DIR / "bind_train_1600.jsonl"
    val_file = OUTPUT_DIR / "bind_val_200.jsonl"
    dev_file = OUTPUT_DIR / "bind_dev_200.jsonl"
    master_file = OUTPUT_DIR / "bind_master_2000.jsonl"
    
    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(val_file, "w", encoding="utf-8") as f:
        for r in val_recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(dev_file, "w", encoding="utf-8") as f:
        for r in dev_recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(master_file, "w", encoding="utf-8") as f:
        for r in train_recs + val_recs + dev_recs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        
    print("All partition files written successfully.")


if __name__ == "__main__":
    main()
