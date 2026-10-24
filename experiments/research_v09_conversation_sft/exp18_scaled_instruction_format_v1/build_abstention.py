import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.generate_candidate_banks_exp18 import check_gate, check_tokens

tokenizer, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096", verify_checksums=True)

# 15 distinct domains with diverse parameters
DOMAINS = [
    ("server_cluster", "Host node {name} operating at {metric1} percent CPU with {metric2} GB RAM active.", "CPU utilization", "Host node {name} is operating at {metric1} percent CPU.", "ambient intake temperature in Celsius", "The telemetry dispatch does not record the ambient temperature."),
    ("power_station", "Generator Unit-{name} discharged {metric1} cubic meters of cooling water per second into reservoir {metric2}.", "cooling water discharge rate", "Generator Unit-{name} discharged {metric1} cubic meters per second.", "electrical grid megawatt output", "The power station log does not state the megawatt output."),
    ("cargo_vessel", "Container vessel {name} loaded {metric1} standard freight containers at dock sector {metric2}.", "number of containers loaded", "Container vessel {name} loaded {metric1} standard freight containers.", "current bunker fuel oil reserve", "The harbor manifest does not mention the bunker fuel reserve."),
    ("weather_station", "Meteorological station {name} recorded wind gusts of {metric1} kilometers per hour from direction {metric2}.", "recorded wind gust speed", "Station {name} recorded gusts of {metric1} kilometers per hour.", "barometric pressure in millibars", "The meteorological report does not specify the barometric pressure."),
    ("chemical_reactor", "Batch reactor {name} maintained internal reaction pressure at {metric1} kilopascals during phase {metric2}.", "reaction pressure in kilopascals", "Batch reactor {name} maintained {metric1} kilopascals pressure.", "catalyst mass concentration in grams", "The reactor monitoring sheet does not contain the catalyst mass."),
    ("aviation_flight", "Transport aircraft {name} climbed to cruise altitude of {metric1} thousand feet over checkpoint {metric2}.", "assigned cruise altitude", "Aircraft {name} climbed to {metric1} thousand feet.", "total fuel consumption in gallons", "The flight log does not state the total fuel consumption."),
    ("transit_depot", "Rapid transit train {name} completed {metric1} passenger circuit runs along track corridor {metric2}.", "completed passenger circuit runs", "Train {name} completed {metric1} passenger circuit runs.", "traction motor operating temperature", "The depot maintenance report does not record the motor temperature."),
    ("geothermal_well", "Borehole well {name} tapped geothermal steam at {metric1} degrees Celsius in extraction zone {metric2}.", "steam temperature in Celsius", "Borehole well {name} tapped steam at {metric1} degrees Celsius.", "dissolved mineral salinity concentration", "The geothermal dispatch does not specify the mineral salinity."),
    ("water_treatment", "Filtration facility {name} processed {metric1} million liters of river inflow across basin {metric2}.", "volume of river inflow processed", "Facility {name} processed {metric1} million liters of inflow.", "residual chlorine concentration in parts per million", "The filtration report does not mention the chlorine concentration."),
    ("grain_elevator", "Agricultural terminal {name} unloaded {metric1} metric tons of winter barley into silo sector {metric2}.", "metric tons of winter barley unloaded", "Terminal {name} unloaded {metric1} metric tons of barley.", "grain internal moisture percentage", "The storage manifest does not state the grain moisture percentage."),
    ("mining_conveyor", "Haulage conveyor {name} transferred {metric1} tons of crushed granite ore to processing plant {metric2}.", "tons of granite ore transferred", "Conveyor {name} transferred {metric1} tons of granite ore.", "electric motor drive current in amperes", "The mining production record does not contain the motor drive current."),
    ("pipeline_station", "Pumping station {name} delivered {metric1} barrels of refined fuel daily through transmission line {metric2}.", "daily barrels of refined fuel delivered", "Station {name} delivered {metric1} barrels of fuel daily.", "suction flange static pressure", "The pipeline summary does not record the suction flange pressure."),
    ("optical_foundry", "Vacuum coating chamber {name} applied {metric1} nanometers of optical silica onto substrate batch {metric2}.", "thickness of optical silica applied", "Chamber {name} applied {metric1} nanometers of silica.", "chamber residual vacuum pressure in torrs", "The coating log does not specify the chamber vacuum pressure."),
    ("telecom_tower", "Cellular base station {name} routed {metric1} gigabytes of mobile data through transmitter array {metric2}.", "gigabytes of mobile data routed", "Base station {name} routed {metric1} gigabytes of data.", "backup battery state of charge percentage", "The station status log does not mention the backup battery charge."),
    ("foundry_furnace", "Electric arc furnace {name} poured {metric1} metric tons of molten structural alloy into ladle {metric2}.", "metric tons of structural alloy poured", "Furnace {name} poured {metric1} metric tons of alloy.", "measured slag viscosity rating", "The furnace melt record does not state the slag viscosity rating."),
]

NAMES = [
    "Orion", "Pegasus", "Cygnus", "Vela", "Centauri", "Aquila", "Cassiopeia", "Andromeda", "Perseus", "Auriga",
    "Taurus", "Hydra", "Lynx", "Phoenix", "Carina", "Draco", "Cepheus", "Corvus", "Fornax", "Lupus",
    "Volans", "Pavo", "Tucana", "Indus", "Columba", "Norma", "Octans", "Pyxis", "Antlia", "Circinus",
    "Scutum", "Sagitta", "Pictor", "Caelum", "Musca", "Mensa", "Reticulum", "Microscopium", "Telescopium", "Sculptor",
    "Fenix", "Polaris", "Sirius", "Arcturus", "Vega", "Rigel", "Procyon", "Betelgeuse", "Altair", "Aldebaran",
    "Spica", "Antares", "Pollux", "Deneb", "Regulus", "Castor", "Gacrux", "Bellatrix", "Elnath", "Miaplacidus",
    "Alnilam", "Alnitak", "Alioth", "Dubhe", "Mirfak", "Wezen", "Sargas", "Kaus", "Avior", "Alkaid",
    "Menkalinan", "Atria", "Alhena", "Achernar", "Canopus", "Capella", "Shaula", "Alcor", "Alphecca", "Mintaka",
    "Saiph", "Algol", "Diphda", "Hamal", "Kochab", "Rasalhague", "Algieba", "Mizar", "Nunki", "Mirzam",
    "Adhara", "Alphard", "Enif", "Ankaa", "Markab", "Scheat", "Algenib", "Zaurak", "Caph", "Segin",
    "Navi", "Alsephina", "Suhail", "Muhlifain", "Aspidiske", "Grumium", "Rastaban", "Eltanin", "Kornephoros", "Yed",
    "Sabik", "Cebalrai", "Kraz", "Gienah", "Porrima", "Vindemiatrix", "Zubenelgenubi", "Zubeneschamali", "Brachium", "Dschubba",
    "Graffias", "Fang", "Acrab", "Al Niyat", "Lesath", "Kappe", "Alnasl", "Polis", "Kaus Media", "Ascella",
    "Arkab", "Albaldah", "Nashira", "Deneb Algedi", "Dabih", "Algedi", "Sadalmelik", "Sadalsuud", "Sadachbia", "Skat",
    "Biham", "Homam", "Matar", "Baham", "Salm", "Alrescha", "Torcular", "Botein", "Sheratan", "Mesarthim",
    "Zaurac", "Beid", "Keid", "Angetenar", "Azha", "Acamar", "Theemin", "Sceptrum", "Zibal", "Mira"
]

print(f"Names count: {len(NAMES)}")
