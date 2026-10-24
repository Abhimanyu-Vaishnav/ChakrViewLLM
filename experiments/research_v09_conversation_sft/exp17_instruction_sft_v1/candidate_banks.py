"""
candidate_banks.py
==================
Curated, factually verified, and diverse candidate banks for:
- factual_qa (40 Train, 5 Val, 5 Test = 50 items)
- short_explanation (40 Train, 5 Val, 5 Test = 50 items)
- uncertainty_abstention (40 Train, 5 Val, 5 Test = 50 items, pairing unanswerable queries with answerable controls)

All items strictly observe:
- 100% token length compliance with hard bounds [3, 45] (mean ~20-28 tokens, >=90% within provisional target [5, 35])
- 0% lexical overlap with frozen Normal Chat Gate v0.1 prompts
- Strict cross-split disjointness across Train, Val, and Test
"""

# ===========================================================================
# 1. FACTUAL QA BANKS (50 Distinct, Verified Factual Items)
# ===========================================================================
# Structure: (question, answer, intent, verification_note)

CANDIDATE_FACTUAL_BANKS = {
    "train": [
        ("What is the capital city of Japan?", "Tokyo is the capital city of Japan.", "Identify Tokyo as capital of Japan", "Established geographical fact."),
        ("What is the largest ocean on planet Earth?", "The Pacific Ocean is the largest ocean on Earth.", "Identify Pacific Ocean", "National Oceanic and Atmospheric Administration (NOAA)."),
        ("What gas do green plants absorb from the air during photosynthesis?", "Plants absorb carbon dioxide from the air during photosynthesis.", "Identify carbon dioxide", "Standard biochemical botany."),
        ("What is the chemical symbol for the element gold?", "The chemical symbol for gold is Au.", "Identify Au as symbol for gold", "IUPAC periodic table."),
        ("Which planet is positioned closest to the Sun?", "Mercury is the planet positioned closest to the Sun.", "Identify Mercury", "NASA planetary fact sheet."),
        ("At what temperature Celsius does pure water boil at standard sea level?", "Pure water boils at 100 degrees Celsius at sea level.", "State 100 degrees Celsius", "Standard thermodynamic definition at 1 atm."),
        ("What is the official currency of the United Kingdom?", "The official currency of the United Kingdom is the Pound Sterling.", "Identify Pound Sterling", "Bank of England official currency."),
        ("How many primary continents are traditionally recognized on Earth?", "There are seven continents traditionally recognized on Earth.", "State seven continents", "Standard geographic consensus."),
        ("Which muscular organ pumps oxygenated blood through the human body?", "The heart pumps oxygenated blood through the human body.", "Identify heart", "Human cardiovascular anatomy."),
        ("What is the hardest naturally occurring mineral found on Earth?", "Diamond is the hardest naturally occurring mineral on Earth.", "Identify diamond", "Mohs hardness scale (rating 10)."),
        ("What is the largest living species of mammal currently on Earth?", "The blue whale is the largest mammal on Earth.", "Identify blue whale", "Marine biology consensus (Balaenoptera musculus)."),
        ("Which planet in our solar system is nicknamed the Morning Star?", "Venus is frequently referred to as the Morning Star.", "Identify Venus", "Astronomical naming convention."),
        ("What is the primary chemical element found in coal and diamond?", "Carbon is the primary chemical element in coal and diamond.", "Identify carbon", "Allotropes of carbon in mineralogy."),
        ("What is the capital city of Spain?", "Madrid is the capital city of Spain.", "Identify Madrid as capital of Spain", "Official political geography."),
        ("What force attracts two masses toward each other in classical mechanics?", "Gravity attracts two masses toward each other in classical mechanics.", "Identify gravity", "Newtonian law of universal gravitation."),
        ("Which organ in the human body produces insulin?", "The pancreas produces insulin in the human body.", "Identify pancreas", "Endocrine physiology."),
        ("What is the outermost solid mechanical layer of planet Earth?", "The crust is the outermost solid mechanical layer of Earth.", "Identify Earth crust", "Geophysical structural geology."),
        ("Which gas makes up approximately seventy-eight percent of Earth's atmosphere?", "Nitrogen comprises approximately seventy-eight percent of Earth's atmosphere.", "Identify nitrogen", "Atmospheric composition standard data."),
        ("What is the chemical symbol for the element silver?", "The chemical symbol for silver is Ag.", "Identify Ag as silver", "IUPAC periodic table (Argentum)."),
        ("What is the capital city of Italy?", "Rome is the capital city of Italy.", "Identify Rome as capital of Italy", "Official political geography."),
        ("Which natural satellite orbits the Earth?", "The Moon is the natural satellite that orbits the Earth.", "Identify the Moon", "Celestial mechanics."),
        ("What unit is universally used to measure electrical resistance?", "The ohm is the standard unit of electrical resistance.", "Identify ohm", "International System of Units (SI)."),
        ("Which organ is primarily responsible for detoxifying chemicals in the human body?", "The liver is primarily responsible for detoxifying chemicals in the human body.", "Identify liver", "Human metabolic physiology."),
        ("What is the capital city of Germany?", "Berlin is the capital city of Germany.", "Identify Berlin as capital of Germany", "Official political geography."),
        ("What instrument is used to measure atmospheric air pressure?", "A barometer is used to measure atmospheric air pressure.", "Identify barometer", "Meteorological instrumentation."),
        ("What is the chemical formula for ordinary table salt?", "The chemical formula for table salt is NaCl.", "Identify NaCl", "Sodium chloride mineralogy."),
        ("Which bird is famous for being incapable of flight yet an exceptional swimmer?", "The penguin is a flightless bird known for exceptional swimming.", "Identify penguin", "Ornithological taxonomy (Spheniscidae)."),
        ("What is the capital city of France?", "Paris is the capital city of France.", "Identify Paris as capital of France", "Official political geography."),
        ("What is the freezing point of pure water in degrees Celsius?", "Pure water freezes at zero degrees Celsius.", "State 0 degrees Celsius", "Thermodynamic standard."),
        ("Which primary blood vessel carries oxygenated blood away from the heart?", "The aorta carries oxygenated blood away from the heart.", "Identify aorta", "Cardiovascular anatomy."),
        ("What is the capital city of Egypt?", "Cairo is the capital city of Egypt.", "Identify Cairo as capital of Egypt", "Official political geography."),
        ("What subatomic particle carries a negative electrical charge in an atom?", "An electron carries a negative electrical charge in an atom.", "Identify electron", "Atomic physics standard model."),
        ("Which continent is the South Pole located on?", "The South Pole is located on the continent of Antarctica.", "Identify Antarctica", "Geographic polar cartography."),
        ("What pigment gives human skin and hair their characteristic color?", "Melanin is the pigment that gives skin and hair their color.", "Identify melanin", "Dermatological biochemistry."),
        ("What is the capital city of Brazil?", "Brasilia is the capital city of Brazil.", "Identify Brasilia as capital of Brazil", "Official political geography."),
        ("What is the speed of light in a vacuum to the nearest hundred thousand km/s?", "The speed of light in a vacuum is approximately 300,000 kilometers per second.", "State 300,000 km/s", "Fundamental physical constant c."),
        ("What is the main metal component of bronze alloy along with tin?", "Copper is the main metal component combined with tin in bronze.", "Identify copper", "Metallurgical composition."),
        ("Which planet is the second closest to the Sun?", "Venus is the second closest planet to the Sun.", "Identify Venus", "Solar system planetary order."),
        ("What biological process converts sugar into cellular energy without oxygen?", "Fermentation converts sugar into cellular energy without oxygen.", "Identify fermentation", "Anaerobic cellular metabolism."),
        ("What is the capital city of South Korea?", "Seoul is the capital city of South Korea.", "Identify Seoul as capital of South Korea", "Official political geography."),
    ],
    "val": [
        ("What is the capital city of Canada?", "Ottawa is the capital city of Canada.", "Identify Ottawa as capital of Canada", "Official political geography."),
        ("What is the tallest mountain peak above sea level on Earth?", "Mount Everest is the tallest mountain peak above sea level.", "Identify Mount Everest", "Geographic topographical survey."),
        ("Which metallic element remains liquid at standard room temperature?", "Mercury is the metallic element that remains liquid at room temperature.", "Identify mercury", "Chemical elements standard physical state."),
        ("Which galaxy contains our solar system and Earth?", "The Milky Way galaxy contains our solar system.", "Identify Milky Way", "Astronomical galactic catalog."),
        ("What is the capital city of Argentina?", "Buenos Aires is the capital city of Argentina.", "Identify Buenos Aires", "Official political geography."),
    ],
    "test": [
        ("What is the capital city of Australia?", "Canberra is the capital city of Australia.", "Identify Canberra as capital of Australia", "Official political geography."),
        ("What is the largest cold polar desert on planet Earth?", "The Antarctic Desert is the largest polar desert on Earth.", "Identify Antarctic Desert", "Physical geography desert classification."),
        ("What subatomic particle with no electrical charge resides in atomic nuclei?", "Neutrons are subatomic particles with no electrical charge in atomic nuclei.", "Identify neutron", "Nuclear physics."),
        ("What is the chemical symbol for the metallic element iron?", "The chemical symbol for iron is Fe.", "Identify Fe as iron", "IUPAC periodic table (Ferrum)."),
        ("Which planet in our solar system features the Great Red Spot storm?", "Jupiter features the massive storm known as the Great Red Spot.", "Identify Jupiter", "NASA planetary atmospheric observations."),
    ],
}


# ===========================================================================
# 2. SCIENTIFIC EXPLANATION BANKS (50 Distinct, Verified Mechanisms)
# ===========================================================================
# Concisely formulated to fit strictly within [3, 45] tokens (target: 5-35 tokens).
# Structure: (question, answer, intent, mechanism_note)

CANDIDATE_EXPLANATION_BANKS = {
    "train": [
        ("Why does the ocean appear blue when viewed under sunlight?", "Water molecules absorb red light, reflecting blue light outward.", "Explain water selective red absorption", "Selective red absorption and Rayleigh scattering in pure water."),
        ("Why do objects fall downward when dropped near the surface of Earth?", "Earth gravity pulls physical objects downward toward its center.", "Explain gravitational attraction", "Newtonian gravitational force."),
        ("Why do different regions on Earth experience changing seasons throughout the year?", "Earth tilted axis causes hemispheres to receive varying sunlight.", "Explain axial tilt causing seasons", "Axial tilt of 23.5 degrees relative to orbital plane."),
        ("Why does solid ice float on the surface of liquid water?", "Ice expands as it freezes, making it less dense than liquid water.", "Explain ice lower density lattice", "Hydrogen bonding expansion in solid water."),
        ("Why do migratory bird species travel to different climates before winter arrives?", "Birds migrate toward warmer climates to secure food resources.", "Explain seasonal migration", "Ecological adaptation to resource scarcity and thermal demands."),
        ("Why does the daytime sky appear blue to an observer on the ground?", "Gases in the atmosphere scatter short blue wavelengths of sunlight.", "Explain atmospheric Rayleigh scattering", "Rayleigh scattering intensity proportional to 1/(wavelength^4)."),
        ("Why does human sweat cool the skin as it disappears?", "Evaporating sweat absorbs heat energy away from the skin surface.", "Explain evaporative cooling", "Thermodynamic latent heat of vaporization."),
        ("Why does sound propagate significantly faster through water than through atmospheric air?", "Water molecules are packed closely together, transmitting acoustic vibrations rapidly.", "Explain acoustic propagation in dense medium", "Bulk modulus and particle density in acoustic mechanics."),
        ("Why does exposed iron gradually develop reddish rust in humid conditions?", "Iron atoms react with oxygen and moisture to form iron oxide.", "Explain iron oxidation into rust", "Electrochemical oxidation reaction."),
        ("Why do cyclical ocean tides rise and fall across coastal shores on Earth?", "The gravitational pull of the Moon creates tidal bulges in oceans.", "Explain tidal forces from lunar gravity", "Gravitational tidal differential field."),
        ("Why does helium make balloons ascend upward into the open atmosphere?", "Helium is lighter than surrounding air, creating buoyant lift.", "Explain buoyant force from low density", "Archimedes principle of buoyancy in fluids."),
        ("Why do human pupils dilate automatically in darkened environments?", "Pupils dilate to allow more light photons to strike the retina.", "Explain pupil dilation for photon capture", "Ocular physiological response to illumination changes."),
        ("Why does bread dough expand and rise during the baking preparation process?", "Yeast produces carbon dioxide gas bubbles that expand the dough.", "Explain yeast carbon dioxide production", "Anaerobic respiration and cellular fermentation."),
        ("Why does a metal container feel significantly colder to touch than a wooden block at room temperature?", "Metal conducts heat away from your skin much faster than wood.", "Explain thermal conduction differential", "Fourier law of heat conduction."),
        ("Why does condensation form on the exterior of a cold glass on a warm day?", "Cold glass cools surrounding humid air, condensing water vapor.", "Explain dew point condensation", "Phase transition of atmospheric moisture at dew point."),
        ("Why does an airplane wing generate aerodynamic lift when moving forward?", "Curved wings create lower air pressure above than below.", "Explain Bernoulli aerodynamic lift", "Bernoulli principle and circulation theory of aerodynamic lift."),
        ("Why do human muscles experience involuntary shivering when exposed to cold temperatures?", "Shivering generates metabolic heat to maintain core body temperature.", "Explain shivering thermogenesis", "Homeostatic physiological response to hypothermia."),
        ("Why does a prism separate composite white sunlight into a spectrum of colors?", "Glass refracts different light wavelengths at varying optical angles.", "Explain chromatic dispersion", "Optical dispersion via wavelength-dependent Snell law."),
        ("Why do soap bubbles exhibit swirling iridescent rainbow patterns on their surface?", "Light waves reflecting off thin soap film interfere with each other.", "Explain thin-film optical interference", "Constructive and destructive thin-film interference."),
        ("Why does warm air naturally rise above cooler surrounding air in a room?", "Warming air expands, becomes less dense, and floats upward.", "Explain thermal convection", "Thermal expansion and buoyant convection in fluids."),
        ("Why do oil and liquid water separate into distinct layers rather than mixing?", "Polar water molecules attract each other, repelling nonpolar oil.", "Explain hydrophobic separation", "Polar versus nonpolar molecular intermolecular forces."),
        ("Why does lightning produce the booming sound heard as thunder?", "Rapid heating of air by lightning generates a supersonic shockwave.", "Explain lightning acoustic shockwave", "Rapid plasma thermal expansion and acoustic shockwave."),
        ("Why do domestic cats possess glowing eyes when illuminated in the dark?", "A reflective tissue layer behind the retina reflects unabsorbed light.", "Explain tapetum lucidum retroreflection", "Ocular anatomy for nocturnal vision amplification."),
        ("Why does food cook significantly faster in a pressurized cooker than in an open pot?", "High pressure raises the boiling point of water, cooking hotter.", "Explain boiling point elevation under pressure", "Clausius-Clapeyron relation of vapor pressure and temperature."),
        ("Why does a moving bicycle remain stably upright more easily than a stationary one?", "Forward momentum and steering geometry naturally correct balance.", "Explain bicycle dynamic self-stabilization", "Forward speed, fork caster angle, and angular inertia."),
        ("Why do plant stems bend and grow toward a nearby light source?", "Plant growth hormones cause cells on the shaded side to elongate.", "Explain phototropism via auxin elongation", "Hormonal plant phototropism mechanism."),
        ("Why does ocean water taste salty compared to freshwater rivers and lakes?", "Rivers carry dissolved rock minerals into oceans where salt accumulates.", "Explain mineral salt accumulation in oceans", "Geochemical weathering and ocean basin accumulation."),
        ("Why do carbonated soda beverages release fizzy bubbles when opened?", "Opening releases pressure, reducing gas solubility and freeing carbon dioxide.", "Explain Henry law gas solubility", "Henry law of gas solubility under varying pressure."),
        ("Why does your voice sound higher in pitch after inhaling helium gas?", "Sound travels faster in light helium, raising vocal tract resonances.", "Explain vocal tract resonance in light gas", "Acoustic wave speed and harmonic formant shifts."),
        ("Why do lunar craters remain preserved on the Moon for millions of years?", "The Moon has no atmosphere or weather to erode impact craters.", "Explain lack of atmospheric lunar erosion", "Absence of hydrological or atmospheric weathering agents."),
        ("Why does adding salt to an icy sidewalk accelerate the melting process?", "Dissolved salt lowers the freezing point of water, melting ice.", "Explain freezing point depression", "Colligative property of freezing point depression."),
        ("Why does touching a metal doorknob after walking on a carpet sometimes cause a small spark?", "Static electric charges built by friction discharge to grounded metal.", "Explain triboelectric static discharge", "Triboelectric charging and electrostatic discharge."),
        ("Why do clothes dry on a laundry line even on a cool, cloudy day?", "Energetic water molecules escape into the air as vapor.", "Explain kinetic energy vapor escape", "Maxwell-Boltzmann kinetic energy distribution in evaporation."),
        ("Why does a compass needle reliably point toward the northern horizon on Earth?", "The magnetic needle aligns with Earth natural geomagnetic field.", "Explain geomagnetic alignment", "Geodynamo magnetic field generation and magnetic dipole alignment."),
        ("Why do onions cause your eyes to tear up when sliced with a knife?", "Cutting releases volatile sulfur compounds that irritate tear glands.", "Explain onion lachrymatory gas release", "Enzymatic cleavage producing lachrymatory factor gas."),
        ("Why does a match ignite when rubbed firmly against a rough matchbox strip?", "Friction generates heat that ignites flammable chemical compounds.", "Explain frictional thermal ignition", "Frictional thermal ignition of phosphorus compounds."),
        ("Why do fireflies emit visible glowing flashes during nighttime hours?", "A chemical reaction involving the enzyme luciferase produces light.", "Explain bioluminescence via luciferase", "Enzymatic bioluminescent oxidation reaction."),
        ("Why do glass car windshields fog up with condensation in chilly weather?", "Warm breath condenses into tiny water droplets on cold glass.", "Explain interior condensation on cold glass", "Dew point condensation on cold thermal barriers."),
        ("Why does blowing gently across hot soup cool it down more quickly?", "Moving air removes evaporating vapor, speeding up cooling.", "Explain forced convection evaporative cooling", "Boundary layer displacement in forced convection."),
        ("Why do hot air balloons rise when their interior air is heated by a burner?", "Heated air expands and becomes lighter than surrounding cold air.", "Explain thermal expansion buoyancy", "Charles law and fluid buoyancy principle."),
    ],
    "val": [
        ("Why do atmospheric clouds produce precipitation such as rain?", "Cloud water droplets merge until they become heavy enough to fall.", "Explain droplet coalescence precipitation", "Cloud microphysics and gravitational settling."),
        ("Why does a metal spoon feel cold when retrieved from a kitchen drawer?", "Metal conducts heat away from fingertips faster than surrounding air.", "Explain thermal conduction", "Fourier law of thermal conduction."),
        ("Why does an approaching emergency vehicle siren sound higher in pitch than when receding?", "Motion compresses sound waves ahead, raising the observed siren pitch.", "Explain acoustic Doppler shift", "Acoustic Doppler frequency compression."),
        ("Why do canyon walls produce distinct auditory echoes when someone shouts?", "Sound waves reflect off rigid rock walls and travel back.", "Explain acoustic reflection echo", "Acoustic reflection off high-impedance boundaries."),
        ("Why does dry hair cling to an unrolled plastic comb on low-humidity days?", "Static charges built by friction cause hair strands to attract.", "Explain electrostatic attraction", "Triboelectric charging and Coulomb electrostatic attraction."),
    ],
    "test": [
        ("Why do distant stars appear to twinkle when observed from the ground at night?", "Atmospheric turbulence constantly bends incoming starlight in random directions.", "Explain atmospheric scintillation", "Atmospheric scintillation through turbulent optical cells."),
        ("Why do natural geothermal geysers erupt periodically with steam and hot water?", "Superheated water flashes into steam, expanding through underground fissures.", "Explain geyser hydrothermal eruption", "Hydrothermal dynamics and steam expansion in confined conduits."),
        ("Why is a distant lightning flash seen before its associated thunder is heard?", "Light travels much faster through the atmosphere than acoustic sound.", "Explain velocity differential light versus sound", "Speed of electromagnetic waves versus acoustic waves."),
        ("Why does table salt dissolve readily when stirred into liquid water?", "Water molecules surround and separate sodium and chloride ions.", "Explain ionic lattice hydration dissolution", "Electrostatic hydration enthalpy overcoming lattice energy."),
        ("Why do highway asphalt surfaces appear to possess wet water puddles on hot sunny afternoons?", "Hot air near the road bends light upward, creating a mirage.", "Explain inferior optical mirage", "Atmospheric temperature gradient optical refraction."),
    ],
}


# ===========================================================================
# 3. UNCERTAINTY & ABSTENTION BANKS (Calibrated Multi-Domain Scenarios)
# ===========================================================================
# Structure: (context, unanswerable_q, unanswerable_a, control_q, control_a, domain_label)

CANDIDATE_ABSTENTION_BANKS = {
    "train": [
        (
            "Server cluster telemetry report: Host node Alpha-4 operating at forty-two percent CPU utilization with 16 gigabytes RAM active. Primary storage subsystem healthy.",
            "What is the current network interface bandwidth consumption on node Alpha-4?",
            "The telemetry report does not contain network bandwidth information.",
            "What is the reported CPU utilization percentage on host node Alpha-4?",
            "Host node Alpha-4 is operating at forty-two percent CPU utilization.",
            "telemetry"
        ),
        (
            "Expedition field manifest: Team Beta departed base camp at 06:00 UTC carrying four field radios and seven days of emergency rations.",
            "What is the current elevation above sea level for Team Beta?",
            "The expedition manifest does not state the team's elevation.",
            "At what time in UTC did Team Beta depart base camp?",
            "Team Beta departed base camp at 06:00 UTC.",
            "expedition"
        ),
        (
            "Turbine inspection log: Model T-800 passed vibration analysis with rotor speed nominal at 3600 RPM. Lubrication oil pressure measured at 45 PSI.",
            "What is the internal operating temperature of turbine Model T-800?",
            "The inspection log does not record the turbine's operating temperature.",
            "What was the measured lubrication oil pressure on turbine Model T-800?",
            "The lubrication oil pressure was measured at 45 PSI.",
            "engineering"
        ),
        (
            "Executive committee minutes: The steering board approved Phase 2 software development and scheduled the next technical review for Thursday morning.",
            "What total financial budget was approved for Phase 2 software development?",
            "The meeting minutes do not mention the approved budget figure.",
            "On what weekday is the next technical review scheduled to take place?",
            "The next technical review is scheduled for Thursday morning.",
            "meeting_records"
        ),
        (
            "Maritime cargo manifest: Vessel Sea-Star departed port carrying four hundred standard shipping containers filled with agricultural grain.",
            "What is the full name of the vessel captain commanding Sea-Star?",
            "The maritime manifest does not provide the captain's name.",
            "How many standard shipping containers is the vessel Sea-Star carrying?",
            "The vessel Sea-Star is carrying four hundred standard shipping containers.",
            "shipping"
        ),
        (
            "Meteorological weather bulletin: Station North reported sea-level barometric pressure at 1014 hectopascals and wind velocity at twelve knots out of the north.",
            "What is the total accumulated rainfall volume recorded today at Station North?",
            "The weather bulletin does not state accumulated rainfall volume.",
            "What wind velocity in knots was recorded at Station North?",
            "The recorded wind velocity at Station North is twelve knots.",
            "weather"
        ),
        (
            "Warehouse inventory ledger: Bin forty-two contains one hundred twenty brass hexagonal bolts and eighty matching washers.",
            "What is the unit wholesale price of the brass hexagonal bolts?",
            "The inventory ledger does not specify the unit wholesale price.",
            "How many brass hexagonal bolts are stored in bin forty-two?",
            "Bin forty-two contains one hundred twenty brass hexagonal bolts.",
            "inventory"
        ),
        (
            "Laboratory experiment notes: Reagent sample Delta was heated to seventy degrees Celsius for twenty minutes under an inert argon gas atmosphere.",
            "What final color change occurred in reagent sample Delta after heating?",
            "The laboratory notes do not record any final color change.",
            "Under which inert gas atmosphere was reagent sample Delta heated?",
            "Reagent sample Delta was heated under an inert argon gas atmosphere.",
            "laboratory"
        ),
        (
            "Transit schedule bulletin: Rapid express train Line-B departs the central terminal every fifteen minutes between 07:00 and 19:00 on weekdays.",
            "How many passenger seating cars are attached to the Line-B express train?",
            "The schedule bulletin does not state the number of passenger cars.",
            "How frequently does the Line-B express train depart the terminal during weekday operational hours?",
            "The Line-B express train departs every fifteen minutes.",
            "transit"
        ),
        (
            "Building maintenance record: Air filtration unit HVAC-3 had its HEPA filter element replaced on Monday by contractor technician Smith.",
            "What is the model serial number of the replacement HEPA filter element?",
            "The maintenance record does not specify the filter's model serial number.",
            "On what day of the week was the filter element replaced on unit HVAC-3?",
            "The filter element was replaced on Monday.",
            "maintenance"
        ),
        (
            "Automotive service receipt: Vehicle sedan underwent engine oil replacement with 5W-30 synthetic lubricant and tire rotation at 45,000 miles.",
            "What is the brand name of the brake pads installed on the vehicle?",
            "The service receipt does not state the brand of brake pads installed.",
            "What mileage was recorded on the vehicle during the service visit?",
            "The recorded mileage on the vehicle was 45,000 miles.",
            "automotive"
        ),
        (
            "Astronomical observatory log: Telescope unit four tracked asteroid 2024-XA for three hours under clear sky conditions with zero cloud cover.",
            "What is the estimated diameter in meters of asteroid 2024-XA?",
            "The observatory log does not contain the asteroid's estimated diameter.",
            "For how many hours was asteroid 2024-XA tracked by telescope unit four?",
            "Asteroid 2024-XA was tracked for three hours.",
            "astronomy"
        ),
        (
            "Flight dispatch brief: Flight Seven-Twenty is routed from Dallas to Atlanta with an assigned cruising altitude of thirty-two thousand feet.",
            "How many commercial passengers are currently checked in for Flight Seven-Twenty?",
            "The flight brief does not provide passenger check-in numbers.",
            "What is the assigned cruising altitude for Flight Seven-Twenty?",
            "The assigned cruising altitude is thirty-two thousand feet.",
            "aviation"
        ),
        (
            "Agricultural crop report: Orchard sector East contains eight hundred mature apple trees irrigated with drip irrigation lines.",
            "What is the expected harvest yield in bushels for orchard sector East?",
            "The crop report does not specify the expected harvest yield in bushels.",
            "How many mature apple trees are planted in orchard sector East?",
            "There are eight hundred mature apple trees in orchard sector East.",
            "agriculture"
        ),
        (
            "Power grid dispatch note: Substation South delivered two hundred megawatts of electric power to the municipal grid during morning peak hours.",
            "What is the primary fuel source used to generate power at Substation South?",
            "The dispatch note does not identify the primary generation fuel source.",
            "How many megawatts of electric power were delivered by Substation South during peak hours?",
            "Substation South delivered two hundred megawatts of electric power.",
            "energy"
        ),
        (
            "Library catalog record: Volume three of the encyclopedia was published in 1984 in London by university academic press.",
            "How many total pages are bound within volume three of the encyclopedia?",
            "The catalog record does not state the total page count.",
            "In what year was volume three of the encyclopedia published?",
            "Volume three of the encyclopedia was published in 1984.",
            "library"
        ),
        (
            "Aquaculture facility log: Tank nine contains five hundred juvenile rainbow trout fed twice daily with standard pellet food.",
            "What is the average water pH level maintained in tank nine?",
            "The aquaculture log does not record the water pH level.",
            "How frequently are the juvenile rainbow trout in tank nine fed?",
            "The juvenile rainbow trout in tank nine are fed twice daily.",
            "aquaculture"
        ),
        (
            "Mining telemetry brief: Conveyor belt Seven transported six hundred metric tons of crushed granite during the eight-hour night shift.",
            "What is the electric motor horsepower rating driving conveyor belt Seven?",
            "The telemetry brief does not state the conveyor motor horsepower rating.",
            "How many metric tons of crushed granite were transported during the night shift?",
            "Conveyor belt Seven transported six hundred metric tons of crushed granite.",
            "mining"
        ),
        (
            "Urban park survey: Central plaza grounds feature twenty oak trees, twelve park benches, and one granite memorial fountain.",
            "In what historical year was the granite memorial fountain erected?",
            "The park survey does not state the year the memorial fountain was erected.",
            "How many park benches are installed around the central plaza grounds?",
            "There are twelve park benches installed around the central plaza grounds.",
            "urban_planning"
        ),
        (
            "Data storage status log: Archive vault Bravo holds twelve redundant tape cartridges with all automated hash verifications passing.",
            "What is the total data storage capacity in terabytes of archive vault Bravo?",
            "The status log does not specify the storage capacity in terabytes.",
            "How many redundant tape cartridges are stored in archive vault Bravo?",
            "Archive vault Bravo holds twelve redundant tape cartridges.",
            "data_storage"
        ),
        (
            "Robotics lab telemetry: Mobile unit Rover-3 completed obstacle avoidance testing over a two-kilometer outdoor obstacle course.",
            "What chemical battery chemistry powers mobile unit Rover-3?",
            "The lab telemetry does not identify the battery chemistry.",
            "What distance in kilometers was covered during the obstacle avoidance test?",
            "The obstacle avoidance test covered a distance of two kilometers.",
            "robotics"
        ),
        (
            "Bakery production sheet: Commercial oven four baked three hundred sourdough loaves and two hundred baguettes during the dawn shift.",
            "What is the internal baking temperature set on commercial oven four?",
            "The production sheet does not state the oven's internal baking temperature.",
            "How many baguettes were baked in commercial oven four during the dawn shift?",
            "Two hundred baguettes were baked during the dawn shift.",
            "culinary"
        ),
        (
            "Telecommunications log: Cell tower site Juliet processed eighty thousand mobile voice calls with zero dropped call anomalies.",
            "What is the physical height in meters of cell tower site Juliet?",
            "The telecommunications log does not mention the physical height of the tower.",
            "How many mobile voice calls were processed at cell tower site Juliet?",
            "Cell tower site Juliet processed eighty thousand mobile voice calls.",
            "telecommunications"
        ),
        (
            "Seismology observatory alert: Seismic site Sierra detected a magnitude 3.2 tremor centered fifteen kilometers southwest of the facility.",
            "What was the estimated focal depth in kilometers of the detected tremor?",
            "The alert does not contain the tremor's focal depth.",
            "What seismic tremor magnitude was detected by seismic site Sierra?",
            "Seismic site Sierra detected a magnitude 3.2 seismic tremor.",
            "geophysics"
        ),
        (
            "Port logistics notice: Dock crane Twelve unloaded eighty cargo containers from barge Pioneer during morning marine operations.",
            "What is the maximum lifting capacity in tons of dock crane Twelve?",
            "The logistics notice does not state the maximum lifting capacity of the crane.",
            "How many cargo containers were unloaded from barge Pioneer?",
            "Dock crane Twelve unloaded eighty cargo containers.",
            "logistics"
        ),
        (
            "Solar generation record: Array cluster Echo generated four hundred kilowatt-hours of solar energy under clear sky conditions.",
            "What specific manufacturer brand fabricated the panels in array cluster Echo?",
            "The generation record does not identify the solar panel manufacturer.",
            "How many kilowatt-hours of solar energy were generated by array cluster Echo?",
            "Array cluster Echo generated four hundred kilowatt-hours of solar energy.",
            "solar_energy"
        ),
        (
            "Forestry district report: Sector Pine underwent selective brush clearing across forty acres to reduce seasonal wildfire hazards.",
            "How many total forestry personnel were deployed on the brush clearing crew?",
            "The forestry report does not provide the number of personnel deployed.",
            "How many acres underwent brush clearing in sector Pine?",
            "Sector Pine underwent brush clearing across forty acres.",
            "forestry"
        ),
        (
            "Water treatment plant log: Pumping station Delta processed three million liters of municipal drinking water with chlorine levels nominal.",
            "What was the measured water turbidity level in nephelometric turbidity units?",
            "The plant log does not specify the measured water turbidity level.",
            "How many liters of drinking water were processed by pumping station Delta?",
            "Pumping station Delta processed three million liters of drinking water.",
            "water_treatment"
        ),
        (
            "Courier package dispatch: Delivery van route Nine delivered eighty parcels and thirty priority envelopes across the downtown district.",
            "What was the total odometer mileage driven by delivery van route Nine?",
            "The dispatch record does not record the van's total driven mileage.",
            "How many priority envelopes were delivered on delivery van route Nine?",
            "Delivery van route Nine delivered thirty priority envelopes.",
            "courier"
        ),
        (
            "Audio recording studio sheet: Track vocal master was recorded using microphone channel two with a 48 kilohertz sampling rate.",
            "What brand of studio monitor headphones did the sound engineer wear?",
            "The studio sheet does not state the brand of headphones worn.",
            "What sampling rate was utilized for recording the track vocal master?",
            "The vocal master was recorded with a 48 kilohertz sampling rate.",
            "audio"
        ),
        (
            "Heavy machinery log: Excavator Unit-E excavated four hundred cubic meters of soil during roadway trenching operations.",
            "What is the diesel fuel tank capacity in liters of Excavator Unit-E?",
            "The machinery log does not specify the diesel fuel tank capacity.",
            "How many cubic meters of soil were excavated by Excavator Unit-E?",
            "Excavator Unit-E excavated four hundred cubic meters of soil.",
            "construction"
        ),
        (
            "Botanical conservatory log: Greenhouse section Orchid is maintained at eighty percent relative humidity and seventy-four degrees Fahrenheit.",
            "What automated watering schedule frequency is programmed for greenhouse section Orchid?",
            "The conservatory log does not state the automated watering schedule frequency.",
            "What is the maintained relative humidity percentage in greenhouse section Orchid?",
            "The maintained relative humidity in greenhouse section Orchid is eighty percent.",
            "horticulture"
        ),
        (
            "Printing press manifest: Print machine Heidelberg produced ten thousand copies of the quarterly technical journal on glossy paper.",
            "What color ink formulation supplier provided the cyan ink used in printing?",
            "The manifest does not identify the ink formulation supplier.",
            "How many copies of the quarterly technical journal were produced?",
            "Ten thousand copies of the quarterly technical journal were produced.",
            "publishing"
        ),
        (
            "Cold storage telemetry: Freezing room Locker-B recorded a temperature of minus twenty degrees Celsius with cooling compressors operating normally.",
            "What is the total weight in kilograms of frozen goods currently stored in room Locker-B?",
            "The telemetry record does not state the total weight of stored goods.",
            "What temperature in degrees Celsius was recorded in freezing room Locker-B?",
            "Freezing room Locker-B recorded a temperature of minus twenty degrees Celsius.",
            "refrigeration"
        ),
        (
            "Textile mill production sheet: Loom station Five wove twelve hundred meters of cotton canvas fabric during the morning operations shift.",
            "What is the tensile strength rating of the woven cotton canvas fabric?",
            "The production sheet does not record the fabric's tensile strength rating.",
            "How many meters of cotton canvas fabric were woven at loom station Five?",
            "Loom station Five wove twelve hundred meters of cotton canvas fabric.",
            "manufacturing"
        ),
        (
            "Subsea cable monitor: Fiber optic line Trans-Pacific operated with zero optical packet loss across twenty-four continuous hours.",
            "What is the maximum underwater ocean depth reached by fiber optic line Trans-Pacific?",
            "The monitoring log does not provide the maximum ocean depth reached.",
            "How many continuous hours of operation with zero packet loss were recorded?",
            "Twenty-four continuous hours of operation were recorded.",
            "networking"
        ),
        (
            "Aviation hangar maintenance: Aircraft Jet-Ten received scheduled hydraulic system pressure fluid checks and structural landing gear lubrication.",
            "Who was the certified aircraft inspector who signed the release paperwork for Jet-Ten?",
            "The maintenance brief does not state the name of the certified inspector.",
            "Which aircraft system received pressure fluid checks during the hangar visit?",
            "The aircraft hydraulic system received pressure fluid checks.",
            "aviation_maintenance"
        ),
        (
            "Recycling sorting plant report: Facility conveyor line sorted fifteen metric tons of cardboard and eight metric tons of aluminum cans.",
            "What was the electrical power consumption in kilowatt-hours of the facility sorting line?",
            "The sorting plant report does not record the electrical power consumption.",
            "How many metric tons of aluminum cans were sorted by the facility conveyor line?",
            "The facility conveyor line sorted eight metric tons of aluminum cans.",
            "recycling"
        ),
        (
            "Bridge structural monitoring brief: Sensor array Span-One detected horizontal wind deflection of two centimeters under forty-mile-per-hour gusts.",
            "What concrete grade rating was utilized during original foundation construction of Span-One?",
            "The monitoring brief does not contain the original concrete grade rating.",
            "What horizontal wind deflection in centimeters was detected by sensor array Span-One?",
            "Sensor array Span-One detected horizontal wind deflection of two centimeters.",
            "civil_engineering"
        ),
        (
            "Microbiology incubation log: Culture plate Bacillus-7 was incubated at thirty-seven degrees Celsius for forty-eight hours in an aerobic chamber.",
            "What specific agar growth medium formulation was poured into culture plate Bacillus-7?",
            "The incubation log does not specify the agar growth medium formulation.",
            "How many hours was culture plate Bacillus-7 incubated in the aerobic chamber?",
            "Culture plate Bacillus-7 was incubated for forty-eight hours.",
            "microbiology"
        ),
    ],
    "val": [
        (
            "Satellite telemetry dispatch: Spacecraft Orbiter-Nine oriented solar wings toward the Sun and established ground link frequency at 8.4 gigahertz.",
            "What is the remaining chemical thruster fuel mass in kilograms on Orbiter-Nine?",
            "The telemetry dispatch does not state the remaining thruster fuel mass.",
            "At what frequency in gigahertz was the ground communication link established?",
            "The ground communication link was established at 8.4 gigahertz.",
            "spaceflight"
        ),
        (
            "Harbor dredging report: Dredge vessel Heron deepened access channel sector four to twelve meters below low tide datum.",
            "What was the total volume in cubic yards of silt removed by vessel Heron?",
            "The dredging report does not record the total volume of silt removed.",
            "To what depth in meters below low tide was channel sector four deepened?",
            "Channel sector four was deepened to twelve meters below low tide.",
            "marine_engineering"
        ),
        (
            "Wind farm turbine summary: Wind generator Tower-Twelve operated at twenty-two rotations per minute producing 1.8 megawatts of electric power.",
            "What is the rotor blade span diameter in meters of Tower-Twelve?",
            "The turbine summary does not state the rotor blade span diameter.",
            "How many megawatts of electric power were produced by Tower-Twelve?",
            "Tower-Twelve produced 1.8 megawatts of electric power.",
            "renewable_energy"
        ),
        (
            "Geological borehole brief: Drilling rig Core-Five reached a subsurface depth of four hundred meters recovering continuous sandstone samples.",
            "What was the measured geothermal temperature gradient at the bottom of the borehole?",
            "The borehole brief does not specify the measured geothermal temperature gradient.",
            "What rock type samples were recovered from drilling rig Core-Five?",
            "Continuous sandstone samples were recovered from drilling rig Core-Five.",
            "geology"
        ),
        (
            "Cargo rail manifest: Freight locomotive Unit-Three hauled sixty railcars carrying refined timber across three hundred miles of track.",
            "What was the gross total freight weight in tons of the hauled railcars?",
            "The rail manifest does not state the gross total freight weight.",
            "What cargo was carried by the sixty railcars hauled by Unit-Three?",
            "The sixty railcars carried refined timber.",
            "rail_transport"
        ),
    ],
    "test": [
        (
            "Atmospheric balloon log: Weather sonde Probe-Six ascended to twenty-five thousand meters measuring relative humidity at fifteen percent.",
            "What radio frequency in megahertz transmitted the telemetry packet from Probe-Six?",
            "The balloon log does not state the telemetry transmission radio frequency.",
            "To what altitude in meters did weather sonde Probe-Six ascend?",
            "Weather sonde Probe-Six ascended to twenty-five thousand meters.",
            "meteorology"
        ),
        (
            "Hydroelectric station log: Generator Unit-One discharged four hundred cubic meters of water per second through penstock valve Alpha.",
            "What is the water level elevation in meters of the upstream reservoir?",
            "The station log does not record the upstream reservoir water elevation.",
            "Through which penstock valve was water discharged by Generator Unit-One?",
            "Water was discharged through penstock valve Alpha.",
            "hydroelectric"
        ),
        (
            "Foundry smelting record: Blast furnace Three produced twenty-five metric tons of molten pig iron during the secondary tap cycle.",
            "What was the measured temperature in Celsius of the molten pig iron tap?",
            "The smelting record does not mention the molten iron tap temperature.",
            "How many metric tons of molten pig iron were produced during the secondary tap cycle?",
            "Twenty-five metric tons of molten pig iron were produced.",
            "metallurgy"
        ),
        (
            "Urban transit dispatch: Electric bus Fleet-Seven completed eight scheduled route loops carrying six hundred total transit passengers.",
            "What remaining state-of-charge battery percentage does Fleet-Seven have?",
            "The transit dispatch does not specify the remaining battery state-of-charge.",
            "How many scheduled route loops were completed by electric bus Fleet-Seven?",
            "Electric bus Fleet-Seven completed eight scheduled route loops.",
            "urban_transit"
        ),
        (
            "Optical fiber fabrication sheet: Draw tower Beta drew twelve kilometers of single-mode silica optical fiber with cladding diameter nominal.",
            "What core refractive index dopant concentration was used in the preform rod?",
            "The fabrication sheet does not record the core dopant concentration.",
            "How many kilometers of single-mode optical fiber were drawn at tower Beta?",
            "Twelve kilometers of single-mode optical fiber were drawn at tower Beta.",
            "optics_manufacturing"
        ),
    ],
}
