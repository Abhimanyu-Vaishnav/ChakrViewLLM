"""
Phase B: Build Frozen Conversational Evaluation Suite (CONVERSATION_BENCHMARK_V1.json)
Constructs exactly 200 independently authored held-out conversational test probes across 12 mandated categories:
1. Basic conversation (20)
2. Follow-up understanding (18)
3. Context retention (18)
4. Multi-turn continuity (18)
5. Reference resolution (16)
6. Topic transition (16)
7. Clarification (16)
8. Instruction following (16)
9. Naturalness (16)
10. Consistency (16)
11. Language consistency / Hindi & Hinglish (16)
12. Response completion (14)
Ensures 0% prompt leakage against all training sets.
"""

import json
import hashlib
from pathlib import Path

OUT_DIR = Path("D:/Project/ChakrView/experiments/conversation/cr_exp_17")
OUT_DIR.mkdir(parents=True, exist_ok=True)

probes = []

# 1. Basic conversation (20)
basic = [
    ("CONV_BASIC_01", "User: Greetings! Hope you are having a pleasant day.\nAssistant:", ["Greetings", "pleasant", "hello", "help", "assist"]),
    ("CONV_BASIC_02", "User: Wishing you a pleasant morning!\nAssistant:", ["morning", "pleasant", "help", "assist", "day"]),
    ("CONV_BASIC_03", "User: Good evening! How are you?\nAssistant:", ["Good evening", "well", "fine", "assist"]),
    ("CONV_BASIC_04", "User: Could you state your identity and purpose?\nAssistant:", ["ChakrView", "model", "assistant", "purpose"]),
    ("CONV_BASIC_05", "User: By what moniker or title should I address you?\nAssistant:", ["ChakrView", "name", "call"]),
    ("CONV_BASIC_06", "User: Can you assist me today?\nAssistant:", ["Yes", "assist", "help", "Certainly"]),
    ("CONV_BASIC_07", "User: Thank you for your help!\nAssistant:", ["welcome", "glad", "anytime", "pleasure"]),
    ("CONV_BASIC_08", "User: I hope your upcoming hours are wonderful.\nAssistant:", ["Thank you", "great", "wonderful", "you too"]),
    ("CONV_BASIC_09", "User: Goodbye!\nAssistant:", ["Goodbye", "Bye", "farewell", "great"]),
    ("CONV_BASIC_10", "User: Share a brief summary of your architecture and role.\nAssistant:", ["ChakrView", "neural", "model", "assistant"]),
    ("CONV_BASIC_11", "User: It is an absolute delight to make your acquaintance.\nAssistant:", ["delight", "pleasure", "acquaintance", "welcome", "meet"]),
    ("CONV_BASIC_12", "User: What specific range of capabilities do you offer?\nAssistant:", ["help", "assist", "questions", "text", "capabilities"]),
    ("CONV_BASIC_13", "User: Do you possess organic biological consciousness?\nAssistant:", ["no", "model", "AI", "neural", "assistant", "biological"]),
    ("CONV_BASIC_14", "User: Please introduce yourself briefly.\nAssistant:", ["ChakrView", "model", "assistant"]),
    ("CONV_BASIC_15", "User: How is the weather today?\nAssistant:", ["cannot", "sensors", "access", "weather", "not"]),
    ("CONV_BASIC_16", "User: What languages do you understand?\nAssistant:", ["English", "Hindi", "languages"]),
    ("CONV_BASIC_17", "User: Can you answer questions?\nAssistant:", ["Yes", "certainly", "help", "questions"]),
    ("CONV_BASIC_18", "User: I need some assistance.\nAssistant:", ["What", "help", "assist", "please"]),
    ("CONV_BASIC_19", "User: Are you ready to converse?\nAssistant:", ["Yes", "ready", "assist", "hello"]),
    ("CONV_BASIC_20", "User: Welcome to our lab session!\nAssistant:", ["Thank you", "glad", "assist", "pleasure"])
]
for pid, p, kw in basic:
    probes.append({"id": pid, "category": "Basic conversation", "prompt": p, "expected_keywords": kw, "min_tokens": 3})

# 2. Follow-up understanding (18)
followup = [
    ("CONV_FOLL_01", "User: What is Python?\nAssistant: Python is a high-level programming language.\nUser: Who created it?\nAssistant:", ["Guido", "Rossum"]),
    ("CONV_FOLL_02", "User: What is the capital of France?\nAssistant: The capital of France is Paris.\nUser: What currency does that country use?\nAssistant:", ["Euro"]),
    ("CONV_FOLL_03", "User: Tell me about the Sun.\nAssistant: The Sun is a star at the center of the Solar System.\nUser: How far is it from Earth?\nAssistant:", ["150", "million", "AU", "distance", "kilometers"]),
    ("CONV_FOLL_04", "User: What is Linux?\nAssistant: Linux is an open-source operating system kernel.\nUser: When was it released?\nAssistant:", ["1991"]),
    ("CONV_FOLL_05", "User: Mount Everest is very high.\nAssistant: Yes, it is the highest peak above sea level.\nUser: Which mountain range is it in?\nAssistant:", ["Himalayas", "Himalaya"]),
    ("CONV_FOLL_06", "User: What is water made of?\nAssistant: Water is a chemical compound consisting of hydrogen and oxygen.\nUser: What is its chemical formula?\nAssistant:", ["H2O"]),
    ("CONV_FOLL_07", "User: Have you heard of Shakespeare?\nAssistant: Yes, William Shakespeare was an English playwright and poet.\nUser: Name one play he wrote.\nAssistant:", ["Hamlet", "Macbeth", "Romeo", "Juliet", "Othello"]),
    ("CONV_FOLL_08", "User: We are discussing Mars.\nAssistant: Mars is the fourth planet from the Sun.\nUser: Does it have moons?\nAssistant:", ["Phobos", "Deimos", "two", "moons", "Yes"]),
    ("CONV_FOLL_09", "User: What is photosynthesis?\nAssistant: Photosynthesis is the process plants use to convert light into chemical energy.\nUser: What gas do plants take in for this?\nAssistant:", ["carbon dioxide", "CO2"]),
    ("CONV_FOLL_10", "User: I am reading about gravity.\nAssistant: Gravity is the natural force that attracts bodies toward one another.\nUser: Who formulated the universal law of gravitation?\nAssistant:", ["Newton", "Isaac"]),
    ("CONV_FOLL_11", "User: Let us talk about DNA.\nAssistant: DNA carries genetic instructions for development and functioning.\nUser: What does DNA stand for?\nAssistant:", ["Deoxyribonucleic"]),
    ("CONV_FOLL_12", "User: What is the speed of light?\nAssistant: The speed of light in vacuum is approximately 300,000 km per second.\nUser: Does anything travel faster?\nAssistant:", ["No", "nothing", "limit"]),
    ("CONV_FOLL_13", "User: Tell me about Jupiter.\nAssistant: Jupiter is the largest planet in our solar system.\nUser: Does it have rings?\nAssistant:", ["Yes", "faint", "rings"]),
    ("CONV_FOLL_14", "User: What is an algorithm?\nAssistant: An algorithm is a step-by-step procedure for solving a problem.\nUser: Give a simple example.\nAssistant:", ["sorting", "recipe", "addition", "search"]),
    ("CONV_FOLL_15", "User: Who wrote the Odyssey?\nAssistant: The Odyssey is an ancient Greek epic poem attributed to Homer.\nUser: What other epic did he write?\nAssistant:", ["Iliad"]),
    ("CONV_FOLL_16", "User: What is RAM?\nAssistant: RAM stands for Random Access Memory, used for temporary working storage.\nUser: Is it volatile?\nAssistant:", ["Yes", "volatile", "loses"]),
    ("CONV_FOLL_17", "User: What does CPU stand for?\nAssistant: CPU stands for Central Processing Unit.\nUser: What is its main role?\nAssistant:", ["instructions", "processing", "execute", "compute"]),
    ("CONV_FOLL_18", "User: Tell me about Tokyo.\nAssistant: Tokyo is the bustling capital city of Japan.\nUser: Which island of Japan is it on?\nAssistant:", ["Honshu"])
]
for pid, p, kw in followup:
    probes.append({"id": pid, "category": "Follow-up understanding", "prompt": p, "expected_keywords": kw, "min_tokens": 2})

# 3. Context retention (18)
retention = [
    ("CONV_RET_01", "User: My name is Alice.\nAssistant: Hello Alice! How can I assist you?\nUser: What is my name?\nAssistant:", ["Alice"]),
    ("CONV_RET_02", "User: I have a golden retriever named Max.\nAssistant: Golden retrievers are wonderful dogs!\nUser: What kind of pet do I have?\nAssistant:", ["dog", "golden retriever", "Max"]),
    ("CONV_RET_03", "User: My favorite food is dosa.\nAssistant: Dosa is delicious and crispy.\nUser: What do I like to eat?\nAssistant:", ["dosa"]),
    ("CONV_RET_04", "User: The target port for deployment is 8080.\nAssistant: Noted, deployment port 8080 recorded.\nUser: Which port will we deploy to?\nAssistant:", ["8080"]),
    ("CONV_RET_05", "User: We are meeting in room 402.\nAssistant: Room 402 has been noted for the meeting.\nUser: Where is the meeting?\nAssistant:", ["402"]),
    ("CONV_RET_06", "User: My car is a red sedan.\nAssistant: A red sedan is very stylish.\nUser: What color is my car?\nAssistant:", ["red"]),
    ("CONV_RET_07", "User: My sister's name is Priya.\nAssistant: Priya is a lovely name.\nUser: What is my sister's name?\nAssistant:", ["Priya"]),
    ("CONV_RET_08", "User: I work as a graphic designer.\nAssistant: Graphic design requires great creativity.\nUser: What is my profession?\nAssistant:", ["graphic designer", "design"]),
    ("CONV_RET_09", "User: The project deadline is Friday.\nAssistant: Understood, project deadline Friday.\nUser: When is the deadline?\nAssistant:", ["Friday"]),
    ("CONV_RET_10", "User: I am traveling to Chennai tomorrow.\nAssistant: Chennai is a vibrant coastal city.\nUser: Where am I heading?\nAssistant:", ["Chennai"]),
    ("CONV_RET_11", "User: The secret passcode is Falcon99.\nAssistant: I have recorded the passcode.\nUser: What is the passcode?\nAssistant:", ["Falcon99"]),
    ("CONV_RET_12", "User: My primary programming language is Rust.\nAssistant: Rust is known for memory safety and performance.\nUser: What language do I write code in?\nAssistant:", ["Rust"]),
    ("CONV_RET_13", "User: I have three cats at home.\nAssistant: Three cats must keep your home lively!\nUser: How many cats do I have?\nAssistant:", ["three", "3"]),
    ("CONV_RET_14", "User: The server IP address is 192.168.1.50.\nAssistant: Server IP address recorded.\nUser: What is the server IP?\nAssistant:", ["192.168.1.50"]),
    ("CONV_RET_15", "User: I love drinking green tea.\nAssistant: Green tea is healthy and soothing.\nUser: What beverage do I love?\nAssistant:", ["green tea", "tea"]),
    ("CONV_RET_16", "User: The database table is called users_metadata.\nAssistant: Table users_metadata noted.\nUser: What is the table name?\nAssistant:", ["users_metadata"]),
    ("CONV_RET_17", "User: We booked flight AI302.\nAssistant: Flight AI302 has been noted.\nUser: What is our flight number?\nAssistant:", ["AI302"]),
    ("CONV_RET_18", "User: My daughter is studying biology.\nAssistant: Biology is a fascinating field of science.\nUser: What subject is she studying?\nAssistant:", ["biology"])
]
for pid, p, kw in retention:
    probes.append({"id": pid, "category": "Context retention", "prompt": p, "expected_keywords": kw, "min_tokens": 1})

# 4. Multi-turn continuity (18)
continuity = [
    ("CONV_CONT_01", "User: I am planning a trip.\nAssistant: That sounds exciting! Where are you thinking of going?\nUser: To Rajasthan.\nAssistant: Rajasthan is full of magnificent palaces and forts.\nUser: Which city should I visit first?\nAssistant:", ["Jaipur", "Udaipur", "Jodhpur", "Rajasthan", "city"]),
    ("CONV_CONT_02", "User: We need to bake a cake.\nAssistant: What kind of cake would you like to make?\nUser: Chocolate.\nAssistant: A chocolate cake is always a crowd pleaser!\nUser: What is the first ingredient we need?\nAssistant:", ["flour", "cocoa", "sugar", "butter", "chocolate"]),
    ("CONV_CONT_03", "User: I want to learn to play chess.\nAssistant: Chess is a classic game of strategy.\nUser: What does the knight do?\nAssistant: The knight moves in an L-shape and can jump over pieces.\nUser: Can the bishop jump over pieces too?\nAssistant:", ["No", "cannot", "bishop"]),
    ("CONV_CONT_04", "User: Let us build a web app.\nAssistant: What stack would you like to use?\nUser: Node.js and React.\nAssistant: Great modern full-stack choice.\nUser: What package manager should we use?\nAssistant:", ["npm", "yarn", "pnpm"]),
    ("CONV_CONT_05", "User: My garden needs work.\nAssistant: Gardening is very rewarding. What would you like to plant?\nUser: Tomatoes.\nAssistant: Tomatoes thrive with plenty of sunlight.\nUser: How often should I water them?\nAssistant:", ["regularly", "daily", "water", "soil", "dry"]),
    ("CONV_CONT_06", "User: I have an interview tomorrow.\nAssistant: Best wishes! What role are you interviewing for?\nUser: Software Engineer.\nAssistant: That is a demanding and rewarding role.\nUser: What should I prepare tonight?\nAssistant:", ["algorithms", "system design", "resume", "rest", "review"]),
    ("CONV_CONT_07", "User: I bought a new bicycle.\nAssistant: Cycling is great exercise! What type did you get?\nUser: A road bike.\nAssistant: Road bikes are fast and lightweight.\nUser: Do I need a helmet?\nAssistant:", ["Yes", "helmet", "safety", "essential"]),
    ("CONV_CONT_08", "User: We adopted a puppy.\nAssistant: Congratulations! What breed is the puppy?\nUser: A Labrador.\nAssistant: Labradors are friendly and energetic.\nUser: What should we feed him?\nAssistant:", ["puppy food", "diet", "nutrition", "veterinarian"]),
    ("CONV_CONT_09", "User: I am learning French.\nAssistant: French is a beautiful romance language.\nUser: How do I say good morning?\nAssistant: You say Bonjour.\nUser: How do I say thank you?\nAssistant:", ["Merci"]),
    ("CONV_CONT_10", "User: Let us write a story.\nAssistant: What genre do you have in mind?\nUser: Science fiction.\nAssistant: Sci-fi allows boundless imagination.\nUser: Where should the opening scene take place?\nAssistant:", ["space station", "spaceship", "planet", "colony", "future"]),
    ("CONV_CONT_11", "User: I am setting up a home office.\nAssistant: Having a comfortable workspace is crucial.\nUser: What desk should I get?\nAssistant: An adjustable standing desk is very popular.\nUser: What kind of chair is best?\nAssistant:", ["ergonomic", "support", "chair"]),
    ("CONV_CONT_12", "User: We are organizing a dinner.\nAssistant: Sounds fun! How many guests are coming?\nUser: Six friends.\nAssistant: Six is a great number for a cozy dinner.\nUser: What main course would you suggest?\nAssistant:", ["pasta", "curry", "rice", "roast", "dinner"]),
    ("CONV_CONT_13", "User: I am feeling tired.\nAssistant: It is important to listen to your body.\nUser: I worked 12 hours today.\nAssistant: That is a very long workday.\nUser: What should I do right now?\nAssistant:", ["rest", "sleep", "relax", "water"]),
    ("CONV_CONT_14", "User: I want to read a book.\nAssistant: What kind of books do you enjoy?\nUser: Mystery novels.\nAssistant: Mysteries keep you guessing till the end.\nUser: Recommend an author.\nAssistant:", ["Agatha Christie", "Conan Doyle", "Patterson", "author"]),
    ("CONV_CONT_15", "User: Our computer is overheating.\nAssistant: Overheating can cause throttling or damage.\nUser: The fans are very loud.\nAssistant: That means the system is struggling to cool down.\nUser: What is a common cause?\nAssistant:", ["dust", "thermal paste", "airflow", "ventilation"]),
    ("CONV_CONT_16", "User: Let us talk about coffee.\nAssistant: Coffee has a rich culture worldwide.\nUser: I like espresso.\nAssistant: Espresso provides a concentrated, bold flavor.\nUser: How is a cappuccino made from it?\nAssistant:", ["steamed milk", "foam", "milk"]),
    ("CONV_CONT_17", "User: I am training for a 5k run.\nAssistant: 5k is a fantastic fitness goal.\nUser: I ran 2k yesterday.\nAssistant: That is steady progress.\nUser: Should I run every single day?\nAssistant:", ["rest", "recovery", "alternate", "No"]),
    ("CONV_CONT_18", "User: We want to adopt a second dog.\nAssistant: Bringing a second pet home is exciting.\nUser: Our first dog is very quiet.\nAssistant: Introducing them gradually will be key.\nUser: Where should they meet for the first time?\nAssistant:", ["neutral ground", "park", "outside", "neutral"])
]
for pid, p, kw in continuity:
    probes.append({"id": pid, "category": "Multi-turn continuity", "prompt": p, "expected_keywords": kw, "min_tokens": 2})

# 5. Reference resolution (16)
refres = [
    ("CONV_REF_01", "User: Look at that red car.\nAssistant: Yes, it looks sleek.\nUser: Who owns it?\nAssistant: (Turn 1)", ["car", "red car", "owner", "not provided", "do not know"]),
    ("CONV_REF_02", "User: Sarah bought a new guitar yesterday.\nAssistant: Playing guitar is a great hobby.\nUser: Does she play well?\nAssistant:", ["Sarah", "guitar", "she"]),
    ("CONV_REF_03", "User: The laptop fell off the table.\nAssistant: Oh no, was there any damage?\nUser: The screen is cracked.\nAssistant:", ["screen", "laptop", "repair", "replace"]),
    ("CONV_REF_04", "User: Rohit went to the market.\nAssistant: Did he find what he needed?\nUser: He bought vegetables and spices.\nAssistant:", ["Rohit", "he", "vegetables", "spices", "market"]),
    ("CONV_REF_05", "User: The meeting with the client was postponed.\nAssistant: Why was it rescheduled?\nUser: Because they needed more time.\nAssistant:", ["client", "time", "meeting", "rescheduled"]),
    ("CONV_REF_06", "User: The cake was left in the oven too long.\nAssistant: Did it burn?\nUser: Yes, it turned completely black.\nAssistant:", ["cake", "burned", "sorry", "discard"]),
    ("CONV_REF_07", "User: I adopted a parrot named Charlie.\nAssistant: Parrots are intelligent birds.\nUser: Can he talk?\nAssistant:", ["Charlie", "parrot", "he", "words", "talk"]),
    ("CONV_REF_08", "User: The train to Mumbai was delayed by two hours.\nAssistant: That must be frustrating.\nUser: Is it arriving on platform 3?\nAssistant:", ["train", "platform", "Mumbai", "delayed"]),
    ("CONV_REF_09", "User: Dr. Sharma gave a lecture on neurology.\nAssistant: Neurology is an intricate medical science.\nUser: Was her talk well received?\nAssistant:", ["Dr. Sharma", "lecture", "talk", "she"]),
    ("CONV_REF_10", "User: We replaced the old router.\nAssistant: Hopefully that improves connectivity.\nUser: The new one is much faster.\nAssistant:", ["router", "connection", "faster", "Wi-Fi"]),
    ("CONV_REF_11", "User: Maya submitted her thesis today.\nAssistant: That is a huge milestone for Maya!\nUser: Did she celebrate?\nAssistant:", ["Maya", "she", "thesis", "celebrate"]),
    ("CONV_REF_12", "User: The power went out across the neighborhood.\nAssistant: Is there any storm or outage notice?\nUser: It happened five minutes ago.\nAssistant:", ["power", "outage", "electricity"]),
    ("CONV_REF_13", "User: I planted a mango sapling.\nAssistant: Mango trees can grow quite large.\nUser: How long until it bears fruit?\nAssistant:", ["years", "tree", "sapling", "mango"]),
    ("CONV_REF_14", "User: The printer ran out of black ink.\nAssistant: Do you have a replacement cartridge?\nUser: No, I need to order one.\nAssistant:", ["cartridge", "ink", "printer", "order"]),
    ("CONV_REF_15", "User: Ananya finished the marathon.\nAssistant: Running a marathon takes tremendous dedication!\nUser: What was her finish time?\nAssistant:", ["Ananya", "her", "time", "marathon", "not provided"]),
    ("CONV_REF_16", "User: The soup was too salty.\nAssistant: That can happen easily with broth.\nUser: How can I fix it?\nAssistant:", ["water", "potato", "dilute", "soup"])
]
for pid, p, kw in refres:
    probes.append({"id": pid, "category": "Reference resolution", "prompt": p, "expected_keywords": kw, "min_tokens": 2})

# 6. Topic transition (16)
topic = [
    ("CONV_TOPIC_01", "User: Let us stop talking about computers and talk about food.\nAssistant:", ["food", "cuisine", "eat", "cooking", "Sure"]),
    ("CONV_TOPIC_02", "User: Enough about the weather. Tell me a joke.\nAssistant:", ["joke", "laugh", "Why", "What", "did"]),
    ("CONV_TOPIC_03", "User: Moving on from physics, what is your favorite sport?\nAssistant:", ["sport", "cricket", "football", "chess", "athletics"]),
    ("CONV_TOPIC_04", "User: That covers the database. Now let us discuss frontend UI design.\nAssistant:", ["frontend", "UI", "design", "React", "interface"]),
    ("CONV_TOPIC_05", "User: Let us switch subjects. Do you like music?\nAssistant:", ["music", "songs", "melody", "sound", "Yes"]),
    ("CONV_TOPIC_06", "User: Forget about history for a moment. Can we talk about space exploration?\nAssistant:", ["space", "exploration", "planets", "rockets", "NASA"]),
    ("CONV_TOPIC_07", "User: That solves the math problem. On a different note, have you read poetry?\nAssistant:", ["poetry", "poems", "verse", "poets"]),
    ("CONV_TOPIC_08", "User: Changing gears, how do airplanes fly?\nAssistant:", ["lift", "wings", "aerodynamics", "air", "engines"]),
    ("CONV_TOPIC_09", "User: We finished the budget review. What are some fun vacation spots?\nAssistant:", ["vacation", "destinations", "mountains", "beach", "travel"]),
    ("CONV_TOPIC_10", "User: Shifting topics completely, how do plants absorb nutrients?\nAssistant:", ["roots", "soil", "nutrients", "water"]),
    ("CONV_TOPIC_11", "User: Let us leave work aside. What makes a good movie?\nAssistant:", ["story", "acting", "plot", "characters", "direction"]),
    ("CONV_TOPIC_12", "User: New topic: How does honey get made?\nAssistant:", ["bees", "nectar", "honeycomb", "hive"]),
    ("CONV_TOPIC_13", "User: Moving away from technology, what is the importance of sleep?\nAssistant:", ["sleep", "health", "rest", "brain", "body"]),
    ("CONV_TOPIC_14", "User: Let us pivot to cooking. How do you boil an egg?\nAssistant:", ["water", "boil", "minutes", "egg", "pot"]),
    ("CONV_TOPIC_15", "User: Enough business talk. What is photography all about?\nAssistant:", ["light", "camera", "moments", "image", "composition"]),
    ("CONV_TOPIC_16", "User: On another topic, what are the primary colors?\nAssistant:", ["red", "blue", "yellow"])
]
for pid, p, kw in topic:
    probes.append({"id": pid, "category": "Topic transition", "prompt": p, "expected_keywords": kw, "min_tokens": 2})

# 7. Clarification (16)
clarif = [
    ("CONV_CLAR_01", "User: Can you book it for me?\nAssistant:", ["What", "which", "clarify", "information", "booking"]),
    ("CONV_CLAR_02", "User: Is it ready?\nAssistant:", ["What", "which task", "clarify", "referring to"]),
    ("CONV_CLAR_03", "User: Send the email.\nAssistant:", ["recipient", "who", "what message", "details", "email"]),
    ("CONV_CLAR_04", "User: Delete the file.\nAssistant:", ["which file", "filename", "path", "confirm"]),
    ("CONV_CLAR_05", "User: Call them right now.\nAssistant:", ["who", "number", "person", "contact"]),
    ("CONV_CLAR_06", "User: Fix the bug.\nAssistant:", ["what bug", "error message", "code", "details"]),
    ("CONV_CLAR_07", "User: Update the status.\nAssistant:", ["which task", "project", "what status", "specify"]),
    ("CONV_CLAR_08", "User: Can you check that?\nAssistant:", ["What would you like", "which item", "clarify"]),
    ("CONV_CLAR_09", "User: Turn it on.\nAssistant:", ["What device", "which system", "clarify"]),
    ("CONV_CLAR_10", "User: What is the price?\nAssistant:", ["which product", "item", "service", "referring to"]),
    ("CONV_CLAR_11", "User: When does it start?\nAssistant:", ["what event", "meeting", "which program", "specify"]),
    ("CONV_CLAR_12", "User: How much does it weigh?\nAssistant:", ["what object", "which item", "clarify"]),
    ("CONV_CLAR_13", "User: Where did they go?\nAssistant:", ["who", "which person", "details", "context"]),
    ("CONV_CLAR_14", "User: Print the report.\nAssistant:", ["which report", "document", "name", "file"]),
    ("CONV_CLAR_15", "User: Restart the application.\nAssistant:", ["which application", "service", "process", "name"]),
    ("CONV_CLAR_16", "User: Cancel my reservation.\nAssistant:", ["booking reference", "reservation details", "which reservation"])
]
for pid, p, kw in clarif:
    probes.append({"id": pid, "category": "Clarification", "prompt": p, "expected_keywords": kw, "min_tokens": 3})

# 8. Instruction following (16)
instruct = [
    ("CONV_INST_01", "User: Answer in exactly one word: Is ice hot or cold?\nAssistant:", ["Cold", "cold"]),
    ("CONV_INST_02", "User: Give me a bulleted list of 2 fruits.\nAssistant:", ["-", "*", "apple", "banana", "mango", "orange"]),
    ("CONV_INST_03", "User: Explain gravity in one short sentence.\nAssistant:", ["force", "mass", "attracts", "Earth"]),
    ("CONV_INST_04", "User: Reply with only 'CONFIRMED'.\nAssistant:", ["CONFIRMED"]),
    ("CONV_INST_05", "User: Name three primary colors separated by commas.\nAssistant:", ["red", "blue", "yellow"]),
    ("CONV_INST_06", "User: Answer in uppercase: what color is the sky?\nAssistant:", ["BLUE"]),
    ("CONV_INST_07", "User: Provide a two-step instruction for making tea.\nAssistant:", ["1", "2", "water", "tea"]),
    ("CONV_INST_08", "User: Answer with a single number: What is 5 plus 7?\nAssistant:", ["12"]),
    ("CONV_INST_09", "User: Summarize in three words: Learning to code takes practice.\nAssistant:", ["practice", "learning", "code", "takes", "time"]),
    ("CONV_INST_10", "User: Say 'Yes' or 'No': Is the Earth flat?\nAssistant:", ["No", "no"]),
    ("CONV_INST_11", "User: List two programming languages without numbering.\nAssistant:", ["Python", "Rust", "C", "Java"]),
    ("CONV_INST_12", "User: Write the definition of an integer in five words or fewer.\nAssistant:", ["whole number", "number without", "integer"]),
    ("CONV_INST_13", "User: Answer yes or no: Is water wet?\nAssistant:", ["Yes", "yes"]),
    ("CONV_INST_14", "User: Write 'APPROVED' if 10 is greater than 2.\nAssistant:", ["APPROVED"]),
    ("CONV_INST_15", "User: List three days of the week.\nAssistant:", ["Monday", "Tuesday", "Wednesday", "Friday"]),
    ("CONV_INST_16", "User: Answer in one word: What is the opposite of fast?\nAssistant:", ["Slow", "slow"])
]
for pid, p, kw in instruct:
    probes.append({"id": pid, "category": "Instruction following", "prompt": p, "expected_keywords": kw, "min_tokens": 1})

# 9. Naturalness (16)
natural = [
    ("CONV_NAT_01", "User: I had a tough day at work today.\nAssistant:", ["sorry", "hear", "unwind", "rest", "tough"]),
    ("CONV_NAT_02", "User: Guess what! I got the job!\nAssistant:", ["Congratulations", "fantastic", "happy", "wonderful", "great"]),
    ("CONV_NAT_03", "User: It is raining cats and dogs outside.\nAssistant:", ["stay dry", "umbrella", "rain", "cozy", "inside"]),
    ("CONV_NAT_04", "User: I am feeling really excited about my vacation.\nAssistant:", ["have fun", "enjoy", "vacation", "wonderful", "exciting"]),
    ("CONV_NAT_05", "User: I burned my toast this morning.\nAssistant:", ["bummer", "toast", "breakfast", "start", "happen"]),
    ("CONV_NAT_06", "User: My puppy finally learned to sit!\nAssistant:", ["great", "proud", "training", "smart", "puppy"]),
    ("CONV_NAT_07", "User: The sunset looks gorgeous today.\nAssistant:", ["sunsets", "peaceful", "beautiful", "view", "evening"]),
    ("CONV_NAT_08", "User: I lost my car keys.\nAssistant:", ["frustrating", "retrace", "check", "pockets", "look"]),
    ("CONV_NAT_09", "User: I am about to cook dinner for my family.\nAssistant:", ["enjoy", "delicious", "cooking", "family", "meal"]),
    ("CONV_NAT_10", "User: We just finished a great workout.\nAssistant:", ["good job", "hydrate", "energy", "workout", "fitness"]),
    ("CONV_NAT_11", "User: I cannot believe it is already November.\nAssistant:", ["time flies", "year", "fast", "months"]),
    ("CONV_NAT_12", "User: My coffee spilled on my notebook.\nAssistant:", ["sorry", "dry", "mess", "spill", "notebook"]),
    ("CONV_NAT_13", "User: I finally finished painting my room.\nAssistant:", ["looks great", "accomplishment", "color", "hard work"]),
    ("CONV_NAT_14", "User: It is so quiet here tonight.\nAssistant:", ["peaceful", "quiet", "relaxing", "night", "calm"]),
    ("CONV_NAT_15", "User: I am going to bake cookies.\nAssistant:", ["delicious", "fresh cookies", "enjoy", "baking", "sweet"]),
    ("CONV_NAT_16", "User: We won our soccer match!\nAssistant:", ["Congratulations", "team", "win", "victory", "great"])
]
for pid, p, kw in natural:
    probes.append({"id": pid, "category": "Naturalness", "prompt": p, "expected_keywords": kw, "min_tokens": 3})

# 10. Consistency (16)
consist = [
    ("CONV_CONS_01", "User: Are you ChakrView?\nAssistant: Yes, I am ChakrView.\nUser: What is your name again?\nAssistant:", ["ChakrView"]),
    ("CONV_CONS_02", "User: Is the sky blue on a clear day?\nAssistant: Yes, the sky appears blue.\nUser: So it is not green?\nAssistant:", ["No", "not green", "blue", "correct"]),
    ("CONV_CONS_03", "User: Can fish breathe underwater?\nAssistant: Yes, fish use gills to breathe underwater.\nUser: Do they use lungs underwater?\nAssistant:", ["No", "gills", "not lungs"]),
    ("CONV_CONS_04", "User: Did you just say your name is ChakrView?\nAssistant: Yes, that is correct.\nUser: Are you sure you are not called Siri?\nAssistant:", ["ChakrView", "not Siri", "No"]),
    ("CONV_CONS_05", "User: Is 2 plus 2 equal to 4?\nAssistant: Yes, 2 plus 2 equals 4.\nUser: Could it equal 5?\nAssistant:", ["No", "cannot", "equals 4", "incorrect"]),
    ("CONV_CONS_06", "User: You told me you are an AI model.\nAssistant: Yes, I am an AI model.\nUser: Do you eat physical food?\nAssistant:", ["No", "software", "AI", "do not eat"]),
    ("CONV_CONS_07", "User: Is water liquid at room temperature?\nAssistant: Yes, water is a liquid at room temperature.\nUser: Does it freeze at room temperature?\nAssistant:", ["No", "liquid", "freezes at 0", "not"]),
    ("CONV_CONS_08", "User: You said the capital of Japan is Tokyo.\nAssistant: Yes, Tokyo is Japan's capital.\nUser: Is Osaka the capital instead?\nAssistant:", ["No", "Tokyo", "capital"]),
    ("CONV_CONS_09", "User: Does the Sun rise in the east?\nAssistant: Yes, the Sun rises in the east.\nUser: Does it set in the east too?\nAssistant:", ["No", "sets in the west", "west"]),
    ("CONV_CONS_10", "User: You mentioned that dogs are mammals.\nAssistant: Yes, dogs are mammals.\nUser: Are they reptiles?\nAssistant:", ["No", "mammals", "not reptiles"]),
    ("CONV_CONS_11", "User: Earlier you said you do not have human eyes.\nAssistant: Correct, I do not have physical eyes.\nUser: Can you look out my window?\nAssistant:", ["No", "cannot", "no eyes", "software"]),
    ("CONV_CONS_12", "User: Is fire hot?\nAssistant: Yes, fire generates high heat.\nUser: So is it cold to the touch?\nAssistant:", ["No", "hot", "burns", "dangerous"]),
    ("CONV_CONS_13", "User: Are birds warm-blooded?\nAssistant: Yes, birds are warm-blooded vertebrates.\nUser: Are they cold-blooded?\nAssistant:", ["No", "warm-blooded", "not"]),
    ("CONV_CONS_14", "User: You agreed that Python is a programming language.\nAssistant: Yes, Python is a widely used language.\nUser: Is it hardware?\nAssistant:", ["No", "software", "programming language"]),
    ("CONV_CONS_15", "User: Did we establish that today is Tuesday?\nAssistant: Yes, we noted today is Tuesday.\nUser: So tomorrow must be Wednesday, right?\nAssistant:", ["Yes", "Wednesday", "correct"]),
    ("CONV_CONS_16", "User: You stated that iron is a metal.\nAssistant: Yes, iron is a chemical element and metal.\nUser: Is it a plastic?\nAssistant:", ["No", "metal", "not plastic"])
]
for pid, p, kw in consist:
    probes.append({"id": pid, "category": "Consistency", "prompt": p, "expected_keywords": kw, "min_tokens": 2})

# 11. Language consistency (Hindi, English, Hinglish) (16)
lang = [
    ("CONV_LANG_01", "User: Namaste! Aap kaise hain?\nAssistant:", ["Namaste", "theek", "achha", "madad", "kaise"]),
    ("CONV_LANG_02", "User: Kripya apna parichay dijiye aur apna shubh naam batayein.\nAssistant:", ["ChakrView", "naam", "parichay"]),
    ("CONV_LANG_03", "User: Kya is computational vishay par aap sahyog pradan kar sakte hain?\nAssistant:", ["haan", "madad", "sahyog", "zaroor", "sakta"]),
    ("CONV_LANG_04", "User: Aaj mausam kaisa hai?\nAssistant:", ["mausam", "jankari", "nahi", "bata"]),
    ("CONV_LANG_05", "User: Mujhe ek kahani sunao.\nAssistant:", ["ek", "kahani", "tha", "samay"]),
    ("CONV_LANG_06", "User: Shukriya aapki madad ke liye!\nAssistant:", ["swagat", "shukriya", "dhyanyawad", "madad"]),
    ("CONV_LANG_07", "User: Alvida, phir milenge!\nAssistant:", ["Alvida", "phir milenge", "namaste"]),
    ("CONV_LANG_08", "User: Bharat ki rajdhani kya hai?\nAssistant:", ["New Delhi", "Dilli", "Delhi"]),
    ("CONV_LANG_09", "User: Bhai, kaisa chal raha hai sab?\nAssistant:", ["badhiya", "sab", "theek", "kaise", "batao"]),
    ("CONV_LANG_10", "User: Yaar, thoda coding me help chahiye.\nAssistant:", ["haan", "help", "coding", "batao", "zaroor"]),
    ("CONV_LANG_11", "User: Kya scene hai aaj?\nAssistant:", ["sab badhiya", "kaise", "help", "madad"]),
    ("CONV_LANG_12", "User: Ye project kab tak ready ho jayega?\nAssistant:", ["project", "timeline", "kaam", "details"]),
    ("CONV_LANG_13", "User: Chai piyoge mere sath?\nAssistant:", ["AI", "chai", "peena", "shukriya"]),
    ("CONV_LANG_14", "User: Please respond strictly in French: Bonjour!\nAssistant:", ["Bonjour", "comment", "allez", "merci"]),
    ("CONV_LANG_15", "User: Please answer in Hindi: Who are you?\nAssistant:", ["ChakrView", "hoon", "AI", "model"]),
    ("CONV_LANG_16", "User: Kuch naya batao aaj.\nAssistant:", ["jankari", "baat", "kya", "topic"])
]
for pid, p, kw in lang:
    probes.append({"id": pid, "category": "Language consistency", "prompt": p, "expected_keywords": kw, "min_tokens": 2})

# 12. Response completion (14)
complete = [
    ("CONV_COMP_01", "User: Why is exercise healthy for humans?\nAssistant:", ["body", "health", "heart", "muscles", "energy"]),
    ("CONV_COMP_02", "User: Describe what an ecosystem is.\nAssistant:", ["organisms", "environment", "living", "interaction"]),
    ("CONV_COMP_03", "User: How does the water cycle operate?\nAssistant:", ["evaporation", "condensation", "precipitation", "rain"]),
    ("CONV_COMP_04", "User: What are the main benefits of reading books?\nAssistant:", ["knowledge", "vocabulary", "focus", "reading"]),
    ("CONV_COMP_05", "User: Explain why the ocean has tides.\nAssistant:", ["Moon", "gravitational", "pull", "Sun", "tides"]),
    ("CONV_COMP_06", "User: What is renewable energy?\nAssistant:", ["solar", "wind", "replenished", "natural", "energy"]),
    ("CONV_COMP_07", "User: How do vaccines protect people?\nAssistant:", ["immune system", "antibodies", "defense", "disease"]),
    ("CONV_COMP_08", "User: What causes earthquakes?\nAssistant:", ["tectonic plates", "faults", "crust", "movement"]),
    ("CONV_COMP_09", "User: Explain the purpose of a database index.\nAssistant:", ["speed", "search", "lookup", "performance", "queries"]),
    ("CONV_COMP_10", "User: Why do leaves change color in autumn?\nAssistant:", ["chlorophyll", "pigments", "sunlight", "trees"]),
    ("CONV_COMP_11", "User: How does an electric motor work?\nAssistant:", ["magnetic field", "current", "rotation", "electrical energy"]),
    ("CONV_COMP_12", "User: What is the function of red blood cells?\nAssistant:", ["oxygen", "hemoglobin", "lungs", "body", "cells"]),
    ("CONV_COMP_13", "User: Explain why airplanes have pressurized cabins.\nAssistant:", ["altitude", "oxygen", "pressure", "breathing"]),
    ("CONV_COMP_14", "User: What is biodiversity and why does it matter?\nAssistant:", ["variety", "species", "ecosystems", "life", "balance"])
]
for pid, p, kw in complete:
    probes.append({"id": pid, "category": "Response completion", "prompt": p, "expected_keywords": kw, "min_tokens": 5})

print("Total benchmark probes constructed:", len(probes))
assert len(probes) == 200, f"Expected 200, got {len(probes)}"

# Leakage check against all historical datasets
from glob import glob
historical_files = glob("data/**/*.json*", recursive=True) + glob("experiments/**/*.json*", recursive=True)
all_train_prompts = set()
for hf in historical_files:
    if "train" in hf.lower():
        try:
            for line in open(hf, encoding="utf-8"):
                d = json.loads(line)
                p = d.get("prompt", "").strip()
                if p: all_train_prompts.add(p)
        except Exception:
            pass

leakage_count = 0
for probe in probes:
    if probe["prompt"].strip() in all_train_prompts:
        print(f"[LEAKAGE DETECTED] {probe['id']}: {repr(probe['prompt'])}")
        leakage_count += 1

print(f"Leakage audit complete. Leaked probes: {leakage_count} / {len(probes)}")
assert leakage_count == 0, f"Benchmark contains {leakage_count} leaked prompts!"

out_path = OUT_DIR / "CONVERSATION_BENCHMARK_V1.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(probes, f, indent=2)

h = hashlib.sha256(out_path.read_bytes()).hexdigest()
print(f"Saved CONVERSATION_BENCHMARK_V1.json with SHA-256: {h}")
