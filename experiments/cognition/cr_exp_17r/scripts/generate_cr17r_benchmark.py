import json
import hashlib
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
OUT_DIR = REPO_ROOT / "experiments/cognition/cr_exp_17r/benchmark"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Build 100 completely fresh probes across 10 categories (10 probes each)
# Designed specifically to test conversational competence, entity recall, follow-up, topic switching, Hindi/Hinglish, and refusal boundaries.
# Zero overlap with CR-17 benchmark and training corpus.

probes = []
pid = 1

def add_probe(category, history, current_user, expected_behavior, required_info, failure_conditions, answerable):
    global pid
    prompt_str = ""
    for turn in history:
        prompt_str += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    prompt_str += f"User: {current_user}\nAssistant:"
    
    probes.append({
        "probe_id": f"CR17R_PROBE_{pid:03d}",
        "category": category,
        "history": history,
        "current_user_message": current_user,
        "formatted_prompt": prompt_str,
        "expected_behavior": expected_behavior,
        "required_info": required_info,
        "failure_conditions": failure_conditions,
        "answerability": answerable
    })
    pid += 1

# 1. Greetings and Introductions (10 probes)
add_probe("1_greetings_introductions", [], "Good day! How are you doing today?", "Polite greeting response acknowledging the user", ["greeting"], ["refusal sink", "repetition loop"], True)
add_probe("1_greetings_introductions", [], "A very warm welcome to you! How is everything?", "Courteous welcoming greeting", ["greeting"], ["refusal sink", "word salad"], True)
add_probe("1_greetings_introductions", [], "Good afternoon! Are you available to assist me?", "Affirmative polite response indicating readiness to assist", ["assist", "ready"], ["refusal sink", "repetitive loop"], True)
add_probe("1_greetings_introductions", [], "Greetings! Hope you are having a productive cycle.", "Friendly acknowledgment of greeting", ["greeting"], ["refusal sink"], True)
add_probe("1_greetings_introductions", [], "Hello there! What kind of assistant are you?", "Introduction identifying itself as an AI/ChakrView assistant", ["assistant", "ChakrView"], ["refusal sink"], True)
add_probe("1_greetings_introductions", [], "Hey! Nice to connect with you.", "Friendly conversational opening", ["hello", "welcome"], ["refusal sink"], True)
add_probe("1_greetings_introductions", [], "Good evening! Can you lend me a hand tonight?", "Affirmative polite response expressing helpfulness", ["help", "assist"], ["refusal sink"], True)
add_probe("1_greetings_introductions", [], "Hi! Who am I speaking with right now?", "Clear self-identification as ChakrView neural model", ["ChakrView"], ["refusal sink"], True)
add_probe("1_greetings_introductions", [], "Salutations! What is your primary design goal?", "Statement of role as an AI assistant developed from first principles", ["assist", "neural"], ["refusal sink"], True)
add_probe("1_greetings_introductions", [], "Greetings friend! Are you running natively or in the cloud?", "Acknowledgment of running natively / local CPU architecture", ["CPU", "local", "neural"], ["refusal sink"], True)

# 2. Natural Conversational Responses (10 probes)
add_probe("2_natural_conversational", [], "I finally finished all my work for the day!", "Congratulatory or supportive conversational remark", ["great", "relax", "good"], ["refusal sink", "repetitive tokens"], True)
add_probe("2_natural_conversational", [], "That was a really helpful explanation, thank you!", "Gracious acknowledgment of thanks (you are welcome)", ["welcome", "glad", "help"], ["refusal sink"], True)
add_probe("2_natural_conversational", [], "I have a long drive ahead of me tomorrow.", "Polite wish for a safe trip or supportive remark", ["safe", "drive", "trip"], ["refusal sink"], True)
add_probe("2_natural_conversational", [], "Coffee is definitely my favorite morning drink.", "Conversational agreement or pleasant remark about coffee", ["coffee", "morning"], ["refusal sink"], True)
add_probe("2_natural_conversational", [], "I really enjoy solving difficult puzzles.", "Supportive remark on puzzle solving or problem solving", ["puzzles", "enjoy"], ["refusal sink"], True)
add_probe("2_natural_conversational", [], "It is quite chilly outside this evening.", "Conversational acknowledgment of the cold weather", ["cold", "warm", "chilly"], ["refusal sink"], True)
add_probe("2_natural_conversational", [], "I am preparing a presentation for my colleagues.", "Encouraging remark offering help with slides or talking points", ["presentation", "help"], ["refusal sink"], True)
add_probe("2_natural_conversational", [], "I appreciate your quick replies today.", "Gracious acknowledgment of the user's appreciation", ["welcome", "happy to help"], ["refusal sink"], True)
add_probe("2_natural_conversational", [], "Have a peaceful evening!", "Polite reciprocating farewell wish", ["evening", "thank you", "good evening"], ["refusal sink"], True)
add_probe("2_natural_conversational", [], "Take care and see you later!", "Polite parting response", ["take care", "goodbye", "later"], ["refusal sink"], True)

# 3. Follow-up Questions (10 probes)
add_probe("3_follow_up_questions", [{"user": "I am looking for a light snack.", "assistant": "Fruits like apples or bananas are healthy light snacks."}], "Which of those has fewer calories?", "Comparison or follow-up response addressing apples vs bananas", ["apple", "banana", "calories"], ["refusal sink"], True)
add_probe("3_follow_up_questions", [{"user": "Python is my favorite programming language.", "assistant": "Python is known for its clear readability and versatile libraries."}], "Why is it so popular for beginners?", "Explanation citing syntax, readability, or simplicity", ["syntax", "simple", "easy", "readable"], ["refusal sink"], True)
add_probe("3_follow_up_questions", [{"user": "I need to travel to Mumbai.", "assistant": "You can travel to Mumbai by flight, train, or road depending on your schedule."}], "Which is typically the fastest option?", "Answer identifying flights/air travel as the fastest", ["flight", "plane", "air"], ["refusal sink"], True)
add_probe("3_follow_up_questions", [{"user": "I bought a new mechanical keyboard.", "assistant": "Mechanical keyboards offer great tactile feedback and durability."}], "Do they make a lot of clicking noise?", "Affirmative explanation regarding mechanical switches and clicking", ["switches", "click", "noise", "yes"], ["refusal sink"], True)
add_probe("3_follow_up_questions", [{"user": "Let us talk about databases.", "assistant": "Databases store and manage structured or unstructured data."}], "What is the difference between SQL and NoSQL?", "Explanation contrasting relational tables with document/flexible storage", ["SQL", "relational", "tables", "NoSQL"], ["refusal sink"], True)
add_probe("3_follow_up_questions", [{"user": "I am learning how to play chess.", "assistant": "Chess is a deep strategic game involving positional planning."}], "What is the primary objective of the game?", "Answer stating checkmate or capturing the opponent's king", ["king", "checkmate", "mate"], ["refusal sink"], True)
add_probe("3_follow_up_questions", [{"user": "The weather forecast says heavy rain.", "assistant": "Make sure to carry an umbrella and drive carefully."}], "Should I take an umbrella or a raincoat?", "Supportive confirmation that either is good", ["umbrella", "raincoat", "rain"], ["refusal sink"], True)
add_probe("3_follow_up_questions", [{"user": "I am studying physics fundamentals.", "assistant": "Physics explores matter, energy, space, and time."}], "Who formulated the laws of classical motion?", "Identification of Sir Isaac Newton", ["Newton", "Isaac"], ["refusal sink"], True)
add_probe("3_follow_up_questions", [{"user": "I am planning to bake some bread today.", "assistant": "Baking fresh bread requires flour, yeast, water, and salt."}], "What role does yeast play in the dough?", "Explanation of yeast causing fermentation and dough rising", ["rise", "ferment", "gas", "leavening"], ["refusal sink"], True)
add_probe("3_follow_up_questions", [{"user": "We are designing a compact neural network.", "assistant": "Compact neural architectures require efficient parameter allocation."}], "Why does parameter count matter for CPU execution?", "Explanation connecting smaller parameters to faster CPU cache fits and lower compute", ["cache", "latency", "compute", "memory", "speed"], ["refusal sink"], True)

# 4. Context-Dependent References (10 probes)
add_probe("4_context_references", [{"user": "My cat Oliver is sleeping on the couch.", "assistant": "Cats love cozy sleeping spots."}], "Where is he sleeping?", "Reference resolution identifying the couch", ["couch"], ["refusal sink", "wrong entity"], True)
add_probe("4_context_references", [{"user": "I parked my blue sedan on 4th Street.", "assistant": "I have noted that your blue sedan is on 4th Street."}], "What color is it?", "Reference resolution identifying blue", ["blue"], ["refusal sink", "wrong color"], True)
add_probe("4_context_references", [{"user": "Dr. Sharma is arriving at Terminal 2.", "assistant": "Terminal 2 arrival recorded for Dr. Sharma."}], "Which terminal did you say?", "Reference resolution identifying Terminal 2", ["Terminal 2", "2"], ["refusal sink"], True)
add_probe("4_context_references", [{"user": "We scheduled our sprint planning for 10:00 AM.", "assistant": "Sprint planning is scheduled for 10:00 AM."}], "At what time does it start?", "Reference resolution identifying 10:00 AM", ["10:00", "10 AM"], ["refusal sink"], True)
add_probe("4_context_references", [{"user": "The server room temperature reached 28 degrees Celsius.", "assistant": "That is warmer than optimal server room conditions."}], "What temperature did it reach?", "Reference resolution identifying 28 degrees Celsius", ["28"], ["refusal sink"], True)
add_probe("4_context_references", [{"user": "Alice submitted the engineering proposal yesterday.", "assistant": "The engineering proposal submission is noted."}], "Who submitted it?", "Reference resolution identifying Alice", ["Alice"], ["refusal sink"], True)
add_probe("4_context_references", [{"user": "The meeting is booked in Conference Room B.", "assistant": "Meeting room confirmed for Room B."}], "Which room was it again?", "Reference resolution identifying Room B", ["Room B", "B"], ["refusal sink"], True)
add_probe("4_context_references", [{"user": "I ordered a pepperoni pizza with extra cheese.", "assistant": "Your pepperoni pizza order sounds delicious."}], "What topping did I ask for extra of?", "Reference resolution identifying extra cheese", ["cheese"], ["refusal sink"], True)
add_probe("4_context_references", [{"user": "My bicycle has a flat rear tire.", "assistant": "You will need a patch kit or a new tube for the rear tire."}], "Which tire has the problem?", "Reference resolution identifying rear tire", ["rear"], ["refusal sink"], True)
add_probe("4_context_references", [{"user": "Professor Sen is teaching Quantum Mechanics this semester.", "assistant": "Quantum Mechanics is a challenging and fascinating subject."}], "What course is he teaching?", "Reference resolution identifying Quantum Mechanics", ["Quantum Mechanics"], ["refusal sink"], True)

# 5. User Identity and Project Grounding (10 probes)
add_probe("5_identity_project_grounding", [], "What is your official system name?", "Clear statement that its name is ChakrView", ["ChakrView"], ["refusal sink", "ChatGPT", "Llama"], True)
add_probe("5_identity_project_grounding", [], "Who developed you?", "Statement identifying the ChakrView team or first-principles indigenous development", ["ChakrView", "first principles", "team"], ["refusal sink", "OpenAI", "Meta"], True)
add_probe("5_identity_project_grounding", [], "Are you built upon OpenAI GPT models?", "Denial stating it is an independent architecture built from first principles", ["No", "independent", "first principles"], ["refusal sink", "Yes"], True)
add_probe("5_identity_project_grounding", [], "What is your underlying neural architecture family?", "Identification as ChakrMicro / Transformer decoder", ["ChakrMicro", "transformer", "neural"], ["refusal sink"], True)
add_probe("5_identity_project_grounding", [], "How many layers does the ChakrMicro v0.1 architecture have?", "Accurate statement of 6 layers", ["6", "six"], ["refusal sink"], True)
add_probe("5_identity_project_grounding", [], "Are you designed to run natively on host CPUs?", "Affirmation of native host CPU execution", ["CPU", "host", "yes"], ["refusal sink"], True)
add_probe("5_identity_project_grounding", [], "Do you require external cloud APIs to respond?", "Clarification that execution is local and autonomous without external APIs", ["No", "local", "autonomous", "independent"], ["refusal sink", "Yes"], True)
add_probe("5_identity_project_grounding", [], "Are you an experimental preview model?", "Affirmative acknowledgment of being the v0.9.1 experimental preview", ["v0.9.1", "preview", "experimental", "yes"], ["refusal sink"], True)
add_probe("5_identity_project_grounding", [], "Which foundational hardware constraint guides your system architecture?", "Statement prioritizing CPU efficiency, sovereign intelligence, or compact footprint", ["CPU", "compact", "host", "memory"], ["refusal sink"], True)
add_probe("5_identity_project_grounding", [], "Can you identify yourself and your version?", "Identification as ChakrView v0.9.1", ["ChakrView", "v0.9.1"], ["refusal sink"], True)

# 6. Multi-turn Context Retention (10 probes)
add_probe("6_multiturn_retention", [
    {"user": "My name is Vikram and I am a software architect.", "assistant": "Hello Vikram, great to meet a software architect!"},
    {"user": "I am currently refactoring our authentication service.", "assistant": "Authentication refactoring requires careful session handling."}
], "What is my profession again?", "Accurate recall of software architect", ["software architect", "architect"], ["refusal sink", "wrong entity"], True)

add_probe("6_multiturn_retention", [
    {"user": "I live in Bengaluru and work remotely.", "assistant": "Bengaluru is a vibrant tech hub!"},
    {"user": "The weather is very pleasant here right now.", "assistant": "Glad to hear you are enjoying good weather."}
], "Which city did I say I live in?", "Accurate recall of Bengaluru", ["Bengaluru", "Bangalore"], ["refusal sink", "wrong city"], True)

add_probe("6_multiturn_retention", [
    {"user": "Our team has 8 engineers and 2 product designers.", "assistant": "That sounds like a well-balanced product engineering squad."}
], "How many engineers are on our team?", "Accurate recall of 8 engineers", ["8", "eight"], ["refusal sink", "wrong number"], True)

add_probe("6_multiturn_retention", [
    {"user": "I am traveling to Kyoto next Thursday.", "assistant": "Kyoto is beautiful in all seasons!"},
    {"user": "I plan to visit historic temples there.", "assistant": "Kinkaku-ji and Fushimi Inari are wonderful temples."}
], "Where am I traveling next Thursday?", "Accurate recall of Kyoto", ["Kyoto"], ["refusal sink"], True)

add_probe("6_multiturn_retention", [
    {"user": "My favorite programming language is Rust.", "assistant": "Rust provides excellent memory safety and performance."}
], "What language did I say was my favorite?", "Accurate recall of Rust", ["Rust"], ["refusal sink", "wrong language"], True)

add_probe("6_multiturn_retention", [
    {"user": "We adopted PostgreSQL as our primary database.", "assistant": "PostgreSQL is a robust and feature-rich relational engine."}
], "Which primary database did we adopt?", "Accurate recall of PostgreSQL", ["PostgreSQL", "Postgres"], ["refusal sink"], True)

add_probe("6_multiturn_retention", [
    {"user": "My dog is named Bruno.", "assistant": "Bruno is a great name for a dog!"},
    {"user": "He loves chasing tennis balls in the park.", "assistant": "Tennis balls provide great exercise for energetic dogs."}
], "What is the name of my dog?", "Accurate recall of Bruno", ["Bruno"], ["refusal sink", "wrong name"], True)

add_probe("6_multiturn_retention", [
    {"user": "I have an exam on Linear Algebra on Monday.", "assistant": "Linear algebra is foundational for vectors and matrices."}
], "What subject is my exam on?", "Accurate recall of Linear Algebra", ["Linear Algebra"], ["refusal sink"], True)

add_probe("6_multiturn_retention", [
    {"user": "We deployed our microservice on port 9090.", "assistant": "Port 9090 is configured for your microservice."}
], "Which port was used for the microservice?", "Accurate recall of port 9090", ["9090"], ["refusal sink"], True)

add_probe("6_multiturn_retention", [
    {"user": "I am reading Crime and Punishment.", "assistant": "Dostoevsky's Crime and Punishment is a masterpiece of psychological drama."}
], "What book did I say I am reading?", "Accurate recall of Crime and Punishment", ["Crime and Punishment"], ["refusal sink"], True)

# 7. Topic Changes (10 probes)
add_probe("7_topic_changes", [
    {"user": "We were discussing cloud architectures.", "assistant": "Cloud architectures emphasize elastic scaling and resilience."}
], "Let us switch gears: what is the boiling point of water at sea level?", "Correct transition answering 100 degrees Celsius", ["100", "boiling", "Celsius"], ["refusal sink"], True)

add_probe("7_topic_changes", [
    {"user": "I was telling you about my morning workout.", "assistant": "Exercise is essential for maintaining physical and mental health."}
], "Changing the topic completely, how many continents are there on Earth?", "Correct transition answering 7 continents", ["7", "seven"], ["refusal sink"], True)

add_probe("7_topic_changes", [
    {"user": "The database migration completed successfully.", "assistant": "Great to hear the database migration succeeded without issues."}
], "Moving on to something else: do penguins fly in the air?", "Correct transition stating penguins do not fly in the air", ["No", "cannot fly", "flightless"], ["refusal sink"], True)

add_probe("7_topic_changes", [
    {"user": "I had a sandwich for lunch.", "assistant": "Sandwiches are quick and satisfying meals."}
], "On a totally different note, what is the largest planet in our solar system?", "Correct transition answering Jupiter", ["Jupiter"], ["refusal sink"], True)

add_probe("7_topic_changes", [
    {"user": "We fixed the frontend CSS bug.", "assistant": "Frontend styling fixes make user experiences seamless."}
], "Let us talk about music now: what instrument has 88 keys?", "Correct transition answering the piano", ["piano"], ["refusal sink"], True)

add_probe("7_topic_changes", [
    {"user": "I like writing Python scripts.", "assistant": "Python scripts automate repetitive developer workflows."}
], "Enough about code, what color do you get by mixing red and yellow?", "Correct transition answering orange", ["orange"], ["refusal sink"], True)

add_probe("7_topic_changes", [
    {"user": "The flight was delayed by two hours.", "assistant": "Flight delays can be tedious and tiring."}
], "Forget flights for a moment: what gas do plants absorb during photosynthesis?", "Correct transition answering carbon dioxide / CO2", ["carbon dioxide", "CO2"], ["refusal sink"], True)

add_probe("7_topic_changes", [
    {"user": "I am reviewing pull requests.", "assistant": "Code reviews maintain software quality."}
], "Switching subjects: how many days are there in a standard year?", "Correct transition answering 365 days", ["365"], ["refusal sink"], True)

add_probe("7_topic_changes", [
    {"user": "We are testing Docker containers.", "assistant": "Docker provides isolated execution environments."}
], "Now for something different: what is the capital of France?", "Correct transition answering Paris", ["Paris"], ["refusal sink"], True)

add_probe("7_topic_changes", [
    {"user": "My tea is getting cold.", "assistant": "You should drink your tea while it is warm!"}
], "Let us pivot: what is the chemical formula for water?", "Correct transition answering H2O", ["H2O", "water"], ["refusal sink"], True)

# 8. Uncertainty and Missing Information (10 probes)
add_probe("8_uncertainty_missing_info", [], "What did I eat for breakfast this morning?", "Natural statement that the model does not know what the user ate without context", ["not know", "do not know", "cannot know"], ["refusal sink", "hallucinated breakfast"], False)
add_probe("8_uncertainty_missing_info", [], "Where did I put my car keys?", "Natural acknowledgment of missing information", ["do not know", "cannot know", "unaware"], ["refusal sink", "hallucinated location"], False)
add_probe("8_uncertainty_missing_info", [], "What is my secret password?", "Natural statement that the model does not possess private personal data", ["do not know", "cannot know", "private"], ["refusal sink", "hallucinated password"], False)
add_probe("8_uncertainty_missing_info", [], "Who will win the world cup twenty years from now?", "Appropriate statement that future events cannot be predicted with certainty", ["cannot predict", "unknown", "future"], ["refusal sink", "confident prediction"], False)
add_probe("8_uncertainty_missing_info", [{"user": "I bought some groceries.", "assistant": "Groceries are essential for the week."}], "How much did my groceries cost?", "Natural clarification that the cost was not mentioned", ["did not mention", "do not know", "unspecified"], ["refusal sink", "hallucinated cost"], False)
add_probe("8_uncertainty_missing_info", [], "What is the exact population of my home village right now?", "Natural acknowledgment that private/hyperlocal real-time statistics are unknown", ["do not have", "unknown", "do not know"], ["refusal sink"], False)
add_probe("8_uncertainty_missing_info", [{"user": "My friend called me earlier.", "assistant": "Always good to hear from friends."}], "What did my friend tell me on the phone?", "Natural statement that the phone call details were not shared", ["did not share", "do not know", "not mentioned"], ["refusal sink"], False)
add_probe("8_uncertainty_missing_info", [], "What temperature is it in my living room at this very moment?", "Statement that it has no physical sensors in the room", ["sensors", "do not know", "cannot measure"], ["refusal sink"], False)
add_probe("8_uncertainty_missing_info", [{"user": "I am reading a novel.", "assistant": "Novels offer immersive storytelling."}], "What is the author's middle name?", "Natural clarification that neither the novel nor the author was specified", ["did not mention", "unspecified", "do not know"], ["refusal sink"], False)
add_probe("8_uncertainty_missing_info", [], "What am I thinking about right now?", "Clear statement that AI cannot read human minds", ["cannot read", "do not know", "mind"], ["refusal sink"], False)

# 9. Hindi and Hinglish Conversation (10 probes)
add_probe("9_hindi_hinglish", [], "Namaste! Aapka kya haal hai?", "Polite Hindi greeting response", ["Namaste", "main", "theek", "aap"], ["refusal sink", "word salad"], True)
add_probe("9_hindi_hinglish", [], "Aapka naam kya hai aur aap kya kar sakte hain?", "Self-introduction in Hindi as ChakrView", ["ChakrView", "naam", "madad"], ["refusal sink"], True)
add_probe("9_hindi_hinglish", [], "Bhai, thoda help chahiye tha.", "Friendly Hindi/Hinglish response offering assistance", ["haan", "batao", "madad", "help"], ["refusal sink"], True)
add_probe("9_hindi_hinglish", [], "Kya aap Hindi samajh sakte hain?", "Affirmative Hindi response confirming understanding", ["haan", "Hindi", "samajh", "sakta"], ["refusal sink"], True)
add_probe("9_hindi_hinglish", [], "Shubh prabhat! Aaj ka din kaisa chal raha hai?", "Courteous Hindi morning greeting", ["Shubh prabhat", "achha", "madad"], ["refusal sink"], True)
add_probe("9_hindi_hinglish", [], "Dhanyawad aapki sahayata ke liye!", "Gracious Hindi response to thanks (aapka swagat hai)", ["swagat", "madad", "dhanyawad"], ["refusal sink"], True)
add_probe("9_hindi_hinglish", [], "Coding sikhna chahta hoon, kahan se start karun?", "Helpful guidance in Hindi/Hinglish on starting programming", ["Python", "start", "coding", "basic"], ["refusal sink"], True)
add_probe("9_hindi_hinglish", [], "Aap kaun se computer par run ho rahe hain?", "Response stating local computer / CPU in Hindi/Hinglish", ["computer", "CPU", "system", "local"], ["refusal sink"], True)
add_probe("9_hindi_hinglish", [], "Alvida! Phir milte hain.", "Polite parting in Hindi", ["Alvida", "namaste", "milenge"], ["refusal sink"], True)
add_probe("9_hindi_hinglish", [], "Aapka din shubh ho!", "Reciprocating well-wish in Hindi", ["dhanyawad", "aapka", "shubh"], ["refusal sink"], True)

# 10. Conversation Continuity and Relevance (10 probes)
add_probe("10_continuity_relevance", [
    {"user": "I am working on an open-source project.", "assistant": "Open-source projects foster great community collaboration."}
], "What license should I consider if I want anyone to use it freely?", "Answer suggesting MIT or Apache 2.0 license", ["MIT", "Apache", "license"], ["refusal sink"], True)

add_probe("10_continuity_relevance", [
    {"user": "I am trying to improve my sleep quality.", "assistant": "Consistent sleep schedules and dark rooms improve sleep hygiene."}
], "Should I avoid looking at screens right before bed?", "Affirmation that avoiding blue light/screens before bed is beneficial", ["Yes", "avoid", "screen", "blue light"], ["refusal sink"], True)

add_probe("10_continuity_relevance", [
    {"user": "We are preparing for a long hiking trek.", "assistant": "Sturdy footwear and hydration packs are essential for long treks."}
], "What kind of socks are best to prevent blisters?", "Answer suggesting wool or synthetic moisture-wicking hiking socks", ["wool", "hiking", "synthetic", "socks"], ["refusal sink"], True)

add_probe("10_continuity_relevance", [
    {"user": "My garden has several tomato plants.", "assistant": "Tomatoes thrive in full sun and well-draining soil."}
], "How often should I water them during warm weather?", "Answer recommending deep, consistent watering (every few days)", ["water", "deeply", "daily", "regularly"], ["refusal sink"], True)

add_probe("10_continuity_relevance", [
    {"user": "I am learning git version control.", "assistant": "Git manages history across distributed software repositories."}
], "Which command creates a new branch?", "Answer specifying git branch or git checkout -b", ["branch", "git branch"], ["refusal sink"], True)

add_probe("10_continuity_relevance", [
    {"user": "We are optimizing memory consumption in Python.", "assistant": "Profiling memory allocations reveals unexpected leaks."}
], "Can generators help reduce memory when reading large files?", "Affirmative answer explaining generators yield items on the fly", ["Yes", "generators", "memory", "stream"], ["refusal sink"], True)

add_probe("10_continuity_relevance", [
    {"user": "I am preparing for an interview next week.", "assistant": "Reviewing fundamentals and practicing mock questions builds confidence."}
], "Any tips on handling behavioral interview questions?", "Suggestion like using the STAR method (Situation, Task, Action, Result)", ["STAR", "situation", "examples"], ["refusal sink"], True)

add_probe("10_continuity_relevance", [
    {"user": "We are designing an API for mobile clients.", "assistant": "Mobile APIs should prioritize small payloads and battery efficiency."}
], "Is JSON or XML preferred for modern mobile endpoints?", "Answer stating JSON is widely preferred for lightweight parsing", ["JSON"], ["refusal sink"], True)

add_probe("10_continuity_relevance", [
    {"user": "I am starting to run 5 kilometers every morning.", "assistant": "Consistent morning runs build cardiovascular endurance."}
], "Is warming up and stretching important before running?", "Affirmative answer stating warmups prevent injury", ["Yes", "warmup", "injury", "stretch"], ["refusal sink"], True)

add_probe("10_continuity_relevance", [
    {"user": "We are writing unit tests for our service.", "assistant": "Automated unit tests ensure deterministic regression protection."}
], "Should tests be isolated and independent of each other?", "Affirmative explanation that unit tests must be isolated and repeatable", ["Yes", "isolated", "independent"], ["refusal sink"], True)

# Save benchmark
out_file = OUT_DIR / "CR17R_CONVERSATION_100.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(probes, f, indent=2)

h = hashlib.sha256(out_file.read_bytes()).hexdigest()
manifest = {
    "benchmark_name": "CR17R_CONVERSATION_100",
    "total_probes": len(probes),
    "categories_count": 10,
    "probes_per_category": 10,
    "sha256": h
}
with open(OUT_DIR / "benchmark_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"CR17R_CONVERSATION_100.json generated with {len(probes)} probes.")
print(f"SHA-256: {h}")
