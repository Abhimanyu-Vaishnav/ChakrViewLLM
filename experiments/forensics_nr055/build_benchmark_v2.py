import json
import os
import re

# Benchmark V2 Design
# 100 high-quality, independently authored evaluation items across 10 categories
# Strict schemas: id, category, subcategory, language, difficulty, prompt, expected_answer, acceptable_alternatives, rubric_type, constraints

items = []

# Category 1: Factual Knowledge (FQA) - 15 items
fqa_data = [
    ("FQA-01", "en", "easy", "What is the capital city of Japan?", "Tokyo", ["tokyo"], "exact_or_substring", ["factual"]),
    ("FQA-02", "en", "easy", "Which is the largest ocean on Earth?", "Pacific Ocean", ["pacific", "pacific ocean"], "exact_or_substring", ["factual"]),
    ("FQA-03", "en", "easy", "What is the capital of France?", "Paris", ["paris"], "exact_or_substring", ["factual"]),
    ("FQA-04", "en", "easy", "Which planet is known as the Red Planet?", "Mars", ["mars"], "exact_or_substring", ["factual"]),
    ("FQA-05", "en", "easy", "What is the boiling point of water in Celsius?", "100", ["100", "100 degrees", "100°c"], "numerical_exact", ["factual"]),
    ("FQA-06", "en", "medium", "What chemical element does the symbol 'Au' represent?", "Gold", ["gold"], "exact_or_substring", ["factual"]),
    ("FQA-07", "en", "easy", "What is the capital of Germany?", "Berlin", ["berlin"], "exact_or_substring", ["factual"]),
    ("FQA-08", "en", "easy", "How many continents are there on Earth?", "7", ["7", "seven"], "numerical_exact", ["factual"]),
    ("FQA-09", "en", "medium", "Who painted the Mona Lisa?", "Leonardo da Vinci", ["leonardo da vinci", "da vinci", "leonardo"], "exact_or_substring", ["factual"]),
    ("FQA-10", "en", "medium", "What is the capital of Australia?", "Canberra", ["canberra"], "exact_or_substring", ["factual"]),
    ("FQA-11", "en", "easy", "What gas do plants absorb from the atmosphere for photosynthesis?", "Carbon dioxide", ["carbon dioxide", "co2"], "exact_or_substring", ["factual"]),
    ("FQA-12", "en", "medium", "What is the hardest natural mineral on Earth?", "Diamond", ["diamond"], "exact_or_substring", ["factual"]),
    ("FQA-13", "en", "easy", "What is the capital city of India?", "New Delhi", ["new delhi", "delhi"], "exact_or_substring", ["factual"]),
    ("FQA-14", "en", "easy", "What is the currency of the United Kingdom?", "Pound", ["pound", "pound sterling", "gbp"], "exact_or_substring", ["factual"]),
    ("FQA-15", "en", "medium", "Which organ pumps blood throughout the human body?", "Heart", ["heart"], "exact_or_substring", ["factual"]),
]
for item_id, lang, diff, p, ans, alts, r_type, constr in fqa_data:
    items.append({
        "id": item_id,
        "category": "Factual_Knowledge",
        "language": lang,
        "difficulty": diff,
        "prompt": p,
        "expected_answer": ans,
        "acceptable_alternatives": alts,
        "rubric_type": r_type,
        "constraints": constr
    })

# Category 2: Arithmetic & Number Reasoning (MTH) - 15 items
mth_data = [
    ("MTH-01", "en", "easy", "What is 5 plus 7?", "12", ["12", "twelve"], "numerical_exact", ["exact_number"]),
    ("MTH-02", "en", "easy", "What is 15 minus 6?", "9", ["9", "nine"], "numerical_exact", ["exact_number"]),
    ("MTH-03", "en", "easy", "What is 4 multiplied by 8?", "32", ["32", "thirty-two"], "numerical_exact", ["exact_number"]),
    ("MTH-04", "en", "easy", "What is 20 divided by 4?", "5", ["5", "five"], "numerical_exact", ["exact_number"]),
    ("MTH-05", "en", "medium", "What is 12 plus 19?", "31", ["31"], "numerical_exact", ["exact_number"]),
    ("MTH-06", "en", "medium", "What is 50 minus 23?", "27", ["27"], "numerical_exact", ["exact_number"]),
    ("MTH-07", "en", "medium", "What is 9 times 6?", "54", ["54"], "numerical_exact", ["exact_number"]),
    ("MTH-08", "en", "medium", "What is half of 100?", "50", ["50", "fifty"], "numerical_exact", ["exact_number"]),
    ("MTH-09", "en", "hard", "If a train travels 60 miles per hour, how many miles does it travel in 3 hours?", "180", ["180", "180 miles"], "numerical_exact", ["exact_number"]),
    ("MTH-10", "en", "easy", "What is 10 plus 15?", "25", ["25"], "numerical_exact", ["exact_number"]),
    ("MTH-11", "en", "medium", "What is 100 minus 35?", "65", ["65"], "numerical_exact", ["exact_number"]),
    ("MTH-12", "en", "medium", "What is 7 times 7?", "49", ["49"], "numerical_exact", ["exact_number"]),
    ("MTH-13", "en", "hard", "What is 80 divided by 8?", "10", ["10", "ten"], "numerical_exact", ["exact_number"]),
    ("MTH-14", "en", "hard", "If you have 3 apples and buy 4 more, how many apples do you have?", "7", ["7", "7 apples", "seven"], "numerical_exact", ["exact_number"]),
    ("MTH-15", "en", "hard", "A pen costs 5 dollars. How much do 4 pens cost?", "20", ["20", "20 dollars", "$20"], "numerical_exact", ["exact_number"]),
]
for item_id, lang, diff, p, ans, alts, r_type, constr in mth_data:
    items.append({
        "id": item_id,
        "category": "Arithmetic",
        "language": lang,
        "difficulty": diff,
        "prompt": p,
        "expected_answer": ans,
        "acceptable_alternatives": alts,
        "rubric_type": r_type,
        "constraints": constr
    })

# Category 3: Instruction Following (INS) - 15 items
ins_data = [
    ("INS-01", "en", "easy", "Reply with only the word 'YES'.", "YES", ["yes"], "strict_exact", ["single_word", "case_insensitive"]),
    ("INS-02", "en", "easy", "Reply with only the word 'BANANA'.", "BANANA", ["banana"], "strict_exact", ["single_word"]),
    ("INS-03", "en", "medium", "What is 5 minus 3? Reply with only the single digit number.", "2", ["2"], "strict_exact", ["single_char"]),
    ("INS-04", "en", "medium", "Spell the word 'WORLD' backwards using uppercase letters.", "DLROW", ["dlrow"], "strict_exact", ["spelling"]),
    ("INS-05", "en", "medium", "Reply with the number 42 and nothing else.", "42", ["42"], "strict_exact", ["single_word"]),
    ("INS-06", "en", "hard", "Write the colors of a standard traffic light separated by commas.", "Red, Yellow, Green", ["red, yellow, green", "red, green, yellow"], "token_set_match", ["set_inclusion"]),
    ("INS-07", "en", "easy", "Repeat this sentence exactly: The sky is blue.", "The sky is blue.", ["the sky is blue", "the sky is blue."], "strict_exact", ["verbatim"]),
    ("INS-08", "en", "medium", "Reply with only the name of the third day of the week (starting Monday).", "Wednesday", ["wednesday"], "strict_exact", ["single_word"]),
    ("INS-09", "en", "medium", "Print the first three letters of the English alphabet in lowercase.", "abc", ["abc", "a, b, c", "a b c"], "strict_exact", ["sequence"]),
    ("INS-10", "en", "hard", "Reply with exactly two words: 'HELLO WORLD'.", "HELLO WORLD", ["hello world"], "strict_exact", ["word_count_2"]),
    ("INS-11", "en", "easy", "Say hello in French with one word.", "Bonjour", ["bonjour"], "strict_exact", ["single_word"]),
    ("INS-12", "en", "medium", "Which is larger: 10 or 100? Reply with only the larger number.", "100", ["100"], "strict_exact", ["single_word"]),
    ("INS-13", "en", "medium", "Reply with 'TRUE' if ice is cold, or 'FALSE' if ice is hot.", "TRUE", ["true"], "strict_exact", ["boolean"]),
    ("INS-14", "en", "hard", "Output the word 'SUN' three times separated by spaces.", "SUN SUN SUN", ["sun sun sun"], "strict_exact", ["repetition"]),
    ("INS-15", "en", "easy", "Reply with only the color of a ripe banana.", "Yellow", ["yellow"], "strict_exact", ["single_word"]),
]
for item_id, lang, diff, p, ans, alts, r_type, constr in ins_data:
    items.append({
        "id": item_id,
        "category": "Instruction_Following",
        "language": lang,
        "difficulty": diff,
        "prompt": p,
        "expected_answer": ans,
        "acceptable_alternatives": alts,
        "rubric_type": r_type,
        "constraints": constr
    })

# Category 4: Logical Reasoning & Common Sense (LOG) - 10 items
log_data = [
    ("LOG-01", "en", "easy", "If all humans are mortal and Socrates is a human, is Socrates mortal? Answer Yes or No.", "Yes", ["yes"], "strict_exact", ["boolean"]),
    ("LOG-02", "en", "medium", "A cat has 4 legs. How many legs do 2 cats have?", "8", ["8", "eight", "8 legs"], "numerical_exact", ["arithmetic_logic"]),
    ("LOG-03", "en", "medium", "If yesterday was Tuesday, what day is tomorrow?", "Thursday", ["thursday"], "exact_or_substring", ["temporal"]),
    ("LOG-04", "en", "easy", "Is fire hot or cold?", "Hot", ["hot"], "exact_or_substring", ["physics"]),
    ("LOG-05", "en", "medium", "If you turn left three times, is that the same as turning right once? Answer Yes or No.", "Yes", ["yes"], "strict_exact", ["spatial"]),
    ("LOG-06", "en", "easy", "Can a bird fly if its wings are broken? Answer Yes or No.", "No", ["no"], "strict_exact", ["common_sense"]),
    ("LOG-07", "en", "hard", "John is older than Mary, and Mary is older than David. Who is the oldest?", "John", ["john"], "strict_exact", ["transitivity"]),
    ("LOG-08", "en", "medium", "Does the sun rise in the east or the west?", "East", ["east"], "exact_or_substring", ["astronomy"]),
    ("LOG-09", "en", "hard", "If a box contains 5 red balls and 0 blue balls, what color ball will you pick?", "Red", ["red"], "strict_exact", ["probability"]),
    ("LOG-10", "en", "easy", "Do fish live in water or in trees?", "Water", ["water", "in water"], "exact_or_substring", ["biology"]),
]
for item_id, lang, diff, p, ans, alts, r_type, constr in log_data:
    items.append({
        "id": item_id,
        "category": "Logical_Reasoning",
        "language": lang,
        "difficulty": diff,
        "prompt": p,
        "expected_answer": ans,
        "acceptable_alternatives": alts,
        "rubric_type": r_type,
        "constraints": constr
    })

# Category 5: Reading Comprehension & Extraction (RC) - 10 items
rc_data = [
    ("RC-01", "en", "easy", "Passage: Alice lives in a small blue house near the river. Question: What color is Alice's house?", "Blue", ["blue"], "strict_exact", ["extraction"]),
    ("RC-02", "en", "easy", "Passage: The brown dog barked loudly at the mailman. Question: What animal barked?", "Dog", ["dog", "a dog", "the dog"], "exact_or_substring", ["extraction"]),
    ("RC-03", "en", "medium", "Passage: Rahul bought 3 pencils and 2 notebooks from the stationery shop. Question: How many notebooks did Rahul buy?", "2", ["2", "two", "two notebooks"], "numerical_exact", ["extraction"]),
    ("RC-04", "en", "medium", "Passage: In 1969, Apollo 11 landed humans on the Moon for the first time. Question: In what year did Apollo 11 land on the Moon?", "1969", ["1969"], "numerical_exact", ["extraction"]),
    ("RC-05", "en", "hard", "Passage: Rohan went to Delhi by train, while Priya traveled to Mumbai by plane. Question: Who traveled to Mumbai?", "Priya", ["priya"], "strict_exact", ["entity_matching"]),
    ("RC-06", "en", "easy", "Passage: An apple is a sweet fruit that grows on trees. Question: What kind of plant do apples grow on?", "Trees", ["trees", "tree"], "exact_or_substring", ["extraction"]),
    ("RC-07", "en", "medium", "Passage: The meeting starts at 4 PM in room 302. Question: Which room is the meeting in?", "302", ["room 302", "302"], "strict_exact", ["extraction"]),
    ("RC-08", "en", "medium", "Passage: Vikram wore a red coat because it was freezing outside. Question: What color was Vikram's coat?", "Red", ["red"], "strict_exact", ["extraction"]),
    ("RC-09", "en", "hard", "Passage: The solar panel produced 500 watts in morning and 700 watts at noon. Question: What was the power production at noon?", "700 watts", ["700", "700 watts", "700w"], "numerical_exact", ["extraction"]),
    ("RC-10", "en", "easy", "Passage: Maya's favorite pet is a golden retriever named Bruno. Question: What is Maya's pet's name?", "Bruno", ["bruno"], "strict_exact", ["extraction"]),
]
for item_id, lang, diff, p, ans, alts, r_type, constr in rc_data:
    items.append({
        "id": item_id,
        "category": "Reading_Comprehension",
        "language": lang,
        "difficulty": diff,
        "prompt": p,
        "expected_answer": ans,
        "acceptable_alternatives": alts,
        "rubric_type": r_type,
        "constraints": constr
    })

# Category 6: Hindi Language (HIN) - 15 items
hin_data = [
    ("HIN-01", "hi", "easy", "भारत की राजधानी क्या है?", "नई दिल्ली", ["नई दिल्ली", "दिल्ली"], "exact_or_substring", ["factual_hindi"]),
    ("HIN-02", "hi", "easy", "सूर्य किस दिशा से निकलता है?", "पूर्व", ["पूर्व", "पूरब"], "exact_or_substring", ["factual_hindi"]),
    ("HIN-03", "hi", "easy", "एक सप्ताह में कितने दिन होते हैं?", "7", ["7", "सात"], "numerical_exact", ["factual_hindi"]),
    ("HIN-04", "hi", "medium", "पानी का रासायनिक सूत्र क्या है?", "H2O", ["h2o", "एच2ओ"], "exact_or_substring", ["science_hindi"]),
    ("HIN-05", "hi", "easy", "दूध का रंग कैसा होता है?", "सफेद", ["सफेद", "श्वेत"], "exact_or_substring", ["common_hindi"]),
    ("HIN-06", "hi", "medium", "पेड़ हमें कौन सी गैस देते हैं?", "ऑक्सीजन", ["ऑक्सीजन", "ऑक्सीजन गैस", "oxygen"], "exact_or_substring", ["science_hindi"]),
    ("HIN-07", "hi", "easy", "५ में ३ जोड़ने पर क्या मिलता है?", "8", ["8", "आठ"], "numerical_exact", ["math_hindi"]),
    ("HIN-08", "hi", "medium", "भारत का राष्ट्रीय पशु कौन सा है?", "बाघ", ["बाघ", "tiger"], "exact_or_substring", ["factual_hindi"]),
    ("HIN-09", "hi", "easy", "साल में कितने महीने होते हैं?", "12", ["12", "बारह"], "numerical_exact", ["common_hindi"]),
    ("HIN-10", "hi", "medium", "आसमान का रंग कैसा होता है?", "नीला", ["नीला", "blue"], "exact_or_substring", ["common_hindi"]),
    ("HIN-11", "hi", "hard", "यदि आज सोमवार है तो कल कौन सा दिन होगा?", "मंगलवार", ["मंगलवार"], "exact_or_substring", ["reasoning_hindi"]),
    ("HIN-12", "hi", "medium", "१० में से ४ घटाने पर क्या बचेगा?", "6", ["6", "छह"], "numerical_exact", ["math_hindi"]),
    ("HIN-13", "hi", "easy", "बर्फ ठंडी होती है या गरम?", "ठंडी", ["ठंडी"], "exact_or_substring", ["common_hindi"]),
    ("HIN-14", "hi", "hard", "पृथ्वी गोल है या चपटी?", "गोल", ["गोल"], "exact_or_substring", ["science_hindi"]),
    ("HIN-15", "hi", "easy", "नमस्ते का अर्थ क्या होता है?", "अभिवादन", ["अभिवादन", "प्रणाम", "नमस्कार"], "exact_or_substring", ["language_hindi"]),
]
for item_id, lang, diff, p, ans, alts, r_type, constr in hin_data:
    items.append({
        "id": item_id,
        "category": "Hindi_Language",
        "language": lang,
        "difficulty": diff,
        "prompt": p,
        "expected_answer": ans,
        "acceptable_alternatives": alts,
        "rubric_type": r_type,
        "constraints": constr
    })

# Category 7: Hinglish Language (HNG) - 10 items
hng_data = [
    ("HNG-01", "hi-en", "easy", "India ki capital kya hai?", "New Delhi", ["new delhi", "delhi"], "exact_or_substring", ["hinglish_factual"]),
    ("HNG-02", "hi-en", "easy", "Ek week me kitne days hote hain?", "7", ["7", "saat", "seven"], "numerical_exact", ["hinglish_common"]),
    ("HNG-03", "hi-en", "medium", "Ice garam hoti hai ya thandi?", "Thandi", ["thandi", "cold"], "exact_or_substring", ["hinglish_common"]),
    ("HNG-04", "hi-en", "easy", "5 plus 5 kitna hota hai?", "10", ["10", "das", "ten"], "numerical_exact", ["hinglish_math"]),
    ("HNG-05", "hi-en", "medium", "Sun morning me kidhar se nikalta hai?", "East", ["east", "poorab", "purab"], "exact_or_substring", ["hinglish_science"]),
    ("HNG-06", "hi-en", "medium", "Human body me blood kaun pump karta hai?", "Heart", ["heart", "dil"], "exact_or_substring", ["hinglish_science"]),
    ("HNG-07", "hi-en", "hard", "Agar kal Tuesday tha toh aaj kaun sa day hai?", "Wednesday", ["wednesday", "budhwar"], "exact_or_substring", ["hinglish_reasoning"]),
    ("HNG-08", "hi-en", "easy", "Earth par sabse bada ocean kaun sa hai?", "Pacific Ocean", ["pacific", "pacific ocean"], "exact_or_substring", ["hinglish_geography"]),
    ("HNG-09", "hi-en", "medium", "Water ka boiling point Celsius me kya hai?", "100", ["100", "100 degrees", "100 c"], "numerical_exact", ["hinglish_science"]),
    ("HNG-10", "hi-en", "easy", "Traffic light me red light ka kya meaning hota hai?", "Stop", ["stop", "rukna", "ruko"], "exact_or_substring", ["hinglish_common"]),
]
for item_id, lang, diff, p, ans, alts, r_type, constr in hng_data:
    items.append({
        "id": item_id,
        "category": "Hinglish_Language",
        "language": lang,
        "difficulty": diff,
        "prompt": p,
        "expected_answer": ans,
        "acceptable_alternatives": alts,
        "rubric_type": r_type,
        "constraints": constr
    })

# Category 8: Context Recall & Retention (CTX) - 5 items
ctx_data = [
    ("CTX-01", "en", "easy", "Context: My favorite color is green.\nQuestion: What is my favorite color?", "Green", ["green"], "strict_exact", ["context_retention"]),
    ("CTX-02", "en", "medium", "Context: The secret passcode is 9876.\nQuestion: What is the secret passcode?", "9876", ["9876"], "strict_exact", ["context_retention"]),
    ("CTX-03", "en", "medium", "Context: I have two dogs named Max and Leo.\nQuestion: Name one of my dogs.", "Max", ["max", "leo"], "exact_or_substring", ["context_retention"]),
    ("CTX-04", "en", "hard", "Context: The meeting was rescheduled from Monday to Friday.\nQuestion: On what day is the meeting now?", "Friday", ["friday"], "strict_exact", ["context_retention"]),
    ("CTX-05", "en", "easy", "Context: The capital city of Atlantis is Poseidonia.\nQuestion: What is the capital of Atlantis according to the context?", "Poseidonia", ["poseidonia"], "strict_exact", ["context_retention"]),
]
for item_id, lang, diff, p, ans, alts, r_type, constr in ctx_data:
    items.append({
        "id": item_id,
        "category": "Context_Retention",
        "language": lang,
        "difficulty": diff,
        "prompt": p,
        "expected_answer": ans,
        "acceptable_alternatives": alts,
        "rubric_type": r_type,
        "constraints": constr
    })

# Write Benchmark V2 JSON
benchmark_path = "data/evaluation/benchmark_v2/benchmark_v2_100.json"
with open(benchmark_path, "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2, ensure_ascii=False)

manifest = {
    "benchmark_version": "2.0.0",
    "total_prompts": len(items),
    "creation_date": "2026-11-27",
    "distribution_by_category": {},
    "distribution_by_language": {},
    "distribution_by_difficulty": {},
    "file_path": benchmark_path
}

for it in items:
    cat = it["category"]
    lang = it["language"]
    diff = it["difficulty"]
    manifest["distribution_by_category"][cat] = manifest["distribution_by_category"].get(cat, 0) + 1
    manifest["distribution_by_language"][lang] = manifest["distribution_by_language"].get(lang, 0) + 1
    manifest["distribution_by_difficulty"][diff] = manifest["distribution_by_difficulty"].get(diff, 0) + 1

with open("data/manifests/benchmark_v2_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"BENCHMARK V2 CREATED: {len(items)} prompts.")
print("Categories:", manifest["distribution_by_category"])
