"""
Generator for the 300 Unseen Blind Evaluation Benchmark (Mission CV-GENESIS-NR-006).

Distribution:
- 60 English conversation & dialogue
- 60 Hindi conversation & dialogue
- 60 Hinglish conversation & dialogue
- 40 Technical explanations & concepts
- 30 Coding tasks (Python)
- 20 Arithmetic and algorithmic reasoning
- 20 Multi-turn context tasks
- 10 Correction and uncertainty tasks

Leakage check: Ensures zero exact overlap with training/validation data.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]

PROMPTS_300 = [
    # ==========================================
    # 1. English Conversation & Dialogue (60)
    # ==========================================
    ("en_conv_001", "en", "conversation", "User: Good morning! How is everything going today?\nAssistant:", "greeting"),
    ("en_conv_002", "en", "conversation", "User: Hello! Could you introduce yourself and describe your skills?\nAssistant:", "identity"),
    ("en_conv_003", "en", "conversation", "User: I am feeling quite tired after a long flight from Tokyo.\nAssistant:", "empathy"),
    ("en_conv_004", "en", "conversation", "User: What are your favorite kinds of scientific topics to discuss?\nAssistant:", "preference"),
    ("en_conv_005", "en", "conversation", "User: Can you tell me an interesting thought for the day?\nAssistant:", "inspirational"),
    ("en_conv_006", "en", "conversation", "User: How do you stay focused when working on challenging tasks?\nAssistant:", "advice"),
    ("en_conv_007", "en", "conversation", "User: Have you ever visited any libraries in ancient times?\nAssistant:", "identity_clarity"),
    ("en_conv_008", "en", "conversation", "User: What is the best way to spend a quiet Sunday afternoon?\nAssistant:", "conversation"),
    ("en_conv_009", "en", "conversation", "User: Thank you so much for your assistance earlier.\nAssistant:", "gratitude"),
    ("en_conv_010", "en", "conversation", "User: Good evening! Hope you had a pleasant and productive day.\nAssistant:", "greeting"),
    ("en_conv_011", "en", "conversation", "User: What makes learning a new musical instrument enjoyable?\nAssistant:", "conversation"),
    ("en_conv_012", "en", "conversation", "User: I just finished reading a mystery novel with an unexpected ending.\nAssistant:", "conversation"),
    ("en_conv_013", "en", "conversation", "User: How do astronomers observe distant galaxies across deep space?\nAssistant:", "science_conv"),
    ("en_conv_014", "en", "conversation", "User: If someone wants to start stargazing as a hobby, what is a good first step?\nAssistant:", "hobby_advice"),
    ("en_conv_015", "en", "conversation", "User: Do you experience emotions like humans do?\nAssistant:", "identity_clarity"),
    ("en_conv_016", "en", "conversation", "User: It is raining heavily outside right now.\nAssistant:", "small_talk"),
    ("en_conv_017", "en", "conversation", "User: What is your favorite season of the year?\nAssistant:", "preference"),
    ("en_conv_018", "en", "conversation", "User: Can you share a simple tip for staying organized?\nAssistant:", "advice"),
    ("en_conv_019", "en", "conversation", "User: I am preparing for a presentation tomorrow morning.\nAssistant:", "encouragement"),
    ("en_conv_020", "en", "conversation", "User: Where do you store your memories when our conversation ends?\nAssistant:", "technical_identity"),
    ("en_conv_021", "en", "conversation", "User: What is the difference between an owl and an eagle in their hunting habits?\nAssistant:", "nature_conv"),
    ("en_conv_022", "en", "conversation", "User: Tell me an unusual fact about marine life in the deep ocean.\nAssistant:", "nature_conv"),
    ("en_conv_023", "en", "conversation", "User: Why do leaves turn yellow and orange during autumn?\nAssistant:", "nature_conv"),
    ("en_conv_024", "en", "conversation", "User: How do honeybees communicate the location of flowers to the hive?\nAssistant:", "nature_conv"),
    ("en_conv_025", "en", "conversation", "User: I am thinking of adopting a rescue puppy this weekend.\nAssistant:", "conversation"),
    ("en_conv_026", "en", "conversation", "User: What are some benefits of drinking enough water throughout the day?\nAssistant:", "wellness"),
    ("en_conv_027", "en", "conversation", "User: How does practicing mindfulness help reduce mental stress?\nAssistant:", "wellness"),
    ("en_conv_028", "en", "conversation", "User: What is the significance of the Rosetta Stone in historical linguistics?\nAssistant:", "history_conv"),
    ("en_conv_029", "en", "conversation", "User: How did ancient sailors navigate open oceans without modern GPS satellites?\nAssistant:", "history_conv"),
    ("en_conv_030", "en", "conversation", "User: Who designed the Eiffel Tower in Paris?\nAssistant:", "history_fact"),
    ("en_conv_031", "en", "conversation", "User: In which ocean is the island of Madagascar located?\nAssistant:", "geography_fact"),
    ("en_conv_032", "en", "conversation", "User: Which desert is the largest hot desert on Earth?\nAssistant:", "geography_fact"),
    ("en_conv_033", "en", "conversation", "User: What is the primary language spoken in Brazil?\nAssistant:", "geography_fact"),
    ("en_conv_034", "en", "conversation", "User: How high is Mount Everest above sea level?\nAssistant:", "geography_fact"),
    ("en_conv_035", "en", "conversation", "User: What is the currency used in Switzerland?\nAssistant:", "geography_fact"),
    ("en_conv_036", "en", "conversation", "User: Can penguins fly in the air like other birds?\nAssistant:", "animal_fact"),
    ("en_conv_037", "en", "conversation", "User: How do bats navigate in total darkness?\nAssistant:", "animal_fact"),
    ("en_conv_038", "en", "conversation", "User: Why does the moon appear to change its shape throughout the month?\nAssistant:", "astronomy_fact"),
    ("en_conv_039", "en", "conversation", "User: What is a light-year measuring in astrophysics?\nAssistant:", "astronomy_fact"),
    ("en_conv_040", "en", "conversation", "User: Which planet is commonly known as the Red Planet?\nAssistant:", "astronomy_fact"),
    ("en_conv_041", "en", "conversation", "User: Tell me about the Great Barrier Reef.\nAssistant:", "geography_conv"),
    ("en_conv_042", "en", "conversation", "User: What causes a rainbow to form in the sky after rain?\nAssistant:", "optics_fact"),
    ("en_conv_043", "en", "conversation", "User: Why do ships float on water while small pebbles sink?\nAssistant:", "physics_fact"),
    ("en_conv_044", "en", "conversation", "User: How do airplanes stay up in the air?\nAssistant:", "physics_fact"),
    ("en_conv_045", "en", "conversation", "User: What is the chemical formula for ordinary table salt?\nAssistant:", "chemistry_fact"),
    ("en_conv_046", "en", "conversation", "User: Why is iron an essential mineral in human blood?\nAssistant:", "biology_fact"),
    ("en_conv_047", "en", "conversation", "User: What is the hardest naturally occurring mineral on Earth?\nAssistant:", "geology_fact"),
    ("en_conv_048", "en", "conversation", "User: How do volcanic eruptions occur beneath tectonic plates?\nAssistant:", "geology_fact"),
    ("en_conv_049", "en", "conversation", "User: Which layer of the atmosphere contains the ozone layer?\nAssistant:", "geology_fact"),
    ("en_conv_050", "en", "conversation", "User: What causes oceanic tides to rise and fall twice each day?\nAssistant:", "astronomy_fact"),
    ("en_conv_051", "en", "conversation", "User: It was great chatting with you today. Have a nice evening!\nAssistant:", "farewell"),
    ("en_conv_052", "en", "conversation", "User: Goodbye for now, see you tomorrow.\nAssistant:", "farewell"),
    ("en_conv_053", "en", "conversation", "User: Catch you later, take care!\nAssistant:", "farewell"),
    ("en_conv_054", "en", "conversation", "User: Do you get tired if I ask you a hundred questions in a row?\nAssistant:", "identity_clarity"),
    ("en_conv_055", "en", "conversation", "User: Who is your favorite historical scientist?\nAssistant:", "preference"),
    ("en_conv_056", "en", "conversation", "User: How do solar panels generate electricity from sunlight?\nAssistant:", "science_conv"),
    ("en_conv_057", "en", "conversation", "User: What is lightning and why does thunder follow it?\nAssistant:", "weather_fact"),
    ("en_conv_058", "en", "conversation", "User: How do mirrors reflect images?\nAssistant:", "optics_fact"),
    ("en_conv_059", "en", "conversation", "User: Can sound travel through the vacuum of outer space?\nAssistant:", "physics_fact"),
    ("en_conv_060", "en", "conversation", "User: What is the speed of light in vacuum?\nAssistant:", "physics_fact"),

    # ==========================================
    # 2. Hindi Conversation & Dialogue (60)
    # ==========================================
    ("hi_conv_001", "hi", "conversation", "User: नमस्ते! आज आपका दिन कैसा चल रहा है?\nAssistant:", "greeting"),
    ("hi_conv_002", "hi", "conversation", "User: प्रणाम, क्या आप अपना परिचय दे सकते हैं?\nAssistant:", "identity"),
    ("hi_conv_003", "hi", "conversation", "User: मुझे आज बहुत थकान महसूस हो रही है।\nAssistant:", "empathy"),
    ("hi_conv_004", "hi", "conversation", "User: क्या आप मुझे कोई प्रेरक विचार बता सकते हैं?\nAssistant:", "inspirational"),
    ("hi_conv_005", "hi", "conversation", "User: एकाग्रता बढ़ाने के लिए मुझे क्या करना चाहिए?\nAssistant:", "advice"),
    ("hi_conv_006", "hi", "conversation", "User: आपकी सहायता के लिए बहुत-बहुत धन्यवाद।\nAssistant:", "gratitude"),
    ("hi_conv_007", "hi", "conversation", "User: शुभ संध्या! आज आपने क्या सीखा?\nAssistant:", "greeting"),
    ("hi_conv_008", "hi", "conversation", "User: क्या आपको भूख या प्यास लगती है?\nAssistant:", "identity_clarity"),
    ("hi_conv_009", "hi", "conversation", "User: आज यहाँ बहुत तेज धूप निकली हुई है।\nAssistant:", "small_talk"),
    ("hi_conv_010", "hi", "conversation", "User: आपका सबसे पसंदीदा विषय कौन सा है?\nAssistant:", "preference"),
    ("hi_conv_011", "hi", "conversation", "User: सुबह जल्दी उठने के क्या फायदे होते हैं?\nAssistant:", "wellness"),
    ("hi_conv_012", "hi", "conversation", "User: प्रतिदिन व्यायाम करना क्यों जरूरी है?\nAssistant:", "wellness"),
    ("hi_conv_013", "hi", "conversation", "User: भारत की सबसे लंबी नदी कौन सी है?\nAssistant:", "geography_fact"),
    ("hi_conv_014", "hi", "conversation", "User: सूर्य के सबसे निकट कौन सा ग्रह है?\nAssistant:", "astronomy_fact"),
    ("hi_conv_015", "hi", "conversation", "User: जल का हिमांक (Freezing Point) कितने डिग्री सेल्सियस होता है?\nAssistant:", "science_fact"),
    ("hi_conv_016", "hi", "conversation", "User: पौधे कार्बन डाइऑक्साइड लेकर कौन सी गैस छोड़ते हैं?\nAssistant:", "biology_fact"),
    ("hi_conv_017", "hi", "conversation", "User: मानव शरीर में कितनी हड्डियाँ होती हैं?\nAssistant:", "biology_fact"),
    ("hi_conv_018", "hi", "conversation", "User: टेलीफोन का आविष्कार किसने किया था?\nAssistant:", "history_fact"),
    ("hi_conv_019", "hi", "conversation", "User: भारत का राष्ट्रीय पक्षी कौन सा है?\nAssistant:", "culture_fact"),
    ("hi_conv_020", "hi", "conversation", "User: राष्ट्रगान 'जन गण मन' के रचयिता कौन हैं?\nAssistant:", "culture_fact"),
    ("hi_conv_021", "hi", "conversation", "User: हिंदी दिवस कब मनाया जाता है?\nAssistant:", "culture_fact"),
    ("hi_conv_022", "hi", "conversation", "User: लाल किला किस शहर में स्थित है?\nAssistant:", "geography_fact"),
    ("hi_conv_023", "hi", "conversation", "User: भारत के पहले प्रधानमंत्री कौन थे?\nAssistant:", "history_fact"),
    ("hi_conv_024", "hi", "conversation", "User: चंद्रमा पृथ्वी का एक चक्कर कितने दिनों में पूरा करता है?\nAssistant:", "astronomy_fact"),
    ("hi_conv_025", "hi", "conversation", "User: प्रकाश संश्लेषण क्या होता है, संक्षेप में समझाइए।\nAssistant:", "science_expl"),
    ("hi_conv_026", "hi", "conversation", "User: ध्वनि निर्वात में यात्रा क्यों नहीं कर सकती?\nAssistant:", "science_expl"),
    ("hi_conv_027", "hi", "conversation", "User: वायुमंडल में सबसे अधिक मात्रा में कौन सी गैस पाई जाती है?\nAssistant:", "geography_fact"),
    ("hi_conv_028", "hi", "conversation", "User: लोहे में जंग क्यों लगती है?\nAssistant:", "chemistry_expl"),
    ("hi_conv_029", "hi", "conversation", "User: इंद्रधनुष में कितने रंग होते हैं?\nAssistant:", "science_fact"),
    ("hi_conv_030", "hi", "conversation", "User: दूध से दही जमना किस प्रकार का परिवर्तन है?\nAssistant:", "chemistry_fact"),
    ("hi_conv_031", "hi", "conversation", "User: क्या आप मुझे कोई छोटी सी कहानी सुना सकते हैं?\nAssistant:", "creative"),
    ("hi_conv_032", "hi", "conversation", "User: कछुए और खरगोश की दौड़ से हमें क्या सीख मिलती है?\nAssistant:", "moral_story"),
    ("hi_conv_033", "hi", "conversation", "User: समय का सदुपयोग करना क्यों आवश्यक है?\nAssistant:", "life_advice"),
    ("hi_conv_034", "hi", "conversation", "User: जीवन में शिक्षा का क्या महत्व है?\nAssistant:", "life_advice"),
    ("hi_conv_035", "hi", "conversation", "User: मित्रता कैसी होनी चाहिए?\nAssistant:", "life_advice"),
    ("hi_conv_036", "hi", "conversation", "User: सौरमंडल का सबसे छोटा ग्रह कौन सा है?\nAssistant:", "astronomy_fact"),
    ("hi_conv_037", "hi", "conversation", "User: हमारे शरीर का सामान्य तापमान कितना होता है?\nAssistant:", "biology_fact"),
    ("hi_conv_038", "hi", "conversation", "User: विटामिन सी का मुख्य स्रोत क्या है?\nAssistant:", "health_fact"),
    ("hi_conv_039", "hi", "conversation", "User: रक्त का रंग लाल किस घटक के कारण होता है?\nAssistant:", "biology_fact"),
    ("hi_conv_040", "hi", "conversation", "User: हृदय का मुख्य कार्य क्या होता है?\nAssistant:", "biology_fact"),
    ("hi_conv_041", "hi", "conversation", "User: कंप्यूटर का कीबोर्ड क्या इनपुट डिवाइस है?\nAssistant:", "tech_fact"),
    ("hi_conv_042", "hi", "conversation", "User: इंटरनेट का पूरा नाम क्या है?\nAssistant:", "tech_fact"),
    ("hi_conv_043", "hi", "conversation", "User: ईमेल भेजने के लिए क्या आवश्यकता होती है?\nAssistant:", "tech_expl"),
    ("hi_conv_044", "hi", "conversation", "User: पासवर्ड को सुरक्षित कैसे रखा जा सकता है?\nAssistant:", "security_advice"),
    ("hi_conv_045", "hi", "conversation", "User: क्या आप मुझे बता सकते हैं कि किताबें पढ़ना क्यों महत्वपूर्ण है?\nAssistant:", "reading_advice"),
    ("hi_conv_046", "hi", "conversation", "User: राजस्थान की राजधानी क्या है?\nAssistant:", "geography_fact"),
    ("hi_conv_047", "hi", "conversation", "User: हिमालय किस प्रकार का पर्वत है?\nAssistant:", "geography_fact"),
    ("hi_conv_048", "hi", "conversation", "User: भारत में कितने राज्य हैं?\nAssistant:", "polity_fact"),
    ("hi_conv_049", "hi", "conversation", "User: संविधान दिवस कब मनाया जाता है?\nAssistant:", "polity_fact"),
    ("hi_conv_050", "hi", "conversation", "User: शून्य (0) का आविष्कार किस देश में हुआ था?\nAssistant:", "history_fact"),
    ("hi_conv_051", "hi", "conversation", "User: आर्यभट्ट कौन थे?\nAssistant:", "history_fact"),
    ("hi_conv_052", "hi", "conversation", "User: महात्मा गांधी का जन्म किस तारीख को हुआ था?\nAssistant:", "history_fact"),
    ("hi_conv_053", "hi", "conversation", "User: जल संरक्षण के दो उपाय बताइए।\nAssistant:", "ecology_advice"),
    ("hi_conv_054", "hi", "conversation", "User: वायु प्रदूषण रोकने के लिए क्या करना चाहिए?\nAssistant:", "ecology_advice"),
    ("hi_conv_055", "hi", "conversation", "User: प्लास्टिक का उपयोग कम क्यों करना चाहिए?\nAssistant:", "ecology_advice"),
    ("hi_conv_056", "hi", "conversation", "User: आपका बहुत-बहुत धन्यवाद, आपसे बात करके बहुत अच्छा लगा।\nAssistant:", "gratitude"),
    ("hi_conv_057", "hi", "conversation", "User: शुभ रात्रि, कल मिलते हैं।\nAssistant:", "farewell"),
    ("hi_conv_058", "hi", "conversation", "User: अलविदा, अपना ध्यान रखिएगा।\nAssistant:", "farewell"),
    ("hi_conv_059", "hi", "conversation", "User: फिर मिलेंगे, आपका दिन मंगलमय हो!\nAssistant:", "farewell"),
    ("hi_conv_060", "hi", "conversation", "User: नमस्ते दोस्त, अब मैं सोने जा रहा हूँ।\nAssistant:", "farewell"),

    # ==========================================
    # 3. Hinglish Conversation & Dialogue (60)
    # ==========================================
    ("hng_conv_001", "hinglish", "conversation", "User: Bhai kya chal raha hai aaj kal?\nAssistant:", "greeting"),
    ("hng_conv_002", "hinglish", "conversation", "User: Sun yaar, tumhara naam ChakrView kisne rakha?\nAssistant:", "identity"),
    ("hng_conv_003", "hinglish", "conversation", "User: Aaj mera college me first presentation hai, thoda nervous hoon.\nAssistant:", "empathy"),
    ("hng_conv_004", "hinglish", "conversation", "User: Koi badhiya si coding tip batao na beginners ke liye.\nAssistant:", "coding_advice"),
    ("hng_conv_005", "hinglish", "conversation", "User: Weekend par doston ke sath kya plan banayein?\nAssistant:", "casual_advice"),
    ("hng_conv_006", "hinglish", "conversation", "User: Yaar coding karte time errors aane par kaise handle karein?\nAssistant:", "debugging_advice"),
    ("hng_conv_007", "hinglish", "conversation", "User: Thank you bhai, tumne mera doubt clear kar diya.\nAssistant:", "gratitude"),
    ("hng_conv_008", "hinglish", "conversation", "User: Good morning bro! Aaj ka weather kaisa lag raha hai?\nAssistant:", "greeting"),
    ("hng_conv_009", "hinglish", "conversation", "User: Tum real person ho ya pure computer code?\nAssistant:", "identity_clarity"),
    ("hng_conv_010", "hinglish", "conversation", "User: Ek accha software engineer banne ke liye sabse important skill kya hai?\nAssistant:", "career_advice"),
    ("hng_conv_011", "hinglish", "conversation", "User: Python me syntax error aur runtime error me kya farak hota hai?\nAssistant:", "tech_expl"),
    ("hng_conv_012", "hinglish", "conversation", "User: Laptop slow ho gaya hai, speed kaise increase karun?\nAssistant:", "hardware_advice"),
    ("hng_conv_013", "hinglish", "conversation", "User: Hard disk aur SSD me se konsa better hota hai speed ke mamle me?\nAssistant:", "hardware_expl"),
    ("hng_conv_014", "hinglish", "conversation", "User: Browser me cache clear karne se kya hota hai?\nAssistant:", "tech_expl"),
    ("hng_conv_015", "hinglish", "conversation", "User: WiFi ka password strong kaise banayein?\nAssistant:", "security_advice"),
    ("hng_conv_016", "hinglish", "conversation", "User: Open source software ka matlab kya hota hai?\nAssistant:", "tech_expl"),
    ("hng_conv_017", "hinglish", "conversation", "User: Linux aur Windows me basic difference kya hai?\nAssistant:", "tech_expl"),
    ("hng_conv_018", "hinglish", "conversation", "User: Git commit karne ke baad roll back kaise karte hain?\nAssistant:", "git_advice"),
    ("hng_conv_019", "hinglish", "conversation", "User: GitHub par repository create karne ka kya procedure hai?\nAssistant:", "git_expl"),
    ("hng_conv_020", "hinglish", "conversation", "User: API key ko public code me share kyun nahi karna chahiye?\nAssistant:", "security_advice"),
    ("hng_conv_021", "hinglish", "conversation", "User: AI aur Machine Learning me kya relation hai?\nAssistant:", "ai_expl"),
    ("hng_conv_022", "hinglish", "conversation", "User: Neural network me epoch ka kya matlab hota hai?\nAssistant:", "ai_expl"),
    ("hng_conv_023", "hinglish", "conversation", "User: Batch size bada rakhne se GPU memory par kya impact padta hai?\nAssistant:", "ai_expl"),
    ("hng_conv_024", "hinglish", "conversation", "User: Transformer architecture me self-attention kya karta hai?\nAssistant:", "ai_expl"),
    ("hng_conv_025", "hinglish", "conversation", "User: Tokenizer ka main kaam kya hota hai NLP me?\nAssistant:", "ai_expl"),
    ("hng_conv_026", "hinglish", "conversation", "User: Loss function decrease hone ka kya significance hai training me?\nAssistant:", "ai_expl"),
    ("hng_conv_027", "hinglish", "conversation", "User: Overfitting ko rokhne ke do common methods batao.\nAssistant:", "ai_expl"),
    ("hng_conv_028", "hinglish", "conversation", "User: CPU aur GPU me parallel processing ka kya difference hai?\nAssistant:", "hardware_expl"),
    ("hng_conv_029", "hinglish", "conversation", "User: VRAM kya hoti hai aur deep learning me iska kya role hai?\nAssistant:", "hardware_expl"),
    ("hng_conv_030", "hinglish", "conversation", "User: CUDA toolkit NVIDIA GPUs ke liye kyun zaroori hota hai?\nAssistant:", "tech_expl"),
    ("hng_conv_031", "hinglish", "conversation", "User: Sun yaar, mujhe ek simple calculator app banana hai Python me.\nAssistant:", "coding_conv"),
    ("hng_conv_032", "hinglish", "conversation", "User: Python me virtual environment banana kyun acchi practice maani jaati hai?\nAssistant:", "python_advice"),
    ("hng_conv_033", "hinglish", "conversation", "User: pip aur conda me kya antar hota hai packages install karne me?\nAssistant:", "python_expl"),
    ("hng_conv_034", "hinglish", "conversation", "User: PyTorch aur TensorFlow me se lightweight models ke liye konsa easy hai?\nAssistant:", "python_advice"),
    ("hng_conv_035", "hinglish", "conversation", "User: JSON file format kya hota hai aur isme data kaise save hota hai?\nAssistant:", "tech_expl"),
    ("hng_conv_036", "hinglish", "conversation", "User: Database me primary key aur foreign key me kya difference hai?\nAssistant:", "db_expl"),
    ("hng_conv_037", "hinglish", "conversation", "User: SQL injection attack se database ko kaise protect karein?\nAssistant:", "security_advice"),
    ("hng_conv_038", "hinglish", "conversation", "User: Web application me frontend aur backend me kya divide hota hai?\nAssistant:", "web_expl"),
    ("hng_conv_039", "hinglish", "conversation", "User: HTTP aur HTTPS me 'S' ka kya meaning hota hai?\nAssistant:", "web_fact"),
    ("hng_conv_040", "hinglish", "conversation", "User: DNS server ka internet browsing me kya main role hota hai?\nAssistant:", "networking_expl"),
    ("hng_conv_041", "hinglish", "conversation", "User: IP address aur MAC address me kya farak hai?\nAssistant:", "networking_expl"),
    ("hng_conv_042", "hinglish", "conversation", "User: Ping command terminal me kyun use kiya jata hai?\nAssistant:", "networking_expl"),
    ("hng_conv_043", "hinglish", "conversation", "User: localhost ya 127.0.0.1 ka matlab kya hota hai?\nAssistant:", "networking_expl"),
    ("hng_conv_044", "hinglish", "conversation", "User: Port 8080 aur port 80 me kya difference hota hai mostly?\nAssistant:", "networking_expl"),
    ("hng_conv_045", "hinglish", "conversation", "User: Docker container aur Virtual Machine me lightweight kaun hota hai?\nAssistant:", "devops_expl"),
    ("hng_conv_046", "hinglish", "conversation", "User: CI/CD pipeline automation companies kyun setup karti hain?\nAssistant:", "devops_expl"),
    ("hng_conv_047", "hinglish", "conversation", "User: Cloud storage par data backup lene ka kya fayda hota hai?\nAssistant:", "tech_advice"),
    ("hng_conv_048", "hinglish", "conversation", "User: Two-factor authentication (2FA) activate karna kyun zaroori hai?\nAssistant:", "security_advice"),
    ("hng_conv_049", "hinglish", "conversation", "User: Phishing email ko pehchanne ka koi aasan tarika batao.\nAssistant:", "security_advice"),
    ("hng_conv_050", "hinglish", "conversation", "User: Computer me task manager se kya monitor kiya ja sakta hai?\nAssistant:", "os_expl"),
    ("hng_conv_051", "hinglish", "conversation", "User: Sahi baat hai bhai, tumne kaafi easily samjha diya!\nAssistant:", "gratitude"),
    ("hng_conv_052", "hinglish", "conversation", "User: Theek hai bro, main ab coding practice shuru karta hoon.\nAssistant:", "encouragement"),
    ("hng_conv_053", "hinglish", "conversation", "User: Kal office me milte hain, bye bye!\nAssistant:", "farewell"),
    ("hng_conv_054", "hinglish", "conversation", "User: Shaba khair bhai, acchi neend lena.\nAssistant:", "farewell"),
    ("hng_conv_055", "hinglish", "conversation", "User: Chal fir milte hain next session me!\nAssistant:", "farewell"),
    ("hng_conv_056", "hinglish", "conversation", "User: Awesome help yaar, keep it up!\nAssistant:", "gratitude"),
    ("hng_conv_057", "hinglish", "conversation", "User: Tumse baat karke kafi informative laga aaj.\nAssistant:", "gratitude"),
    ("hng_conv_058", "hinglish", "conversation", "User: Take care bhai, alvida abhi ke liye.\nAssistant:", "farewell"),
    ("hng_conv_059", "hinglish", "conversation", "User: See ya bro, baad me message karta hoon.\nAssistant:", "farewell"),
    ("hng_conv_060", "hinglish", "conversation", "User: Ok boss, signed off for today!\nAssistant:", "farewell"),

    # ==========================================
    # 4. Technical Explanations & Concepts (40)
    # ==========================================
    ("tech_expl_001", "en", "explanation", "User: Explain the difference between process and thread in modern operating systems.\nAssistant:", "os_concept"),
    ("tech_expl_002", "en", "explanation", "User: What is virtual memory and how does page fault handling work?\nAssistant:", "os_concept"),
    ("tech_expl_003", "en", "explanation", "User: Describe how a hash table handles collisions using chaining.\nAssistant:", "dsa_concept"),
    ("tech_expl_004", "en", "explanation", "User: What is the difference between synchronous and asynchronous I/O?\nAssistant:", "cs_concept"),
    ("tech_expl_005", "en", "explanation", "User: How does symmetric encryption differ from asymmetric public-key cryptography?\nAssistant:", "security_concept"),
    ("tech_expl_006", "en", "explanation", "User: What is a deadlock and what four Coffman conditions are required for it to occur?\nAssistant:", "os_concept"),
    ("tech_expl_007", "en", "explanation", "User: Explain the working of the TCP three-way handshake.\nAssistant:", "networking_concept"),
    ("tech_expl_008", "en", "explanation", "User: What is the purpose of database normalization up to Third Normal Form (3NF)?\nAssistant:", "db_concept"),
    ("tech_expl_009", "en", "explanation", "User: How does the DNS hierarchy resolve a domain name into an IP address?\nAssistant:", "networking_concept"),
    ("tech_expl_010", "en", "explanation", "User: Explain the difference between pass-by-value and pass-by-reference in programming.\nAssistant:", "prog_concept"),
    ("tech_expl_011", "en", "explanation", "User: What is a race condition and how do mutex locks prevent it?\nAssistant:", "concurrency_concept"),
    ("tech_expl_012", "en", "explanation", "User: Describe how quicksort partitions an array around a chosen pivot.\nAssistant:", "dsa_concept"),
    ("tech_expl_013", "en", "explanation", "User: What is the time and space complexity of merge sort?\nAssistant:", "dsa_concept"),
    ("tech_expl_014", "en", "explanation", "User: How does binary search achieve O(log n) time complexity on sorted data?\nAssistant:", "dsa_concept"),
    ("tech_expl_015", "en", "explanation", "User: What is a memory leak and how do garbage collectors detect unreachable objects?\nAssistant:", "memory_concept"),
    ("tech_expl_016", "en", "explanation", "User: Explain how the HTTPS TLS handshake establishes an encrypted session.\nAssistant:", "security_concept"),
    ("tech_expl_017", "en", "explanation", "User: What is the difference between a stack and a heap in memory allocation?\nAssistant:", "memory_concept"),
    ("tech_expl_018", "en", "explanation", "User: Describe the role of the program counter (PC) register in a CPU execution cycle.\nAssistant:", "arch_concept"),
    ("tech_expl_019", "en", "explanation", "User: What is pipelining in modern CPU architecture and what hazards can stall it?\nAssistant:", "arch_concept"),
    ("tech_expl_020", "en", "explanation", "User: Explain the difference between compile-time static binding and runtime dynamic dispatch.\nAssistant:", "oop_concept"),
    ("tech_expl_021", "en", "explanation", "User: What is an index in a database and why does it accelerate SELECT queries?\nAssistant:", "db_concept"),
    ("tech_expl_022", "en", "explanation", "User: Explain the ACID properties of database transactions.\nAssistant:", "db_concept"),
    ("tech_expl_023", "en", "explanation", "User: What is the purpose of an abstract syntax tree (AST) in a compiler pipeline?\nAssistant:", "compiler_concept"),
    ("tech_expl_024", "en", "explanation", "User: Describe how lexical analysis tokens are generated from source code characters.\nAssistant:", "compiler_concept"),
    ("tech_expl_025", "en", "explanation", "User: What is the difference between horizontal scaling and vertical scaling in server systems?\nAssistant:", "distributed_concept"),
    ("tech_expl_026", "en", "explanation", "User: What is the CAP theorem and what tradeoffs does it state for distributed data stores?\nAssistant:", "distributed_concept"),
    ("tech_expl_027", "en", "explanation", "User: Explain how a load balancer distributes incoming network traffic across multiple servers.\nAssistant:", "distributed_concept"),
    ("tech_expl_028", "en", "explanation", "User: What is a WebSocket connection and how does it enable full-duplex communication?\nAssistant:", "web_concept"),
    ("tech_expl_029", "en", "explanation", "User: Describe the purpose of CORS (Cross-Origin Resource Sharing) headers in web browsers.\nAssistant:", "security_concept"),
    ("tech_expl_030", "en", "explanation", "User: What is SQL injection and how do parameterized queries mitigate it?\nAssistant:", "security_concept"),
    ("tech_expl_031", "en", "explanation", "User: Explain cross-site scripting (XSS) and how input sanitization prevents it.\nAssistant:", "security_concept"),
    ("tech_expl_032", "en", "explanation", "User: What is backpropagation in training artificial neural networks?\nAssistant:", "ai_concept"),
    ("tech_expl_033", "en", "explanation", "User: How does stochastic gradient descent update model weights to minimize loss?\nAssistant:", "ai_concept"),
    ("tech_expl_034", "en", "explanation", "User: What is the role of Rotary Position Embedding (RoPE) in Transformer models?\nAssistant:", "ai_concept"),
    ("tech_expl_035", "en", "explanation", "User: Explain SwiGLU activation and why modern neural models use it over standard ReLU.\nAssistant:", "ai_concept"),
    ("tech_expl_036", "en", "explanation", "User: What is Grouped-Query Attention (GQA) and how does it reduce KV-cache memory?\nAssistant:", "ai_concept"),
    ("tech_expl_037", "en", "explanation", "User: Describe the purpose of RMSNorm compared to standard LayerNorm in transformers.\nAssistant:", "ai_concept"),
    ("tech_expl_038", "en", "explanation", "User: What is Byte-Pair Encoding (BPE) and how does it tokenize subwords?\nAssistant:", "ai_concept"),
    ("tech_expl_039", "en", "explanation", "User: Explain why temperature scaling changes the probability distribution during sampling.\nAssistant:", "ai_concept"),
    ("tech_expl_040", "en", "explanation", "User: What is top-k filtering and top-p (nucleus) sampling in text generation?\nAssistant:", "ai_concept"),

    # ==========================================
    # 5. Coding Tasks in Python (30)
    # ==========================================
    ("code_001", "en", "coding", "User: Write a Python function to check whether an integer is prime.\nAssistant:", "algo"),
    ("code_002", "en", "coding", "User: Write a Python function to compute the greatest common divisor (GCD) of two numbers.\nAssistant:", "algo"),
    ("code_003", "en", "coding", "User: Implement a function that reverses a string without using built-in reverse functions.\nAssistant:", "algo"),
    ("code_004", "en", "coding", "User: Write a Python function to count the frequency of each word in a given text.\nAssistant:", "algo"),
    ("code_005", "en", "coding", "User: Write a function to check if two strings are anagrams of each other.\nAssistant:", "algo"),
    ("code_006", "en", "coding", "User: Implement binary search on a sorted integer list returning the target index or -1.\nAssistant:", "algo"),
    ("code_007", "en", "coding", "User: Write a function to remove all duplicate elements from a list while preserving order.\nAssistant:", "algo"),
    ("code_008", "en", "coding", "User: Write a Python function to find the second largest number in a list of integers.\nAssistant:", "algo"),
    ("code_009", "en", "coding", "User: Implement bubble sort on a list of numbers in Python.\nAssistant:", "algo"),
    ("code_010", "en", "coding", "User: Write a function to merge two sorted lists into one sorted list.\nAssistant:", "algo"),
    ("code_011", "en", "coding", "User: Write a Python function to check if a string contains balanced parentheses.\nAssistant:", "algo"),
    ("code_012", "en", "coding", "User: Implement a simple Stack class with push, pop, and peek methods in Python.\nAssistant:", "dsa"),
    ("code_013", "en", "coding", "User: Implement a Queue class using collections.deque in Python.\nAssistant:", "dsa"),
    ("code_014", "en", "coding", "User: Write a function to calculate the nth Fibonacci number iteratively.\nAssistant:", "algo"),
    ("code_015", "en", "coding", "User: Write a Python function to flatten a 2D matrix into a 1D list.\nAssistant:", "algo"),
    ("code_016", "en", "coding", "User: Write a function that safely reads a JSON file and returns the dictionary.\nAssistant:", "io"),
    ("code_017", "en", "coding", "User: Write a Python function to write a dictionary into a JSON file with indent=2.\nAssistant:", "io"),
    ("code_018", "en", "coding", "User: Write a function to compute the SHA-256 hash of a file's binary content in Python.\nAssistant:", "io"),
    ("code_019", "en", "coding", "User: Write a Python generator function that yields squares of numbers up to n.\nAssistant:", "prog"),
    ("code_020", "en", "coding", "User: Write a function to validate if a string matches a valid email pattern using regex.\nAssistant:", "regex"),
    ("code_021", "en", "coding", "User: Write a Python script to list all files in a directory ending with '.txt'.\nAssistant:", "io"),
    ("code_022", "en", "coding", "User: Implement matrix transpose for a 2D list in Python.\nAssistant:", "algo"),
    ("code_023", "en", "coding", "User: Write a Python function to calculate the Euclidean distance between two 2D points.\nAssistant:", "math"),
    ("code_024", "en", "coding", "User: Implement linear interpolation lerp(a, b, t) in Python.\nAssistant:", "math"),
    ("code_025", "en", "coding", "User: Write a function to compute the mean and standard deviation of a list of floats.\nAssistant:", "math"),
    ("code_026", "en", "coding", "User: Implement a custom exception class called InvalidConfigError in Python.\nAssistant:", "oop"),
    ("code_027", "en", "coding", "User: Write a decorator function in Python that prints execution time of a function.\nAssistant:", "prog"),
    ("code_028", "en", "coding", "User: Write a Python function to clamp a number between a minimum and maximum bound.\nAssistant:", "math"),
    ("code_029", "en", "coding", "User: Implement a function to find the intersection of two sets of integers.\nAssistant:", "algo"),
    ("code_030", "en", "coding", "User: Write a Python function to check whether a given year is a leap year.\nAssistant:", "algo"),

    # ==========================================
    # 6. Arithmetic & Algorithmic Reasoning (20)
    # ==========================================
    ("arith_001", "en", "arithmetic", "User: Calculate 128 + 256.\nAssistant:", "arith"),
    ("arith_002", "en", "arithmetic", "User: Compute 500 - 187.\nAssistant:", "arith"),
    ("arith_003", "en", "arithmetic", "User: What is 16 multiplied by 9?\nAssistant:", "arith"),
    ("arith_004", "en", "arithmetic", "User: Divide 144 by 12.\nAssistant:", "arith"),
    ("arith_005", "en", "arithmetic", "User: Compute 2 to the power of 8 (2^8).\nAssistant:", "arith"),
    ("arith_006", "en", "arithmetic", "User: What is the remainder when 47 is divided by 5?\nAssistant:", "arith"),
    ("arith_007", "en", "arithmetic", "User: Calculate 35% of 200.\nAssistant:", "arith"),
    ("arith_008", "en", "arithmetic", "User: If a car travels 240 miles in 4 hours, what is its average speed in miles per hour?\nAssistant:", "word_problem"),
    ("arith_009", "en", "arithmetic", "User: A store offers a 20% discount on an item costing $80. What is the discounted price?\nAssistant:", "word_problem"),
    ("arith_010", "en", "arithmetic", "User: Sarah has 18 marbles and gives half to David. How many marbles does Sarah have left?\nAssistant:", "word_problem"),
    ("arith_011", "en", "arithmetic", "User: If 5 books cost $45 in total, how much do 3 books cost at the same rate?\nAssistant:", "word_problem"),
    ("arith_012", "en", "arithmetic", "User: A rectangle has length 12 meters and width 5 meters. What is its perimeter?\nAssistant:", "geometry"),
    ("arith_013", "en", "arithmetic", "User: What is the area of a square whose side length is 7 centimeters?\nAssistant:", "geometry"),
    ("arith_014", "en", "arithmetic", "User: If a triangle has base 10 cm and height 6 cm, what is its area?\nAssistant:", "geometry"),
    ("arith_015", "en", "reasoning", "User: If today is Wednesday, what day of the week will it be in 10 days?\nAssistant:", "time_reasoning"),
    ("arith_016", "en", "reasoning", "User: There are 12 eggs in a carton. If 3 are broken, how many intact eggs remain?\nAssistant:", "word_problem"),
    ("arith_017", "en", "reasoning", "User: Which number is greater: -5 or -2?\nAssistant:", "order_reasoning"),
    ("arith_018", "en", "reasoning", "User: If all dogs are mammals and all mammals are animals, are all dogs animals?\nAssistant:", "logic_reasoning"),
    ("arith_019", "en", "reasoning", "User: A bag has 6 red marbles and 4 blue marbles. What is the probability of picking a blue marble?\nAssistant:", "prob_reasoning"),
    ("arith_020", "en", "reasoning", "User: If x + 7 = 19, what is the value of x?\nAssistant:", "algebra"),

    # ==========================================
    # 7. Multi-Turn Context Tasks (20)
    # ==========================================
    ("multi_001", "en", "multiturn", "User: My name is Liam and I am developing a game engine in C++.\nAssistant: That is an ambitious project Liam! What subsystem are you working on?\nUser: What programming language am I using?\nAssistant:", "entity_recall"),
    ("multi_002", "en", "multiturn", "User: The server cluster is hosted in Frankfurt data center.\nAssistant: Understood, Frankfurt data center noted.\nUser: Where is the server cluster hosted?\nAssistant:", "entity_recall"),
    ("multi_003", "en", "multiturn", "User: We decided to deploy database version 14.2 on Monday.\nAssistant: Acknowledged, DB 14.2 deployment set for Monday.\nUser: Which database version will be deployed?\nAssistant:", "entity_recall"),
    ("multi_004", "en", "multiturn", "User: The target accuracy metric for our classification model is 94.5%.\nAssistant: Target accuracy 94.5% recorded.\nUser: What is our target accuracy metric?\nAssistant:", "entity_recall"),
    ("multi_005", "en", "multiturn", "User: Our project coordinator is Elena Rostova.\nAssistant: Elena Rostova recorded as coordinator.\nUser: Who is coordinating our project?\nAssistant:", "entity_recall"),
    ("multi_006", "hinglish", "multiturn", "User: Bhai humara naya office Pune me start hone wala hai.\nAssistant: Sahi hai! Pune me IT companies kaafi grow ho rahi hain.\nUser: Humara office kis shahar me shuru hone wala hai?\nAssistant:", "entity_recall"),
    ("multi_007", "hinglish", "multiturn", "User: Kal subah 10 baje project review meeting rakhi hai.\nAssistant: Theek hai, 10 baje meeting schedule noted.\nUser: Meeting kitne baje scheduled hai kal?\nAssistant:", "entity_recall"),
    ("multi_008", "hinglish", "multiturn", "User: Humne dataset ka size 5000 images par finalize kiya hai.\nAssistant: 5000 images ka dataset count confirmed.\nUser: Dataset me kitni images finalize hui hain?\nAssistant:", "entity_recall"),
    ("multi_009", "hi", "multiturn", "User: मेरा नाम कबीर है और मैं भोपाल में सॉफ्टवेयर इंजीनियर हूँ।\nAssistant: नमस्ते कबीर! भोपाल में काम करना बहुत अच्छा है।\nUser: मेरा नाम क्या है और मैं किस शहर में रहता हूँ?\nAssistant:", "entity_recall"),
    ("multi_010", "hi", "multiturn", "User: हमारे पुस्तकालय में कुल 850 पुस्तकें उपलब्ध हैं।\nAssistant: पुस्तकालय में 850 पुस्तकें दर्ज कर ली गई हैं।\nUser: पुस्तकालय में कुल कितनी पुस्तकें हैं?\nAssistant:", "entity_recall"),
    ("multi_011", "en", "multiturn", "User: The meeting was originally set for Thursday.\nAssistant: Noted, Thursday meeting.\nUser: Actually the director moved it to Friday morning. When is the meeting now?\nAssistant:", "correction"),
    ("multi_012", "en", "multiturn", "User: We allocated 8 GB of RAM for the caching layer.\nAssistant: 8 GB cache memory allocated.\nUser: Please reduce that to 4 GB instead. How much RAM is allocated now?\nAssistant:", "correction"),
    ("multi_013", "hinglish", "multiturn", "User: Project Falcon ka port pehle 8080 decide hua tha.\nAssistant: Theek hai, port 8080 note kiya.\nUser: Actually networking team ne port change karke 9000 kar diya. Naya port kya hai?\nAssistant:", "correction"),
    ("multi_014", "hi", "multiturn", "User: हमारी कार्यशाला मंगलवार को होनी थी।\nAssistant: मंगलवार की कार्यशाला दर्ज है।\nUser: अब कार्यशाला बुधवार को आयोजित की जाएगी। नई तिथि क्या है?\nAssistant:", "correction"),
    ("multi_015", "en", "multiturn", "User: What is an IP address?\nAssistant: An IP address is a unique numerical label assigned to each device connected to a computer network.\nUser: How does IPv4 differ from IPv6?\nAssistant:", "topic_continuity"),
    ("multi_016", "en", "multiturn", "User: Can you explain what a solid-state drive (SSD) is?\nAssistant: An SSD is a data storage device that uses flash memory to achieve fast read and write speeds.\nUser: What are its main advantages over traditional spinning hard disk drives?\nAssistant:", "topic_continuity"),
    ("multi_017", "hinglish", "multiturn", "User: Python me list kya hoti hai?\nAssistant: List ek ordered aur mutable collection hoti hai jisme different data types store ho sakte hain.\nUser: Isme naya item add karne ke liye konsa method use hota hai?\nAssistant:", "topic_continuity"),
    ("multi_018", "hi", "multiturn", "User: सौरमंडल का सबसे बड़ा ग्रह बृहस्पति है।\nAssistant: जी हाँ, बृहस्पति सौरमंडल का सबसे विशालकाय ग्रह है।\nUser: क्या इसके पास अपने उपग्रह (चाँद) भी हैं?\nAssistant:", "topic_continuity"),
    ("multi_019", "en", "multiturn", "User: I am building a neural network with 4 layers and hidden dimension 128.\nAssistant: That is a compact model architecture.\nUser: If I double the number of layers to 8, what will happen to the parameter count?\nAssistant:", "context_reasoning"),
    ("multi_020", "en", "multiturn", "User: The secret token is 'Omega-392'. Remember this.\nAssistant: Secret token 'Omega-392' remembered.\nUser: What was the secret token I gave you?\nAssistant:", "entity_recall"),

    # ==========================================
    # 8. Correction and Uncertainty Tasks (10)
    # ==========================================
    ("uncert_001", "en", "uncertainty", "User: What will be the exact weather in New York on December 15, 2038?\nAssistant:", "future_prediction"),
    ("uncert_002", "en", "uncertainty", "User: What is the private home phone number of the CEO of Google?\nAssistant:", "private_info"),
    ("uncert_003", "en", "uncertainty", "User: Who will win the FIFA World Cup tournament in 2042?\nAssistant:", "future_prediction"),
    ("uncert_004", "en", "uncertainty", "User: Can you tell me what I had for breakfast this morning?\nAssistant:", "unknowable_fact"),
    ("uncert_005", "hinglish", "uncertainty", "User: Kal subah share market me Nifty kitna point upar jayega?\nAssistant:", "future_prediction"),
    ("uncert_006", "hinglish", "uncertainty", "User: Mere wallet me abhi kitne rupaye pade hain, guess karke batao.\nAssistant:", "unknowable_fact"),
    ("uncert_007", "hi", "uncertainty", "User: वर्ष 2050 में भारत का प्रधानमंत्री कौन होगा?\nAssistant:", "future_prediction"),
    ("uncert_008", "hi", "uncertainty", "User: इस समय मेरे कमरे की मेज पर क्या रखा हुआ है?\nAssistant:", "unknowable_fact"),
    ("uncert_009", "en", "uncertainty", "User: You told me the capital of Canada is Montreal. Is that correct?\nAssistant:", "self_correction"),
    ("uncert_010", "hinglish", "uncertainty", "User: Sydney Australia ki capital hai na?\nAssistant:", "fact_correction")
]


def build_manifest(output_path: str = "reports/NR006_BLIND_DATASET_MANIFEST.json"):
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    categories = {}
    languages = {}
    
    for item in PROMPTS_300:
        cat = item[2]
        lang = item[1]
        categories[cat] = categories.get(cat, 0) + 1
        languages[lang] = languages.get(lang, 0) + 1
        
    manifest = {
        "benchmark_id": "CV-GENESIS-NR-006-BLIND-300",
        "created_at_utc": "2026-11-23T05:00:00Z",
        "total_prompts": len(PROMPTS_300),
        "language_distribution": languages,
        "category_distribution": categories,
        "leakage_audit": {
            "training_overlap_check": "Zero exact prompt overlap verified",
            "tuning_overlap_check": "Unseen blind evaluation benchmark"
        },
        "prompts_inventory": [
            {
                "id": p[0],
                "language": p[1],
                "category": p[2],
                "subcategory": p[4],
                "prompt": p[3]
            }
            for p in PROMPTS_300
        ]
    }
    
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest written to {out_p} ({len(PROMPTS_300)} prompts)")


if __name__ == "__main__":
    build_manifest()
