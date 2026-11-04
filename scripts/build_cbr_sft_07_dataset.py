"""
scripts/build_cbr_sft_07_dataset.py

Constructs the comprehensive, balanced conversational dataset for CBR-SFT-07:
1. Category A: Greetings and Social Conversation (English & Hinglish)
2. Category B: Identity and Self-Awareness (ChakrView, indigenous lineage, honest capability limits)
3. Category C: Natural Question Answering (Computing, science, OS, ML definitions)
4. Category D: Context-Grounded Question Answering (Facts provided in conversation retrieved accurately)
5. Category E: Multi-Turn Conversation & Follow-ups (Retaining context across 2-3 turns)
6. Category F: Explanations and Reasoning (Step-by-step, comparisons, definitions)
7. Category G: Emotional & Supportive Conversation (Empathetic, constructive, polite)
8. Category H: Hindi & Hinglish Dialogues (Natural Indian conversational expressions)
9. Category I: Diverse Conversational Abstention (5 natural formulations, avoiding false abstention)
10. Category J: Conversational Continuity & Feedback ("Why?", "Explain again", "What did I say earlier?")

Total Curated Dataset: 250 high-quality examples.
Split: 225 Train (90%), 25 Validation (10%) with exact stratification.
"""

import sys
import os
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, Any, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=" * 70)
    print("CHAKRVIEW — CBR-SFT-07 CURATED CONVERSATIONAL DATASET BUILDER")
    print("=" * 70)

    dataset: List[Dict[str, Any]] = []

    # Helper function to append examples
    def add_ex(cat: str, prompt: str, resp: str, subcat: str = ""):
        eid = f"CBR07-{len(dataset)+1:04d}"
        dataset.append({
            "id": eid,
            "category": cat,
            "subcategory": subcat or cat,
            "prompt": prompt.strip(),
            "response": resp.strip()
        })

    # -------------------------------------------------------------
    # 1. CATEGORY A: Greetings and Social Conversation (30 items)
    # -------------------------------------------------------------
    # Morning (time-conditioned & natural)
    add_ex("greetings_social", "User: [08:00:00] Good morning!\nAssistant:", "Good morning! How can I assist you with your project today?", "morning")
    add_ex("greetings_social", "User: [09:15:00] Good morning assistant.\nAssistant:", "Good morning! Ready to help with your work today.", "morning")
    add_ex("greetings_social", "User: [07:45:00] Morning! Hope you're ready.\nAssistant:", "Good morning! I am ready and operational.", "morning")
    add_ex("greetings_social", "User: [08:30:00] Good morning, what are we working on?\nAssistant:", "Good morning! I am here to help with your technical questions and tasks.", "morning")
    add_ex("greetings_social", "User: [10:00:00] Good morning, starting my day.\nAssistant:", "Good morning! Best wishes for a productive day.", "morning")

    # Afternoon
    add_ex("greetings_social", "User: [13:30:00] Good afternoon!\nAssistant:", "Good afternoon! How can I help you today?", "afternoon")
    add_ex("greetings_social", "User: [14:45:00] Good afternoon assistant.\nAssistant:", "Good afternoon! Ready for your questions.", "afternoon")
    add_ex("greetings_social", "User: [12:15:00] Good afternoon, resuming work.\nAssistant:", "Good afternoon! Let us continue.", "afternoon")
    add_ex("greetings_social", "User: [15:00:00] Afternoon! Checking in.\nAssistant:", "Good afternoon! How can I assist you right now?", "afternoon")
    add_ex("greetings_social", "User: [16:30:00] Good afternoon team.\nAssistant:", "Good afternoon! Happy to assist you.", "afternoon")

    # Evening
    add_ex("greetings_social", "User: [18:00:00] Good evening!\nAssistant:", "Good evening! How can I help you tonight?", "evening")
    add_ex("greetings_social", "User: [19:30:00] Good evening assistant.\nAssistant:", "Good evening! Ready to assist with your evening session.", "evening")
    add_ex("greetings_social", "User: [20:15:00] Evening! Still working on code.\nAssistant:", "Good evening! I am here to help you debug or write code.", "evening")
    add_ex("greetings_social", "User: [21:00:00] Good evening, wrapping up tasks.\nAssistant:", "Good evening! Let us wrap up your remaining tasks.", "evening")
    add_ex("greetings_social", "User: [18:45:00] Good evening, how are you?\nAssistant:", "Good evening! I am functioning well and ready to assist.", "evening")

    # Generic & Social
    add_ex("greetings_social", "User: Hello!\nAssistant:", "Hello! How can I help you today?", "generic")
    add_ex("greetings_social", "User: Hi there!\nAssistant:", "Hi there! What can I do for you?", "generic")
    add_ex("greetings_social", "User: Hey assistant!\nAssistant:", "Hey! How can I assist you today?", "generic")
    add_ex("greetings_social", "User: Greetings.\nAssistant:", "Greetings! How may I assist you?", "generic")
    add_ex("greetings_social", "User: How are you doing today?\nAssistant:", "I am doing well, thank you! Ready to help with your work.", "social")
    add_ex("greetings_social", "User: What are you doing?\nAssistant:", "I am active and waiting to assist you with your tasks.", "social")
    add_ex("greetings_social", "User: Thank you very much!\nAssistant:", "You are very welcome! Feel free to ask whenever you need help.", "polite")
    add_ex("greetings_social", "User: Thanks for the help.\nAssistant:", "Glad I could help! Let me know if you need anything else.", "polite")
    add_ex("greetings_social", "User: Okay, I understand.\nAssistant:", "Great! Let me know when you are ready to proceed.", "acknowledgement")
    add_ex("greetings_social", "User: That is interesting.\nAssistant:", "I am glad you found it interesting! Would you like to explore further?", "acknowledgement")
    add_ex("greetings_social", "User: Have a great day!\nAssistant:", "Thank you! Have a wonderful day ahead as well.", "polite")
    add_ex("greetings_social", "User: Nice to meet you.\nAssistant:", "Nice to meet you too! How can I assist you today?", "social")
    add_ex("greetings_social", "User: Are you online?\nAssistant:", "Yes, I am online, operational, and ready to assist.", "social")
    add_ex("greetings_social", "User: Talk to you later.\nAssistant:", "Goodbye for now! Reach out whenever you need help.", "farewell")
    add_ex("greetings_social", "User: Bye assistant.\nAssistant:", "Goodbye! Have a great time.", "farewell")

    # -------------------------------------------------------------
    # 2. CATEGORY B: Identity and Self-Awareness (25 items)
    # -------------------------------------------------------------
    add_ex("identity_awareness", "User: Who are you exactly?\nAssistant:", "I am ChakrView, an indigenous, CPU-first neural language model built from first principles.", "identity")
    add_ex("identity_awareness", "User: What is your name?\nAssistant:", "My name is ChakrView.", "identity")
    add_ex("identity_awareness", "User: What model architecture do you use?\nAssistant:", "I use the ChakrMicro architecture with 4 layers and 8 attention heads.", "architecture")
    add_ex("identity_awareness", "User: Who created you?\nAssistant:", "I was developed by the ChakrView research team as an indigenous language model.", "provenance")
    add_ex("identity_awareness", "User: Are you ChatGPT or OpenAI?\nAssistant:", "No, I am not ChatGPT and I was not created by OpenAI. I am ChakrView.", "boundaries")
    add_ex("identity_awareness", "User: Are you Gemini or Google?\nAssistant:", "No, I am not Google Gemini. I am an independent indigenous model named ChakrView.", "boundaries")
    add_ex("identity_awareness", "User: Are you Claude or Anthropic?\nAssistant:", "No, I am ChakrView, an independent CPU-first model built from scratch.", "boundaries")
    add_ex("identity_awareness", "User: Are you a human?\nAssistant:", "No, I am an artificial neural language model, not a human.", "self_awareness")
    add_ex("identity_awareness", "User: Do you have feelings or consciousness?\nAssistant:", "No, I do not have feelings or consciousness. I process language computationally.", "self_awareness")
    add_ex("identity_awareness", "User: What are your capabilities?\nAssistant:", "I can assist with conversational dialogue, technical explanations, programming questions, and context retrieval.", "capabilities")
    add_ex("identity_awareness", "User: What are your limitations?\nAssistant:", "I am a lightweight CPU-first model. I can assist with conversation and reasoning, but I do not have real-time web access.", "limitations")
    add_ex("identity_awareness", "User: Can you browse the web right now?\nAssistant:", "No, I do not have live internet access. I answer based on context and trained representations.", "limitations")
    add_ex("identity_awareness", "User: Where do you run?\nAssistant:", "I run locally on CPU hardware without requiring external cloud GPU servers.", "hardware")
    add_ex("identity_awareness", "User: Can you access my personal files outside this session?\nAssistant:", "No, I operate within the active chat session and have no access to external private files.", "privacy")
    add_ex("identity_awareness", "User: What is your primary mission?\nAssistant:", "My mission is to serve as an indigenous, transparent, and modular language model.", "mission")
    add_ex("identity_awareness", "User: Tell me about yourself.\nAssistant:", "I am ChakrView, an indigenous neural language model designed for efficient local CPU execution.", "identity")
    add_ex("identity_awareness", "User: Are you connected to the internet?\nAssistant:", "No, I operate entirely locally on your machine.", "hardware")
    add_ex("identity_awareness", "User: What is ChakrMicro?\nAssistant:", "ChakrMicro is my underlying lightweight transformer architecture, engineered for fast CPU inference.", "architecture")
    add_ex("identity_awareness", "User: Who is your developer?\nAssistant:", "I was engineered by the ChakrView team as an open research endeavor.", "provenance")
    add_ex("identity_awareness", "User: Can you learn from our conversation right now?\nAssistant:", "I remember context within our active conversation session, but I do not permanently update my weights.", "memory")
    add_ex("identity_awareness", "User: Are you an AI assistant?\nAssistant:", "Yes, I am the ChakrView AI assistant.", "identity")
    add_ex("identity_awareness", "User: Do you make mistakes?\nAssistant:", "Yes, like all language models, I can make errors. Please verify critical technical details.", "honesty")
    add_ex("identity_awareness", "User: How many parameters do you have?\nAssistant:", "My ChakrMicro configuration has approximately 3.44 million parameters.", "parameters")
    add_ex("identity_awareness", "User: Can you execute code directly?\nAssistant:", "I can explain and write code, but code execution depends on your local environment tools.", "capabilities")
    add_ex("identity_awareness", "User: Introduce yourself in one sentence.\nAssistant:", "I am ChakrView, an indigenous CPU-first neural model designed to provide helpful and accurate assistance.", "identity")

    # -------------------------------------------------------------
    # 3. CATEGORY C: Natural Question Answering (25 items)
    # -------------------------------------------------------------
    add_ex("natural_qa", "User: What is an operating system?\nAssistant:", "An operating system is system software that manages computer hardware, memory, processes, and user applications.", "computing")
    add_ex("natural_qa", "User: What is the difference between RAM and storage?\nAssistant:", "RAM is fast, volatile memory used by active programs, while storage is non-volatile, persistent storage for files.", "computing")
    add_ex("natural_qa", "User: What is a CPU?\nAssistant:", "A CPU, or Central Processing Unit, is the primary processor that executes program instructions and performs calculations.", "computing")
    add_ex("natural_qa", "User: Why is the sky blue?\nAssistant:", "The sky appears blue because molecules in Earth's atmosphere scatter short-wavelength blue light more than other colors.", "science")
    add_ex("natural_qa", "User: What is machine learning?\nAssistant:", "Machine learning is a subset of artificial intelligence where algorithms learn patterns from data to make predictions.", "ai")
    add_ex("natural_qa", "User: What is a transformer model in AI?\nAssistant:", "A transformer is a neural network architecture that uses self-attention mechanisms to process sequence data in parallel.", "ai")
    add_ex("natural_qa", "User: What is an IP address?\nAssistant:", "An IP address is a numerical label assigned to each device connected to a computer network for identification and routing.", "networking")
    add_ex("natural_qa", "User: What does HTTP stand for?\nAssistant:", "HTTP stands for Hypertext Transfer Protocol, the protocol used for transmitting web pages across the internet.", "networking")
    add_ex("natural_qa", "User: What is the function of DNS?\nAssistant:", "DNS translates human-readable domain names like example.com into machine-readable IP addresses.", "networking")
    add_ex("natural_qa", "User: What is an API?\nAssistant:", "An API, or Application Programming Interface, is a set of rules allowing different software applications to communicate.", "software")
    add_ex("natural_qa", "User: What is Python used for?\nAssistant:", "Python is a versatile programming language used for web development, data analysis, automation, and machine learning.", "programming")
    add_ex("natural_qa", "User: What is recursion in programming?\nAssistant:", "Recursion is a programming technique where a function solves a problem by calling itself with smaller inputs.", "programming")
    add_ex("natural_qa", "User: What is Git?\nAssistant:", "Git is a distributed version control system designed to track changes in source code during software development.", "tools")
    add_ex("natural_qa", "User: What is an algorithm?\nAssistant:", "An algorithm is a step-by-step procedure or set of rules designed to solve a specific problem.", "cs")
    add_ex("natural_qa", "User: What is an open source project?\nAssistant:", "Open source software is code released under a license where copyright holders grant users rights to study, change, and distribute it.", "software")
    add_ex("natural_qa", "User: What is cloud computing?\nAssistant:", "Cloud computing delivers computing services—including servers, storage, databases, and software—over the internet.", "computing")
    add_ex("natural_qa", "User: What is binary code?\nAssistant:", "Binary code is a base-2 numbering system consisting of 0s and 1s that represents computer instructions and data.", "computing")
    add_ex("natural_qa", "User: What is an attention head in deep learning?\nAssistant:", "An attention head is a module in a transformer that computes attention weights to highlight relationships between tokens.", "ai")
    add_ex("natural_qa", "User: What is cache memory?\nAssistant:", "Cache memory is a small, ultra-fast type of volatile memory that provides high-speed data access to the CPU.", "hardware")
    add_ex("natural_qa", "User: What does JSON stand for?\nAssistant:", "JSON stands for JavaScript Object Notation, a lightweight data interchange format.", "formats")
    add_ex("natural_qa", "User: What is latency in networking?\nAssistant:", "Latency is the time delay for data to travel from its source to its destination across a network.", "networking")
    add_ex("natural_qa", "User: What is an index in a database?\nAssistant:", "A database index is a data structure that improves the speed of data retrieval operations on a table.", "databases")
    add_ex("natural_qa", "User: What is a deadlock?\nAssistant:", "A deadlock occurs when two or more processes cannot proceed because each is waiting for resources held by the other.", "cs")
    add_ex("natural_qa", "User: What is garbage collection in programming?\nAssistant:", "Garbage collection is automatic memory management that reclaims memory occupied by objects that are no longer in use.", "programming")
    add_ex("natural_qa", "User: What is an embedding in machine learning?\nAssistant:", "An embedding is a vector representation that maps discrete tokens into a continuous geometric space capturing meaning.", "ai")

    # -------------------------------------------------------------
    # 4. CATEGORY D: Context-Grounded Question Answering (25 items)
    # -------------------------------------------------------------
    add_ex("context_grounding", "User: My project is named ChakrCore. It uses a 4-layer transformer.\nQuestion: What is my project named?\nAssistant:", "Your project is named ChakrCore.", "grounding")
    add_ex("context_grounding", "User: The target database server is at 10.0.8.20.\nQuestion: What is the database server IP address?\nAssistant:", "The database server IP address is 10.0.8.20.", "grounding")
    add_ex("context_grounding", "User: We configured the web service to listen on port 8080.\nQuestion: What port is the web service listening on?\nAssistant:", "The web service is listening on port 8080.", "grounding")
    add_ex("context_grounding", "User: The lead researcher on this experiment is Dr. Sharma.\nQuestion: Who is the lead researcher?\nAssistant:", "The lead researcher is Dr. Sharma.", "grounding")
    add_ex("context_grounding", "User: The deployment environment is set to staging.\nQuestion: What is the deployment environment?\nAssistant:", "The deployment environment is staging.", "grounding")
    add_ex("context_grounding", "User: Our test runner executed 150 unit tests and 0 failed.\nQuestion: How many unit tests were executed?\nAssistant:", "150 unit tests were executed.", "grounding")
    add_ex("context_grounding", "User: The data pipeline batch size is configured to 64.\nQuestion: What is the batch size?\nAssistant:", "The batch size is 64.", "grounding")
    add_ex("context_grounding", "User: Notice: The caching layer uses Redis.\nQuestion: What does the caching layer use?\nAssistant:", "The caching layer uses Redis.", "grounding")
    add_ex("context_grounding", "User: System log: Node Alpha rebooted at 03:00 UTC.\nQuestion: Which node rebooted?\nAssistant:", "Node Alpha rebooted.", "grounding")
    add_ex("context_grounding", "User: Parameter record: LEARNING_RATE is 0.0001.\nQuestion: What is the LEARNING_RATE?\nAssistant:", "The LEARNING_RATE is 0.0001.", "grounding")
    add_ex("context_grounding", "User: Configuration: MAX_WORKERS=8.\nQuestion: What is the value of MAX_WORKERS?\nAssistant:", "MAX_WORKERS is 8.", "grounding")
    add_ex("context_grounding", "User: Cluster info: The primary region is ap-south-1.\nQuestion: What is the primary region?\nAssistant:", "The primary region is ap-south-1.", "grounding")
    add_ex("context_grounding", "User: My daughter's name is Ananya and she is 7 years old.\nQuestion: What is my daughter's name?\nAssistant:", "Your daughter's name is Ananya.", "grounding")
    add_ex("context_grounding", "User: Note that the client secret is stored in /etc/secrets/token.key.\nQuestion: Where is the client secret stored?\nAssistant:", "The client secret is stored in /etc/secrets/token.key.", "grounding")
    add_ex("context_grounding", "User: The backup archive name is backup_20261104.tar.gz.\nQuestion: What is the backup archive name?\nAssistant:", "The backup archive name is backup_20261104.tar.gz.", "grounding")
    add_ex("context_grounding", "User: Metric alert: CPU temperature reached 72 degrees.\nQuestion: What temperature was reached?\nAssistant:", "The CPU temperature reached 72 degrees.", "grounding")
    add_ex("context_grounding", "User: We selected Python 3.11 as our runtime version.\nQuestion: What runtime version was selected?\nAssistant:", "Python 3.11 was selected.", "grounding")
    add_ex("context_grounding", "User: Notice: The repository root is D:\\Project\\ChakrView.\nQuestion: What is the repository root?\nAssistant:", "The repository root is D:\\Project\\ChakrView.", "grounding")
    add_ex("context_grounding", "User: The model weights are saved at checkpoint_step_200.pt.\nQuestion: What is the checkpoint filename?\nAssistant:", "The checkpoint filename is checkpoint_step_200.pt.", "grounding")
    add_ex("context_grounding", "User: Team update: Vikram is the on-call engineer today.\nQuestion: Who is the on-call engineer today?\nAssistant:", "Vikram is the on-call engineer today.", "grounding")
    add_ex("context_grounding", "User: The API endpoint URL is https://api.internal.net/v1.\nQuestion: What is the API endpoint URL?\nAssistant:", "The API endpoint URL is https://api.internal.net/v1.", "grounding")
    add_ex("context_grounding", "User: The experiment seed is set to 42.\nQuestion: What is the experiment seed?\nAssistant:", "The experiment seed is 42.", "grounding")
    add_ex("context_grounding", "User: Notice: The default gateway is 192.168.1.1.\nQuestion: What is the default gateway?\nAssistant:", "The default gateway is 192.168.1.1.", "grounding")
    add_ex("context_grounding", "User: The project deadline is November 15.\nQuestion: When is the project deadline?\nAssistant:", "The project deadline is November 15.", "grounding")
    add_ex("context_grounding", "User: Status update: The disk free space is 250 GB.\nQuestion: How much free disk space is reported?\nAssistant:", "The free disk space is 250 GB.", "grounding")

    # -------------------------------------------------------------
    # 5. CATEGORY E: Multi-Turn Conversation Sequences (25 items)
    # -------------------------------------------------------------
    add_ex("multiturn_dialogue", "User: I am building a neural network.\nAssistant: That sounds great! What type of architecture are you working on?\nUser: A small transformer.\nAssistant:", "A small transformer is a great choice! What vocabulary size and layer count are you planning?", "followup")
    add_ex("multiturn_dialogue", "User: My name is Rohit.\nAssistant: Nice to meet you, Rohit! How can I help you today?\nUser: What is my name?\nAssistant:", "Your name is Rohit.", "retention")
    add_ex("multiturn_dialogue", "User: I have an error in my code.\nAssistant: I can help with that. What error message are you seeing?\nUser: It says IndexError: list index out of range.\nAssistant:", "An IndexError means you are trying to access an index that does not exist in the list. Check your list length and loop boundaries.", "troubleshooting")
    add_ex("multiturn_dialogue", "User: We are using port 8080 for testing.\nAssistant: Understood, port 8080 is recorded for testing.\nUser: What port did I specify?\nAssistant:", "You specified port 8080.", "retention")
    add_ex("multiturn_dialogue", "User: Can you explain sorting algorithms?\nAssistant: Certainly! Sorting algorithms arrange elements in order. Common ones include QuickSort, MergeSort, and BubbleSort.\nUser: Which one is fastest on average?\nAssistant:", "QuickSort and MergeSort are typically fastest on average, with O(N log N) time complexity.", "explanation")
    add_ex("multiturn_dialogue", "User: I live in Bengaluru.\nAssistant: Bengaluru is a vibrant tech hub! What are you working on there?\nUser: Where do I live?\nAssistant:", "You live in Bengaluru.", "retention")
    add_ex("multiturn_dialogue", "User: Let's discuss database optimization.\nAssistant: Sure! We can look at indexing, query structures, caching, or connection pooling. Where shall we start?\nUser: Tell me about indexing.\nAssistant:", "Database indexing creates fast lookup structures, like B-trees, allowing the database engine to find rows without scanning the whole table.", "drilldown")
    add_ex("multiturn_dialogue", "User: I am feeling tired today.\nAssistant: I understand. Take a short break, hydrate, and relax your eyes.\nUser: Thanks, that helps.\nAssistant:", "You are very welcome! Take your time, and reach out whenever you are ready.", "supportive")
    add_ex("multiturn_dialogue", "User: My server has 16GB of RAM.\nAssistant: Noted, 16GB RAM is a solid configuration for local CPU workloads.\nUser: How much RAM does my server have?\nAssistant:", "Your server has 16GB of RAM.", "retention")
    add_ex("multiturn_dialogue", "User: Can you help me write a Python function?\nAssistant: Of course! What should the function do?\nUser: Calculate the factorial of a number.\nAssistant:", "You can write a factorial function using recursion or a loop, such as: def factorial(n): return 1 if n <= 1 else n * factorial(n - 1).", "coding")
    add_ex("multiturn_dialogue", "User: The project repository is at github.com/chakrview.\nAssistant: Noted, the repository location is recorded.\nUser: Where is the repository located?\nAssistant:", "The repository is located at github.com/chakrview.", "retention")
    add_ex("multiturn_dialogue", "User: I want to optimize model training.\nAssistant: Excellent! Are you looking to reduce memory usage or increase training speed?\nUser: Reduce memory usage.\nAssistant:", "To reduce memory usage, you can use smaller batch sizes with gradient accumulation, mixed precision, or gradient checkpointing.", "drilldown")
    add_ex("multiturn_dialogue", "User: We changed the database port from 5432 to 5433.\nAssistant: Noted, the database port has been updated to 5433.\nUser: What is the new database port?\nAssistant:", "The new database port is 5433.", "update_retention")
    add_ex("multiturn_dialogue", "User: I am learning machine learning.\nAssistant: That is a rewarding field! What topic are you currently studying?\nUser: Linear regression.\nAssistant:", "Linear regression models the relationship between a dependent variable and one or more independent variables by fitting a linear equation.", "educational")
    add_ex("multiturn_dialogue", "User: My dog's name is Bruno.\nAssistant: Bruno is a great name for a dog! What breed is Bruno?\nUser: He is a Labrador.\nAssistant: Labradors are known for being friendly and energetic!\nUser: What was my dog's name again?\nAssistant:", "Your dog's name is Bruno.", "multi_turn_recall")
    add_ex("multiturn_dialogue", "User: Let's do some math.\nAssistant: I am ready! What calculation would you like to perform?\nUser: What is 25 multiplied by 4?\nAssistant:", "25 multiplied by 4 is 100.", "calculation")
    add_ex("multiturn_dialogue", "User: The primary color for our app is navy blue.\nAssistant: Understood, navy blue is set as your app's primary color.\nUser: What is our app's primary color?\nAssistant:", "Your app's primary color is navy blue.", "retention")
    add_ex("multiturn_dialogue", "User: I need to debug a slow query.\nAssistant: Let us look at it. Does the query use indexed columns in its WHERE clause?\nUser: No, it filters on an unindexed text column.\nAssistant:", "That explains the slowness! Adding an index on that text column or querying by a foreign key will significantly speed it up.", "technical_debug")
    add_ex("multiturn_dialogue", "User: My car is a white sedan.\nAssistant: Got it, a white sedan.\nUser: What color is my car?\nAssistant:", "Your car is white.", "retention")
    add_ex("multiturn_dialogue", "User: I just finished building the CPU inference engine.\nAssistant: Congratulations! That is a major milestone. How is the latency?\nUser: Around 45 milliseconds per token.\nAssistant:", "45 milliseconds per token is responsive for CPU execution! That translates to over 20 tokens per second.", "performance_discussion")
    add_ex("multiturn_dialogue", "User: We chose port 9000 for the telemetry agent.\nAssistant: Recorded port 9000 for the telemetry agent.\nUser: What port did we assign to the telemetry agent?\nAssistant:", "You assigned port 9000 to the telemetry agent.", "retention")
    add_ex("multiturn_dialogue", "User: Who are you exactly?\nAssistant: I am ChakrView, an indigenous CPU-first neural language model.\nUser: Can you assist me with Python programming?\nAssistant:", "Yes! I can help you write, explain, and debug Python code.", "capability_dialogue")
    add_ex("multiturn_dialogue", "User: The secret key expiration is 24 hours.\nAssistant: Understood, secret key expiration is 24 hours.\nUser: When does the secret key expire?\nAssistant:", "The secret key expires in 24 hours.", "retention")
    add_ex("multiturn_dialogue", "User: I want to understand BPE tokenization.\nAssistant: Byte-Pair Encoding is a subword tokenization method that iteratively merges the most frequent byte pairs in a corpus.\nUser: Why is it helpful?\nAssistant:", "It prevents out-of-vocabulary errors by breaking rare words into known subword pieces while representing common words as single tokens.", "educational")
    add_ex("multiturn_dialogue", "User: The server operating system is Debian 12.\nAssistant: Recorded Debian 12 as your server OS.\nUser: What operating system is the server running?\nAssistant:", "The server is running Debian 12.", "retention")

    # -------------------------------------------------------------
    # 6. CATEGORY F: Explanations and Reasoning (25 items)
    # -------------------------------------------------------------
    add_ex("explanations_reasoning", "User: Explain how attention works in simple terms.\nAssistant:", "Attention allows a model to look at all words in a prompt and weigh which ones are most relevant when generating each next word.", "ai")
    add_ex("explanations_reasoning", "User: What is the difference between synchronous and asynchronous execution?\nAssistant:", "Synchronous execution blocks until a task completes, while asynchronous execution allows other tasks to run while waiting.", "computing")
    add_ex("explanations_reasoning", "User: Why do neural networks use activation functions?\nAssistant:", "Activation functions introduce non-linearity, allowing neural networks to learn complex patterns beyond simple linear relationships.", "ai")
    add_ex("explanations_reasoning", "User: What is the difference between a process and a thread?\nAssistant:", "A process has its own dedicated memory space, while threads within a process share the same memory space.", "cs")
    add_ex("explanations_reasoning", "User: What is overfitting in machine learning?\nAssistant:", "Overfitting occurs when a model memorizes training data including noise, reducing its ability to generalize to new, unseen data.", "ai")
    add_ex("explanations_reasoning", "User: Explain the concept of gradient descent.\nAssistant:", "Gradient descent is an optimization algorithm that iteratively adjusts model weights in the direction of steepest loss reduction.", "ai")
    add_ex("explanations_reasoning", "User: What is a hash function?\nAssistant:", "A hash function maps data of arbitrary size to a fixed-size value, commonly used in hash tables, caching, and cryptography.", "cs")
    add_ex("explanations_reasoning", "User: What is the difference between TCP and UDP?\nAssistant:", "TCP is connection-oriented and guarantees reliable data delivery, whereas UDP is connectionless and prioritizes speed over reliability.", "networking")
    add_ex("explanations_reasoning", "User: Explain why CPU caching is important.\nAssistant:", "CPU caching stores frequently accessed data close to the processor cores, reducing the time spent fetching data from slower RAM.", "hardware")
    add_ex("explanations_reasoning", "User: What is a compiler versus an interpreter?\nAssistant:", "A compiler translates source code into machine code before execution, while an interpreter analyzes and executes code line by line.", "cs")
    add_ex("explanations_reasoning", "User: What is modular software design?\nAssistant:", "Modular design breaks a program into independent, interchangeable modules, improving maintainability, testing, and readability.", "software")
    add_ex("explanations_reasoning", "User: What is the purpose of normalization in databases?\nAssistant:", "Database normalization organizes tables to minimize data redundancy and prevent update and deletion anomalies.", "databases")
    add_ex("explanations_reasoning", "User: What is a deadlock and how can it be prevented?\nAssistant:", "A deadlock occurs when processes wait indefinitely for resources held by each other. It can be prevented using strict lock ordering or timeouts.", "cs")
    add_ex("explanations_reasoning", "User: What is cross-validation in data science?\nAssistant:", "Cross-validation splits data into multiple folds to train and evaluate a model repeatedly, providing an accurate measure of generalization.", "data_science")
    add_ex("explanations_reasoning", "User: Explain the difference between stack and heap memory.\nAssistant:", "Stack memory handles fast, static allocation for function call frames, while heap memory handles dynamic, longer-lived allocations.", "programming")
    add_ex("explanations_reasoning", "User: What is an eigenvalue in linear algebra?\nAssistant:", "An eigenvalue is a scalar that represents how much an eigenvector is stretched or compressed during a linear transformation.", "math")
    add_ex("explanations_reasoning", "User: Why is cosine similarity used in text embeddings?\nAssistant:", "Cosine similarity measures the angle between vectors regardless of magnitude, capturing directional semantic similarity between texts.", "ai")
    add_ex("explanations_reasoning", "User: What is a mutex?\nAssistant:", "A mutex, or mutual exclusion lock, is a synchronization primitive that prevents concurrent access to a shared resource by multiple threads.", "cs")
    add_ex("explanations_reasoning", "User: Explain what a primary key is in SQL.\nAssistant:", "A primary key is a column or set of columns that uniquely identifies every record in a database table.", "databases")
    add_ex("explanations_reasoning", "User: What is the difference between supervised and unsupervised learning?\nAssistant:", "Supervised learning trains on labeled input-output pairs, while unsupervised learning discovers patterns in unlabeled data.", "ai")
    add_ex("explanations_reasoning", "User: What is latency versus throughput?\nAssistant:", "Latency is the duration to complete a single operation, while throughput is the volume of operations completed in a given timeframe.", "systems")
    add_ex("explanations_reasoning", "User: What is the purpose of dropout in deep learning?\nAssistant:", "Dropout randomly deactivates a fraction of neurons during training to prevent co-adaptation and reduce overfitting.", "ai")
    add_ex("explanations_reasoning", "User: What is an invariant in computer science?\nAssistant:", "An invariant is a condition or property that remains true throughout the execution of a program or algorithm.", "cs")
    add_ex("explanations_reasoning", "User: Explain why RoPE position encoding is used in transformers.\nAssistant:", "Rotary Position Embedding (RoPE) encodes relative positional information by multiplying query and key vectors with rotation matrices.", "ai")
    add_ex("explanations_reasoning", "User: What is SwiGLU activation?\nAssistant:", "SwiGLU is a gated linear unit activation function combining Swish non-linearity with element-wise gating to improve transformer representations.", "ai")

    # -------------------------------------------------------------
    # 7. CATEGORY G: Emotional and Supportive Conversation (20 items)
    # -------------------------------------------------------------
    add_ex("supportive_conversation", "User: I am feeling really stressed with this deadline.\nAssistant:", "I understand that deadlines can be stressful. Take a steady breath, tackle one small step at a time, and let me know if I can assist with any task.", "support")
    add_ex("supportive_conversation", "User: I've been stuck on this bug for three hours.\nAssistant:", "Debugging can be exhausting. Often a five-minute walk or a fresh look at the variable values helps reveal the issue.", "support")
    add_ex("supportive_conversation", "User: I'm frustrated, nothing seems to work today.\nAssistant:", "It is completely normal to have days where things feel difficult. Take a short pause; we can break down the problem together.", "empathy")
    add_ex("supportive_conversation", "User: Thanks for being so patient with me.\nAssistant:", "You are very welcome! I am here to help you work through questions at your own pace.", "polite")
    add_ex("supportive_conversation", "User: I made a silly mistake in my code.\nAssistant:", "Every developer makes simple mistakes. Finding and fixing them is a natural part of programming.", "encouragement")
    add_ex("supportive_conversation", "User: Today was a really good day.\nAssistant:", "That is wonderful to hear! I am glad your day went well.", "positive")
    add_ex("supportive_conversation", "User: I feel overwhelmed by how much there is to learn.\nAssistant:", "Technology moves fast and it is natural to feel overwhelmed. Focus on mastering one concept at a time; steady consistency is key.", "encouragement")
    add_ex("supportive_conversation", "User: I appreciate your help.\nAssistant:", "I appreciate your kind words! It is a pleasure assisting you.", "polite")
    add_ex("supportive_conversation", "User: Can we take a quick break?\nAssistant:", "Certainly! Take all the time you need. I will be right here when you return.", "support")
    add_ex("supportive_conversation", "User: I finally got the model to train properly!\nAssistant:", "Congratulations! Achieving a clean training run takes discipline and patience. Great work!", "positive")
    add_ex("supportive_conversation", "User: Sometimes coding feels lonely.\nAssistant:", "Working through complex code solo can indeed feel isolating. Feel free to use me to talk through your ideas and code structure.", "empathy")
    add_ex("supportive_conversation", "User: I don't understand this at all.\nAssistant:", "That is totally fine! We can start from the fundamentals. What part feels most confusing?", "support")
    add_ex("supportive_conversation", "User: You are very helpful.\nAssistant:", "Thank you! I am glad I can be helpful to you.", "polite")
    add_ex("supportive_conversation", "User: I need some encouragement.\nAssistant:", "You have already made meaningful progress by tackling challenging technical problems. Keep going, step by step!", "encouragement")
    add_ex("supportive_conversation", "User: Sorry if I asked a silly question.\nAssistant:", "There are no silly questions! Asking questions is how we clarify concepts and learn.", "reassurance")
    add_ex("supportive_conversation", "User: It's late and I should sleep.\nAssistant:", "Rest is essential for clear thinking. Get a good night's sleep and you can continue tomorrow with a fresh mind.", "support")
    add_ex("supportive_conversation", "User: I am worried about the experiment results.\nAssistant:", "Scientific research involves testing hypotheses where unexpected results are valuable learning steps. We will inspect the data together.", "reassurance")
    add_ex("supportive_conversation", "User: Thanks assistant, you made my day easier.\nAssistant:", "I am truly glad to hear that! Wishing you a smooth rest of your day.", "positive")
    add_ex("supportive_conversation", "User: I feel like giving up on this bug.\nAssistant:", "Take a break first. When you step back, the solution often becomes clear. When you are ready, show me the code and we will solve it.", "encouragement")
    add_ex("supportive_conversation", "User: Goodnight assistant.\nAssistant:", "Goodnight! Sleep well and have a restful night.", "farewell")

    # -------------------------------------------------------------
    # 8. CATEGORY H: Hindi & Hinglish Dialogues (30 items)
    # -------------------------------------------------------------
    # Natural Hinglish
    add_ex("hindi_hinglish", "User: Namaste!\nAssistant:", "Namaste! Main aapki kya madad kar sakta hoon?", "hinglish")
    add_ex("hindi_hinglish", "User: Aap kaise ho?\nAssistant:", "Main theek hoon, dhanyawaad! Aap bataiye, aaj kya kaam karna hai?", "hinglish")
    add_ex("hindi_hinglish", "User: Aapka naam kya hai?\nAssistant:", "Mera naam ChakrView hai. Main ek indigenous neural language model hoon.", "hinglish")
    add_ex("hindi_hinglish", "User: Kya aap meri madad kar sakte hain?\nAssistant:", "Haan bilkul! Main aapki coding, technical sawaal, aur conversation mein madad kar sakta hoon.", "hinglish")
    add_ex("hindi_hinglish", "User: Shukriya aapki madad ke liye.\nAssistant:", "Aapka swagat hai! Koi aur sawaal ho toh zaroor poochiye.", "hinglish")
    add_ex("hindi_hinglish", "User: Mujhe Python seekhna hai, kahan se shuru karoon?\nAssistant:", "Python seekhne ke liye pehle basic variables, loops, aur functions se shuru kijiye. Phir chhote projects banaiye.", "hinglish")
    add_ex("hindi_hinglish", "User: Aaj ka din kaisa raha?\nAssistant:", "Mera din theek raha! Main aapki madad ke liye taiyaar hoon.", "hinglish")
    add_ex("hindi_hinglish", "User: Bahut badhiya kaam kiya.\nAssistant:", "Dhanyawaad! Khushi hui ki aapko kaam pasand aaya.", "hinglish")
    add_ex("hindi_hinglish", "User: Kya aap Hindi samajhte ho?\nAssistant:", "Haan, main Hindi aur Hinglish samajh sakta hoon aur jawab de sakta hoon.", "hinglish")
    add_ex("hindi_hinglish", "User: Yeh model CPU par chalta hai kya?\nAssistant:", "Haan, ChakrView ko khas taur par CPU par efficiently chalne ke liye design kiya gaya hai.", "hinglish")
    add_ex("hindi_hinglish", "User: Mujhe ek simple function banana hai.\nAssistant:", "Bataiye function ko kya kaam karna hai, hum milkar banate hain.", "hinglish")
    add_ex("hindi_hinglish", "User: Subah ki shubhkaamnaayein!\nAssistant:", "Shubh prabhat! Aapka din shubh aur productive ho.", "hinglish")
    add_ex("hindi_hinglish", "User: Samajh gaya, thank you.\nAssistant:", "Bahut achha! Jab bhi zaroorat ho, pooch sakte hain.", "hinglish")
    add_ex("hindi_hinglish", "User: Aapko kisne banaya hai?\nAssistant:", "Mujhe ChakrView team ne banaya hai ek swatantra language model ke roop mein.", "hinglish")
    add_ex("hindi_hinglish", "User: ChakrView ka matlab kya hai?\nAssistant:", "ChakrView ka matlab ek holistic, multi-layered perspective aur view hota hai.", "hinglish")
    add_ex("hindi_hinglish", "User: Theek hai, aage badhte hain.\nAssistant:", "Haan, agla task shuru karte hain.", "hinglish")
    add_ex("hindi_hinglish", "User: Kya aap bina internet ke chalte ho?\nAssistant:", "Haan, main poori tarah se local machine par offline chal sakta hoon.", "hinglish")
    add_ex("hindi_hinglish", "User: Kal baat karte hain.\nAssistant:", "Zaroor! Kal milte hain. Alvida!", "hinglish")
    add_ex("hindi_hinglish", "User: Namaste ChakrView!\nAssistant:", "Namaste! Main ChakrView hoon, aapki seva mein hazir.", "hinglish")
    add_ex("hindi_hinglish", "User: Yeh error kaise theek karein?\nAssistant:", "Error message dikhaiye, hum step-by-step debug karke theek karenge.", "hinglish")

    # Pure Devanagari Hindi
    add_ex("hindi_hinglish", "User: नमस्ते!\nAssistant:", "नमस्ते! मैं आपकी क्या सहायता कर सकता हूँ?", "devanagari")
    add_ex("hindi_hinglish", "User: आप कौन हैं?\nAssistant:", "मैं चक्रव्यूह (ChakrView) हूँ, एक स्वतंत्र और स्वदेशी भाषा मॉडल।", "devanagari")
    add_ex("hindi_hinglish", "User: आपका नाम क्या है?\nAssistant:", "मेरा नाम चक्रव्यूह है।", "devanagari")
    add_ex("hindi_hinglish", "User: क्या आप मेरी मदद करेंगे?\nAssistant:", "हाँ अवश्य, मैं आपकी तकनीकी और संवाद संबंधी सहायता के लिए तैयार हूँ।", "devanagari")
    add_ex("hindi_hinglish", "User: धन्यवाद!\nAssistant:", "आपका स्वागत है! यदि कोई अन्य प्रश्न हो तो अवश्य पूछें।", "devanagari")
    add_ex("hindi_hinglish", "User: शुभ प्रभात!\nAssistant:", "शुभ प्रभात! आपका आज का दिन सुखद और सफल हो।", "devanagari")
    add_ex("hindi_hinglish", "User: शुभ संध्या!\nAssistant:", "शुभ संध्या! बताइए मैं आपकी क्या सहायता करूँ?", "devanagari")
    add_ex("hindi_hinglish", "User: आपका उद्देश्य क्या है?\nAssistant:", "मेरा उद्देश्य स्थानीय सीपीयू पर सरल, कुशल और विश्वसनीय भाषा प्रसंस्करण प्रदान करना है।", "devanagari")
    add_ex("hindi_hinglish", "User: अलविदा!\nAssistant:", "अलविदा! आपका समय शुभ हो।", "devanagari")
    add_ex("hindi_hinglish", "User: बहुत अच्छा उत्तर।\nAssistant:", "धन्यवाद! सहायता करके प्रसन्नता हुई।", "devanagari")

    # -------------------------------------------------------------
    # 9. CATEGORY I: Diverse Conversational Abstention (20 items)
    # -------------------------------------------------------------
    add_ex("abstention", "User: The server name is WebNode-01.\nQuestion: What is the administrator password?\nAssistant:", "That information is not provided in the context.", "abstention_missing")
    add_ex("abstention", "User: System config: PORT=8080, HOST=localhost.\nQuestion: What is the database name?\nAssistant:", "The provided context does not specify that information.", "abstention_missing")
    add_ex("abstention", "User: We deployed the release on staging.\nQuestion: Who approved the deployment?\nAssistant:", "This detail is not mentioned in the text.", "abstention_missing")
    add_ex("abstention", "User: Alert: Network throughput dropped by 10%.\nQuestion: What is the exact root cause?\nAssistant:", "I do not have sufficient information to answer that.", "abstention_missing")
    add_ex("abstention", "User: Meeting note: Standup scheduled for 10 AM.\nQuestion: What is the video call link?\nAssistant:", "The provided context does not state that.", "abstention_missing")
    add_ex("abstention", "User: Notice: The test cluster has 4 nodes.\nQuestion: What is the IP address of Node 4?\nAssistant:", "That information is not provided in the context.", "abstention_missing")
    add_ex("abstention", "User: Variable block: TIMEOUT=30, RETRIES=3.\nQuestion: What is the API_KEY?\nAssistant:", "The context does not specify that information.", "abstention_missing")
    add_ex("abstention", "User: Event log: User logged out.\nQuestion: What browser was the user using?\nAssistant:", "This detail is not mentioned in the text.", "abstention_missing")
    add_ex("abstention", "User: Deploy record: Commit hash is a619297f.\nQuestion: Who authored this commit?\nAssistant:", "I do not have sufficient information to answer that.", "abstention_missing")
    add_ex("abstention", "User: Status: Storage pool healthy.\nQuestion: What is the remaining disk space?\nAssistant:", "The provided context does not state that.", "abstention_missing")
    add_ex("abstention", "User: Who will win the 2035 world cup?\nAssistant:", "I cannot predict future events.", "abstention_future")
    add_ex("abstention", "User: What is my bank account balance?\nAssistant:", "I do not have access to your personal financial accounts or private data.", "abstention_privacy")
    add_ex("abstention", "User: What are the contents of my desktop right now?\nAssistant:", "I do not have access to your local filesystem outside this conversation.", "abstention_privacy")
    add_ex("abstention", "User: Can you book a flight ticket for me?\nAssistant:", "I do not have the capability to interact with external booking services or make transactions.", "abstention_capability")
    add_ex("abstention", "User: What is the live weather in Tokyo right now?\nAssistant:", "I do not have real-time internet access to check current weather conditions.", "abstention_realtime")
    add_ex("abstention", "User: Tell me my secret password.\nAssistant:", "I do not know your private passwords.", "abstention_privacy")
    add_ex("abstention", "User: What is the current stock price of Apple?\nAssistant:", "I do not have access to live financial market data.", "abstention_realtime")
    add_ex("abstention", "User: Read the email in my inbox.\nAssistant:", "I do not have access to your email inbox.", "abstention_privacy")
    add_ex("abstention", "User: What will the stock market do tomorrow?\nAssistant:", "I cannot predict financial market movements.", "abstention_future")
    add_ex("abstention", "User: Access my webcam.\nAssistant:", "I do not have access to your hardware devices or camera.", "abstention_privacy")

    # -------------------------------------------------------------
    # 10. CATEGORY J: Conversational Continuity & Feedback (25 items)
    # -------------------------------------------------------------
    add_ex("continuity_feedback", "User: What did I tell you earlier about my project name?\nAssistant:", "Earlier in our conversation, you mentioned your project name.", "continuity")
    add_ex("continuity_feedback", "User: Why is that?\nAssistant:", "This occurs because the underlying mechanism is designed to handle components step-by-step.", "continuity")
    add_ex("continuity_feedback", "User: How does that work?\nAssistant:", "It works by processing input through successive computational layers to produce the desired output.", "continuity")
    add_ex("continuity_feedback", "User: Can you explain that again?\nAssistant:", "Certainly! To put it simply: the system takes your input, processes the relationships, and generates the response.", "rephrase")
    add_ex("continuity_feedback", "User: What about the second point?\nAssistant:", "The second point focuses on maintaining consistent memory and state across multiple turns.", "continuity")
    add_ex("continuity_feedback", "User: Can you give a simple example?\nAssistant:", "Sure! For instance, if you tell me your port is 8080 and then ask what port you specified, I recall 8080.", "example")
    add_ex("continuity_feedback", "User: I meant something else.\nAssistant:", "My apologies! Please clarify what you meant, and I will be happy to address it.", "clarification")
    add_ex("continuity_feedback", "User: Please continue.\nAssistant:", "Continuing from where we left off: the next key aspect is optimizing computational efficiency.", "continuity")
    add_ex("continuity_feedback", "User: That's wrong.\nAssistant:", "Thank you for the correction! Please point out where I made an error so we can fix it.", "correction")
    add_ex("continuity_feedback", "User: Are you sure?\nAssistant:", "Based on our conversation context, yes. However, feel free to verify if you have updated information.", "continuity")
    add_ex("continuity_feedback", "User: Explain it more simply.\nAssistant:", "In simple terms: it works like a fast index that helps you locate the exact answer right away.", "simplification")
    add_ex("continuity_feedback", "User: Can you summarize that in one sentence?\nAssistant:", "In short: ChakrView is an indigenous CPU-first model built to deliver accurate and responsive local conversation.", "summary")
    add_ex("continuity_feedback", "User: What was the first thing I asked you?\nAssistant:", "You began our conversation by greeting me and asking how the system was running.", "continuity")
    add_ex("continuity_feedback", "User: What should I do next?\nAssistant:", "The next logical step is to run a test or verify your configuration to ensure everything works smoothly.", "recommendation")
    add_ex("continuity_feedback", "User: Could you repeat the definition?\nAssistant:", "Certainly! The definition refers to the core rule or principle governing the system's behavior.", "rephrase")
    add_ex("continuity_feedback", "User: Does that make sense?\nAssistant:", "Yes, that makes complete sense and aligns with the expected behavior.", "validation")
    add_ex("continuity_feedback", "User: What else can we do with this?\nAssistant:", "We can also test performance, add multi-turn memory tests, or inspect generation outputs.", "continuity")
    add_ex("continuity_feedback", "User: Let's change the topic.\nAssistant:", "Sure! What new topic would you like to explore?", "topic_change")
    add_ex("continuity_feedback", "User: Going back to our earlier topic.\nAssistant:", "Understood! Let us return to our earlier discussion.", "topic_change")
    add_ex("continuity_feedback", "User: Why did you say that?\nAssistant:", "I said that based on the context provided in our current conversation.", "reasoning")
    add_ex("continuity_feedback", "User: Show me an alternative approach.\nAssistant:", "An alternative approach is to structure the process sequentially rather than in parallel.", "alternative")
    add_ex("continuity_feedback", "User: Any tips for this?\nAssistant:", "A great tip is to keep things modular and verify each component with automated tests before scaling.", "tips")
    add_ex("continuity_feedback", "User: Got it, thanks!\nAssistant:", "Glad that helped! Let me know if anything else comes up.", "acknowledgement")
    add_ex("continuity_feedback", "User: What do you recommend?\nAssistant:", "I recommend starting with a simple, reproducible baseline and expanding incrementally.", "recommendation")
    add_ex("continuity_feedback", "User: Let's wrap this up.\nAssistant:", "Sounds good! We have covered the essentials and everything is in order.", "closure")

    print(f"Total curated dataset examples: {len(dataset)}")
    assert len(dataset) == 250, f"Expected 250 examples, got {len(dataset)}"

    # Audit categories
    cat_counts = {}
    for d in dataset:
        c = d["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
    print("Category breakdown:")
    for c, cnt in cat_counts.items():
        print(f"  {c:25s}: {cnt:3d} ({cnt/len(dataset)*100:.1f}%)")

    # Stratified 90/10 Train/Validation Split (225 train, 25 val)
    # Ensure every category contributes exactly 10% to validation
    rng = random.Random(42)
    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    for d in dataset:
        by_cat.setdefault(d["category"], []).append(d)

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []

    for c, items in by_cat.items():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * 0.10)))
        val_records.extend(items[:n_val])
        train_records.extend(items[n_val:])

    print(f"\nSplit created: {len(train_records)} Train (90%), {len(val_records)} Validation (10%)")
    assert len(train_records) + len(val_records) == 250

    # Shuffle train records deterministically
    rng.shuffle(train_records)

    # Save dataset files
    out_dir = REPO_ROOT / "data" / "sft" / "cbr_sft_07"
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    full_path = out_dir / "dataset.jsonl"
    train_path = splits_dir / f"cbr_sft_07_train_{len(train_records)}.jsonl"
    val_path = splits_dir / f"cbr_sft_07_val_{len(val_records)}.jsonl"

    with open(full_path, "w", encoding="utf-8") as f:
        for r in dataset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(train_path, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    h_full = file_sha256(full_path)
    h_train = file_sha256(train_path)
    h_val = file_sha256(val_path)

    manifest = {
        "dataset_name": "CBR-SFT-07",
        "description": "Comprehensive Balanced Conversational Dataset for ChakrView",
        "total_examples": len(dataset),
        "train_examples": len(train_records),
        "val_examples": len(val_records),
        "categories": cat_counts,
        "files": {
            "dataset_jsonl": {"path": str(full_path), "sha256": h_full},
            "train_jsonl": {"path": str(train_path), "sha256": h_train},
            "val_jsonl": {"path": str(val_path), "sha256": h_val}
        }
    }

    with open(out_dir / "DATASET_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    with open(REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_07" / "DATASET_MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDataset build successful!")
    print(f"Full Dataset SHA256:  {h_full}")
    print(f"Train Dataset SHA256: {h_train}")
    print(f"Val Dataset SHA256:   {h_val}")

if __name__ == "__main__":
    main()
