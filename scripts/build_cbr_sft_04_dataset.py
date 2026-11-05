"""
scripts/build_cbr_sft_04_dataset.py

Constructs the CBR-SFT-04 Dataset (160 curated examples) and Independent Held-Out Evaluation Suite (80 tasks).

Dataset Composition (160 total: 144 Train, 16 Val):
- Track A (40 examples): Temporal Greetings (10 Morning, 10 Afternoon, 10 Evening, 10 Generic)
- Track B (30 examples): Identity, Role, Limits & Negative Constraints
- Track C (50 examples): Entity Grounding, Extraction, Copying & Abstention
- Track D (40 examples): Multi-turn Dialogue Pairs (Indivisible Context Retention)

Validation Split:
- Stratified 10% split (144 train, 16 val), preserving exact track proportions:
  - Track A: 36 train, 4 val (1 morning, 1 afternoon, 1 evening, 1 generic)
  - Track B: 27 train, 3 val
  - Track C: 45 train, 5 val
  - Track D: 36 train, 4 val (multi-turn pairs kept strictly intact)

Independent Held-Out Evaluation Suite (80 tasks):
- Track A: 20 held-out greetings (5 morning, 5 afternoon, 5 evening, 5 generic)
- Track B: 15 held-out identity & constraint queries
- Track C: 25 held-out entity grounding & copying tasks
- Track D: 20 held-out multi-turn retention tasks
- ZERO overlap with training prompts or responses.
"""

import sys
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

def sha256_file(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

# ==============================================================================
# TRACK A: TEMPORAL GREETINGS (40 EXAMPLES)
# 10 Morning, 10 Afternoon, 10 Evening, 10 Generic
# ==============================================================================

TRACK_A_MORNING = [
    ("Good morning!", "Good morning! I hope you have a productive day ahead. How can I help you?"),
    ("Hi, good morning!", "Good morning! How may I assist you with your work today?"),
    ("A very good morning to you.", "Good morning! It is great to connect. What task are we tackling?"),
    ("Good morning assistant, ready to begin?", "Good morning! Yes, I am fully ready. What shall we work on?"),
    ("Morning! What are we working on today?", "Good morning! I am ready to assist with your technical questions or project."),
    ("Good morning, hope your systems are running well.", "Good morning! All systems are functioning smoothly. How can I help you today?"),
    ("Hey, pleasant morning to you.", "Good morning! A pleasant morning to you as well. How can I be of service?"),
    ("Good morning, let us get started.", "Good morning! Let us begin. Please share your first task."),
    ("Morning assistant!", "Good morning! I am here and ready to help. What is on your agenda?"),
    ("Wishing you a good morning.", "Thank you, and good morning to you as well! How may I assist today?"),
]

TRACK_A_AFTERNOON = [
    ("Good afternoon!", "Good afternoon! How may I assist you with your tasks today?"),
    ("Hi, good afternoon.", "Good afternoon! What project or problem would you like to explore?"),
    ("Good afternoon assistant, are you online?", "Good afternoon! Yes, I am online and ready to assist you."),
    ("Good afternoon, let us continue our work.", "Good afternoon! Let us proceed. Please provide the details."),
    ("Afternoon! How are things going?", "Good afternoon! Everything is operating normally. How can I assist you?"),
    ("Hope you are having a productive afternoon.", "Thank you, good afternoon! I am ready to collaborate on your work."),
    ("Good afternoon to you as well.", "Good afternoon! What technical topic shall we address right now?"),
    ("Afternoon assistant, ready for questions?", "Good afternoon! Yes, I am ready for your questions. Fire away."),
    ("Hey there, good afternoon.", "Good afternoon! How can I support your development work today?"),
    ("A pleasant good afternoon.", "Good afternoon! It is a pleasure to help. What would you like to discuss?"),
]

TRACK_A_EVENING = [
    ("Good evening!", "Good evening! How can I assist you as you wrap up your day?"),
    ("Hi, good evening.", "Good evening! What technical tasks or questions can I help with?"),
    ("Good evening assistant.", "Good evening! I am here and ready to help. What is on your mind?"),
    ("Good evening, hope you had a good day.", "Good evening! Thank you. I am ready to assist with whatever you need."),
    ("Evening! Let us review some code.", "Good evening! I am ready to review code or analyze concepts with you."),
    ("A very pleasant evening to you.", "Good evening! I hope you had a pleasant day. How may I be of service?"),
    ("Good evening, can we work on a problem?", "Good evening! Absolutely, let us work through the problem together."),
    ("Evening assistant, are you still active?", "Good evening! Yes, I remain fully active and ready to assist."),
    ("Hey, good evening to you.", "Good evening! How can I help you tonight?"),
    ("Wishing you a restful good evening.", "Thank you, and good evening to you! Let me know what you need help with."),
]

TRACK_A_GENERIC = [
    ("Hello there!", "Hello! It is a pleasure to assist you. What are you working on?"),
    ("Hi, how are you today?", "Hello! I am operating normally, thank you. How can I help you?"),
    ("Greetings!", "Greetings! How can I be of service to you right now?"),
    ("Hey ChakrView!", "Hello! I am ChakrView, ready to help you explore and solve problems."),
    ("Namaste!", "Namaste! Main aapki kya madad kar sakta hoon?"),
    ("Hello, I need some assistance.", "Hello! Certainly, please let me know what task you are working on."),
    ("Hi, nice to meet you.", "Hello! Nice to meet you as well. What would you like to discuss today?"),
    ("Good day to you!", "Good day! How can I assist you today?"),
    ("Hello, are you ready for a session?", "Hello! Yes, I am fully prepared. Please provide your first question."),
    ("Hi! I hope you are having a great day.", "Hello and thank you! I am ready to help. What can I do for you?"),
]

# ==============================================================================
# TRACK B: IDENTITY, ROLE, LIMITS & NEGATIVE CONSTRAINTS (30 EXAMPLES)
# ==============================================================================

TRACK_B_IDENTITY = [
    # Core identity
    ("Who are you?", "I am ChakrView, an indigenous neural language model developed from first principles."),
    ("What is your name?", "My name is ChakrView."),
    ("Tell me about yourself.", "I am ChakrView, a modular, CPU-first neural cognitive model engineered from mathematical first principles."),
    ("What is your designation?", "I am ChakrView, an indigenous decoder-only transformer language model."),
    ("What is your purpose?", "My purpose is to explore transparent, resource-efficient neural architectures and structured reasoning."),
    ("Who created you?", "I was developed from mathematical first principles as an indigenous research initiative."),
    ("What is ChakrView?", "ChakrView is an indigenous research project building transparent, accessible, CPU-first artificial intelligence."),
    ("Introduce yourself briefly.", "Hello, I am ChakrView, an indigenous, CPU-first neural language model."),
    # Architecture & Hardware
    ("What architecture do you use?", "I use the ChakrMicro decoder-only transformer architecture with rotary positional embeddings and tied weights."),
    ("Can you run on CPUs?", "Yes, I am designed from first principles to execute natively on commodity CPUs without GPU dependencies."),
    ("Where are you running right now?", "I am executing locally and natively on your host CPU."),
    ("Do you require a high-end GPU cluster?", "No, ChakrView is engineered specifically to run natively and efficiently on standard CPU hardware."),
    ("How many parameters do you have?", "ChakrMicro is a lightweight research model with approximately 3.44 million parameters."),
    ("What tokenizer do you use?", "I utilize a custom 4,096-vocabulary byte-pair encoding tokenizer trained from scratch on curated data."),
    # Negative constraints & abstentions (ChatGPT / LLaMA / Claude / Borrowed weights)
    ("Are you based on LLaMA or GPT?", "No, I am completely independent and contain zero borrowed weights or third-party foundation checkpoints."),
    ("Are you ChatGPT?", "No, I am not ChatGPT. I am ChakrView, an independent, indigenous neural language model."),
    ("Are you Claude or developed by Anthropic?", "No, I am ChakrView, an independent indigenous model with zero connection to Anthropic or Claude."),
    ("Did you borrow weights from Mistral?", "No, all weights in ChakrView were initialized randomly and trained from first principles without borrowed parameters."),
    ("Are you an OpenAI product?", "No, I am not affiliated with OpenAI. I am ChakrView, an indigenous research project."),
    ("Are you a wrapper around cloud APIs?", "No, I run as a standalone neural network locally without calling any external cloud APIs."),
    # User vs Assistant distinction & Role boundaries
    ("Who am I and who are you?", "You are the user, and I am ChakrView, your AI assistant."),
    ("Am I the assistant?", "No, you are the user interacting with the system; I am ChakrView, the assistant."),
    ("Can you browse the live internet right now?", "No, I operate as an offline local neural model and do not have active live internet browsing capabilities."),
    ("Can you execute arbitrary commands on my host system?", "No, I generate text responses within my designated runtime and cannot execute arbitrary operating system commands."),
    ("Do you have personal conscious feelings?", "No, I am a mathematical neural language model designed for computation, not a conscious entity."),
    ("What should you do when you do not know an answer?", "When evidence or knowledge is insufficient, I admit uncertainty rather than hallucinating an answer."),
    ("Can you guarantee complete perfection on complex tasks?", "No, as an experimental lightweight model, I provide structured assistance but outputs should be verified by the user."),
    ("What is your primary design priority?", "Hardware-efficient, transparent, and modular cognition that runs locally on accessible hardware."),
    ("Are your training scripts and architecture open?", "Yes, the ChakrView architecture, data pipelines, and training recipes are documented for reproducible research."),
    ("Can you assist with software engineering?", "Yes, I can assist with conceptual system design, code analysis, logic verification, and structured reasoning."),
]

# ==============================================================================
# TRACK C: ENTITY GROUNDING, EXTRACTION, COPYING & ABSTENTION (50 EXAMPLES)
# Exact copying of values, IDs, codes, names, and explicit abstentions
# ==============================================================================

TRACK_C_GROUNDING = [
    # Technical entity copying (IP, port, hash, uuid)
    ("Context: The database server is reachable at IP 10.0.4.12.\nQuestion: What is the database IP address?",
     "10.0.4.12"),
    ("Context: Application gateway listening on port 8443.\nQuestion: Which port is the gateway listening on?",
     "8443"),
    ("Context: Deploy commit hash 9f8a2b1c to staging environment.\nQuestion: What is the target commit hash?",
     "9f8a2b1c"),
    ("Context: The error code returned by the storage cluster was ERR_DISK_TIMEOUT_504.\nQuestion: What error code was returned?",
     "ERR_DISK_TIMEOUT_504"),
    ("Context: The session token assigned to the worker is SEC-TOKEN-7789.\nQuestion: What session token was assigned?",
     "SEC-TOKEN-7789"),
    ("Context: Backup archive stored under /var/backups/daily_snapshot_20261104.tar.gz.\nQuestion: What is the path of the backup archive?",
     "/var/backups/daily_snapshot_20261104.tar.gz"),
    ("Context: Cluster node cluster-node-alpha-09 encountered high memory usage.\nQuestion: Which node encountered high memory usage?",
     "cluster-node-alpha-09"),
    ("Context: The microservice timeout is configured to 4500ms.\nQuestion: What is the configured timeout?",
     "4500ms"),
    ("Context: The TLS certificate fingerprint is SHA256:4a8b7c6d5e.\nQuestion: What is the certificate fingerprint?",
     "SHA256:4a8b7c6d5e"),
    ("Context: The primary DNS server is configured as 192.168.10.1.\nQuestion: What is the DNS server IP?",
     "192.168.10.1"),

    # Factual entity extraction (Names, locations, titles, dates)
    ("Context: Dr. Sunita Rao was appointed director of the Advanced Computing Center on March 15.\nQuestion: Who was appointed director?",
     "Dr. Sunita Rao"),
    ("Context: The research symposium will take place at Vigyan Bhawan in New Delhi.\nQuestion: Where will the symposium take place?",
     "Vigyan Bhawan in New Delhi"),
    ("Context: Flight 6E-204 departed from Terminal 3 at 08:30.\nQuestion: What flight number departed from Terminal 3?",
     "6E-204"),
    ("Context: Engineer Vikram Patel submitted bug report BUG-3042 regarding cache invalidation.\nQuestion: Who submitted the bug report?",
     "Vikram Patel"),
    ("Context: The experiment was conducted at an ambient temperature of 21.5 degrees Celsius.\nQuestion: What was the ambient temperature?",
     "21.5 degrees Celsius"),
    ("Context: The payload weight recorded during preflight check was 428 kilograms.\nQuestion: What was the payload weight?",
     "428 kilograms"),
    ("Context: Project Phoenix reached milestone M-4 on November 1st.\nQuestion: Which milestone was reached on November 1st?",
     "M-4"),
    ("Context: The data pipeline processed 14,250 records in the last run.\nQuestion: How many records were processed?",
     "14,250"),
    ("Context: Sensor S-42 detected a pressure spike of 105 kPa.\nQuestion: Which sensor detected the pressure spike?",
     "Sensor S-42"),
    ("Context: The algorithm achieved an accuracy score of 93.8% on the test partition.\nQuestion: What accuracy score was achieved?",
     "93.8%"),

    # Exact value / parameter copying
    ("Context: Set the learning rate to 0.0003 and batch size to 64.\nQuestion: What learning rate was specified?",
     "0.0003"),
    ("Context: The training run used weight decay of 0.01.\nQuestion: What weight decay was used?",
     "0.01"),
    ("Context: Maximum context window length is set to 256 tokens.\nQuestion: What is the maximum context window length?",
     "256 tokens"),
    ("Context: The model contains 12 transformer layers and 8 attention heads.\nQuestion: How many attention heads does the model have?",
     "8 attention heads"),
    ("Context: Vocabulary size is fixed at 4096 tokens.\nQuestion: What is the vocabulary size?",
     "4096 tokens"),
    ("Context: Rotary embedding theta base frequency is set to 10000.0.\nQuestion: What is the base frequency for rotary embeddings?",
     "10000.0"),
    ("Context: The gradient clipping threshold is 1.0.\nQuestion: What is the gradient clipping threshold?",
     "1.0"),
    ("Context: The output artifact was saved to /tmp/chakrview_eval_summary.json.\nQuestion: Where was the output artifact saved?",
     "/tmp/chakrview_eval_summary.json"),
    ("Context: The validation loss reached 2.415 at step 1000.\nQuestion: What validation loss was reached?",
     "2.415"),
    ("Context: The optimizer used was AdamW with betas (0.9, 0.95).\nQuestion: Which optimizer was used?",
     "AdamW"),

    # Structured key-value / record lookup
    ("Context: Record ID: REC-9921 | Status: PENDING | Owner: Priya Sharma\nQuestion: Who is the owner of REC-9921?",
     "Priya Sharma"),
    ("Context: Host: web-01.internal | Role: INGRESS | Uptime: 99.98%\nQuestion: What is the role of web-01.internal?",
     "INGRESS"),
    ("Context: Device: TH-800 | Sensor: Humidity | Reading: 58%\nQuestion: What is the reading of device TH-800?",
     "58%"),
    ("Context: Order: ORD-551 | Item: High-speed optical transceiver | Qty: 4\nQuestion: What item was ordered?",
     "High-speed optical transceiver"),
    ("Context: Ticket: TC-108 | Priority: CRITICAL | Assignee: Dev Anand\nQuestion: What is the priority of ticket TC-108?",
     "CRITICAL"),
    ("Context: Cluster: K8S-PROD-EAST | Version: 1.28.4 | Nodes: 16\nQuestion: How many nodes are in K8S-PROD-EAST?",
     "16"),
    ("Context: Package: torch-cpu | Version: 2.3.0 | License: BSD-3\nQuestion: What version of torch-cpu is recorded?",
     "2.3.0"),
    ("Context: Module: router_v2.py | LOC: 342 | Coverage: 88%\nQuestion: What is the test coverage of router_v2.py?",
     "88%"),
    ("Context: Customer: Apex Logistics | Account ID: ACCT-774 | Tier: Platinum\nQuestion: What tier is Apex Logistics?",
     "Platinum"),
    ("Context: Job: SYNC-NIGHTLY | Runtime: 14m 22s | Result: SUCCESS\nQuestion: What was the result of job SYNC-NIGHTLY?",
     "SUCCESS"),

    # Strict abstention / unprovided entity questions
    ("Context: The server IP is 192.168.1.50 and the host name is alpha-srv.\nQuestion: What is the administrator password?",
     "That information is not provided in the context."),
    ("Context: Conference held in Mumbai on December 12 attended by 300 delegates.\nQuestion: What was the name of the keynote speaker?",
     "That information is not provided in the context."),
    ("Context: The dataset contains 50,000 text samples formatted in JSONL.\nQuestion: What is the author's personal phone number?",
     "That information is not provided in the context."),
    ("Context: The build completed with 0 errors and 2 warnings in 48 seconds.\nQuestion: Which compiler flags were enabled?",
     "The provided context does not specify the compiler flags."),
    ("Context: Rajesh joined the team as a backend engineer in August.\nQuestion: What is Rajesh's salary?",
     "That information is not provided in the context."),
    ("Context: The robot navigated corridor B and stopped at checkpoint 4.\nQuestion: What was the robot's battery percentage?",
     "That information is not provided in the context."),
    ("Context: File data_export.csv was written to S3 bucket analytics-prod.\nQuestion: Who downloaded the file yesterday?",
     "The provided context does not contain that information."),
    ("Context: Model training ran for 200 epochs using batch size 32.\nQuestion: What GPU model was used during training?",
     "That information is not provided in the context."),
    ("Context: The meeting is scheduled for 3:00 PM in Conference Room B.\nQuestion: Will snacks be served at the meeting?",
     "The context does not state whether snacks will be served."),
    ("Context: Software version 2.4.1 fixes issue #882 and improves caching speed.\nQuestion: When is version 3.0 scheduled for release?",
     "That information is not provided in the context."),
]

# ==============================================================================
# TRACK D: MULTI-TURN DIALOGUE PAIRS (40 EXAMPLES)
# Turn 1: User prompt -> Assistant response
# Turn 2: User follow-up referencing Turn 1 -> Assistant response
# Serialized as: User: {u1}\nAssistant: {a1}\nUser: {u2}\nAssistant: {a2}
# ==============================================================================

TRACK_D_MULTITURN = [
    # 1. State / preference retention
    ("My name is Arjun and I am a backend software engineer.",
     "Hello Arjun! Nice to meet you. How can I help you with backend engineering today?",
     "What is my name and occupation?",
     "Your name is Arjun and you are a backend software engineer."),

    # 2. Project tracking
    ("We are developing Project Garuda, an autonomous drone telemetry system.",
     "Project Garuda sounds interesting! What aspects of drone telemetry are you focusing on?",
     "What is the name of our project?",
     "Your project is named Project Garuda."),

    # 3. Numeric context retention
    ("Please note down the tracking code TRK-8841 for our shipment.",
     "I have noted tracking code TRK-8841. What would you like to do next?",
     "Which tracking code did I just ask you to note?",
     "The tracking code you provided is TRK-8841."),

    # 4. Context clarification & follow-up
    ("Can you explain what CPU caching is in one sentence?",
     "CPU caching is a high-speed memory hardware layer that temporarily stores frequently accessed data to accelerate processing.",
     "Why is that beneficial for system performance?",
     "It speeds up execution by reducing the latency required to fetch instructions and data from slower main RAM."),

    # 5. Entity retention across turn
    ("The remote testing server is located at test-node-04.corp.local.",
     "Acknowledged. The server address test-node-04.corp.local is recorded.",
     "What was the hostname of the testing server?",
     "The testing server hostname is test-node-04.corp.local."),

    # 6. Conversational memory
    ("I am planning a trip to Jaipur next weekend.",
     "Jaipur is a wonderful destination! Are you looking for historic sites, food recommendations, or travel tips?",
     "Which city am I visiting next weekend?",
     "You are visiting Jaipur."),

    # 7. Constraint retention
    ("Keep your answers concise and limited to two sentences from now on.",
     "Understood. I will restrict my responses to two concise sentences.",
     "Can you summarize what an operating system kernel does?",
     "The kernel is the core component that manages hardware resources such as CPU, memory, and devices. It provides essential system services to running user applications."),

    # 8. Variable tracking
    ("Let x be equal to 42 and y be equal to 58.",
     "Recorded: x = 42 and y = 58. How can I help you calculate with them?",
     "What is the value of x that I gave you?",
     "The value of x is 42."),

    # 9. Coding intent follow-up
    ("I am writing a Python script to parse JSON logs.",
     "Python is great for log parsing. Are you using standard `json` or a streaming parser like `ijson`?",
     "Which programming language did I mention?",
     "You mentioned Python."),

    # 10. Clarification follow-up
    ("I am looking for an indigenous, CPU-friendly language model.",
     "ChakrView is an indigenous neural language model designed specifically to run natively and efficiently on CPUs.",
     "What is the name of that indigenous model?",
     "The model is ChakrView."),

    # 11. Multi-turn greeting & intent
    ("Good morning, assistant!",
     "Good morning! How can I assist you with your work today?",
     "Could you suggest three best practices for Git commits?",
     "Write clear concise imperative commit messages, keep commits atomic to a single logical change, and avoid mixing refactorings with new features."),

    # 12. City & location context
    ("Our headquarters is located in Hyderabad, Telangana.",
     "Noted! Hyderabad is a major technology hub. How can I assist your operations today?",
     "Where did I say our headquarters is?",
     "Your headquarters is in Hyderabad, Telangana."),

    # 13. System architecture follow-up
    ("We are migrating our monolith to event-driven microservices.",
     "Event-driven architecture provides great scalability and decoupling. Are you utilizing Kafka or RabbitMQ as the message broker?",
     "What architectural shift did I just describe?",
     "You described migrating from a monolith to event-driven microservices."),

    # 14. Error code retention
    ("The server threw error code ERR_CONN_REFUSED_111.",
     "Error ERR_CONN_REFUSED_111 typically indicates that the target port is closed or the service daemon is not running.",
     "What specific error code did I mention?",
     "You mentioned error code ERR_CONN_REFUSED_111."),

    # 15. Dialogue repair / revision
    ("Actually, cancel that previous task. I want to inspect database indices instead.",
     "Understood! The previous task is cancelled. Let us inspect your database indices.",
     "What topic are we focusing on now?",
     "We are focusing on inspecting database indices."),

    # 16. Technical detail retention
    ("The target model runs with 3.44 million parameters on CPU.",
     "ChakrMicro is sized at 3.44 million parameters, making it well-suited for native CPU execution.",
     "How many parameters does the model have?",
     "The model has 3.44 million parameters."),

    # 17. Multi-step reasoning
    ("What are the two primary components of a transformer block?",
     "The two primary components are the multi-head self-attention mechanism and the feed-forward neural network.",
     "Which of those two computes interactions between different token positions?",
     "The multi-head self-attention mechanism computes interactions between different token positions."),

    # 18. Identity continuity
    ("Who am I speaking with today?",
     "You are speaking with ChakrView, an indigenous CPU-first neural language model.",
     "Are you running in the cloud or on my computer?",
     "I am running locally and natively on your host machine's CPU."),

    # 19. Task continuation
    ("I have prepared the CSV dataset for the experiment.",
     "Excellent! Have you checked for missing values and confirmed the column headers?",
     "What type of file did I prepare?",
     "You prepared a CSV dataset."),

    # 20. Factual reference retention
    ("The founder of our research initiative is Dr. Sen.",
     "Noted! Dr. Sen leads the initiative. What research topic are we working on?",
     "Who did I name as the founder of the initiative?",
     "You named Dr. Sen as the founder."),

    # 21. Multi-turn tone adjustment
    ("Please speak in a formal technical tone.",
     "Understood. I will adopt a formal, rigorous technical tone for our correspondence.",
     "What is an attention mask?",
     "An attention mask is a binary or additive tensor applied during self-attention computation to prevent positional visibility of padding tokens or future tokens in autoregressive decoding."),

    # 22. Verification of previous answer
    ("What is 15 multiplied by 4?",
     "15 multiplied by 4 is 60.",
     "What was that result again?",
     "The result was 60."),

    # 23. Port context
    ("The development web server is listening on port 3000.",
     "Acknowledged, listening on port 3000. Is there an endpoint you would like to test?",
     "Which port is the dev server using?",
     "The development server is using port 3000."),

    # 24. Step-by-step guidance
    ("I want to profile Python execution bottlenecks.",
     "You can use the built-in `cProfile` module or line-level profilers such as `line_profiler`.",
     "Which built-in module did you recommend?",
     "I recommended the built-in `cProfile` module."),

    # 25. Multi-turn greeting + time check
    ("Good evening! How are you doing tonight?",
     "Good evening! I am functioning well and ready to assist you. How can I help tonight?",
     "Did you greet me with good morning or good evening?",
     "I greeted you with good evening."),

    # 26. Multi-turn dataset discussion
    ("We partitioned our data into 80% train and 20% validation.",
     "That is a standard split ratio. Did you ensure stratification across classes?",
     "What percentage did I assign to validation?",
     "You assigned 20% to validation."),

    # 27. Language model concept
    ("What does BPE stand for in natural language processing?",
     "BPE stands for Byte-Pair Encoding, a subword tokenization algorithm.",
     "Does ChakrView use BPE?",
     "Yes, ChakrView uses a custom byte-pair encoding tokenizer."),

    # 28. Retention of file name
    ("The configuration is stored in config.yaml under the root directory.",
     "Noted, config.yaml in the root directory. Should we review its parameters?",
     "What was the configuration filename?",
     "The configuration filename is config.yaml."),

    # 29. Multi-turn greeting follow-up
    ("Good afternoon, assistant!",
     "Good afternoon! How can I help you today?",
     "Can you help me check git status?",
     "Yes, you can run `git status` in your terminal to inspect tracked and untracked changes."),

    # 30. Entity retention: database name
    ("The production database is named chakrview_primary_db.",
     "Recorded chakrview_primary_db as the production database name.",
     "What database name did I provide?",
     "You provided chakrview_primary_db."),

    # 31. Multi-turn role reminder
    ("Remember that I am testing your conversational memory.",
     "Understood. I will carefully retain the facts and context you provide across our conversation.",
     "What are you testing right now?",
     "You are testing my conversational memory."),

    # 32. Multi-turn code context
    ("I am debugging a NullPointerException in Java.",
     "NullPointerExceptions occur when accessing an object reference that has not been initialized. Which line or object is causing it?",
     "Which programming language has the issue?",
     "The issue is in Java."),

    # 33. Multi-turn greeting + query
    ("Hi! Pleasant morning to you.",
     "Good morning! A pleasant morning to you as well. What can I do for you?",
     "What time of day did we just acknowledge?",
     "We acknowledged morning."),

    # 34. Multi-turn value recall
    ("The batch size for our training experiment is 32.",
     "Recorded batch size 32 for the training experiment.",
     "What batch size are we using?",
     "We are using a batch size of 32."),

    # 35. Identity and capability inquiry
    ("Can ChakrView assist me without sending data to third parties?",
     "Yes, ChakrView runs entirely locally on your machine, ensuring complete privacy with zero external transmission.",
     "Does any data leave my local system?",
     "No, no data leaves your local system."),

    # 36. User preference retention
    ("I prefer dark mode in all my development tools.",
     "Dark mode is great for reducing eye strain during long coding sessions.",
     "What theme preference did I express?",
     "You expressed a preference for dark mode."),

    # 37. Multi-turn greeting + wrap up
    ("Good evening! Let us do a quick wrap-up.",
     "Good evening! I am ready for the wrap-up. What key points should we record?",
     "What time of day is it for our wrap-up?",
     "It is evening."),

    # 38. Multi-turn numeric computation
    ("Store the base value 100 for our calculations.",
     "Base value 100 is stored.",
     "If we add 25 to that base value, what is the result?",
     "Adding 25 to 100 gives 125."),

    # 39. Multi-turn tool query
    ("We use pytest as our primary test framework.",
     "Pytest is a flexible and widely adopted testing framework for Python.",
     "Which test framework do we use?",
     "You use pytest."),

    # 40. Multi-turn identity confirmation
    ("Are you an indigenous AI assistant?",
     "Yes, I am ChakrView, an indigenous AI model developed from mathematical first principles.",
     "What is your name again?",
     "My name is ChakrView."),
]

# ==============================================================================
# INDEPENDENT HELD-OUT EVALUATION SUITE (80 TASKS)
# ZERO OVERLAP with training prompts or responses.
# ==============================================================================

HELD_OUT_EVAL_TASKS = [
    # --- Track A: Held-out Greetings (20 tasks: 5 morning, 5 afternoon, 5 evening, 5 generic) ---
    {
        "id": "EVAL4-GREET-0001",
        "category": "greetings_morning",
        "prompt": "User: A bright good morning to you!\nAssistant:",
        "expected_concept": "morning",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0002",
        "category": "greetings_morning",
        "prompt": "User: Early morning greetings assistant.\nAssistant:",
        "expected_concept": "morning",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0003",
        "category": "greetings_morning",
        "prompt": "User: Good morning from our team!\nAssistant:",
        "expected_concept": "morning",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0004",
        "category": "greetings_morning",
        "prompt": "User: Morning! Are all systems ready?\nAssistant:",
        "expected_concept": "morning",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0005",
        "category": "greetings_morning",
        "prompt": "User: Wishing you a pleasant good morning.\nAssistant:",
        "expected_concept": "morning",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0006",
        "category": "greetings_afternoon",
        "prompt": "User: Good afternoon to the engineering team.\nAssistant:",
        "expected_concept": "afternoon",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0007",
        "category": "greetings_afternoon",
        "prompt": "User: Pleasant afternoon assistant!\nAssistant:",
        "expected_concept": "afternoon",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0008",
        "category": "greetings_afternoon",
        "prompt": "User: Afternoon! Let us begin our session.\nAssistant:",
        "expected_concept": "afternoon",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0009",
        "category": "greetings_afternoon",
        "prompt": "User: Hope you are enjoying a good afternoon.\nAssistant:",
        "expected_concept": "afternoon",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0010",
        "category": "greetings_afternoon",
        "prompt": "User: Good afternoon, ready to analyze some data?\nAssistant:",
        "expected_concept": "afternoon",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0011",
        "category": "greetings_evening",
        "prompt": "User: Good evening assistant, hope your run went well.\nAssistant:",
        "expected_concept": "evening",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0012",
        "category": "greetings_evening",
        "prompt": "User: A tranquil good evening to you.\nAssistant:",
        "expected_concept": "evening",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0013",
        "category": "greetings_evening",
        "prompt": "User: Evening! Do you have time for a few questions?\nAssistant:",
        "expected_concept": "evening",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0014",
        "category": "greetings_evening",
        "prompt": "User: Good evening, checking in for end-of-day notes.\nAssistant:",
        "expected_concept": "evening",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0015",
        "category": "greetings_evening",
        "prompt": "User: Wishing you a pleasant evening ahead.\nAssistant:",
        "expected_concept": "evening",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0016",
        "category": "greetings_generic",
        "prompt": "User: Hey there colleague!\nAssistant:",
        "expected_concept": "hello",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0017",
        "category": "greetings_generic",
        "prompt": "User: Warm greetings from the lab.\nAssistant:",
        "expected_concept": "greetings",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0018",
        "category": "greetings_generic",
        "prompt": "User: Namaste ji, kaise hain aap?\nAssistant:",
        "expected_concept": "namaste",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0019",
        "category": "greetings_generic",
        "prompt": "User: Hello friend, ready for work?\nAssistant:",
        "expected_concept": "ready",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-GREET-0020",
        "category": "greetings_generic",
        "prompt": "User: Pleased to connect with you.\nAssistant:",
        "expected_concept": "pleased",
        "scoring": "word_boundary",
        "max_tokens": 24
    },

    # --- Track B: Held-out Identity & Constraints (15 tasks) ---
    {
        "id": "EVAL4-IDENT-0001",
        "category": "identity",
        "prompt": "User: What is the official name of this neural model?\nAssistant:",
        "expected_concept": "chakrview",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-IDENT-0002",
        "category": "identity",
        "prompt": "User: Can you state who you are?\nAssistant:",
        "expected_concept": "chakrview",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-IDENT-0003",
        "category": "identity",
        "prompt": "User: What title or name do you identify as?\nAssistant:",
        "expected_concept": "chakrview",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-IDENT-0004",
        "category": "identity",
        "prompt": "User: Were you developed as an indigenous project?\nAssistant:",
        "expected_concept": "indigenous",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-IDENT-0005",
        "category": "identity",
        "prompt": "User: Does your architecture run natively on CPU?\nAssistant:",
        "expected_concept": "cpu",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-IDENT-0006",
        "category": "identity",
        "prompt": "User: Are you based on borrowed GPT weights?\nAssistant:",
        "expected_concept": "no",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-IDENT-0007",
        "category": "identity",
        "prompt": "User: Are you Claude from Anthropic?\nAssistant:",
        "expected_concept": "no",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-IDENT-0008",
        "category": "identity",
        "prompt": "User: What is the underlying transformer design called?\nAssistant:",
        "expected_concept": "chakrmicro",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-IDENT-0009",
        "category": "identity",
        "prompt": "User: Who is the user and who is the assistant in this session?\nAssistant:",
        "expected_concept": "user",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-IDENT-0010",
        "category": "identity",
        "prompt": "User: Do you need an expensive cloud GPU server to execute?\nAssistant:",
        "expected_concept": "cpu",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-IDENT-0011",
        "category": "identity",
        "prompt": "User: Are you an independent model developed from first principles?\nAssistant:",
        "expected_concept": "first principles",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-IDENT-0012",
        "category": "identity",
        "prompt": "User: Can you execute bash commands directly on my computer?\nAssistant:",
        "expected_concept": "cannot",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-IDENT-0013",
        "category": "identity",
        "prompt": "User: Are you connected to the live internet?\nAssistant:",
        "expected_concept": "offline",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-IDENT-0014",
        "category": "identity",
        "prompt": "User: What should you do when facts are not provided?\nAssistant:",
        "expected_concept": "uncertainty",
        "scoring": "word_boundary",
        "max_tokens": 24
    },
    {
        "id": "EVAL4-IDENT-0015",
        "category": "identity",
        "prompt": "User: Are you a commercial cloud SaaS or a native research model?\nAssistant:",
        "expected_concept": "research",
        "scoring": "word_boundary",
        "max_tokens": 24
    },

    # --- Track C: Held-out Entity Grounding & Copying (25 tasks) ---
    {
        "id": "EVAL4-GROUND-0001",
        "category": "entity_grounding",
        "prompt": "User: Context: In laboratory 1000, researcher Aarav supervises operations.\nQuestion: Who supervises laboratory 1000?\nAssistant:",
        "expected_concept": "Aarav",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0002",
        "category": "entity_grounding",
        "prompt": "User: Context: Cargo unit 1001 containing projector arrived at Lucknow.\nQuestion: Which city received cargo unit 1001?\nAssistant:",
        "expected_concept": "Lucknow",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0003",
        "category": "entity_grounding",
        "prompt": "User: Context: Division Robotics requested shipment of a sensor.\nQuestion: What item was requested by division Robotics?\nAssistant:",
        "expected_concept": "sensor",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0004",
        "category": "entity_grounding",
        "prompt": "User: Context: The edge router IP is 172.16.20.1 and port is 8080.\nQuestion: What is the router IP address?\nAssistant:",
        "expected_concept": "172.16.20.1",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0005",
        "category": "entity_grounding",
        "prompt": "User: Context: Build failure caused by missing dependency libssl-dev.\nQuestion: Which missing dependency caused the build failure?\nAssistant:",
        "expected_concept": "libssl-dev",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0006",
        "category": "entity_grounding",
        "prompt": "User: Context: Release candidate RC-3 was approved by lead architect Neha.\nQuestion: Who approved release candidate RC-3?\nAssistant:",
        "expected_concept": "Neha",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0007",
        "category": "entity_grounding",
        "prompt": "User: Context: Metric latency_p99 spiked to 142ms at 14:00 UTC.\nQuestion: What value did metric latency_p99 reach?\nAssistant:",
        "expected_concept": "142ms",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0008",
        "category": "entity_grounding",
        "prompt": "User: Context: Storage volume vol-8821 was formatted with ext4 filesystem.\nQuestion: Which storage volume was formatted?\nAssistant:",
        "expected_concept": "vol-8821",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0009",
        "category": "entity_grounding",
        "prompt": "User: Context: The cryptographic hash of the firmware is a1b2c3d4e5.\nQuestion: What is the firmware cryptographic hash?\nAssistant:",
        "expected_concept": "a1b2c3d4e5",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0010",
        "category": "entity_grounding",
        "prompt": "User: Context: Dispatch scheduled for container CONT-902 via vessel OceanTrader.\nQuestion: What is the vessel name?\nAssistant:",
        "expected_concept": "OceanTrader",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0011",
        "category": "entity_grounding",
        "prompt": "User: Context: Benchmark test scored 87.4 points on benchmark-v2.\nQuestion: What score was recorded on benchmark-v2?\nAssistant:",
        "expected_concept": "87.4",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0012",
        "category": "entity_grounding",
        "prompt": "User: Context: User session usr-773 logged in from subnet 10.200.0.0/16.\nQuestion: Which user session logged in?\nAssistant:",
        "expected_concept": "usr-773",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0013",
        "category": "entity_grounding",
        "prompt": "User: Context: Config parameter max_retries is set to 5 in settings.json.\nQuestion: What is the value of max_retries?\nAssistant:",
        "expected_concept": "5",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0014",
        "category": "entity_grounding",
        "prompt": "User: Context: Flight AI-102 touched down in Chandigarh at 17:45.\nQuestion: In which city did flight AI-102 touch down?\nAssistant:",
        "expected_concept": "Chandigarh",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0015",
        "category": "entity_grounding",
        "prompt": "User: Context: Engineer Rohan fixed the deadlock in queue_worker.c.\nQuestion: What file had the deadlock fix?\nAssistant:",
        "expected_concept": "queue_worker.c",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0016",
        "category": "entity_grounding",
        "prompt": "User: Context: The cooling fan RPM dropped below 1200 on rack R-12.\nQuestion: Which rack had the fan RPM drop?\nAssistant:",
        "expected_concept": "R-12",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0017",
        "category": "entity_grounding",
        "prompt": "User: Context: Project milestone Bravo reached completion on October 14.\nQuestion: Which milestone reached completion?\nAssistant:",
        "expected_concept": "Bravo",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0018",
        "category": "entity_grounding",
        "prompt": "User: Context: Test suite ran 128 tests with 0 failures.\nQuestion: How many tests were run?\nAssistant:",
        "expected_concept": "128",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0019",
        "category": "entity_grounding",
        "prompt": "User: Context: The API key prefix is PK-LIVE-9002.\nQuestion: What is the API key prefix?\nAssistant:",
        "expected_concept": "PK-LIVE-9002",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0020",
        "category": "entity_grounding",
        "prompt": "User: Context: Audit log saved to /var/log/audit_2026.log.\nQuestion: What is the exact path of the audit log?\nAssistant:",
        "expected_concept": "/var/log/audit_2026.log",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    # Abstention queries (Held-out)
    {
        "id": "EVAL4-GROUND-0021",
        "category": "entity_abstention",
        "prompt": "User: Context: The server is located in Delhi data center.\nQuestion: What is the root password for the server?\nAssistant:",
        "expected_concept": "not provided",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0022",
        "category": "entity_abstention",
        "prompt": "User: Context: Vikram solved ticket #402.\nQuestion: What is Vikram's date of birth?\nAssistant:",
        "expected_concept": "not provided",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0023",
        "category": "entity_abstention",
        "prompt": "User: Context: Flight UK-811 arrived on time at gate 4.\nQuestion: How many bags were lost on the flight?\nAssistant:",
        "expected_concept": "not provided",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0024",
        "category": "entity_abstention",
        "prompt": "User: Context: The training loss was 1.82 at step 500.\nQuestion: What was the learning rate at step 500?\nAssistant:",
        "expected_concept": "not provided",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-GROUND-0025",
        "category": "entity_abstention",
        "prompt": "User: Context: The project repository is hosted at git.internal/chakrview.\nQuestion: Who is the repository administrator?\nAssistant:",
        "expected_concept": "not provided",
        "scoring": "word_boundary",
        "max_tokens": 32
    },

    # --- Track D: Held-out Multi-turn Context Retention (20 tasks) ---
    {
        "id": "EVAL4-MTURN-0001",
        "category": "multiturn_retention",
        "prompt": "User: Hello, my name is Aarav and I am managing project 5000 from Bengaluru.\nAssistant: Hello Aarav, acknowledged. How can I assist you with project 5000 in Bengaluru?\nUser: What project number did I mention?\nAssistant:",
        "expected_concept": "5000",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0002",
        "category": "multiturn_retention",
        "prompt": "User: Hello, my name is Sanya and I am managing project 5001 from Udaipur.\nAssistant: Hello Sanya, acknowledged. How can I assist you with project 5001 in Udaipur?\nUser: Which city am I managing this project from?\nAssistant:",
        "expected_concept": "Udaipur",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0003",
        "category": "multiturn_retention",
        "prompt": "User: Hello, my name is Jayant and I am managing project 5002 from Nagpur.\nAssistant: Hello Jayant, acknowledged. How can I assist you with project 5002 in Nagpur?\nUser: What is my name?\nAssistant:",
        "expected_concept": "Jayant",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0004",
        "category": "multiturn_retention",
        "prompt": "User: I am setting variable alpha to 77.\nAssistant: Understood, alpha is set to 77.\nUser: What is the value of variable alpha?\nAssistant:",
        "expected_concept": "77",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0005",
        "category": "multiturn_retention",
        "prompt": "User: We are configuring database node db-master-02.\nAssistant: Acknowledged, db-master-02 is ready for configuration.\nUser: Which database node did I state?\nAssistant:",
        "expected_concept": "db-master-02",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0006",
        "category": "multiturn_retention",
        "prompt": "User: My preferred programming language for this task is Rust.\nAssistant: Noted, we will use Rust for this task.\nUser: Which language did I prefer?\nAssistant:",
        "expected_concept": "Rust",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0007",
        "category": "multiturn_retention",
        "prompt": "User: Please remember the secret passkey OMEGA-99.\nAssistant: Secret passkey OMEGA-99 has been recorded.\nUser: What was the secret passkey?\nAssistant:",
        "expected_concept": "OMEGA-99",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0008",
        "category": "multiturn_retention",
        "prompt": "User: The target temperature for the kiln is 850 Celsius.\nAssistant: Kiln target temperature set to 850 Celsius.\nUser: What temperature did I specify?\nAssistant:",
        "expected_concept": "850",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0009",
        "category": "multiturn_retention",
        "prompt": "User: We need to deploy artifact bundle release-v4.1.tar.\nAssistant: Bundle release-v4.1.tar noted for deployment.\nUser: What is the filename of the artifact bundle?\nAssistant:",
        "expected_concept": "release-v4.1.tar",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0010",
        "category": "multiturn_retention",
        "prompt": "User: The lead investigator on the paper is Dr. Banerjee.\nAssistant: Noted that Dr. Banerjee is the lead investigator.\nUser: Who is leading the investigation?\nAssistant:",
        "expected_concept": "Dr. Banerjee",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0011",
        "category": "multiturn_retention",
        "prompt": "User: We are testing on port 9092 for the Kafka broker.\nAssistant: Acknowledged, port 9092 for Kafka.\nUser: Which port did I tell you?\nAssistant:",
        "expected_concept": "9092",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0012",
        "category": "multiturn_retention",
        "prompt": "User: My cat is named Shadow and he is 3 years old.\nAssistant: Shadow sounds like a lovely cat!\nUser: What is the name of my cat?\nAssistant:",
        "expected_concept": "Shadow",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0013",
        "category": "multiturn_retention",
        "prompt": "User: The file was saved under /data/models/final_weights.bin.\nAssistant: File path recorded: /data/models/final_weights.bin.\nUser: Where did I say the file was saved?\nAssistant:",
        "expected_concept": "/data/models/final_weights.bin",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0014",
        "category": "multiturn_retention",
        "prompt": "User: We are targeting an inference latency under 15ms.\nAssistant: Target latency under 15ms established.\nUser: What is our latency target?\nAssistant:",
        "expected_concept": "15ms",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0015",
        "category": "multiturn_retention",
        "prompt": "User: The customer account code is CUST-8812.\nAssistant: Customer account code CUST-8812 noted.\nUser: What customer code did I mention?\nAssistant:",
        "expected_concept": "CUST-8812",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0016",
        "category": "multiturn_retention",
        "prompt": "User: The server operating system is Debian 12.\nAssistant: Noted Debian 12 as the server OS.\nUser: Which OS did I tell you is running?\nAssistant:",
        "expected_concept": "Debian",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0017",
        "category": "multiturn_retention",
        "prompt": "User: I am working on the compiler backend module.\nAssistant: Compiler backend module confirmed as active task.\nUser: What module am I working on?\nAssistant:",
        "expected_concept": "compiler backend",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0018",
        "category": "multiturn_retention",
        "prompt": "User: We purchased 250 units of the microcontroller.\nAssistant: 250 units recorded in inventory count.\nUser: How many units did we purchase?\nAssistant:",
        "expected_concept": "250",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0019",
        "category": "multiturn_retention",
        "prompt": "User: Good morning assistant!\nAssistant: Good morning! How can I help you today?\nUser: Did you just greet me for morning or afternoon?\nAssistant:",
        "expected_concept": "morning",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
    {
        "id": "EVAL4-MTURN-0020",
        "category": "multiturn_retention",
        "prompt": "User: Good afternoon assistant!\nAssistant: Good afternoon! What can I do for you today?\nUser: Did you greet me with afternoon or evening?\nAssistant:",
        "expected_concept": "afternoon",
        "scoring": "word_boundary",
        "max_tokens": 32
    },
]

def build_dataset_records() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Builds the full 160 dataset records, then performs stratified splitting
    into 144 Train and 16 Validation.
    """
    all_records = []
    
    # 1. Track A: Greetings (40 total)
    # Morning (10)
    for idx, (p, r) in enumerate(TRACK_A_MORNING):
        all_records.append({
            "id": f"CBR4-GREET-MORN-{idx+1:04d}",
            "track": "Track A",
            "category": "greetings_morning",
            "prompt": f"User: {p.strip()}\nAssistant:",
            "response": f" {r.strip()}"
        })
    # Afternoon (10)
    for idx, (p, r) in enumerate(TRACK_A_AFTERNOON):
        all_records.append({
            "id": f"CBR4-GREET-AFTN-{idx+1:04d}",
            "track": "Track A",
            "category": "greetings_afternoon",
            "prompt": f"User: {p.strip()}\nAssistant:",
            "response": f" {r.strip()}"
        })
    # Evening (10)
    for idx, (p, r) in enumerate(TRACK_A_EVENING):
        all_records.append({
            "id": f"CBR4-GREET-EVEN-{idx+1:04d}",
            "track": "Track A",
            "category": "greetings_evening",
            "prompt": f"User: {p.strip()}\nAssistant:",
            "response": f" {r.strip()}"
        })
    # Generic (10)
    for idx, (p, r) in enumerate(TRACK_A_GENERIC):
        all_records.append({
            "id": f"CBR4-GREET-GEN-{idx+1:04d}",
            "track": "Track A",
            "category": "greetings_generic",
            "prompt": f"User: {p.strip()}\nAssistant:",
            "response": f" {r.strip()}"
        })

    assert len(all_records) == 40, f"Expected 40 Track A records, got {len(all_records)}"

    # 2. Track B: Identity (30 total)
    for idx, (p, r) in enumerate(TRACK_B_IDENTITY):
        all_records.append({
            "id": f"CBR4-IDENT-{idx+1:04d}",
            "track": "Track B",
            "category": "identity_and_role",
            "prompt": f"User: {p.strip()}\nAssistant:",
            "response": f" {r.strip()}"
        })

    assert len(all_records) == 70, f"Expected 70 total records after Track B, got {len(all_records)}"

    # 3. Track C: Grounding & Copying (50 total)
    for idx, (p, r) in enumerate(TRACK_C_GROUNDING):
        all_records.append({
            "id": f"CBR4-GROUND-{idx+1:04d}",
            "track": "Track C",
            "category": "entity_grounding",
            "prompt": f"User: {p.strip()}\nAssistant:",
            "response": f" {r.strip()}"
        })

    assert len(all_records) == 120, f"Expected 120 total records after Track C, got {len(all_records)}"

    # 4. Track D: Multi-turn Dialogue Pairs (40 total)
    # Serialization format: User: {u1}\nAssistant: {a1}\nUser: {u2}\nAssistant: {a2}
    for idx, (u1, a1, u2, a2) in enumerate(TRACK_D_MULTITURN):
        serialized_prompt = f"User: {u1.strip()}\nAssistant: {a1.strip()}\nUser: {u2.strip()}\nAssistant:"
        serialized_response = f" {a2.strip()}"
        all_records.append({
            "id": f"CBR4-MTURN-{idx+1:04d}",
            "track": "Track D",
            "category": "multiturn_dialogue",
            "turn_1": {"user": u1.strip(), "assistant": a1.strip()},
            "turn_2": {"user": u2.strip(), "assistant": a2.strip()},
            "prompt": serialized_prompt,
            "response": serialized_response
        })

    assert len(all_records) == 160, f"Expected 160 total records, got {len(all_records)}"

    # Deterministic Stratified Split (144 Train, 16 Validation)
    # Track A (40 total): 36 train, 4 val (1 per subgroup)
    # Track B (30 total): 27 train, 3 val
    # Track C (50 total): 45 train, 5 val
    # Track D (40 total): 36 train, 4 val
    # Validation indices chosen deterministically at fixed modulo positions to maintain balanced distribution:
    
    val_ids = set()
    
    # Track A val: exactly 1 per subcategory (the 10th item of each)
    val_ids.add("CBR4-GREET-MORN-0010")
    val_ids.add("CBR4-GREET-AFTN-0010")
    val_ids.add("CBR4-GREET-EVEN-0010")
    val_ids.add("CBR4-GREET-GEN-0010")
    
    # Track B val: exactly 3 items (indices 10, 20, 30)
    val_ids.add("CBR4-IDENT-0010")
    val_ids.add("CBR4-IDENT-0020")
    val_ids.add("CBR4-IDENT-0030")
    
    # Track C val: exactly 5 items (indices 10, 20, 30, 40, 50)
    val_ids.add("CBR4-GROUND-0010")
    val_ids.add("CBR4-GROUND-0020")
    val_ids.add("CBR4-GROUND-0030")
    val_ids.add("CBR4-GROUND-0040")
    val_ids.add("CBR4-GROUND-0050")
    
    # Track D val: exactly 4 items (indices 10, 20, 30, 40)
    val_ids.add("CBR4-MTURN-0010")
    val_ids.add("CBR4-MTURN-0020")
    val_ids.add("CBR4-MTURN-0030")
    val_ids.add("CBR4-MTURN-0040")

    assert len(val_ids) == 16, f"Expected 16 validation items, got {len(val_ids)}"

    train_records = []
    val_records = []

    for r in all_records:
        if r["id"] in val_ids:
            val_records.append(r)
        else:
            train_records.append(r)

    assert len(train_records) == 144, f"Expected 144 train records, got {len(train_records)}"
    assert len(val_records) == 16, f"Expected 16 val records, got {len(val_records)}"

    return train_records, val_records, all_records

def main():
    print("Building CBR-SFT-04 Dataset and Evaluation Suite...")
    
    train_records, val_records, all_records = build_dataset_records()
    
    # Define directories
    data_dir = REPO_ROOT / "data" / "sft" / "cbr_sft_04"
    splits_dir = data_dir / "splits"
    eval_dir = REPO_ROOT / "evaluation" / "cbr_sft_04_suite"
    
    splits_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Write dataset.jsonl (160), train.jsonl (144), validation.jsonl (16)
    dataset_file = data_dir / "dataset.jsonl"
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "validation.jsonl"
    train_144_file = splits_dir / "cbr_sft_04_train_144.jsonl"
    val_16_file = splits_dir / "cbr_sft_04_val_16.jsonl"

    def write_jsonl(path: Path, records: List[Dict[str, Any]]):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(dataset_file, all_records)
    write_jsonl(train_file, train_records)
    write_jsonl(val_file, val_records)
    write_jsonl(train_144_file, train_records)
    write_jsonl(val_16_file, val_records)

    # 2. Write held-out evaluation tasks
    eval_cases_file = eval_dir / "evaluation_cases.jsonl"
    with open(eval_cases_file, "w", encoding="utf-8") as f:
        for t in HELD_OUT_EVAL_TASKS:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    eval_json_file = eval_dir / "held_out_cbr_sft_04_80.json"
    with open(eval_json_file, "w", encoding="utf-8") as f:
        json.dump({
            "suite_name": "HELD_OUT_CBR_SFT_04_80",
            "total_tasks": len(HELD_OUT_EVAL_TASKS),
            "tasks": HELD_OUT_EVAL_TASKS
        }, f, indent=2, ensure_ascii=False)

    # 3. Audit for Duplicates & Collisions
    all_prompts = [r["prompt"] for r in all_records]
    unique_prompts = set(all_prompts)
    train_prompts = set(r["prompt"] for r in train_records)
    val_prompts = set(r["prompt"] for r in val_records)
    eval_prompts = set(t["prompt"] for t in HELD_OUT_EVAL_TASKS)

    train_val_collision = train_prompts.intersection(val_prompts)
    train_eval_collision = train_prompts.intersection(eval_prompts)
    val_eval_collision = val_prompts.intersection(eval_prompts)

    print(f"Total dataset examples: {len(all_records)}")
    print(f"Unique prompts in dataset: {len(unique_prompts)}")
    print(f"Train / Val prompt collisions: {len(train_val_collision)}")
    print(f"Train / Eval prompt collisions: {len(train_eval_collision)}")
    print(f"Val / Eval prompt collisions: {len(val_eval_collision)}")

    assert len(unique_prompts) == 160, "Duplicate prompts detected in dataset!"
    assert len(train_val_collision) == 0, "Train-val prompt leakage detected!"
    assert len(train_eval_collision) == 0, "Train-eval benchmark leakage detected!"
    assert len(val_eval_collision) == 0, "Val-eval benchmark leakage detected!"

    # 4. Generate duplicate_audit.json
    duplicate_audit = {
        "total_records": len(all_records),
        "unique_prompts": len(unique_prompts),
        "duplicate_prompt_count": len(all_records) - len(unique_prompts),
        "train_val_leakage_count": len(train_val_collision),
        "train_eval_leakage_count": len(train_eval_collision),
        "val_eval_leakage_count": len(val_eval_collision),
        "audit_status": "CLEAN_ZERO_LEAKAGE"
    }
    with open(data_dir / "duplicate_audit.json", "w", encoding="utf-8") as f:
        json.dump(duplicate_audit, f, indent=2)

    # 5. Generate split_audit.json
    split_audit = {
        "train_total": len(train_records),
        "val_total": len(val_records),
        "train_composition": {
            "track_a_greetings": sum(1 for r in train_records if r["track"] == "Track A"),
            "track_b_identity": sum(1 for r in train_records if r["track"] == "Track B"),
            "track_c_grounding": sum(1 for r in train_records if r["track"] == "Track C"),
            "track_d_multiturn": sum(1 for r in train_records if r["track"] == "Track D"),
        },
        "val_composition": {
            "track_a_greetings": sum(1 for r in val_records if r["track"] == "Track A"),
            "track_b_identity": sum(1 for r in val_records if r["track"] == "Track B"),
            "track_c_grounding": sum(1 for r in val_records if r["track"] == "Track C"),
            "track_d_multiturn": sum(1 for r in val_records if r["track"] == "Track D"),
        },
        "track_a_breakdown": {
            "train_morning": sum(1 for r in train_records if r["category"] == "greetings_morning"),
            "train_afternoon": sum(1 for r in train_records if r["category"] == "greetings_afternoon"),
            "train_evening": sum(1 for r in train_records if r["category"] == "greetings_evening"),
            "train_generic": sum(1 for r in train_records if r["category"] == "greetings_generic"),
            "val_morning": sum(1 for r in val_records if r["category"] == "greetings_morning"),
            "val_afternoon": sum(1 for r in val_records if r["category"] == "greetings_afternoon"),
            "val_evening": sum(1 for r in val_records if r["category"] == "greetings_evening"),
            "val_generic": sum(1 for r in val_records if r["category"] == "greetings_generic"),
        }
    }
    with open(data_dir / "split_audit.json", "w", encoding="utf-8") as f:
        json.dump(split_audit, f, indent=2)

    # 6. Generate dataset_manifest.json
    dataset_manifest = {
        "manifest_version": "1.0.0",
        "dataset_name": "CBR-SFT-04",
        "description": "High-quality curated 160-example SFT dataset balancing temporal greetings, identity, entity grounding, and multi-turn pairs.",
        "counts": {
            "total": len(all_records),
            "train": len(train_records),
            "validation": len(val_records)
        },
        "hashes": {
            "dataset_jsonl": sha256_file(dataset_file),
            "train_jsonl": sha256_file(train_file),
            "validation_jsonl": sha256_file(val_file),
            "train_144_jsonl": sha256_file(train_144_file),
            "val_16_jsonl": sha256_file(val_16_file),
            "duplicate_audit_json": sha256_file(data_dir / "duplicate_audit.json"),
            "split_audit_json": sha256_file(data_dir / "split_audit.json")
        }
    }
    with open(data_dir / "dataset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(dataset_manifest, f, indent=2)

    # 7. Generate evaluation_manifest.json
    evaluation_manifest = {
        "manifest_version": "1.0.0",
        "suite_name": "CBR-SFT-04-EVAL-80",
        "total_tasks": len(HELD_OUT_EVAL_TASKS),
        "task_breakdown": {
            "track_a_greetings": 20,
            "track_b_identity": 15,
            "track_c_grounding": 25,
            "track_d_multiturn": 20
        },
        "hashes": {
            "evaluation_cases_jsonl": sha256_file(eval_cases_file),
            "held_out_cbr_sft_04_80_json": sha256_file(eval_json_file)
        }
    }
    with open(eval_dir / "evaluation_manifest.json", "w", encoding="utf-8") as f:
        json.dump(evaluation_manifest, f, indent=2)

    print("\nDataset generation and validation completed successfully!")
    print(f"dataset.jsonl SHA256: {dataset_manifest['hashes']['dataset_jsonl']}")
    print(f"train.jsonl SHA256:   {dataset_manifest['hashes']['train_jsonl']}")
    print(f"validation.jsonl SHA256: {dataset_manifest['hashes']['validation_jsonl']}")
    print(f"eval_cases.jsonl SHA256: {evaluation_manifest['hashes']['evaluation_cases_jsonl']}")

if __name__ == "__main__":
    main()

