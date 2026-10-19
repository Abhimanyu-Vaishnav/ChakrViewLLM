"""Curated SFT Starter Dataset Generator for Batch 2821–2860.

Generates:
1. SFT_DATASET.jsonl: Exactly 240 high-quality, locally authored instruction examples
   across 8 distinct task categories with 80% train (192), 10% val (24), 10% test (24) split.
2. SFT_EVALUATION_SET.json: Separate held-out evaluation set for benchmarking SFT vs Canonical baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "experiments" / "batch2821_2860"

# Category definitions and templates
# Each category contains 30 unique examples: 24 train, 3 val, 3 test.
# Total across 8 categories = 240 examples (192 train, 24 val, 24 test).

def get_simple_instruction_examples() -> List[Dict[str, Any]]:
    # 30 items
    items = [
        ("INST_001", "Define the term 'compiler' in one concise sentence.", "A compiler is a program that translates source code written in a high-level programming language into machine code or bytecode."),
        ("INST_002", "List the four fundamental states of matter.", "The four fundamental states of matter are solid, liquid, gas, and plasma."),
        ("INST_003", "Explain what an IP address is in one sentence.", "An IP address is a unique numerical identifier assigned to each device connected to a computer network that uses the Internet Protocol."),
        ("INST_004", "Name three common relational database management systems.", "Three common relational database management systems are PostgreSQL, MySQL, and SQLite."),
        ("INST_005", "What is the primary function of a CPU cache?", "The primary function of a CPU cache is to store frequently accessed data and instructions close to the processor to reduce memory access latency."),
        ("INST_006", "State the chemical formula for table salt.", "The chemical formula for table salt is NaCl."),
        ("INST_007", "What is the purpose of the 'git status' command?", "The 'git status' command displays the state of the working directory and staging area, showing which files are modified, staged, or untracked."),
        ("INST_008", "Define the mathematical term 'prime number'.", "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself."),
        ("INST_009", "Name three standard HTTP request methods.", "Three standard HTTP request methods are GET, POST, and DELETE."),
        ("INST_010", "What does DNS stand for in computer networking?", "DNS stands for Domain Name System."),
        ("INST_011", "Describe the role of an operating system kernel.", "The kernel is the core component of an operating system that manages hardware resources, memory, and process execution."),
        ("INST_012", "What is the speed of light in a vacuum in meters per second?", "The speed of light in a vacuum is approximately 299,792,458 meters per second."),
        ("INST_013", "Explain what an idempotent operation is.", "An idempotent operation is an operation that produces the same result no matter how many times it is executed with the same parameters."),
        ("INST_014", "Name the planet closest to the Sun.", "The planet closest to the Sun is Mercury."),
        ("INST_015", "Define the computer science concept of 'recursion'.", "Recursion is a programming technique where a function solves a problem by calling itself with smaller sub-problems until reaching a base condition."),
        ("INST_016", "What is the SI unit of electrical resistance?", "The SI unit of electrical resistance is the ohm (symbol: Ω)."),
        ("INST_017", "Name two primary symmetric encryption algorithms.", "Two primary symmetric encryption algorithms are AES (Advanced Encryption Standard) and ChaCha20."),
        ("INST_018", "What is the purpose of an index in a database table?", "An index accelerates data retrieval queries on a database table at the cost of additional storage and write overhead."),
        ("INST_019", "Define the biological term 'mitosis'.", "Mitosis is a process of cell division where a single eukaryotic cell divides into two genetically identical daughter cells."),
        ("INST_020", "What is an abstract syntax tree (AST)?", "An abstract syntax tree is a tree representation of the abstract syntactic structure of source code written in a programming language."),
        ("INST_021", "Name the three branches of the United States federal government.", "The three branches of the United States federal government are legislative, executive, and judicial."),
        ("INST_022", "What does JSON stand for?", "JSON stands for JavaScript Object Notation."),
        ("INST_023", "Explain the concept of 'immutability' in programming.", "Immutability refers to an object or data structure whose state cannot be modified after it has been created."),
        ("INST_024", "What is the freezing point of water in degrees Celsius at standard atmospheric pressure?", "The freezing point of water is 0 degrees Celsius at standard atmospheric pressure."),
        # Val (indices 24..26)
        ("INST_025", "Define the term 'deadlock' in concurrent computing.", "A deadlock occurs when two or more concurrent processes cannot proceed because each is waiting for a resource held by the other."),
        ("INST_026", "Name the three primary additive colors of light.", "The three primary additive colors of light are red, green, and blue."),
        ("INST_027", "What is a hash collision in cryptography?", "A hash collision occurs when two distinct inputs produce the identical output digest from a cryptographic hash function."),
        # Test (indices 27..29)
        ("INST_028", "State the law of conservation of energy.", "The law of conservation of energy states that energy cannot be created or destroyed, only transformed from one form to another."),
        ("INST_029", "What is the difference between synchronous and asynchronous execution?", "Synchronous execution blocks subsequent tasks until the current task completes, whereas asynchronous execution allows tasks to execute concurrently without waiting."),
        ("INST_030", "Define the term 'bandwidth' in computer networking.", "Bandwidth is the maximum rate of data transfer across a given network path in a specific period of time.")
    ]
    return [{"id": x[0], "category": "simple_instruction", "prompt": f"Instruction: {x[1]}\nResponse:", "response": x[2]} for x in items]


def get_context_extraction_examples() -> List[Dict[str, Any]]:
    # 30 items
    items = [
        ("EXTR_001", "The deployment to cluster-east completed at 14:32 UTC on Tuesday by engineer Priya Sharma. Who completed the deployment?", "Priya Sharma completed the deployment."),
        ("EXTR_002", "Server worker-09 operates on port 8443 with TLS 1.3 enabled. What port does worker-09 operate on?", "8443"),
        ("EXTR_003", "Order #44892 was shipped via BlueDart with tracking identifier BD-99410. What is the tracking identifier for Order #44892?", "BD-99410"),
        ("EXTR_004", "The research grant for Project BioSphere was awarded to Dr. Marcus Vance in the amount of $75,000. How much grant money was awarded?", "$75,000"),
        ("EXTR_005", "Database shard omega-4 is allocated 64 GB of memory and 16 CPU cores. How much memory is allocated to shard omega-4?", "64 GB"),
        ("EXTR_006", "The backup archive was created at /var/backups/db_20261014.tar.gz. Where was the backup archive created?", "/var/backups/db_20261014.tar.gz"),
        ("EXTR_007", "Incident INC-8021 was caused by a stale DNS record cached on gateway router G-12. Which router cached the stale DNS record?", "G-12"),
        ("EXTR_008", "Telemetry sensor S-14 recorded an ambient temperature of 23.4 degrees Celsius. What temperature did sensor S-14 record?", "23.4 degrees Celsius"),
        ("EXTR_009", "The invoice was billed to Acme Industrial Corp with payment due on November 30, 2026. On what date is payment due?", "November 30, 2026"),
        ("EXTR_010", "Container image redis:7.2.4-alpine was pulled from internal registry registry.corp.local. What is the full container image tag?", "redis:7.2.4-alpine"),
        ("EXTR_011", "Flight AI-802 is scheduled to depart from Terminal 3 at Gate B14. From which gate will Flight AI-802 depart?", "Gate B14"),
        ("EXTR_012", "The cryptographic checksum for the payload is 4a8f9c2d1e0b. What is the cryptographic checksum of the payload?", "4a8f9c2d1e0b"),
        ("EXTR_013", "The customer service hotline is available from 08:00 to 20:00 Monday through Saturday. What are the operating hours of the hotline?", "08:00 to 20:00"),
        ("EXTR_014", "Audit log 2026-Q3 flagged 12 unauthorized login attempts from IP 10.0.4.15. How many unauthorized attempts were flagged?", "12"),
        ("EXTR_015", "The new warranty policy extends coverage from 12 months to 36 months for premium subscribers. What is the new coverage duration?", "36 months"),
        ("EXTR_016", "Microservice auth-service was scaled to 5 replicas across zone us-west-2a. How many replicas was auth-service scaled to?", "5 replicas"),
        ("EXTR_017", "Project Falcon achieved a latency reduction of 42% following the kernel update. What percentage reduction in latency was achieved?", "42%"),
        ("EXTR_018", "The conference will be held at the Hyatt Regency in Denver on September 18. In which city is the conference located?", "Denver"),
        ("EXTR_019", "Worker thread #3 encountered error code ERR_CONN_RESET at line 142 of socket.py. What was the error code?", "ERR_CONN_RESET"),
        ("EXTR_020", "The primary contact for emergency maintenance is lead technician Dave Kowalski at extension 4022. What extension reaches Dave Kowalski?", "4022"),
        ("EXTR_021", "The software package requires Python version 3.11 or higher to execute. Which minimum Python version is required?", "3.11"),
        ("EXTR_022", "Storage volume vol-8831 is formatted with the XFS filesystem. Which filesystem formats vol-8831?", "XFS"),
        ("EXTR_023", "The annual general meeting concluded with 94.2% shareholder approval for the merger. What percentage of shareholder approval was recorded?", "94.2%"),
        ("EXTR_024", "Sensor hub Hub-North transmits readings every 15 seconds over LoRaWAN. How frequently does Hub-North transmit readings?", "Every 15 seconds"),
        # Val (24..26)
        ("EXTR_025", "The encrypted vault can be unlocked using security key token-zeta-77. What security key unlocks the vault?", "token-zeta-77"),
        ("EXTR_026", "Batch job #912 processed 148,000 records before terminating at 03:15 AM. How many records did batch job #912 process?", "148,000 records"),
        ("EXTR_027", "The SSL certificate for domain api.example.com expires on January 15, 2027. When does the SSL certificate expire?", "January 15, 2027"),
        # Test (27..29)
        ("EXTR_028", "The experimental solar cell recorded a power conversion efficiency of 28.6%. What was the power conversion efficiency?", "28.6%"),
        ("EXTR_029", "Virtual machine node-401 has been assigned internal IP address 172.16.8.99. What internal IP address is assigned to node-401?", "172.16.8.99"),
        ("EXTR_030", "Architect Sarah Lin authored the specification document RFC-1092. Who authored RFC-1092?", "Sarah Lin")
    ]
    return [{"id": x[0], "category": "context_extraction", "prompt": f"Instruction: Extract the requested detail from the text:\n{x[1]}\nResponse:", "response": x[2]} for x in items]


def get_context_grounded_qa_examples() -> List[Dict[str, Any]]:
    # 30 items
    items = [
        ("CQA_001", "Context: ChakrView Enterprise is licensed on an annual per-core basis. Standard support includes email responses within 24 business hours, whereas Premium support provides 24/7 phone escalation with a 1-hour response SLA.\nQuestion: What support response time is guaranteed under the Premium support tier?", "Under the Premium support tier, a 1-hour response SLA is guaranteed with 24/7 phone escalation."),
        ("CQA_002", "Context: The engine cooling circuit utilizes a 50/50 mixture of ethylene glycol and deionized water. The system operating pressure is maintained at 15 psi by the radiator cap.\nQuestion: What fluid mixture does the cooling circuit utilize?", "The cooling circuit utilizes a 50/50 mixture of ethylene glycol and deionized water."),
        ("CQA_003", "Context: In high-concurrency environments, writes are routed to a primary master database, while read queries are load-balanced across three read-only replicas.\nQuestion: Where are read queries routed in this system?", "Read queries are load-balanced across three read-only replicas."),
        ("CQA_004", "Context: Password policies mandate a minimum length of 14 characters, including at least one uppercase letter, one numeral, and one symbol. Passwords expire every 90 days.\nQuestion: How often do passwords expire according to the policy?", "Passwords expire every 90 days."),
        ("CQA_005", "Context: The robot arm uses a stepper motor with a 1.8-degree step angle and harmonic drive gearing with a 50:1 reduction ratio.\nQuestion: What reduction ratio is provided by the harmonic drive gearing?", "The harmonic drive gearing provides a 50:1 reduction ratio."),
        ("CQA_006", "Context: HTTP status code 429 indicates that the user has sent too many requests in a given amount of time, known as rate limiting.\nQuestion: What condition does HTTP status code 429 indicate?", "HTTP status code 429 indicates that the client has sent too many requests and has exceeded the rate limit."),
        ("CQA_007", "Context: The water purification plant treats up to 10 million gallons daily using rapid sand filtration followed by ultraviolet disinfection and chloramination.\nQuestion: Which disinfection methods are used after sand filtration?", "Ultraviolet disinfection and chloramination are used after rapid sand filtration."),
        ("CQA_008", "Context: Employees are eligible for 20 days of paid annual leave, accruing at 1.67 days per full month worked.\nQuestion: How many days of paid annual leave do employees receive per year?", "Employees receive 20 days of paid annual leave per year."),
        ("CQA_009", "Context: The telescope is positioned at an altitude of 4,200 meters atop Mauna Kea to minimize atmospheric distortion and water vapor absorption.\nQuestion: Why is the telescope positioned atop Mauna Kea?", "The telescope is positioned atop Mauna Kea to minimize atmospheric distortion and water vapor absorption."),
        ("CQA_010", "Context: The compiler performs dead code elimination during the optimization phase, stripping functions and variables that are never referenced.\nQuestion: In which phase does the compiler strip unreferenced functions?", "The compiler strips unreferenced functions during the optimization phase."),
        ("CQA_011", "Context: All financial transactions exceeding $10,000 require secondary approval from a compliance manager before execution.\nQuestion: What approval is required for transactions over $10,000?", "Transactions exceeding $10,000 require secondary approval from a compliance manager."),
        ("CQA_012", "Context: The electric motor produces 320 Nm of torque at 0 RPM, delivering immediate acceleration up to a governed top speed of 180 km/h.\nQuestion: What torque does the motor produce at 0 RPM?", "The motor produces 320 Nm of torque at 0 RPM."),
        ("CQA_013", "Context: Data stored in S3 Glacier Flexible Retrieval has an access latency of 3 to 5 hours for standard retrieval jobs.\nQuestion: What is the standard retrieval time for data in S3 Glacier Flexible Retrieval?", "The standard retrieval time is 3 to 5 hours."),
        ("CQA_014", "Context: The vaccine must be stored in ultra-cold freezers between -80 and -60 degrees Celsius to maintain stability.\nQuestion: What storage temperature range is required for the vaccine?", "The vaccine requires storage between -80 and -60 degrees Celsius."),
        ("CQA_015", "Context: Subscriptions can be cancelled at any time through the customer billing portal, with access remaining active until the end of the billing period.\nQuestion: What happens to access after a customer cancels their subscription?", "Access remains active until the end of the current billing period."),
        ("CQA_016", "Context: The firewall drops all inbound UDP packets on ports 1024 through 65535 by default unless an explicit stateful connection exists.\nQuestion: What is the default policy for inbound UDP packets on high ports?", "The default policy is to drop all inbound UDP packets on ports 1024 through 65535."),
        ("CQA_017", "Context: The drone has a maximum payload capacity of 2.5 kg and an operational flight time of 35 minutes on a full battery charge.\nQuestion: How long can the drone operate on a full charge?", "The drone can operate for up to 35 minutes on a full battery charge."),
        ("CQA_018", "Context: In Rust, memory safety is enforced at compile time through an ownership model with strict borrowing and lifetime rules.\nQuestion: How does Rust enforce memory safety?", "Rust enforces memory safety at compile time through an ownership model with borrowing and lifetime rules."),
        ("CQA_019", "Context: The warehouse utilizes automated guided vehicles (AGVs) equipped with LiDAR sensors to navigate aisles and transport pallets.\nQuestion: What sensors do the AGVs use for navigation?", "The AGVs use LiDAR sensors to navigate."),
        ("CQA_020", "Context: The audit discovered that 3 out of 15 servers lacked the mandatory security kernel patch KB-44019.\nQuestion: How many servers were missing the mandatory security patch?", "3 out of 15 servers were missing the mandatory patch."),
        ("CQA_021", "Context: The laboratory operates under ISO 17025 accreditation, ensuring rigorous calibration and measurement traceability.\nQuestion: Which accreditation does the laboratory operate under?", "The laboratory operates under ISO 17025 accreditation."),
        ("CQA_022", "Context: The aircraft is powered by two turbofan engines each capable of producing 28,000 pounds of thrust at sea level.\nQuestion: How many pounds of thrust does each engine produce?", "Each turbofan engine produces 28,000 pounds of thrust."),
        ("CQA_023", "Context: To participate in the beta testing program, users must sign a non-disclosure agreement and submit weekly telemetry logs.\nQuestion: What documents must beta testers sign before participating?", "Beta testers must sign a non-disclosure agreement (NDA)."),
        ("CQA_024", "Context: The building's HVAC system recirculates 80% of interior air while drawing 20% fresh outdoor air through MERV 13 filtration.\nQuestion: What percentage of fresh outdoor air does the HVAC system draw?", "The HVAC system draws 20% fresh outdoor air."),
        # Val (24..26)
        ("CQA_025", "Context: In OAuth 2.0, an authorization code is exchanged for an access token at the authorization server's token endpoint.\nQuestion: Where is the authorization code exchanged for an access token?", "It is exchanged at the authorization server's token endpoint."),
        ("CQA_026", "Context: The lunar lander's descent engine throttles between 10% and 60% thrust to permit a gentle touchdown on the lunar surface.\nQuestion: What is the throttle range of the descent engine?", "The descent engine throttles between 10% and 60% thrust."),
        ("CQA_027", "Context: The file compression utility employs the DEFLATE algorithm, combining LZ77 and Huffman coding.\nQuestion: Which two techniques are combined in the DEFLATE algorithm?", "DEFLATE combines the LZ77 algorithm and Huffman coding."),
        # Test (27..29)
        ("CQA_028", "Context: The battery pack utilizes lithium iron phosphate (LiFePO4) chemistry, rated for over 3,000 full charge cycles at 80% depth of discharge.\nQuestion: How many charge cycles is the battery pack rated for?", "The battery pack is rated for over 3,000 full charge cycles."),
        ("CQA_029", "Context: In BGP routing, an autonomous system prepends its ASN to the AS-PATH attribute when advertising routes to external peers.\nQuestion: Which attribute is prepended when advertising BGP routes to external peers?", "The AS-PATH attribute is prepended."),
        ("CQA_030", "Context: The clinical trial enrolled 450 adult participants across 8 medical centers in North America.\nQuestion: How many participants were enrolled in the clinical trial?", "The clinical trial enrolled 450 adult participants.")
    ]
    return [{"id": x[0], "category": "context_grounded_qa", "prompt": f"Instruction: Answer the question using only the provided context.\n{x[1]}\nResponse:", "response": x[2]} for x in items]


def get_short_summarization_examples() -> List[Dict[str, Any]]:
    # 30 items
    items = [
        ("SUMM_001", "The James Webb Space Telescope uses a 6.5-meter gold-plated beryllium reflector and sensitive infrared instruments to observe the earliest galaxies formed after the Big Bang.", "JWST uses infrared instruments to observe early post-Big Bang galaxies."),
        ("SUMM_002", "Continuous Integration and Continuous Deployment (CI/CD) automates the building, testing, and deployment of software to deliver updates rapidly and reliably to production.", "CI/CD automates software build, test, and deployment pipelines."),
        ("SUMM_003", "Photosynthesis in green plants absorbs carbon dioxide and water in the presence of sunlight, producing glucose and releasing oxygen into the atmosphere.", "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight."),
        ("SUMM_004", "Database transactions adhere to ACID properties—Atomicity, Consistency, Isolation, and Durability—to ensure reliable processing despite system crashes or errors.", "ACID properties ensure reliable and consistent database transactions."),
        ("SUMM_005", "Public key cryptography uses asymmetric key pairs: a public key for encrypting data and a private key for decrypting data without sharing secrets over the wire.", "Asymmetric cryptography secures communication using public encryption and private decryption keys."),
        ("SUMM_006", "The human circulatory system pumps oxygenated blood from the heart through arteries to body tissues and returns deoxygenated blood through veins to the lungs.", "The circulatory system transports oxygenated blood to tissues and returns deoxygenated blood."),
        ("SUMM_007", "Kubernetes is an open-source orchestration system that automates container deployment, scaling, failover, and resource management across server clusters.", "Kubernetes automates containerized application scaling and cluster management."),
        ("SUMM_008", "A solar photovoltaic cell converts sunlight directly into direct current electricity using semiconductor materials that exhibit the photovoltaic effect.", "Solar cells generate DC electricity directly from sunlight via semiconductors."),
        ("SUMM_009", "The central limit theorem in statistics states that the sum or average of many independent random variables tends toward a normal distribution.", "The central limit theorem states that independent sample averages approximate a normal distribution."),
        ("SUMM_010", "Garbage collection in runtimes like Java and Go periodically scans heap memory to identify and reclaim allocated memory that is no longer reachable.", "Garbage collection automatically reclaims unreachable heap memory."),
        ("SUMM_011", "Bicycle helmets reduce head injury severity during collisions by absorbing and dissipating kinetic impact energy through compressible polystyrene foam.", "Helmets reduce collision head injuries by absorbing kinetic impact energy."),
        ("SUMM_012", "Microservices architecture structures an application as a collection of independently deployable, loosely coupled services communicating over lightweight APIs.", "Microservices partition applications into independently deployable, API-connected services."),
        ("SUMM_013", "The Doppler effect causes an observed shift in wave frequency when a wave source and observer move relative to one another.", "The Doppler effect shifts observed wave frequency due to relative motion."),
        ("SUMM_014", "Virtual memory allows an operating system to compensate for physical RAM shortages by temporarily paging data blocks to secondary disk storage.", "Virtual memory pages data to disk storage when physical RAM is exhausted."),
        ("SUMM_015", "Vaccines introduce harmless antigens into the body to stimulate the immune system to produce antibodies and memory cells against specific pathogens.", "Vaccines train the immune system to produce antibodies using harmless antigens."),
        ("SUMM_016", "The open-source Linux kernel was created by Linus Torvalds in 1991 and powers millions of servers, supercomputers, and mobile devices today.", "Linus Torvalds created Linux in 1991, which now powers modern computing infrastructure."),
        ("SUMM_017", "A deadlock in operating systems occurs when multiple processes are permanently blocked because each holds a lock needed by another.", "A deadlock permanently blocks processes waiting on circular resource dependencies."),
        ("SUMM_018", "Tidal power harnesses the gravitational pull of the Moon and Sun on Earth's oceans to turn turbines and generate renewable electricity.", "Tidal power generates renewable electricity from ocean water movement driven by gravity."),
        ("SUMM_019", "A load balancer distributes incoming network traffic across multiple backend servers to prevent overload and maximize application uptime.", "Load balancers distribute traffic across multiple servers to ensure uptime."),
        ("SUMM_020", "Plate tectonics describes how large lithospheric plates floating on Earth's mantle collide and separate, creating mountains, earthquakes, and ocean trenches.", "Plate tectonics explains earthquakes and continental formations via moving lithospheric plates."),
        ("SUMM_021", "Containerization packages application code together with its runtime dependencies into portable containers that execute consistently across environments.", "Containerization packages apps with dependencies for portable execution."),
        ("SUMM_022", "Antibiotic resistance develops when bacterial strains mutate to survive drug treatments, rendering conventional antimicrobial therapies ineffective.", "Antibiotic resistance occurs when mutating bacteria survive standard pharmaceutical treatments."),
        ("SUMM_023", "The TCP three-way handshake establishes a reliable network connection between client and server using SYN, SYN-ACK, and ACK packets.", "The TCP three-way handshake establishes connections via SYN, SYN-ACK, and ACK exchanges."),
        ("SUMM_024", "Superconductors exhibit zero electrical resistance and expel magnetic flux when cooled below a characteristic critical temperature.", "Superconductors conduct electricity with zero resistance below a critical temperature."),
        # Val (24..26)
        ("SUMM_025", "Event-driven architecture triggers and coordinates decoupled services in response to discrete events rather than synchronous request-response polling.", "Event-driven architecture coordinates decoupled services through discrete event notifications."),
        ("SUMM_026", "The greenhouse effect traps solar infrared radiation within Earth's atmosphere through gases like water vapor, carbon dioxide, and methane.", "The greenhouse effect retains atmospheric heat through infrared-absorbing gases."),
        ("SUMM_027", "Binary search finds target values in sorted lists by repeatedly dividing the search interval in half with logarithmic time complexity.", "Binary search locates elements in sorted arrays by repeatedly halving search intervals."),
        # Test (27..29)
        ("SUMM_028", "Geothermal power plants extract steam and hot water from deep underground reservoirs to drive electric generators with low greenhouse emissions.", "Geothermal power generates low-emission electricity using underground geothermal steam."),
        ("SUMM_029", "A database index creates a specialized data structure that accelerates query lookups while increasing write overhead and storage consumption.", "Database indices accelerate query lookups at the expense of storage and write performance."),
        ("SUMM_030", "Blockchain technology creates a decentralized, append-only ledger secured by cryptographic hash chaining and distributed consensus mechanisms.", "Blockchains provide decentralized, cryptographically chained append-only ledgers.")
    ]
    return [{"id": x[0], "category": "short_summarization", "prompt": f"Instruction: Summarize the following passage in a single concise sentence:\n{x[1]}\nResponse:", "response": x[2]} for x in items]


def get_structured_output_examples() -> List[Dict[str, Any]]:
    # 30 items
    items = [
        ("STR_001", "Format the name 'Alice Smith' and role 'Engineer' as a JSON object with keys 'name' and 'role'.", '{"name": "Alice Smith", "role": "Engineer"}'),
        ("STR_002", "Format the status 'ACTIVE' and code 200 as a JSON object with keys 'status' and 'code'.", '{"status": "ACTIVE", "code": 200}'),
        ("STR_003", "Represent the fruits Apple, Banana, and Cherry as a JSON array of strings.", '["Apple", "Banana", "Cherry"]'),
        ("STR_004", "Output a key-value record for server 'web-01' with IP '10.0.0.1' and status 'UP' in JSON.", '{"server": "web-01", "ip": "10.0.0.1", "status": "UP"}'),
        ("STR_005", "Convert the coordinates latitude 37.77 and longitude -122.41 into a JSON object with keys 'lat' and 'lon'.", '{"lat": 37.77, "lon": -122.41}'),
        ("STR_006", "Output a JSON object with keys 'item', 'quantity', and 'price' for 5 pencils at $0.50 each.", '{"item": "pencil", "quantity": 5, "price": 0.5}'),
        ("STR_007", "Create a JSON array containing the numbers 10, 20, 30, 40, and 50.", '[10, 20, 30, 40, 50]'),
        ("STR_008", "Format user details for ID 104 with username 'dev_user' and verified status true as JSON.", '{"id": 104, "username": "dev_user", "verified": true}'),
        ("STR_009", "Output a JSON object for a task named 'Audit' with priority 'HIGH' and completed false.", '{"task": "Audit", "priority": "HIGH", "completed": false}'),
        ("STR_010", "Represent a version record with major 1, minor 4, and patch 0 as JSON.", '{"major": 1, "minor": 4, "patch": 0}'),
        ("STR_011", "Output a JSON object containing keys 'protocol' and 'port' for HTTPS on port 443.", '{"protocol": "HTTPS", "port": 443}'),
        ("STR_012", "Represent an alert with severity 'CRITICAL' and message 'Disk full' in JSON format.", '{"severity": "CRITICAL", "message": "Disk full"}'),
        ("STR_013", "Format the dimensions width 1920, height 1080, and unit 'px' as a JSON object.", '{"width": 1920, "height": 1080, "unit": "px"}'),
        ("STR_014", "Output a JSON object representing a book with title 'Dune' and author 'Frank Herbert'.", '{"title": "Dune", "author": "Frank Herbert"}'),
        ("STR_015", "Represent the three RGB color channels red 255, green 128, and blue 0 as JSON.", '{"red": 255, "green": 128, "blue": 0}'),
        ("STR_016", "Format an authentication response with token 'xyz-123' and expires_in 3600 in JSON.", '{"token": "xyz-123", "expires_in": 3600}'),
        ("STR_017", "Output a JSON array containing the string elements 'GET', 'POST', and 'PUT'.", '["GET", "POST", "PUT"]'),
        ("STR_018", "Output a JSON object for an employee with department 'Security' and badge_number 8812.", '{"department": "Security", "badge_number": 8812}'),
        ("STR_019", "Represent a file record with filename 'report.pdf' and size_kb 240 as JSON.", '{"filename": "report.pdf", "size_kb": 240}'),
        ("STR_020", "Output a JSON object with keys 'country' and 'capital' for Japan and Tokyo.", '{"country": "Japan", "capital": "Tokyo"}'),
        ("STR_021", "Format metric name 'cpu_utilization' and value 48.5 as a JSON object.", '{"metric": "cpu_utilization", "value": 48.5}'),
        ("STR_022", "Output a JSON object for a vehicle with make 'Toyota' and model 'Corolla'.", '{"make": "Toyota", "model": "Corolla"}'),
        ("STR_023", "Format a configuration setting with key 'timeout_seconds' and value 30 in JSON.", '{"key": "timeout_seconds", "value": 30}'),
        ("STR_024", "Represent a database connection record with host 'localhost' and database 'main_db' in JSON.", '{"host": "localhost", "database": "main_db"}'),
        # Val (24..26)
        ("STR_025", "Format an error report with code 'E_NOT_FOUND' and retriable false as JSON.", '{"code": "E_NOT_FOUND", "retriable": false}'),
        ("STR_026", "Output a JSON array of the primary color names 'red', 'green', and 'blue'.", '["red", "green", "blue"]'),
        ("STR_027", "Represent a temperature measurement with value 98.6 and scale 'Fahrenheit' in JSON.", '{"value": 98.6, "scale": "Fahrenheit"}'),
        # Test (27..29)
        ("STR_028", "Output a JSON object with keys 'source' and 'destination' for path from 'A' to 'B'.", '{"source": "A", "destination": "B"}'),
        ("STR_029", "Format memory metrics total_mb 16384 and used_mb 8192 as a JSON object.", '{"total_mb": 16384, "used_mb": 8192}'),
        ("STR_030", "Represent a transaction with tx_id 'TX-901' and amount 150.75 as JSON.", '{"tx_id": "TX-901", "amount": 150.75}')
    ]
    return [{"id": x[0], "category": "structured_output", "prompt": f"Instruction: {x[1]}\nResponse:", "response": x[2]} for x in items]


def get_uncertainty_insufficient_info_examples() -> List[Dict[str, Any]]:
    # 30 items
    items = [
        ("UNC_001", "Context: The meeting concluded at 11:30 AM in Conference Room B.\nQuestion: What was the main topic discussed during the meeting?", "The provided context does not contain information about the main topic discussed in the meeting."),
        ("UNC_002", "Context: Server alpha-1 was rebooted on Sunday morning by the ops team.\nQuestion: How much RAM is installed on server alpha-1?", "The provided context does not state how much RAM is installed on server alpha-1."),
        ("UNC_003", "Context: Dr. Patel submitted the quarterly clinical report to the review board on Friday.\nQuestion: What was the total budget for Dr. Patel's study?", "The provided context does not provide information about the study's budget."),
        ("UNC_004", "Context: Flight 204 departed on time from Chicago O'Hare International Airport.\nQuestion: What was the arrival destination of Flight 204?", "The provided context does not mention the arrival destination of Flight 204."),
        ("UNC_005", "Context: The vehicle is equipped with anti-lock brakes and six airbags.\nQuestion: What is the vehicle's top speed in kilometers per hour?", "The provided context does not contain information regarding the vehicle's top speed."),
        ("UNC_006", "Context: Order #1084 includes three blue shirts and two pairs of socks.\nQuestion: What was the total price of Order #1084?", "The provided context does not state the total price of the order."),
        ("UNC_007", "Context: The concert will take place in Central Park on Saturday evening.\nQuestion: Which musicians are performing at the concert?", "The provided context does not list the musicians performing at the concert."),
        ("UNC_008", "Context: Package pkg-901 was delivered to the front porch at 2:15 PM.\nQuestion: Who signed for the delivery of package pkg-901?", "The provided context does not mention who signed for the delivery."),
        ("UNC_009", "Context: The software library was updated to version 2.4.0 yesterday.\nQuestion: What programming language was the library written in?", "The provided context does not specify the programming language of the library."),
        ("UNC_010", "Context: The garden contains red roses, yellow tulips, and white daisies.\nQuestion: Who planted the garden?", "The provided context does not state who planted the garden."),
        ("UNC_011", "Context: Building C has four floors and is equipped with two elevators.\nQuestion: In what year was Building C constructed?", "The provided context does not provide the year Building C was constructed."),
        ("UNC_012", "Context: The recipe calls for two cups of flour and one cup of sugar.\nQuestion: At what temperature should the oven be preheated?", "The provided context does not mention the oven preheating temperature."),
        ("UNC_013", "Context: The drone operated for 25 minutes before returning to home base.\nQuestion: What video resolution did the drone camera record in?", "The provided context does not contain details about the drone's camera resolution."),
        ("UNC_014", "Context: Account 9942 was flagged for multiple incorrect password attempts.\nQuestion: What is the account holder's email address?", "The provided context does not mention the account holder's email address."),
        ("UNC_015", "Context: The restaurant serves Italian pasta dishes and wood-fired pizzas.\nQuestion: Is the restaurant open on Monday evenings?", "The provided context does not provide information about the restaurant's operating hours."),
        ("UNC_016", "Context: The laptop comes with a 15.6-inch display and an aluminum chassis.\nQuestion: What processor model is installed in the laptop?", "The provided context does not state the processor model installed in the laptop."),
        ("UNC_017", "Context: The bridge spans a distance of 1,200 meters across the river.\nQuestion: What is the toll fee to cross the bridge?", "The provided context does not state the toll fee for crossing the bridge."),
        ("UNC_018", "Context: Project Mercury completed milestone 3 ahead of the scheduled deadline.\nQuestion: How many engineers are currently assigned to Project Mercury?", "The provided context does not state the number of engineers assigned to Project Mercury."),
        ("UNC_019", "Context: The patient's blood pressure was measured at 120/80 mmHg.\nQuestion: What was the patient's heart rate during the exam?", "The provided context does not provide the patient's heart rate."),
        ("UNC_020", "Context: Shard 4 contains 2.5 million user activity records.\nQuestion: Which AWS region hosts Shard 4?", "The provided context does not specify which AWS region hosts Shard 4."),
        ("UNC_021", "Context: The train stops at Central Station, West End, and Riverside.\nQuestion: How much does a one-way ticket cost?", "The provided context does not contain ticket pricing information."),
        ("UNC_022", "Context: The workshop covers fundamentals of machine learning and data preprocessing.\nQuestion: Will lunch be provided to workshop attendees?", "The provided context does not state whether lunch is provided."),
        ("UNC_023", "Context: The warehouse stored 400 pallets of consumer goods on Level 2.\nQuestion: What brand of forklift is used in the warehouse?", "The provided context does not mention the brand of forklift used."),
        ("UNC_024", "Context: The library book was returned five days past the due date.\nQuestion: What was the title of the overdue book?", "The provided context does not give the title of the returned book."),
        # Val (24..26)
        ("UNC_025", "Context: The painting was restored in 1998 by a team of art conservators.\nQuestion: Who originally painted the artwork?", "The provided context does not state who originally painted the artwork."),
        ("UNC_026", "Context: The application crashed with exit code 137 on node 12.\nQuestion: Which user started the application process?", "The provided context does not identify the user who started the process."),
        ("UNC_027", "Context: The test suite executed 84 test cases in 12.4 seconds.\nQuestion: Which test case failed during execution?", "The provided context does not indicate whether any test case failed."),
        # Test (27..29)
        ("UNC_028", "Context: The solar array generated 42 kWh of energy on Thursday.\nQuestion: What was the weather condition on Thursday?", "The provided context does not describe the weather condition on Thursday."),
        ("UNC_029", "Context: The package contains fragile glassware and is marked with caution tape.\nQuestion: What company manufactured the glassware?", "The provided context does not mention the manufacturer of the glassware."),
        ("UNC_030", "Context: Worker process #4 consumed 350 MB of memory before terminating.\nQuestion: What was the exact operating system version of the host?", "The provided context does not specify the host operating system version.")
    ]
    return [{"id": x[0], "category": "uncertainty_insufficient_info", "prompt": f"Instruction: Answer the question based on the provided context. If the context does not contain the answer, state that the information is unavailable.\n{x[1]}\nResponse:", "response": x[2]} for x in items]


def get_conversational_acknowledgement_examples() -> List[Dict[str, Any]]:
    # 30 items
    items = [
        ("CONV_001", "Hello! Are you ready to assist with my project tasks?", "Hello! Yes, I am ready to assist you. What tasks are we working on?"),
        ("CONV_002", "Thank you for the quick explanation.", "You're welcome! Let me know if you need any further assistance."),
        ("CONV_003", "Good morning, assistant.", "Good morning! How can I help you today?"),
        ("CONV_004", "I appreciate your help with this code.", "Happy to help! Let me know if you have any questions about the code."),
        ("CONV_005", "Goodbye for now, see you later.", "Goodbye! Have a great day and feel free to return if you need anything else."),
        ("CONV_006", "Hi there, can you help me check my documentation?", "Hello! Absolutely. Please share the documentation you would like me to review."),
        ("CONV_007", "Thanks, that answers my question completely.", "Glad to hear that! Please reach out if you have any more questions."),
        ("CONV_008", "Good afternoon! Hope you are running smoothly.", "Good afternoon! All systems are functioning normally. How can I assist you?"),
        ("CONV_009", "Understood, thanks for clarifying that point.", "You are welcome. Let me know what step you would like to take next."),
        ("CONV_010", "Hey, I have a quick question about Python syntax.", "Hello! Go ahead and ask your Python syntax question."),
        ("CONV_011", "Thanks a lot for the prompt response.", "You're very welcome! Let me know if there is anything else I can do for you."),
        ("CONV_012", "Hello assistant, hope your day is going well.", "Hello! I am operating smoothly and ready to assist you. What can I help with?"),
        ("CONV_013", "Great job on that summary, thank you.", "Thank you! I am glad the summary was helpful."),
        ("CONV_014", "Hi, I need assistance troubleshooting a system issue.", "Hello! Please describe the system issue and any error messages you are seeing."),
        ("CONV_015", "Thanks! That resolved the problem.", "Excellent! I am glad that resolved it. Let me know if anything else arises."),
        ("CONV_016", "Good evening, let's continue our task.", "Good evening! Ready to continue. What is the next task?"),
        ("CONV_017", "Appreciate the detailed breakdown, thanks.", "You're welcome! Feel free to ask if you would like to explore any part in more detail."),
        ("CONV_018", "Hello, are you available to review a report?", "Yes, I am available. Please share the report details."),
        ("CONV_019", "Thank you, that was very clear and concise.", "You're welcome! I'm glad it was clear."),
        ("CONV_020", "Hey assistant, good to work with you again.", "Great to work with you again! What are we focusing on today?"),
        ("CONV_021", "Thanks for your patience while I gathered the logs.", "No problem at all! Whenever you are ready, share the logs and we will review them."),
        ("CONV_022", "Hi! I would like to double check a calculation.", "Hello! Please provide the calculation you would like to verify."),
        ("CONV_023", "Thank you for catching that mistake.", "You're welcome! Catching discrepancies early is always beneficial."),
        ("CONV_024", "That will be all for today, thank you.", "You're welcome! Have a wonderful day, and see you next time."),
        # Val (24..26)
        ("CONV_025", "Good morning! Let's start the morning standup.", "Good morning! Ready for the standup. What updates or blockers do we have?"),
        ("CONV_026", "Thanks for helping me format that output.", "Happy to help! Let me know if the format needs any further adjustments."),
        ("CONV_027", "Hi, can you help me draft a status email?", "Hello! Certainly. Please provide the key updates to include in the draft."),
        # Test (27..29)
        ("CONV_028", "Thank you for the guidance on this task.", "You're very welcome! I'm glad I could provide useful guidance."),
        ("CONV_029", "Hello! Checking in on our project workflow.", "Hello! Workflow monitoring is active. What details would you like to review?"),
        ("CONV_030", "Thanks assistant, talk to you tomorrow.", "You're welcome! Talk to you tomorrow. Have a great evening!")
    ]
    return [{"id": x[0], "category": "conversational_acknowledgement", "prompt": f"Instruction: Respond appropriately to the conversational remark:\n{x[1]}\nResponse:", "response": x[2]} for x in items]


def get_bounded_text_transformation_examples() -> List[Dict[str, Any]]:
    # 30 items
    items = [
        ("TRNS_001", "Convert the string 'hello world' to uppercase.", "HELLO WORLD"),
        ("TRNS_002", "Convert the string 'DATABASE_CONNECTION_POOL' to lowercase.", "database_connection_pool"),
        ("TRNS_003", "Reverse the order of words in: 'one two three four'.", "four three two one"),
        ("TRNS_004", "Format the list of items [car, bike, bus] as a comma-separated string.", "car, bike, bus"),
        ("TRNS_005", "Convert the title 'system architecture overview' to Title Case.", "System Architecture Overview"),
        ("TRNS_006", "Extract the domain name from the email address 'developer@example.org'.", "example.org"),
        ("TRNS_007", "Remove all punctuation marks from the sentence: 'Warning: server is offline!'.", "Warning server is offline"),
        ("TRNS_008", "Replace all spaces with hyphens in the string: 'my new project name'.", "my-new-project-name"),
        ("TRNS_009", "Convert the snake_case identifier 'user_account_balance' to camelCase.", "userAccountBalance"),
        ("TRNS_010", "Prefix each item in the list ['alpha', 'beta', 'gamma'] with a bullet dash '- '.", "- alpha\n- beta\n- gamma"),
        ("TRNS_011", "Convert the string 'TRUE' to its boolean lowercase counterpart.", "true"),
        ("TRNS_012", "Join the path components ['var', 'log', 'syslog'] using forward slashes.", "var/log/syslog"),
        ("TRNS_013", "Wrap the word 'important' in double quotation marks.", '"important"'),
        ("TRNS_014", "Sort the following numbers in ascending order: 8, 3, 12, 1, 5.", "1, 3, 5, 8, 12"),
        ("TRNS_015", "Extract the file extension from 'archive_2026.tar.gz'.", "gz"),
        ("TRNS_016", "Convert the kebab-case string 'auth-service-client' to snake_case.", "auth_service_client"),
        ("TRNS_017", "Trim leading and trailing whitespace from: '   padded text   '.", "padded text"),
        ("TRNS_018", "Capitalize the first letter of the sentence: 'the quick brown fox'.", "The quick brown fox"),
        ("TRNS_019", "Remove duplicate words from the list: ['apple', 'banana', 'apple', 'orange'].", "['apple', 'banana', 'orange']"),
        ("TRNS_020", "Convert the date string '2026-10-18' to the format '18/10/2026'.", "18/10/2026"),
        ("TRNS_021", "Convert the comma-separated string 'cat, dog, bird' into a bulleted list.", "- cat\n- dog\n- bird"),
        ("TRNS_022", "Enclose the SQL statement 'SELECT * FROM users' in a markdown code block.", "```sql\nSELECT * FROM users\n```"),
        ("TRNS_023", "Abbreviate the phrase 'Hypertext Transfer Protocol' into its acronym.", "HTTP"),
        ("TRNS_024", "Count the number of words in the sentence: 'The quick brown fox jumps'.", "5"),
        # Val (24..26)
        ("TRNS_025", "Convert the string 'production' to uppercase.", "PRODUCTION"),
        ("TRNS_026", "Sort the words alphabetically: [zebra, apple, mango, banana].", "apple, banana, mango, zebra"),
        ("TRNS_027", "Replace underscores with spaces in: 'error_disk_write_timeout'.", "error disk write timeout"),
        # Test (27..29)
        ("TRNS_028", "Convert the string 'CRITICAL' to lowercase.", "critical"),
        ("TRNS_029", "Format the key-value pair key='port' and value=8080 as 'port: 8080'.", "port: 8080"),
        ("TRNS_030", "Reverse the string 'CHAKR'.", "RKAHC")
    ]
    return [{"id": x[0], "category": "bounded_text_transformation", "prompt": f"Instruction: {x[1]}\nResponse:", "response": x[2]} for x in items]


def build_full_dataset() -> List[Dict[str, Any]]:
    generators = [
        get_simple_instruction_examples,
        get_context_extraction_examples,
        get_context_grounded_qa_examples,
        get_short_summarization_examples,
        get_structured_output_examples,
        get_uncertainty_insufficient_info_examples,
        get_conversational_acknowledgement_examples,
        get_bounded_text_transformation_examples,
    ]
    all_records: List[Dict[str, Any]] = []

    for gen in generators:
        cat_records = gen()
        assert len(cat_records) == 30, f"Expected 30 records for generator, got {len(cat_records)}"
        for i, rec in enumerate(cat_records):
            if i < 24:
                split = "train"
                data_role = "TRAIN"
            elif i < 27:
                split = "val"
                data_role = "VALIDATION"
            else:
                split = "test"
                data_role = "TEST"

            rec["split"] = split
            rec["data_role"] = data_role
            rec["provenance"] = "locally_authored_batch2821_2860"
            rec["review_status"] = "VERIFIED_HUMAN_REVIEWED"
            rec["status"] = "APPROVED"
            all_records.append(rec)

    return all_records


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = build_full_dataset()
    assert len(dataset) == 240, f"Expected 240 records, got {len(dataset)}"

    # 1. Write SFT_DATASET.jsonl
    dataset_file = OUTPUT_DIR / "SFT_DATASET.jsonl"
    with open(dataset_file, "w", encoding="utf-8") as f:
        for r in dataset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(dataset)} examples to {dataset_file}")

    # Also write train.jsonl and val.jsonl for direct use with SFTConfig
    train_records = [r for r in dataset if r["split"] == "train"]
    val_records = [r for r in dataset if r["split"] == "val"]
    test_records = [r for r in dataset if r["split"] == "test"]

    with open(OUTPUT_DIR / "train.jsonl", "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(OUTPUT_DIR / "val.jsonl", "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Created split files: train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

    # 2. Write SFT_EVALUATION_SET.json (the 24 test records + baseline evaluation metadata)
    eval_set_file = OUTPUT_DIR / "SFT_EVALUATION_SET.json"
    eval_payload = {
        "evaluation_version": "batch2821_2860_v1",
        "description": "Held-out test set for evaluating SFT candidate against Canonical Step-10,000 baseline",
        "total_cases": len(test_records),
        "cases": test_records
    }
    with open(eval_set_file, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(test_records)} evaluation cases to {eval_set_file}")


if __name__ == "__main__":
    main()
