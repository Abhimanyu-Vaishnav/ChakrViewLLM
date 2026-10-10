# ChakrView — Stage D Computer Systems QA Founder Review Packet (Batch 1 Focus: 25 Train-Origin Candidates)

**Document Reference:** `docs/reviews/STEP_09_FOUNDER_REVIEW_PACKET.md`  
**Official Review Queue:** `checkpoints/experiments/neural_learning_v02/stage_d_design/qa_review_queue_v0.3.0.jsonl`  
**Audited Advisory (v0.2):** `checkpoints/experiments/neural_learning_v02/stage_d_design/qa_review_advisory_batch1_v0.2.json`  
**Tracked Review Tool:** `tools/qa_review/founder_review_workflow.py`  
**Total Candidates in Queue:** 30  
**Scope of this Packet:** Exactly the **25 potentially reviewable train-origin candidates**.

---

## 1. Executive Summary & Authoritative Candidate Accounting

All candidate counts and split memberships in this packet are independently derived from candidate records in `qa_review_queue_v0.3.0.jsonl` and `qa_review_advisory_batch1_v0.2.json`:

| Category / Pool | Count | Candidate IDs | Status & Restrictions |
| :--- | :---: | :--- | :--- |
| **In-Scope: Direct Grounding (`APPROVE`)** | **16** | `QA-REV-001`, `003`, `004`, `005`, `007`, `008`, `011`, `012`, `014`, `015`, `016`, `018`, `022`, `024`, `026`, `027` | Direct verbatim support from source passage; verified file & line provenance. |
| **In-Scope: Source-Supported Revisions (`REVISE`)** | **9** | `QA-REV-002`, `006`, `009`, `010`, `013`, `019`, `020`, `021`, `025` | Technical flaw in draft question/answer corrected to strictly reflect what the source passage actually establishes. |
| **Blocked: Pending Source Enrichment** | **2** | `QA-REV-017`, `QA-REV-023` | **BLOCKED:** Single-sentence sources combine concepts without defining individual roles (HTTP/2 vs HTTP/3/QUIC; hashing vs encryption vs signatures). Requires reference enrichment before review. |
| **Out-of-Scope: Validation Split Only** | **2** | `QA-REV-028` (revise), `QA-REV-029` (approve) | **TRAINING INELIGIBLE:** Originates from `data/processed/stage_b/validation/english.txt`. Evaluation consideration only. |
| **Out-of-Scope: Test Split Only** | **1** | `QA-REV-030` (`APPROVE_TEST_ONLY`) | **TRAINING INELIGIBLE:** Originates from `data/processed/stage_b/test/english.txt`. Test benchmark evaluation only. |
| **Total Candidates in Batch 1** | **30** | Exactly 30 unique records | Official queue status: **30/30 UNREVIEWED (100%)**. |

---

## 2. Founder Review Instructions & Invariants

1. **Non-Negotiable Rule:** Assistant advisory recommendations (`APPROVE`, `REVISE`, `NEEDS_NEW_SOURCE`) are technical guidance only. **Zero items are approved** until the founder records an explicit decision.
2. **Permitted Decisions:**
   - `APPROVE` — Item admitted into the verified train-eligible pool. Requires a non-empty rationale.
   - `REJECT` — Candidate discarded permanently from training pools. Requires a non-empty rationale.
   - `REVISE` — Candidate requires wording or scope adjustment before admission. Requires a non-empty rationale.
3. **Recording Decisions:** The founder can record decisions by running the interactive review tool:
   ```powershell
   .venv\Scripts\python tools/qa_review/founder_review_workflow.py --interactive
   ```
   or by editing the decision fields in this document and applying them via the review tool's export/import workflow.

---

## 3. Part I: Direct Grounding Train-Origin Candidates (16 Items)

The following 16 candidates exhibit direct, unambiguous lexical grounding in their cited source passages.

---

### Candidate 1: QA-REV-001 (operating_systems)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 8)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `APPROVE` (`SOURCE_GROUNDED_DIRECT`)
- **Advisory Analysis:** Exact lexical grounding in passage text; non-generic and factual.

**Source Passage:**
> Operating systems manage virtual memory through page tables, translating virtual addresses to physical RAM addresses via the MMU.

**Draft Question:**
> How do operating systems manage virtual memory and address translation?

**Draft Answer:**
> Operating systems manage virtual memory through page tables, translating virtual addresses to physical RAM addresses via the MMU.

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 2: QA-REV-003 (operating_systems)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 47)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `APPROVE` (`SOURCE_GROUNDED_DIRECT`)
- **Advisory Analysis:** Exact factual grounding on scheduler allocation criteria.

**Source Passage:**
> Operating system schedulers allocate CPU execution slices based on process priority, affinity, and latency targets.

**Draft Question:**
> What criteria do operating system schedulers use to allocate CPU execution slices?

**Draft Answer:**
> Operating system schedulers allocate CPU execution slices based on process priority, affinity, and latency targets.

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 3: QA-REV-004 (operating_systems)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 48)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `APPROVE` (`SOURCE_GROUNDED_DIRECT`)
- **Advisory Analysis:** Direct grounding on context switch overhead factors.

**Source Passage:**
> Context switching between operating system processes incurs overhead from saving CPU register state and invalidating address translation caches.

**Draft Question:**
> What operations cause runtime overhead during an operating system context switch?

**Draft Answer:**
> Context switching between operating system processes incurs overhead from saving CPU register state and invalidating address translation caches.

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 4: QA-REV-005 (cpu_architecture)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 49)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `APPROVE` (`SOURCE_GROUNDED_DIRECT`)
- **Advisory Analysis:** Standard pipeline stages supported verbatim by passage.

**Source Passage:**
> Instruction pipelining overlaps the execution of consecutive instructions across fetch, decode, execute, memory, and writeback stages.

**Draft Question:**
> What execution stages comprise a standard instruction pipelining architecture?

**Draft Answer:**
> Instruction pipelining overlaps the execution of consecutive instructions across fetch, decode, execute, memory, and writeback stages.

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 5: QA-REV-007 (cpu_architecture)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 51)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `APPROVE` (`SOURCE_GROUNDED_DIRECT`)
- **Advisory Analysis:** Direct grounding on branch predictor and BTB micro-architecture.

**Source Passage:**
> Modern branch predictors use branch history tables and return stack buffers to minimize control hazard penalties.

**Draft Question:**
> What micro-architectural components minimize processor pipeline branch misprediction penalties?

**Draft Answer:**
> Modern branch predictors use branch history tables and return stack buffers to minimize control hazard penalties.

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 6: QA-REV-008 (cpu_architecture)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 52)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `APPROVE` (`SOURCE_GROUNDED_DIRECT`)
- **Advisory Analysis:** Direct grounding on out-of-order execution mechanisms (reservation stations, reorder buffers).

**Source Passage:**
> Out-of-order execution cores dynamically reschedule instructions using reservation stations and commit results in program order via reorder buffers.

**Draft Question:**
> What hardware mechanisms do modern out-of-order execution engines employ to manage instructions?

**Draft Answer:**
> Out-of-order execution cores dynamically reschedule instructions using reservation stations and commit results in program order via reorder buffers.

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 7: QA-REV-011 (memory_systems)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 55)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `APPROVE` (`SOURCE_GROUNDED_DIRECT`)
- **Advisory Analysis:** Direct grounding on TLB translation caching.

**Source Passage:**
> Translation lookaside buffers cache recent virtual-to-physical address mappings to avoid expensive page table walks on memory access.

**Draft Question:**
> How do translation lookaside buffers (TLBs) speed up memory address translation?

**Draft Answer:**
> Translation lookaside buffers cache recent virtual-to-physical address mappings to avoid expensive page table walks on memory access.

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 8: QA-REV-012 (memory_systems)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 56)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `APPROVE` (`SOURCE_GROUNDED_DIRECT`)
- **Advisory Analysis:** Direct grounding on memory hierarchy latency vs capacity trade-offs.

**Source Passage:**
> Multi-level memory hierarchies balance access latency against storage capacity across L1, L2, L3 caches, main memory, and non-volatile storage.

**Draft Question:**
> What trade-offs do computer memory hierarchies balance through multi-tiered caching?

**Draft Answer:**
> Multi-level memory hierarchies balance access latency against storage capacity across L1, L2, L3 caches, main memory, and non-volatile storage.

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 9: QA-REV-014 (memory_systems)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 58)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `APPROVE` (`SOURCE_GROUNDED_DIRECT`)
- **Advisory Analysis:** Direct grounding on DMA CPU bypass mechanics.

**Source Passage:**
> Direct memory access controllers transfer bulk data between peripheral devices and system memory without continuous CPU intervention.

**Draft Question:**
> How does direct memory access (DMA) optimize bulk data transfers in a computer system?

**Draft Answer:**
> Direct memory access controllers transfer bulk data between peripheral devices and system memory without continuous CPU intervention.

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 10: QA-REV-015 (networking)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 59)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `APPROVE` (`SOURCE_GROUNDED_DIRECT`)
- **Advisory Analysis:** Direct grounding on TCP ordered delivery and congestion control.

**Source Passage:**
> Transmission Control Protocol provides reliable, ordered, and error-checked delivery of byte streams across IP networks using sliding window flow control.

**Draft Question:**
> What transmission guarantees does Transmission Control Protocol (TCP) provide?

**Draft Answer:**
> Transmission Control Protocol provides reliable, ordered, and error-checked delivery of byte streams across IP networks using sliding window flow control.

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 11: QA-REV-016 (networking)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 60)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `APPROVE` (`SOURCE_GROUNDED_DIRECT`)
- **Advisory Analysis:** Direct grounding on UDP stateless low-latency profile.

**Source Passage:**
> User Datagram Protocol prioritizes low latency over reliability, omitting connection handshake and retransmission for real-time traffic.

**Draft Question:**
> Why is User Datagram Protocol (UDP) preferred for real-time telemetry and streaming?

**Draft Answer:**
> User Datagram Protocol prioritizes low latency over reliability, omitting connection handshake and retransmission for real-time traffic.

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 12: QA-REV-018 (networking)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 62)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `APPROVE` (`SOURCE_GROUNDED_DIRECT`)
- **Advisory Analysis:** Direct grounding on message broker decoupling in event systems.

**Source Passage:**
> Event-driven message brokers decouple producers from consumers through durable topic logs and consumer group offset management.

**Draft Question:**
> What architectural mechanism do event-driven systems use to decouple producers from consumers?

**Draft Answer:**
> Event-driven message brokers decouple producers from consumers through durable topic logs and consumer group offset management.

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 13: QA-REV-022 (databases)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 66)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `APPROVE` (`SOURCE_GROUNDED_DIRECT`)
- **Advisory Analysis:** Direct grounding on query optimizer cost modeling.

**Source Passage:**
> Relational database query optimizers estimate execution plan costs using catalog statistics, index cardinalities, and cost model heuristics.

**Draft Question:**
> What inputs do database query optimizers use to construct relational algebra execution plans?

**Draft Answer:**
> Relational database query optimizers estimate execution plan costs using catalog statistics, index cardinalities, and cost model heuristics.

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 14: QA-REV-024 (security)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 68)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `APPROVE` (`SOURCE_GROUNDED_DIRECT`)
- **Advisory Analysis:** Direct grounding on memory safety vulnerabilities (buffer overflows, UAF).

**Source Passage:**
> Memory safety bugs such as buffer overflows, use-after-free, and dangling pointers represent a primary attack vector in unmanaged systems software.

**Draft Question:**
> What common memory safety vulnerabilities compromise the security of unmanaged computer systems?

**Draft Answer:**
> Memory safety bugs such as buffer overflows, use-after-free, and dangling pointers represent a primary attack vector in unmanaged systems software.

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 15: QA-REV-026 (compilers)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 70)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `APPROVE` (`SOURCE_GROUNDED_DIRECT`)
- **Advisory Analysis:** Direct grounding on compiler optimization passes (SSA, inlining, DCE).

**Source Passage:**
> Modern optimizing compilers apply static single assignment, dead code elimination, and loop vectorization passes to intermediate representations.

**Draft Question:**
> What optimizations do compilers apply to improve generated machine code efficiency?

**Draft Answer:**
> Modern optimizing compilers apply static single assignment, dead code elimination, and loop vectorization passes to intermediate representations.

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 16: QA-REV-027 (compilers)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 71)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `APPROVE` (`SOURCE_GROUNDED_DIRECT`)
- **Advisory Analysis:** Direct grounding on GC mechanisms (tracing vs reference counting).

**Source Passage:**
> Garbage collectors reclaim dynamically allocated heap memory through tracing algorithms or reference counting strategies.

**Draft Question:**
> What two fundamental strategies do garbage collection algorithms use to reclaim memory?

**Draft Answer:**
> Garbage collectors reclaim dynamically allocated heap memory through tracing algorithms or reference counting strategies.

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

## 4. Part II: Source-Supported Revision Candidates (9 Items)

The following 9 candidates contain drafts that either over-generalized, invited external extrapolation, or misstated technical distinctions. Each includes a corrected question and answer strictly anchored in the cited source text.

---

### Candidate 17: QA-REV-002 (operating_systems)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 46)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `REVISE` (`SOURCE_SUPPORTED_REVISION`)
- **Technical Issue:** Draft question asked how hardware privilege rings protect memory, but the passage states virtual memory enables isolated address spaces protected by privilege rings without detailing privilege ring mechanics.
- **Source Limitation:** Bounded to what virtual memory and privilege rings provide; does not extrapolate ring 0/ring 3 transitions.

**Source Passage:**
> Virtual memory allows processes to operate within isolated address spaces protected by hardware privilege rings.

**Original Draft:**
- *Q:* How do hardware privilege rings protect process memory in an operating system?
- *A:* Virtual memory allows processes to operate within isolated address spaces protected by hardware privilege rings.

**Proposed Revision:**
- *Rev Q:* **What mechanism allows operating system processes to operate in protected, isolated address spaces?**
- *Rev A:* **Virtual memory allows processes to operate within isolated address spaces protected by hardware privilege rings.**

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 18: QA-REV-006 (cpu_architecture)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 50)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `REVISE` (`SOURCE_SUPPORTED_REVISION`)
- **Technical Issue:** Draft asked for "three types of hazards" using textbook terminology (data, structural, control). Passage lists data dependencies, structural collisions, and control branch mispredictions.
- **Source Limitation:** Anchored directly in the text's explicit phrasing.

**Source Passage:**
> Hazard detection units resolve data dependencies, structural collisions, and control branch mispredictions.

**Original Draft:**
- *Q:* What three types of hazards do processor hazard detection units resolve?
- *A:* Hazard detection units resolve data dependencies, structural collisions, and control branch mispredictions.

**Proposed Revision:**
- *Rev Q:* **What pipeline execution hazards do processor hazard detection units resolve?**
- *Rev A:* **Hazard detection units resolve data dependencies, structural collisions, and control branch mispredictions.**

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 19: QA-REV-009 (cpu_architecture)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 53)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `REVISE` (`SOURCE_SUPPORTED_REVISION`)
- **Technical Issue:** Draft asked what distinguishes register files from other memory structures, prompting comparative claims not found in the single-sentence passage.
- **Source Limitation:** Bounded strictly to single-cycle latency and intermediate operand storage.

**Source Passage:**
> Register files store intermediate operands with single-cycle access latency.

**Original Draft:**
- *Q:* What latency characteristic distinguishes processor register files from other memory structures?
- *A:* Register files store intermediate operands with single-cycle access latency.

**Proposed Revision:**
- *Rev Q:* **What access latency and functional purpose characterize processor register files?**
- *Rev A:* **Register files store intermediate operands with single-cycle access latency.**

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 20: QA-REV-010 (memory_systems)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 54)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `REVISE` (`SOURCE_SUPPORTED_REVISION`)
- **Technical Issue:** Draft conflated cache coherence with memory consistency guarantees. Cache coherence protocols (MESI/MOESI) ensure uniform memory views across caches for individual memory locations; memory consistency models govern cross-location ordering.
- **Source Limitation:** Confines claim strictly to uniform memory views across threads as stated in the text.

**Source Passage:**
> Cache coherence protocols such as MESI and MOESI ensure uniform memory views across parallel execution threads.

**Original Draft:**
- *Q:* What memory consistency guarantee do cache coherence protocols like MESI and MOESI ensure across parallel threads?
- *A:* Cache coherence protocols such as MESI and MOESI ensure uniform memory views across parallel execution threads.

**Proposed Revision:**
- *Rev Q:* **What memory view do cache coherence protocols such as MESI and MOESI ensure across parallel execution threads?**
- *Rev A:* **Cache coherence protocols such as MESI and MOESI ensure uniform memory views across parallel execution threads.**

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 21: QA-REV-013 (memory_systems)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 57)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `REVISE` (`SOURCE_SUPPORTED_REVISION`)
- **Technical Issue:** Draft asked broadly what "speed advantage" L1 caches provide. The passage specifies sub-nanosecond access latency for frequently referenced memory lines.
- **Source Limitation:** Restricted specifically to sub-nanosecond access latency for frequently referenced lines.

**Source Passage:**
> Level 1 instruction and data caches provide sub-nanosecond access for frequently referenced memory lines.

**Original Draft:**
- *Q:* What speed advantage do Level 1 instruction and data caches provide to processors?
- *A:* Level 1 instruction and data caches provide sub-nanosecond access for frequently referenced memory lines.

**Proposed Revision:**
- *Rev Q:* **What access latency performance do Level 1 instruction and data caches provide for frequently referenced memory lines?**
- *Rev A:* **Level 1 instruction and data caches provide sub-nanosecond access for frequently referenced memory lines.**

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 22: QA-REV-019 (databases)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 63)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `REVISE` (`SOURCE_SUPPORTED_REVISION`)
- **Technical Issue:** Draft asked how Raft/Paxos ensure "consistency" across networks. The passage specifically states they ensure "state machine replication across unreliable networks".
- **Source Limitation:** Strictly aligned with state machine replication across unreliable networks.

**Source Passage:**
> Distributed consensus protocols like Raft and Paxos ensure state machine replication across unreliable networks.

**Original Draft:**
- *Q:* How do distributed consensus protocols like Raft and Paxos ensure consistency across networks?
- *A:* Distributed consensus protocols like Raft and Paxos ensure state machine replication across unreliable networks.

**Proposed Revision:**
- *Rev Q:* **What operational mechanism do distributed consensus protocols like Raft and Paxos ensure across unreliable networks?**
- *Rev A:* **Distributed consensus protocols like Raft and Paxos ensure state machine replication across unreliable networks.**

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 23: QA-REV-020 (databases)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 64)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `REVISE` (`SOURCE_SUPPORTED_REVISION`)
- **Technical Issue:** Draft asked how DBMSs implement ACID transaction properties, which invites a general textbook treatise. Passage specifies two concrete mechanisms: WAL and MVCC.
- **Source Limitation:** Direct grounding on WAL and MVCC without claiming exhaustive coverage of all ACID properties.

**Source Passage:**
> Database management systems implement ACID properties using write-ahead logging and multi-version concurrency control.

**Original Draft:**
- *Q:* How do database management systems implement ACID transaction properties?
- *A:* Database management systems implement ACID properties using write-ahead logging and multi-version concurrency control.

**Proposed Revision:**
- *Rev Q:* **What mechanisms do database management systems use to implement ACID properties according to the source passage?**
- *Rev A:* **Database management systems implement ACID properties using write-ahead logging and multi-version concurrency control.**

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 24: QA-REV-021 (databases)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 65)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `REVISE` (`SOURCE_SUPPORTED_REVISION`)
- **Technical Issue:** Draft asked how on-disk access characteristics differentiate B-trees and LSM trees, which implies an explicit differential comparison not present in the joint sentence.
- **Source Limitation:** Strictly reflects logarithmic search and sequential writes as defined in the source text.

**Source Passage:**
> B-trees and log-structured merge trees organize on-disk records for logarithmic search and sequential writes.

**Original Draft:**
- *Q:* What on-disk access characteristics differentiate B-trees and log-structured merge (LSM) trees?
- *A:* B-trees and log-structured merge trees organize on-disk records for logarithmic search and sequential writes.

**Proposed Revision:**
- *Rev Q:* **How do B-trees and log-structured merge (LSM) trees organize on-disk records for search and write operations?**
- *Rev A:* **B-trees and log-structured merge trees organize on-disk records for logarithmic search and sequential writes.**

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

### Candidate 25: QA-REV-025 (compilers)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 69)
- **Source Hash:** `65c1d78c3e93c0a41c98d9c9662c9c617e44f12171a63cb45bbb0b924543380d`
- **Advisory Recommendation:** `REVISE` (`SOURCE_SUPPORTED_REVISION`)
- **Technical Issue:** Draft asked in the singular what technique BLAS libraries employ, whereas the text identifies three techniques: loop unrolling, register blocking, and cache tiling.
- **Source Limitation:** Reflects all three optimization techniques stated in the passage.

**Source Passage:**
> Linear algebra routines implemented in BLAS libraries utilize loop unrolling, register blocking, and cache tiling to saturate hardware.

**Original Draft:**
- *Q:* What technique do BLAS libraries employ to saturate hardware memory bandwidth and execution units?
- *A:* Linear algebra routines implemented in BLAS libraries utilize loop unrolling, register blocking, and cache tiling to saturate hardware.

**Proposed Revision:**
- *Rev Q:* **What optimization techniques do linear algebra routines in BLAS libraries utilize to saturate hardware?**
- *Rev A:* **Linear algebra routines implemented in BLAS libraries utilize loop unrolling, register blocking, and cache tiling to saturate hardware.**

**Founder Decision:** `_______________` *(Options: APPROVE | REJECT | REVISE)*  
**Founder Rationale / Notes:** `____________________________________________________________________`

---

## 5. Part III: Train-Origin Candidates Blocked for Source Enrichment (2 Items)

These two items originate from the training split but are **BLOCKED from founder approval** because the cited single sentences combine complex concepts without explaining individual component roles.

---

### Blocked 1: QA-REV-017 (networking)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 61)
- **Status:** **BLOCKED (`NEEDS_NEW_SOURCE`)**
- **Why Blocked:** Source passage states: *"Application protocols such as HTTP/2 and HTTP/3 introduce multiplexing, header compression, and QUIC transport."* In computer networking, HTTP/2 introduced multiplexing and HPACK header compression over TCP, whereas HTTP/3 introduced QUIC transport over UDP. Claiming HTTP/2 introduced QUIC is a critical technical error; however, disentangling them introduces facts not present in the citation.
- **Enrichment Requirement:** Author or cite a multi-sentence reference passage in `stage_b` sources that explicitly defines HTTP/2 (TCP/HPACK/multiplexing) and HTTP/3 (UDP/QUIC) separately before this QA item can be admitted.

---

### Blocked 2: QA-REV-023 (security)
- **Split:** `train`
- **Source Locator:** `data/processed/stage_b/train/english.txt` (line 67)
- **Status:** **BLOCKED (`NEEDS_NEW_SOURCE`)**
- **Why Blocked:** Source passage states: *"Cryptographic primitives like SHA-256, AES-GCM, and elliptic curve signatures secure communication channels."* The original question asked what secures channels against "tampering and interception". SHA-256 is an unkeyed hash function providing integrity; AES-GCM provides authenticated encryption (confidentiality + integrity); ECDSA provides authentication and digital signatures. SHA-256 alone does NOT provide confidentiality. The source sentence does not define their distinct roles.
- **Enrichment Requirement:** Enrich source passage to explicitly delineate that AES-GCM provides authenticated encryption (confidentiality + integrity), SHA-256 provides collision-resistant hashing, and elliptic curves provide digital signatures before admission.

---

## 6. Part IV: Evaluation-Only Candidates (3 Items)

These three candidates must remain permanently excluded from the training set.

1. **QA-REV-028 (memory_systems):** Validation split (`data/processed/stage_b/validation/english.txt`, line 4). Defines memory-bound operations. Evaluation pool only.
2. **QA-REV-029 (cpu_architecture):** Validation split (`data/processed/stage_b/validation/english.txt`, line 5). Defines RISC-V modular ISA. Evaluation pool only.
3. **QA-REV-030 (memory_systems):** Test split (`data/processed/stage_b/test/english.txt`, line 4). Cache misses and TLB latency penalties. Test benchmark evaluation only.
