# Starting Point Audit: Batches 541–580
## Neural Intent, Structured Understanding & Tool-Use Learning

**Audit Date:** 2026-10-15  
**Repository:** `D:\Project\ChakrView`  
**Current Branch:** `dev/v0.6-v0.7-validation`  
**Head Commit:** `749ba8c` — `feat(batch501-540): audit scratchpad leakage, train multi-task rule curriculum, and evaluate 60-item generalization`  
**Previous Milestone:** `749ba8c` (Matches expected starting point)

---

### 1. Repository State & Working Tree Audit

#### 1.1 Git Status & Uncommitted Work
- `chakrview/runtime/session.py` has uncommitted modifications:
  - Formats conversational history into canonical `ConversationTemplate` ChatML format (`ChatMessage`, `Role`).
  - Principle #3 strictly dictates: **Preserve user modifications in `chakrview/runtime/session.py`**.
  - No changes will be made to `session.py`. All runtime integrations will occur via external, testable adapters (`chakrview/runtime/session_adapter.py`, `chakrview/runtime/neural_intent_adapter.py`, or `CognitiveRouter`).
- Untracked files exist from prior sprints (such as `docs/sprints/batch103_...`, `tests/test_batch103_...`, etc.). These are preserved untouched.

#### 1.2 Checkpoint SHA-256 Verifications
- **Canonical Checkpoint:** `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - Expected SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
  - Measured SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
  - Status: **VERIFIED IMMUTABLE**
- **Previous Evaluated Candidate:** `experiments/batch501_540/checkpoints/candidate_step200_compositional_rule.pt`
  - Expected SHA-256: `b18e69ebf4f0b396465af3c323f38bcfa84d5a38eabbd4a4220a10d6bfafee98`
  - Measured SHA-256: `b18e69ebf4f0b396465af3c323f38bcfa84d5a38eabbd4a4220a10d6bfafee98`
  - Status: **VERIFIED MATCH**

---

### 2. Hardware and Environment Inspection

- **Operating System:** Windows 11 (64-bit)
- **CPU Cores:** 20 logical cores
- **System RAM:** 31.64 GB total, ~18.57 GB available
- **Python Runtime (.venv_gpu):** Python 3.12, PyTorch 2.14.1+cu126
- **GPU Accelerator:** NVIDIA RTX A1000 6GB Laptop GPU (5.999 GB VRAM available)
- **Training Feasibility:** Training the ~3.4M parameter `ChakrMicro` model on CUDA with batch size 1 to 4 takes <30 seconds for 200–300 steps.

---

### 3. Architecture & Runtime Integration Analysis

#### 3.1 Neural Model Core
- **Model:** `ChakrMicro` (`chakrview/brain/model.py`), ~3.4M parameters.
  - Vocabulary: 4,096 tokens (BPE tokenizer under `release/chakrview-v0.5.0/data/experiments/vocab_4096`).
  - Sequence Length: Up to 256 tokens.
  - Special tokens: `<bos>` = 0, `<eos>` = 1, `<pad>` = 2.
- **Key Finding from Batches 501–540:**
  - The 3.4M neural core cannot perform exact multi-digit compositional arithmetic in autoregressive text (0/10 on novel compositions).
  - However, it achieves strong semantic paraphrase recognition (75%) and clean EOS termination (>93%).
  - Therefore, the neural core is ideally suited for **semantic intent understanding, slot/argument extraction, and cognitive tool routing**, delegating symbolic execution to deterministic governed tools.

#### 3.2 Runtime Request Flow
The current execution path in `GroundedSessionAdapter` (`chakrview/runtime/session_adapter.py`):
1. **User Input:** Received by `execute_turn(session_id, user_input)`.
2. **Cognitive Router:** `CognitiveRouter.classify_route(query)` classifies into routes (`DIRECT_CHAT`, `CONTEXT_QA`, `CALCULATOR_TOOL`, `CONCEPT_DEFINITION`, `USER_MEMORY`, `KNOWLEDGE_RETRIEVAL`, `INSUFFICIENT_EVIDENCE`).
   - *Limitation Identified:* Currently, `CognitiveRouter` uses static regexes and keyword matching (`_extract_math_expression`, `_is_slot_query`, `_extract_concept_definition`).
   - If user phrasing diverges, or in ambiguous/multi-turn contexts, regex rules fail.
3. **Evidence Gate:** `EvidenceGovernedAnsweringGate.process_query(query)` checks for false premises (e.g., "Capital of Mars" -> REFUTE) and indexes document chunks.
4. **Governed Tool Subsystem:** `ToolExecutor` and `ToolRegistry` (`chakrview/runtime/tools.py`):
   - Allowlisted tools: `CalculatorTool` (`tool_id="calculator"`), `TextUtilityTool` (`tool_id="text_utility"`).
   - Enforces strict AST validation for math; zero filesystem/shell authority.
5. **Attribution & Provenance:** Every response produces a `GroundedResponse` with `route_decision`, `component_origin`, `is_supported`, and `support_explanation`.

---

### 4. Integration Strategy for Batches 541–580

1. **Versioned Structured Neural Intent Contract:**
   Define a compact, robust JSON format that the 3.4M neural core can reliably generate within 60–100 tokens:
   ```json
   {"intent": "<INTENT>", "tool": "<TOOL_OR_NULL>", "args": { ... }, "status": "ready"|"clarify"|"unsupported"}
   ```
2. **Supported Intent Set (Mapped to ChakrView Capabilities):**
   - `calculator` -> tool: `calculator`, args: `{"expression": "<EXPR>"}`
   - `text_utility` -> tool: `text_utility`, args: `{"operation": "<OP>", "text": "<STR>"}`
   - `knowledge_lookup` -> tool: `retrieval`, args: `{"query": "<STR>"}`
   - `direct_chat` -> tool: `null`, conversational response
   - `concept_definition` -> tool: `null` / `knowledge`, args: `{"concept": "<CONCEPT>"}`
   - `clarification_needed` -> status: `clarify`, reason for ambiguity
   - `unsupported` -> status: `unsupported`, reason for abstention
3. **Neural Intent Predictor Component:**
   Build `NeuralIntentRouter` / `NeuralStructuredPredictor` that interfaces between user input, neural decoding, schema validation, and `GroundedSessionAdapter`.
4. **Safety Boundaries:**
   - Predictions must strictly pass JSON schema validation.
   - Tool names must match allowlisted tools in `ToolRegistry`.
   - Arguments must be typed and bounded.
   - Raw output is never executed directly.
