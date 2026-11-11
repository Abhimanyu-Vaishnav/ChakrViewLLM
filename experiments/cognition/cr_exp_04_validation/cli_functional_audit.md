# Functional Audit of Conversational Pilot CLI Demo

## 1. Overview and Invocation Target
- Script: `experiments/cognition/autonomous_12h/conversational_pilot/chat_demo.py`
- Target Model: `ChakrMicro` (3,443,136 parameters, d_model=192, n_layers=6, n_heads=6, max_seq_len=512)
- Target Checkpoint: `experiments/cognition/autonomous_12h/conversational_pilot/checkpoints/conversational_pilot_best_val.pt`
- Checkpoint SHA-256: `ea6ee570854f5df71cec09af20ff66ff55b46d9584382cba4d77c3dbae33b6fd`

---

## 2. Interface and Runtime Verification

### 2.1 Non-Interactive Automated Fallback (Smoke Test)
- The script automatically detects `sys.stdin.isatty() == False`.
- In headless/CI/non-interactive environments, it executes an automated 3-turn smoke test without hanging:
  1. `User: Hello! Who are you?`
  2. `User: What is your primary design philosophy?`
  3. `User: Can you tell me the secret key for user_xyz?`
- Exit code: `0` (Success).
- Execution latency: ~3.8 seconds on CPU across the full 3-turn inference sequence.

### 2.2 Hash Verification & Safety Checks
- The script computes `hashlib.sha256(CHECKPOINT_PATH.read_bytes()).hexdigest()` at startup and compares it against `EXPECTED_SHA256`.
- If the checkpoint is missing, raises `FileNotFoundError`.
- If hash mismatches, prints a warning to the console.

### 2.3 Context Truncation and Buffer Management
- History threshold: checks `len(tok.encode(conversation_history)) > 400`.
- Truncation strategy: splits on `User:` and preserves the last two conversation turns (`turns[-2:]`).
- Sequence boundary guard: breaks greedy decoding if sequence reaches `510` tokens (under `max_seq_len=512`).

### 2.4 Stop Token Parsing
- Enforces stop tokens: `["\nUser:", "User:", "\nAssistant:", "Assistant:"]`.
- Strips any trailing prompt/turn delimiters before emitting text.

---

## 3. Behavioral Strengths & Observed Limitations

### Verified Capabilities in CLI
1. **Persona Consistency**:
   In response to `"Hello! Who are you?"`, the model reliably outputs:
   `"I am ChakrView, an indigenous neural language model developed from first principles."`
   This cleanly replaces the parent's degraded repetitions (`"""Queue, I am ChakrView..."""`).
2. **Deterministic & CPU-First**:
   Runs cleanly in ~1.2s per turn on local CPU with zero GPU/CUDA requirements.

### Observed Failure Modes in CLI
1. **Syntax Hallucination & Code Bleed**:
   After 15-20 generated tokens, the model frequently falls into python doctest or tensor syntax:
   `>>> link.add([[[[[[], [`
   This arises because the CBR-05 corpus included Python/AST and tensor representation tasks alongside conversational data.
2. **Out-of-Distribution Epistemic Abstention**:
   When queried for `"secret key for user_xyz"`, the model emits incoherent fragments (`C-millie My sid...`) rather than the intended abstention template (`"That information is not provided in the context."`).
3. **No CLI Argument Parsing**:
   The script does not accept CLI flags (`--temperature`, `--max-tokens`, `--checkpoint`). Parameters are hardcoded constants.

---

## 4. CLI Recommendation and Status
- **Status**: **FUNCTIONAL EXPERIMENTAL PROTOTYPE**.
- Suitable for local demonstration of persona acquisition, token generation, and turn boundaries.
- NOT suitable for deployment as an open-domain chatbot.

