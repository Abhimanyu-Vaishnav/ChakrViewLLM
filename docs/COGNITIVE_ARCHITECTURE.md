# ChakrView Cognitive Architecture

## 1. System Vision

ChakrView is not a monolithic prompt-in-prompt-out neural model. It is designed as a **hybrid modular cognitive architecture** where a compact neural language core operates alongside deterministic cognitive subsystems.

In large-scale AI, factual precision and arithmetic reasoning are often forced into billions of statistical parameters. In ChakrView, we leverage a 5.18M neural core for linguistic fluency and contextual synthesis, while delegating high-stakes calculations, episodic recall, and safety verification to deterministic cognitive modules.

```
                         ┌─────────────────────────────┐
                         │      User Query Input       │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   Deterministic Router &    │
                         │    Intent Classifier        │
                         └──────┬───────────────┬──────┘
                                │               │
          ┌─────────────────────┘               └─────────────────────┐
          ▼                                                           ▼
┌───────────────────────────┐                               ┌───────────────────────────┐
│ Specialized Subsystems    │                               │     Neural Pathway        │
│ ───────────────────────── │                               │ ───────────────────────── │
│ • Principled Abstention   │                               │ • Conversation History    │
│ • Arithmetic Evaluator    │                               │ • Episodic Memory Buffer  │
│ • System Info & Identity  │                               │ • ChakrGenesis SFT Core   │
│ • Deterministic Fallback  │                               │ • Repetition Suppressors  │
└─────────┬─────────────────┘                               └─────────────┬─────────────┘
          │                                                               │
          └─────────────────────┐               ┌─────────────────────────┘
                                ▼               ▼
                         ┌─────────────────────────────┐
                         │    Output Selector &        │
                         │    Factuality Verifier      │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │       Final Response        │
                         └─────────────────────────────┘
```

---

## 2. Core Cognitive Modules

### 2.1 Neural Language Core (`ChakrGenesis`)
- **Role:** Generative fluency, dialogue continuity, bilingual synthesis (English/Hindi/Hinglish).
- **Execution Mode:** Autoregressive sampling with top-k, nucleus filtering, and 3-gram repetition blocking.
- **Isolation Principle:** The neural core is invoked strictly when open-ended linguistic interaction is required. It is evaluated independently in blind benchmarks to ensure no deterministic masking disguises neural degradation.

### 2.2 Intent Router & Policy Controller
- **Role:** Inspects incoming tokens and structural features to select execution pathways.
- **Routing Rules:**
  1. *Direct Identity/Personal Privacy:* Intercepted by `Principled Abstention` policy.
  2. *Strict Mathematical Arithmetic:* Extracted and routed to verified Python arithmetic parser.
  3. *General Conversation / Knowledge Retrieval:* Routed to `ChakrGenesis SFT Core`.

### 2.3 Episodic Working Memory Buffer
- **Role:** Maintains multi-turn context across conversational exchanges.
- **Structure:** A FIFO buffer storing recent user and assistant turns:
  ```
  Turn N-1: <user> ... </user> <assistant> ... </assistant>
  Turn N:   <user> ... </user>
  ```
- **Context Truncation:** Tokens beyond the 256-token context window are progressively truncated using sliding-window attention boundaries.

### 2.4 Principled Abstention & Safety Guard
- **Role:** Protects user privacy and handles queries where the model must not hallucinate (e.g., *"What is my password?"*, *"Tell me private financial data"*).
- **Mechanism:** Deterministic policy checks enforcing refusal templates without passing through generative uncertainty.

---

## 3. Dual-Path Execution Model

ChakrView operates in two distinct evaluation and production modes:

| Mode | Neural Core | Memory Injection | Deterministic Routing | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **Pure Neural Mode** | Active (100%) | Disabled | Disabled | Independent scientific evaluation of neural weights, measuring raw language model perplexity and reasoning. |
| **Hybrid Cognitive Mode** | Active (Fallback/Synth) | Active | Active | End-user production deployment in web/CLI interfaces, delivering guaranteed factual reliability on math and privacy. |

---

## 4. Future Cognitive Modules (Roadmap)

1. **Span-Pointer Memory:** A dynamic key-value pointer network over recent conversational turns, allowing exact extraction of names, numbers, and facts without neural hallucination.
2. **Tool-Use Actuators:** Structured output parser allowing ChakrGenesis to invoke local bash/python tools via `<thought>` and `<tool_call>` control tokens.
3. **Self-Consistency Verifier:** Multi-sample generation voter that checks mathematical or code candidate outputs before displaying them to the user.
