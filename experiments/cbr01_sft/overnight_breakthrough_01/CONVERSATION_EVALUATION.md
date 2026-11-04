# CHAKRVIEW REAL CONVERSATION EVALUATION
## Multi-Turn Transcripts, Empirical Benchmarks & Failure Analysis

---

### 1. Direct Multi-Turn Conversation Transcripts

Evaluated on `sft_checkpoint_step_50.pt` under controlled decoding (`repetition_penalty=1.25`, `no_repeat_ngram_size=3`):

#### Conversation A: Greetings, Identity & Capabilities
```text
User > Hello ChakrView!
ChakrView > Hello! I am ready to help you today?

User > How are you?
ChakrView > I am ChakrView, an indigenous neural language model developed from first principles.

User > What is your name?
ChakrView > My name is ChakrView.

User > What can you do?
ChakrView > I am ChakrView, an indigenous neural language model developed from first principles.

User > Goodbye!
ChakrView > Good afternoon! What can I assist you?
```

#### Conversation B: Hinglish & Multilingual Grounding
```text
User > Namaste!
ChakrView > Namaste! Main aapki kya madad kar sakta hoon?

User > Aapka naam kya hai?
ChakrView > Mera naam ChakrView hai. Main ek indigenous neural language model hoon.

User > Kya aap CPU par chalte hain?
ChakrView > Haan, main natively commodity CPU hardware par execute hota hoon.
```

#### Conversation C: Context Retention & User Facts
```text
User > My name is Abhimanyu.
ChakrView > Good evening! I am ready to assist with your questions will be the context.

User > What is my name?
ChakrView > Your name is Abhimanyu.
```

---

### 2. Failure Analysis & Honest Limitations

1. **Repetitive Reflex on Generic Follow-Ups:**
   When questions fall outside the direct vocabulary representations (e.g. asking deep architectural questions like `"Explain what backpropagation does in gradient descent"`), the model falls back to identity restatements (`"Hello! I am ChakrView..."`) rather than hallucinating external pretraining knowledge.
2. **Context Horizon:**
   In multi-turn chat, preserving more than 2 historical turns causes cross-attention interference due to the small parameter capacity (3.44M parameters, 6 heads, $d_{model}=192$).
3. **Copy-Grounding:**
   Verbatim arbitrary hex string copying remains a known challenge, pointing to the structural need for the planned 20.8M parameter scaling (`docs/research/CHAKRVIEW_20M_SCALING_PROPOSAL.md`).
