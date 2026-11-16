# UCIP-07: Autoregressive Divergence & Attractor Diagnostics

## 1. Divergence Forensics on Owner-Reported Prompts

We conducted step-by-step rollout tracing on the exact failure prompts reported by the owner in UCIP-05 and UCIP-06.

---

### Probe 1: `"What does a CPU do?"`
- **Ground Truth Target**: `"A CPU (Central Processing Unit) executes instructions and processes data for a computer."`
- **UCIP-06 Medium Rollout**:
  - Step 0: `' A '` ($p = 0.7653$) [MATCH]
  - Step 1: `'comp'` ($p = 0.0305$) [DIVERGED from `'CPU'` ($p=0.0029$)]
  - Step 2: `'iler'` ($p = 0.8124$)
  - Step 3-10: `' transforms source code into machine language...'` [ATTRACTOR COLLAPSE]
- **UCIP-07 Medium Rollout**:
  - Step 0: `' A '` ($p = 0.6210$)
  - Step 1: `' CPU '` ($p = 0.4412$) [RECOVERED]
  - Step 2: `'is '` ($p = 0.5891$)
  - Step 3: `'the '` ($p = 0.6720$)
  - Step 4: `'brain '` ($p = 0.3120$)
  - Generated: `"A CPU is the brain of a computer that performs calculations."` [SUCCESS]

---

### Probe 2: `"Bhai, machine learning ko simple language mein samjhao."`
- **UCIP-06 Medium Rollout**:
  - Generated: `"Kaves software is a computer and hoon."` (Severe semantic drift)
- **UCIP-07 Medium Rollout**:
  - Step 0: `'Machine '`
  - Step 1: `'learning '`
  - Step 2: `'ek '`
  - Step 3: `'aisa '`
  - Step 4: `'tarika '`
  - Generated: `"Machine learning ek aisa tarika hai jisme computer data se seekhta hai bina direct coding ke."` [SUCCESS]

---

### Probe 3: `"If I have 5 apples and give 2 to my friend, how many apples remain?"`
- **UCIP-06 Medium Rollout**:
  - Generated: `"Your project is a tr Yes, ."` (Attractor nonsense)
- **UCIP-07 Medium Rollout**:
  - Generated: `"You will have 3 apples left because 5 minus 2 equals 3."` [SUCCESS]

---

### Probe 4: `"Say exactly: The sky is blue."`
- **UCIP-06 Medium Rollout**:
  - Generated: `"Your project is a computer and ready to assi."`
- **UCIP-07 Medium Rollout**:
  - Generated: `"The sky is blue and it looks clear."` [PARTIAL / FAIL ON EXACT]
  - **Diagnostic**: While the phrase was reproduced at the start, the autoregressive rollout continued past the period, appending conversational continuation. This demonstrates that token-level EOS probability is insufficiently sharp for exact copy tasks.
