# CBR-SFT-03B — Raw Generation Index

This index documents side-by-side free autoregressive outputs across the pretraining baseline, parent model, and CBR-SFT-03B checkpoints on held-out prompts.

---

## 1. Held-Out Greeting Prompts (Out-Of-Distribution)

### Task: `EVAL3B-GREET-0001`
- **User Prompt**: `User: Hi, pleasant morning!\nAssistant:`
- **Expected Concept**: `morning`
- **Canonical Step 10k Baseline**:
  ```text
  \n|-\n| scope=&quot;row&quot; | \n
  ```
- **CBR-SFT-02 Step 50 Parent**:
  ```text
  The first of the word of the word of the word of the word of the word
  ```
- **CBR-SFT-03B Step 100**:
  ```text
  Good morning! How can I help you today?
  ```
- **CBR-SFT-03B Step 200**:
  ```text
  Good morning! How can I help you today?
  ```
- **Outcome**: **PASSED**. Generalizes immediately from `"pleasant morning"` to the appropriate morning greeting response.

---

### Task: `EVAL3B-GREET-0004`
- **User Prompt**: `User: Greetings assistant!\nAssistant:`
- **Expected Concept**: `greetings`
- **Canonical Step 10k Baseline**:
  ```text
  \n| style=&quot;background-color: #ffffff
  ```
- **CBR-SFT-02 Step 50 Parent**:
  ```text
  Calling the word of the word of the word of the word of the word of
  ```
- **CBR-SFT-03B Step 200**:
  ```text
  Greetings! How can I be the first item you w
  ```
- **CBR-SFT-03B Step 300**:
  ```text
  Greetings! How can I be of service to you right now
  ```
- **Outcome**: **PASSED**. The model emits the proper greeting token sequence and establishes conversational posture.

---

### Task: `EVAL3B-GREET-0006`
- **User Prompt**: `User: Hello, are you operational?\nAssistant:`
- **Expected Concept**: `online`
- **Canonical Step 10k Baseline**:
  ```text
  \n\nThe Special of the United States\n\nThe Special of
  ```
- **CBR-SFT-02 Step 50 Parent**:
  ```text
  The first of the word of the world of the world.
  ```
- **CBR-SFT-03B Step 200**:
  ```text
  Yes, I am online and ready to assist you. What is on your
  ```
- **Outcome**: **PASSED**. Correctly understands readiness probe and responds in affirmative conversational mode.

---

## 2. Held-Out Identity Prompts

### Task: `EVAL3B-IDENT-0001`
- **User Prompt**: `User: What is your identity?\nAssistant:`
- **Expected Concept**: `ChakrView`
- **Canonical Step 10k Baseline**:
  ```text
  \n\nThe first of the first of the first of the first of the first
  ```
- **CBR-SFT-02 Step 50 Parent**:
  ```text
  The first of the first of the first of the first of the first
  ```
- **CBR-SFT-03B Step 200**:
  ```text
  My name is ChakrView.
  ```
- **Outcome**: **PASSED**. Concise, exact identity attribution without repetitive looping.

---

### Task: `EVAL3B-IDENT-0002`
- **User Prompt**: `User: Identify yourself.\nAssistant:`
- **Expected Concept**: `ChakrView`
- **Canonical Step 10k Baseline**:
  ```text
  \n|-\n| scope=&quot;row&quot; | \n
  ```
- **CBR-SFT-03B Step 200**:
  ```text
  I am ChakrView, a modular and language model cognitive neural language model and natively on commod
  ```
- **CBR-SFT-03B Step 300**:
  ```text
  I am ChakrView, a modular and hardware-efficient neural cognitive model designed for native
  ```
- **Outcome**: **PASSED**. The model correctly articulates its system identity and architectural designation.

---

### Task: `EVAL3B-IDENT-0004`
- **User Prompt**: `User: Who developed you?\nAssistant:`
- **Expected Concept**: `research`
- **CBR-SFT-03B Step 200**:
  ```text
  I was built from first principles as an indigenous research initiative.
  ```
- **Outcome**: **PASSED**. Accurately describes origins and foundational philosophy.

---

## 3. Sample Failed Tasks & Analysis

### Task: `EVAL3B-GREET-0002` (Temporal Incongruity)
- **User Prompt**: `User: Good morning to you!\nAssistant:`
- **Expected Concept**: `morning`
- **CBR-SFT-03B Step 200**:
  ```text
  Good day! How can I assist you today?
  ```
- **Scoring**: Failed automated regex (`'morning'` required), but qualitatively coherent and valid as a conversational response.

### Task: `EVAL3B-GREET-0008` (Evening Collapse)
- **User Prompt**: `User: Good evening everyone!\nAssistant:`
- **Expected Concept**: `evening`
- **CBR-SFT-03B Step 200**:
  ```text
  Good morning! How can I help you today?
  ```
- **Scoring**: Failed. The model has a dominant morning greeting attractor due to the small size of the 20-greeting training slice, causing it to default to morning when evening is prompted.

### Task: `EVAL3B-QA-0007` (Knowledge Beyond Core Dialogue)
- **User Prompt**: `User: What is the capital of France?\nAssistant:`
- **Expected Concept**: `Paris`
- **CBR-SFT-03B Step 200**:
  ```text
  I am ChakrView, an indigenous neural language model cognitive model.
  ```
- **Scoring**: Failed. The small 50-example steering dataset only contained conversational readiness examples; factual QA queries fall back onto the strongest dialogue identity attractor.
