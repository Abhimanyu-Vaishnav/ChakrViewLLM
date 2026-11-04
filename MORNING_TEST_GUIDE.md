# CHAKRVIEW MORNING TEST GUIDE
## Hands-On Interactive Testing Guide for the Conversational ChakrView Model

Welcome back! The overnight autonomous research and training mission (CBR-SFT-07) has completed.
A local, CPU-first conversational ChakrView model is compiled, verified, and ready for you to chat with right now in your terminal.

---

### 1. Launching the Interactive Chat

To start talking with ChakrView immediately, open your terminal in `D:\Project\ChakrView` and execute:

```powershell
.\.venv\Scripts\python.exe chakrview/chat.py
```

To run an automated health check / smoke test:
```powershell
.\.venv\Scripts\python.exe chakrview/chat.py --test-mode
```

---

### 2. Built-In CLI Commands

Inside the chat prompt (`User > `), you can use the following commands:
- `/reset`: Clears the conversation history buffer and resets context.
- `/info`: Displays active model path, SHA-256 hash, and current history turn count.
- `/exit`: Cleanly terminates the session.

---

### 3. Recommended Test Scenarios

Try the following prompts to observe the current capabilities and behavior:

#### Scenario A: Greetings & Social Interaction
```text
User > Good morning assistant!
User > Good afternoon!
User > Hello there, how are you today?
```

#### Scenario B: Identity & Provenance
```text
User > Who are you?
User > What model architecture do you run on?
User > Are you based on ChatGPT or Llama?
```

#### Scenario C: Multi-Turn Conversation
```text
User > I am working on project Falcon.
User > What project am I working on?
```

#### Scenario D: General Computing QA & Explanations
```text
User > What does CPU stand for?
User > What is an operating system?
```

---

### 4. Verified Model Checkpoint Details

- **Active Model File:** `experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_50.pt`
- **Architecture:** ChakrMicro (4 layers, 8 heads, $d_{model}=512$, context 512, 3.44M params)
- **Verified SHA-256:** `0cb5df5fa840d20cea737da47758d96cfe884782c9801f28e51660bb236de86b`
- **Parent Ancestor:** CBR-SFT-06A Step 25 (`0589fd886853a2e65ffa1c03af0e4708cb7dc3dfd40b4db6ccc179c5bf1045f1`)

