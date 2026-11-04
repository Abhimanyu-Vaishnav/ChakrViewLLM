# CHAKRVIEW MORNING TEST GUIDE
## Interactive Terminal Guide for ChakrView Conversational AI

Welcome! The overnight autonomous research and forensic mission has completed.
ChakrView is ready for immediate live conversation in your terminal using the verified, CPU-first fine-tuned model (`sft_checkpoint_step_50.pt`).

---

### 1. Launching Interactive Chat

To start chatting right away in PowerShell:

```powershell
.\.venv\Scripts\python.exe chakrview/chat.py
```

To run the automated verification test:
```powershell
.\.venv\Scripts\python.exe chakrview/chat.py --test-mode
```

---

### 2. Available Terminal Commands

Inside the chat prompt (`User > `):
- `/reset`: Clears the conversation history buffer.
- `/info`: Displays active model path, SHA-256 hash, parameter count, and decoding settings.
- `/exit`: Cleanly terminates the session.

---

### 3. Recommended Scenarios to Try

#### Scenario 1: Greetings & Small Talk
```text
User > Good morning assistant!
User > How are you?
User > Hello there!
```

#### Scenario 2: Identity & Provenance
```text
User > Who are you?
User > What is your name?
User > What can you do?
```

#### Scenario 3: Hinglish Interaction
```text
User > Namaste!
User > Aapka naam kya hai?
```

#### Scenario 4: User Context & Fact Retention
```text
User > My name is Abhimanyu.
User > What is my name?
```

---

### 4. Active Checkpoint Verification
- **Path:** `experiments/cbr01_sft/runs/cbr_sft_07/checkpoints/sft_checkpoint_step_50.pt`
- **SHA-256:** `0cb5df5fa840d20cea737da47758d96cfe884782c9801f28e51660bb236de86b`
- **Parameters:** 3,443,136 (ChakrMicro 6 layers, $d_{model}=192$, tied embeddings)
- **Decoding Configuration:** `temperature=0.0`, `repetition_penalty=1.25`, `no_repeat_ngram_size=3`, `sliding_window=1`
