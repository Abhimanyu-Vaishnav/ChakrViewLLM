import json
from pathlib import Path
import sys

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from chakrview.chat import ConversationalChatCLI

# Audit the runtime prompt assembly directly from chakrview/chat.py
chat_cli = ConversationalChatCLI.__new__(ConversationalChatCLI)
chat_cli.max_history_turns = 1
chat_cli.history = []

# Simulate multi-turn dialogue
turns = [
    ("Hello! My name is Tanvi.", "Hello Tanvi! How can I help?"),
    ("I live in Pune.", "Pune is a great city."),
    ("I work on data science.", "Data science is fascinating."),
    ("What was my name?", "")
]

prompts_at_each_turn = []
for idx, (u, a) in enumerate(turns):
    p = chat_cli.build_prompt(u)
    prompts_at_each_turn.append({
        "turn_index": idx + 1,
        "user_message": u,
        "built_prompt": p,
        "built_prompt_escaped": p.replace("\n", " -- "),
        "turns_in_history_before_call": len(chat_cli.history)
    })
    if a:
        chat_cli.history.append({"user": u, "assistant": a})

with open(REPO_ROOT / "experiments/cognition/cr_exp_18/diagnostics/context_input_traces.json", "w", encoding="utf-8") as f:
    json.dump(prompts_at_each_turn, f, indent=2)

audit_md = f"""# CR-EXP-18: Phase I — Runtime and Inference Context Audit Report

## 1. Runtime Architecture Inspection (`chakrview/chat.py`)
Direct source code inspection of `chakrview/chat.py` reveals the explicit context management mechanism:
```python
# Lines 48-50 in chakrview/chat.py:
self.history: List[Dict[str, str]] = []
# In a 3.44M model, keeping the most recent turn maintains context without token saturation
self.max_history_turns = 1

# Lines 71-75:
def build_prompt(self, user_msg: str) -> str:
    prompt_parts = []
    for turn in self.history[-self.max_history_turns:]:
        prompt_parts.append(f"User: {{turn['user']}}\\nAssistant: {{turn['assistant']}}")
    prompt_parts.append(f"User: {{user_msg}}\\nAssistant:")
    return "\\n".join(prompt_parts)
```

## 2. Definitive Finding: Runtime Truncation Policy
1. **Hard Truncation Ceiling (`max_history_turns = 1`):** In `chat.py`, the CLI runtime **intentionally discards all conversation history prior to the single most recent turn**.
2. **Impact on Turn 4:**
   - At Turn 1: User introduces name `Tanvi`.
   - At Turn 2: User mentions city `Pune`. History retains Turn 1.
   - At Turn 3: User mentions work `Data science`. Turn 1 (`Tanvi`) is **permanently dropped from the input window**.
   - At Turn 4: User asks `"What was my name?"`. The model receives:
     ```
     User: I work on data science.
     Assistant: Data science is fascinating.
     User: What was my name?
     Assistant:
     ```
   - **Causal Result:** The model is asked to recall a name that **physically does not exist in the prompt tensor**. The fact was pruned by `self.history[-self.max_history_turns:]`.

## 3. Discrepancy Between CLI Runtime and Evaluation Scripts
- In `chakrview/chat.py`, `max_history_turns = 1` prevents prompt saturation, but guarantees that multi-turn facts older than 1 turn cannot be retrieved.
- In `evaluate_multiturn.py`, the evaluation script passed the entire history (all 5–8 turns). But because the model was trained on data with at most 2 turns (273/281 examples had User count = 2) and saturated, it also failed when full history was passed.
"""

with open(REPO_ROOT / "experiments/cognition/cr_exp_18/audit/INFERENCE_CONTEXT_AUDIT.md", "w", encoding="utf-8") as f:
    f.write(audit_md)

print("INFERENCE_CONTEXT_AUDIT.md generated successfully.")
