# CR-EXP-19: Phase B — Runtime History Recovery Report

## 1. Problem Statement & Diagnosis
In `chakrview/chat.py`, `self.max_history_turns = 1` previously limited the conversational context window to the single most recent turn:
```python
# Historical code in chat.py:
for turn in self.history[-self.max_history_turns:]:
    prompt_parts.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")
```
This hard truncation caused facts introduced $\ge 2$ turns earlier to be physically dropped from the prompt before sending it to the model.

## 2. Corrected Dynamic Token-Budgeted Implementation
The runtime in `chakrview/chat.py` was refactored with dynamic token budgeting:
- **Total Context Ceiling:** 512 tokens.
- **Maximum Prompt Budget:** 384 tokens (`max_prompt_tokens = 384`).
- **Generation Reserve:** 128 tokens (`max_gen_tokens = 48` defaults, remaining space for EOS safety).
- **Turn Packing Algorithm:**
  1. Computes the token length of the current user turn (`User: {user_msg}\nAssistant:`).
  2. Traverses history backwards (from most recent turn to oldest).
  3. Greedily includes complete turns (User + Assistant) as long as total prompt tokens $\le 384$.
  4. If the budget is exhausted, drops oldest turns first.
  5. Never retains an assistant turn while dropping its corresponding user query.
  6. Returns prompt string along with an audit dictionary tracking retained and discarded turn counts.

## 3. Test Suite Verification
Executed via `experiments/cognition/cr_exp_19/scripts/test_runtime_history.py` (8/8 PASS):
- [x] One-turn conversation: Preserves prompt without history.
- [x] Two-turn conversation: Seamlessly prepends Turn 1.
- [x] Five-turn conversation: Preserves all 4 prior turns within the 384-token budget.
- [x] Eight-turn conversation: Preserves all turns within budget without token overflow.
- [x] Budget pressure: Safely evicts oldest turns when a long turn is injected.
- [x] Complete turn preservation: Never splits user from assistant.
- [x] Reset: `/reset` wipes history cleanly.
- [x] Determinism: Identical inputs yield identical assembled prompt strings.

The software runtime defect in `chakrview/chat.py` is **resolved and verified**.

