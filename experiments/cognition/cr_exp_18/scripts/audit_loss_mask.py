import json
from pathlib import Path
import sys

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.conversation.cr_exp_17.conversation_formatter import ConversationFormatter
from experiments.conversation.cr_exp_17.conversation_loss import ConversationLoss

tok, _ = load_tokenizer_artifacts(str(REPO_ROOT / "data/experiments/vocab_4096"))
formatter = ConversationFormatter(tok, max_seq_len=512)
loss_fn = ConversationLoss(ignore_index=-100)

test_scenarios = [
    ("single_turn", "User: Hi!\nAssistant:", "Hello!"),
    ("two_turn", "User: I am Alice.\nAssistant: Nice to meet you Alice.\nUser: What is my name?\nAssistant:", "Your name is Alice."),
    ("three_turn", "User: Fact 1.\nAssistant: Noted.\nUser: Fact 2.\nAssistant: Recorded.\nUser: Query.\nAssistant:", "Answer is 1 and 2."),
    ("empty_assistant_guard", "User: Hello\nAssistant:", ""),
    ("long_repetition", "User: Tell me a story\nAssistant:", "Once upon a time in a land far away there lived a king and queen.")
]

mask_audits = []
for name, prompt, resp in test_scenarios:
    full_ids, labels = formatter.prepare_training_pair(prompt, resp)
    prompt_len = len(tok.encode(prompt.strip()))
    
    # Verify properties
    is_prompt_masked = all(l == -100 for l in labels[:prompt_len])
    is_resp_supervised = all(l != -100 for l in labels[prompt_len:])
    ends_with_eos = (labels[-1] == 1)
    
    mask_audits.append({
        "scenario": name,
        "prompt": prompt.replace("\n", " "),
        "response": resp,
        "prompt_tokens_len": prompt_len,
        "total_tokens_len": len(full_ids),
        "prompt_fully_masked": is_prompt_masked,
        "response_fully_supervised": is_resp_supervised,
        "ends_with_eos": ends_with_eos,
        "supervised_tokens": [x for x in labels if x != -100]
    })

with open(REPO_ROOT / "experiments/cognition/cr_exp_18/diagnostics/token_level_masks.json", "w", encoding="utf-8") as f:
    json.dump(mask_audits, f, indent=2)

print("Loss mask audit tests complete. All scenarios checked.")
for m in mask_audits:
    print(f"[{m['scenario']}] PromptMasked={m['prompt_fully_masked']}, RespSupervised={m['response_fully_supervised']}, EndsEOS={m['ends_with_eos']}")
