import json
from pathlib import Path
import sys

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.conversation.cr_exp_17.conversation_formatter import ConversationFormatter

tok, _ = load_tokenizer_artifacts(str(REPO_ROOT / "data/experiments/vocab_4096"))
formatter = ConversationFormatter(tok, max_seq_len=512)

# Load 20 diverse conversations from train dataset
train_path = REPO_ROOT / "experiments/conversation/cr_exp_17/data/conversation_train_1080.jsonl"
samples = []
with open(train_path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(json.loads(line))

sample_audits = []
for idx, s in enumerate(samples[:20]):
    prompt = s["prompt"]
    resp = s["response"]
    
    full_ids, labels = formatter.prepare_training_pair(prompt, resp)
    prompt_ids = tok.encode(prompt.strip())
    resp_ids = tok.encode(resp.strip()) + [1]
    
    # Check alignment
    masked_count = sum(1 for x in labels if x == -100)
    supervised_count = sum(1 for x in labels if x != -100)
    supervised_tokens = [x for x in labels if x != -100]
    decoded_supervised = tok.decode([x for x in supervised_tokens if x != 1])
    
    sample_audits.append({
        "sample_id": idx + 1,
        "turns_in_prompt": prompt.count("User:"),
        "prompt_length_chars": len(prompt),
        "prompt_tokens_count": len(prompt_ids),
        "response_tokens_count": len(resp_ids),
        "full_tokens_count": len(full_ids),
        "masked_count": masked_count,
        "supervised_count": supervised_count,
        "decoded_supervised": decoded_supervised,
        "supervised_matches_response": (decoded_supervised.strip() == resp.strip()),
        "truncated": len(full_ids) >= 512,
        "prompt_preview": prompt.replace("\n", " -- "),
        "response_preview": resp
    })

with open(REPO_ROOT / "experiments/cognition/cr_exp_18/diagnostics/sample_formatted_conversations.json", "w", encoding="utf-8") as f:
    json.dump(sample_audits, f, indent=2)

print(f"Audited {len(sample_audits)} sample conversations.")
perfect_matches = sum(1 for s in sample_audits if s["supervised_matches_response"])
print(f"Supervised matches response: {perfect_matches}/{len(sample_audits)}")
