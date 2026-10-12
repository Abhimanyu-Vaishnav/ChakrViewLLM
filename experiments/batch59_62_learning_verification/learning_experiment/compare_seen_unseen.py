import torch
from chakrview.brain.model import ChakrMicro
from chakrview.brain.config import ModelConfig
from experiments.batch42_reproducible_evaluation.evaluation.evaluator import load_tokenizer_artifacts

tokenizer, _ = load_tokenizer_artifacts("release/chakrview-v0.5.0/data/experiments/vocab_4096")
ckpt_path = "experiments/batch51_54_correctness_sprint/learning_experiment/diagnostic_reproduction_step_25.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

model = ChakrMicro(ModelConfig(max_seq_len=512))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Prompt 1: Seen in training (2 + 3)
prompt = "<|im_start|>user\nWhat is 2 + 3?<|im_end|>\n<|im_start|>assistant\n"
p_toks = [0] + tokenizer.encode(prompt)
x = torch.tensor([p_toks], dtype=torch.long)
logits = model(x)[0, -1, :]
probs = torch.softmax(logits, dim=-1)
top_p, top_idx = torch.topk(probs, 5)

print("Overfit diagnostic checkpoint top-5 predictions for SEEN 'What is 2 + 3?':")
for p, idx in zip(top_p.tolist(), top_idx.tolist()):
    tok_str = tokenizer.decode([idx])
    print(f"  Token {idx:4d} ({tok_str!r:15s}): {p*100:.2f}%")

# Prompt 2: Unseen (3 + 4)
prompt_diff = "<|im_start|>user\nWhat is 3 + 4?<|im_end|>\n<|im_start|>assistant\n"
p_toks2 = [0] + tokenizer.encode(prompt_diff)
x2 = torch.tensor([p_toks2], dtype=torch.long)
logits2 = model(x2)[0, -1, :]
probs2 = torch.softmax(logits2, dim=-1)
top_p2, top_idx2 = torch.topk(probs2, 5)

print("\nOverfit diagnostic checkpoint top-5 predictions for UNSEEN 'What is 3 + 4?':")
for p, idx in zip(top_p2.tolist(), top_idx2.tolist()):
    tok_str = tokenizer.decode([idx])
    print(f"  Token {idx:4d} ({tok_str!r:15s}): {p*100:.2f}%")
