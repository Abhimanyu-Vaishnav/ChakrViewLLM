"""
Batch 166: Controlled Neural Copy-Induction Pilot for ChakrMicro.
Compares:
- Arm A: Canonical Base Checkpoint (Step 10,000) fine-tuned with copy-induction loss masking
- Arm B: Untuned Canonical Base Checkpoint (Zero-Shot Control)

Evaluates on held-out test facts (entities and combinations never seen during fine-tuning).
Preserves the canonical checkpoint completely intact; saves experimental weights to separate path.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.conversation import ConversationTemplate, ChatMessage, Role
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
DATA_PATH = Path("experiments/batch161_170_neural_learning/data/split_manifest.json")
EXP_CKPT_PATH = Path("experiments/batch161_170_neural_learning/checkpoints/copy_induction_arm_a.pt")
REPORTS_DIR = Path("experiments/batch161_170_neural_learning/reports")


def build_chatml_example(context: str, query: str, answer: str, tokenizer):
    messages_prompt = [
        ChatMessage(role=Role.USER, content=f"Context: {context}\nQuestion: {query}")
    ]
    prompt_str = ConversationTemplate.format_conversation(messages_prompt, add_generation_prompt=True)
    full_str = prompt_str + answer + "<|im_end|>"
    
    prompt_tokens = tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
    full_tokens = tokenizer.encode(full_str, add_bos=True, add_eos=False)
    
    # Input is all tokens except last
    input_ids = full_tokens[:-1]
    # Target is all tokens shifted by 1
    target_ids = full_tokens[1:]
    
    # Mask prompt positions to -100 (shift by 1 accounts for input/target alignment)
    prompt_len = len(prompt_tokens) - 1
    labels = [-100] * len(target_ids)
    for i in range(prompt_len, len(target_ids)):
        labels[i] = target_ids[i]
        
    return input_ids, labels, prompt_str


def run_copy_induction_pilot(epochs=50, lr=5e-5):
    device = torch.device("cpu")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    train_data = manifest["splits"]["train"]
    val_data = manifest["splits"]["val"]
    test_data = manifest["splits"]["test"]
    
    # Load canonical base model
    cfg = ModelConfig(max_seq_len=512)
    model_arm_a = ChakrMicro(cfg).to(device)
    base_data = torch.load(CANONICAL_BASE_CKPT, map_location=device, weights_only=False)
    model_arm_a.load_state_dict(base_data["model_state_dict"])
    
    # Keep Arm B as unmodified zero-shot control
    model_arm_b = copy.deepcopy(model_arm_a)
    model_arm_b.eval()
    
    optimizer = optim.AdamW(model_arm_a.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Prepare training sequences
    train_sequences = []
    for item in train_data:
        inp, tgt, p_str = build_chatml_example(item["doc"], item["query"], item["value"], tokenizer)
        train_sequences.append((inp, tgt))
        
    t0 = time.perf_counter()
    model_arm_a.train()
    
    for ep in range(epochs):
        ep_loss = 0.0
        optimizer.zero_grad()
        for inp, tgt in train_sequences:
            x = torch.tensor([inp], dtype=torch.long, device=device)
            y = torch.tensor([tgt], dtype=torch.long, device=device)
            logits = model_arm_a(x)
            loss = loss_fn(logits.view(-1, 4096), y.view(-1))
            loss.backward()
            ep_loss += loss.item()
        optimizer.step()
        
    train_time = time.perf_counter() - t0
    print(f"Arm A Training finished in {train_time:.2f}s | Final step loss: {ep_loss / len(train_sequences):.4f}")
    
    # Save Arm A checkpoint
    torch.save({
        "model_state_dict": model_arm_a.state_dict(),
        "parent_checkpoint": str(CANONICAL_BASE_CKPT),
        "epochs": epochs,
        "lr": lr,
    }, EXP_CKPT_PATH)
    
    # Generation Evaluation
    def generate_answer(model, context, query, max_new_tokens=16):
        model.eval()
        messages = [
            ChatMessage(role=Role.USER, content=f"Context: {context}\nQuestion: {query}")
        ]
        prompt = ConversationTemplate.format_conversation(messages, add_generation_prompt=True)
        tokens = tokenizer.encode(prompt, add_bos=True, add_eos=False)
        curr = list(tokens)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                inp = torch.tensor([curr], dtype=torch.long, device=device)
                logits = model(inp)
                next_tok = int(torch.argmax(logits[0, -1, :]).item())
                if next_tok in (1, 3):  # EOS or <|im_end|>
                    break
                curr.append(next_tok)
        gen_tokens = curr[len(tokens):]
        return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    def evaluate_model(model, data_split):
        exact_matches = 0
        samples = []
        for item in data_split:
            gen_text = generate_answer(model, item["doc"], item["query"])
            target = item["value"].strip()
            is_match = (target.lower() in gen_text.lower())
            if is_match:
                exact_matches += 1
            samples.append({
                "entity": item["entity"],
                "query": item["query"],
                "target": target,
                "generated": gen_text[:60],
                "match": is_match,
            })
        em_rate = exact_matches / len(data_split) if data_split else 0.0
        return em_rate, samples

    arm_b_em, arm_b_samples = evaluate_model(model_arm_b, test_data)
    arm_a_em, arm_a_samples = evaluate_model(model_arm_a, test_data)
    
    results = {
        "milestone": "Batch 166 Controlled Neural Copy-Induction Pilot",
        "training_time_sec": round(train_time, 2),
        "held_out_test_cases": len(test_data),
        "arm_b_control_zero_shot": {
            "exact_match_rate": round(arm_b_em, 4),
            "samples": arm_b_samples,
        },
        "arm_a_copy_sft": {
            "exact_match_rate": round(arm_a_em, 4),
            "samples": arm_a_samples,
        }
    }
    
    out_file = REPORTS_DIR / "copy_induction_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results written to {out_file}")
    print(f"Arm B (Control) Test EM: {arm_b_em*100:.1f}% | Arm A (Copy-SFT) Test EM: {arm_a_em*100:.1f}%")


if __name__ == "__main__":
    run_copy_induction_pilot()
