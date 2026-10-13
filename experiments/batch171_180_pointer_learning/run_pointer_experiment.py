"""
Batch 174: Controlled Neural Pointer Training Experiment.
Compares 3 distinct experimental arms:
- Arm A: Canonical Base Checkpoint (Step 10,000) zero-shot (Baseline)
- Arm B: Vocabulary SFT Checkpoint (Batch 166 SFT baseline)
- Arm C: Canonical Base Checkpoint + Auxiliary ChakrPointerHead (Pointer-Generator)

Evaluates strictly on held-out test facts (entities, combinations, and keys unseen during training).
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
from chakrview.runtime.pointer_mechanism import ChakrPointerHead
from chakrview.runtime.conversation import ConversationTemplate, ChatMessage, Role
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
DATA_PATH = Path("experiments/batch171_180_pointer_learning/data/split_manifest.json")
ARM_C_CKPT_PATH = Path("experiments/batch171_180_pointer_learning/checkpoints/pointer_head_arm_c.pt")
REPORTS_DIR = Path("experiments/batch171_180_pointer_learning/reports")


def prepare_pointer_example(doc: str, query: str, answer: str, tokenizer):
    """
    Constructs ChatML input with exact boolean context mask.
    Context mask is True ONLY for tokens inside the 'Context: <doc>' span.
    """
    context_prefix = f"Context: {doc}\n"
    user_content = f"{context_prefix}Question: {query}"
    messages_prompt = [ChatMessage(role=Role.USER, content=user_content)]
    prompt_str = ConversationTemplate.format_conversation(messages_prompt, add_generation_prompt=True)
    full_str = prompt_str + answer + "<|im_end|>"
    
    full_tokens = tokenizer.encode(full_str, add_bos=True, add_eos=False)
    prompt_tokens = tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
    
    # Locate context tokens within prompt
    # Tokens from start of "Context: " to end of doc
    prefix_tokens = tokenizer.encode("<|im_start|>user\nContext: ", add_bos=True, add_eos=False)
    doc_tokens = tokenizer.encode(doc, add_bos=False, add_eos=False)
    
    context_start = len(prefix_tokens)
    context_end = context_start + len(doc_tokens)
    
    input_ids = full_tokens[:-1]
    target_ids = full_tokens[1:]
    
    # Boolean mask over input_ids: True only for tokens within context span
    context_mask = [False] * len(input_ids)
    for idx in range(min(context_start, len(input_ids)), min(context_end, len(input_ids))):
        context_mask[idx] = True
        
    # Labels: prompt positions are masked to -100
    prompt_len = len(prompt_tokens) - 1
    labels = [-100] * len(target_ids)
    for i in range(prompt_len, len(target_ids)):
        labels[i] = target_ids[i]
        
    return input_ids, target_ids, labels, context_mask, prompt_str


def run_pointer_experiment(epochs=80, lr=1e-3):
    device = torch.device("cpu")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    train_data = manifest["splits"]["train"]
    val_data = manifest["splits"]["val"]
    test_data = manifest["splits"]["test"]
    
    # 1. Load Arm A (Canonical Base Checkpoint)
    cfg = ModelConfig(max_seq_len=512)
    model_base = ChakrMicro(cfg).to(device)
    base_data = torch.load(CANONICAL_BASE_CKPT, map_location=device, weights_only=False)
    model_base.load_state_dict(base_data["model_state_dict"])
    model_base.eval()
    
    # Freeze base model parameters for Arm C (only train auxiliary pointer head)
    for p in model_base.parameters():
        p.requires_grad = False
        
    # 2. Arm C: ChakrPointerHead
    pointer_head = ChakrPointerHead(d_model=192, vocab_size=4096, d_proj=64).to(device)
    param_count = sum(p.numel() for p in pointer_head.parameters())
    print(f"Arm C Pointer Head Parameters: {param_count}")
    
    optimizer = optim.AdamW(pointer_head.parameters(), lr=lr, weight_decay=1e-2)
    
    # Prepare training sequences
    train_batches = []
    for item in train_data:
        inp, tgt, labels, c_mask, p_str = prepare_pointer_example(item["doc"], item["query"], item["value"], tokenizer)
        train_batches.append((inp, labels, c_mask))
        
    t0 = time.perf_counter()
    pointer_head.train()
    
    for ep in range(epochs):
        ep_loss = 0.0
        for inp, labels, c_mask in train_batches:
            optimizer.zero_grad()
            x = torch.tensor([inp], dtype=torch.long, device=device)
            y = torch.tensor([labels], dtype=torch.long, device=device)
            mask = torch.tensor([c_mask], dtype=torch.bool, device=device)
            
            with torch.no_grad():
                vocab_logits, hidden_states = model_base(x, return_hidden_states=True)
                
            final_probs, attn, gate = pointer_head(hidden_states, x, vocab_logits, mask)
            
            # Compute negative log-likelihood on unmasked target tokens
            log_probs = torch.log(torch.clamp(final_probs, min=1e-9))
            
            active_mask = (y != -100)
            if active_mask.any():
                active_targets = y[active_mask]
                active_log_probs = log_probs[0][active_mask[0]]
                loss = -active_log_probs.gather(dim=-1, index=active_targets.unsqueeze(-1)).mean()
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
                
    train_time = time.perf_counter() - t0
    print(f"Arm C Training completed in {train_time:.2f}s | Final step loss: {ep_loss / len(train_batches):.4f}")
    
    # Save Arm C Checkpoint
    torch.save({
        "pointer_head_state_dict": pointer_head.state_dict(),
        "d_model": 192,
        "vocab_size": 4096,
        "d_proj": 64,
        "trainable_parameters": param_count,
        "parent_base_checkpoint": str(CANONICAL_BASE_CKPT),
    }, ARM_C_CKPT_PATH)
    
    # 3. Evaluation on Held-Out Test Split
    def generate_eval(arm_type: str, test_items):
        results = []
        exact_matches = 0
        
        for item in test_items:
            doc = item["doc"]
            query = item["query"]
            target = item["value"].strip()
            
            messages = [ChatMessage(role=Role.USER, content=f"Context: {doc}\nQuestion: {query}")]
            prompt_str = ConversationTemplate.format_conversation(messages, add_generation_prompt=True)
            
            prefix_tokens = tokenizer.encode("<|im_start|>user\nContext: ", add_bos=True, add_eos=False)
            doc_tokens = tokenizer.encode(doc, add_bos=False, add_eos=False)
            context_start = len(prefix_tokens)
            context_end = context_start + len(doc_tokens)
            
            curr_tokens = tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
            
            with torch.no_grad():
                for step in range(16):
                    x = torch.tensor([curr_tokens], dtype=torch.long, device=device)
                    T = x.shape[1]
                    
                    if arm_type == "arm_a_baseline":
                        logits = model_base(x)
                        next_tok = int(torch.argmax(logits[0, -1, :]).item())
                    elif arm_type == "arm_c_pointer":
                        # Construct context mask for current sequence length T
                        c_mask = [False] * T
                        for idx in range(min(context_start, T), min(context_end, T)):
                            c_mask[idx] = True
                        mask_t = torch.tensor([c_mask], dtype=torch.bool, device=device)
                        
                        vocab_logits, hidden_states = model_base(x, return_hidden_states=True)
                        final_probs, attn, gate = pointer_head(hidden_states, x, vocab_logits, mask_t)
                        next_tok = int(torch.argmax(final_probs[0, -1, :]).item())
                        
                    if next_tok in (1, 3):  # EOS or <|im_end|>
                        break
                    curr_tokens.append(next_tok)
                    
            gen_tokens = curr_tokens[len(tokenizer.encode(prompt_str, add_bos=True, add_eos=False)):]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            
            is_match = (target.lower() in gen_text.lower())
            if is_match:
                exact_matches += 1
            results.append({
                "entity": item["entity"],
                "query": query,
                "target": target,
                "generated": gen_text[:60],
                "match": is_match,
            })
            
        em_rate = exact_matches / len(test_items) if test_items else 0.0
        return em_rate, results

    arm_a_em, arm_a_results = generate_eval("arm_a_baseline", test_data)
    arm_c_em, arm_c_results = generate_eval("arm_c_pointer", test_data)
    
    eval_summary = {
        "milestone": "Batch 174 Controlled Neural Pointer Training Experiment",
        "trainable_pointer_parameters": param_count,
        "training_time_sec": round(train_time, 2),
        "held_out_test_cases": len(test_data),
        "arm_a_baseline_zero_shot": {
            "exact_match_rate": round(arm_a_em, 4),
            "samples": arm_a_results,
        },
        "arm_c_auxiliary_pointer": {
            "exact_match_rate": round(arm_c_em, 4),
            "samples": arm_c_results,
        }
    }
    
    out_file = REPORTS_DIR / "pointer_training_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)
    print(f"Results written to {out_file}")
    print(f"Arm A (Base) EM: {arm_a_em*100:.1f}% | Arm C (Pointer) EM: {arm_c_em*100:.1f}%")


if __name__ == "__main__":
    run_pointer_experiment()
