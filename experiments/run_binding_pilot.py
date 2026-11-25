"""
NR-015 Phase 3: Train & Evaluate Controlled Binding Pilot (17M)
Tests whether fine-tuning 17M on structured variable binding allows it
to generalize to held-out variable names (delta, gamma) and unseen values.
"""

import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, '.')
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.runtime.inference_17m import ChakrViewInferenceEngine17M

class BindingDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len=128):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        text = item['text']
        # split into prompt and target
        prompt, target = text.split("<assistant>\n")
        prompt_str = prompt + "<assistant>\n"
        target_str = target

        prompt_ids = self.tokenizer.encode(prompt_str)
        target_ids = self.tokenizer.encode(target_str)
        # Append EOS
        eos_id = getattr(self.tokenizer, 'eos_id', 1) or 1
        target_ids.append(eos_id)

        input_ids = prompt_ids + target_ids
        # Label masking: -100 for prompt tokens
        labels = [-100] * len(prompt_ids) + target_ids

        # Truncate or pad
        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        else:
            pad_id = getattr(self.tokenizer, 'pad_id', 2) or 2
            pad_len = self.max_seq_len - len(input_ids)
            input_ids += [pad_id] * pad_len
            labels += [-100] * pad_len

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'raw_item': item
        }

def train_and_eval_pilot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing on device: {device}")

    # Load canonical Stage-D baseline
    engine = ChakrViewInferenceEngine17M(
        checkpoint_path='experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt',
        device=device
    )
    model = engine.model
    tokenizer = engine.tokenizer

    train_ds = BindingDataset("data/synthetic_binding_015/train_binding.json", tokenizer)
    val_ds = BindingDataset("data/synthetic_binding_015/val_binding.json", tokenizer)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    print("\n--- Pre-training Zero-Shot Evaluation on OOD Val Set ---")
    def eval_accuracy(m):
        m.eval()
        correct = 0
        total = 0
        with open("data/synthetic_binding_015/val_binding.json", 'r', encoding='utf-8') as f:
            val_items = json.load(f)
        
        for item in val_items:
            prompt, _ = item['text'].split("<assistant>\n")
            prompt_str = prompt + "<assistant>\n"
            p_ids = torch.tensor(tokenizer.encode(prompt_str), dtype=torch.long, device=device).unsqueeze(0)
            
            with torch.no_grad():
                curr = p_ids
                for _ in range(10): # max 10 tokens for answer
                    out, _ = m(curr)
                    next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)
                    curr = torch.cat([curr, next_tok], dim=1)
                    if next_tok.item() == 1: # EOS
                        break
            gen_text = tokenizer.decode(curr[0, p_ids.size(1):].tolist()).strip()
            total += 1
            if item['answer'] in gen_text:
                correct += 1
        return correct, total, (correct / total) * 100

    init_c, init_tot, init_acc = eval_accuracy(model)
    print(f"Pre-training OOD Val Exact Accuracy: {init_c}/{init_tot} ({init_acc:.1f}%)")

    # Train for 2 epochs (150 steps)
    model.train()
    step = 0
    max_steps = 150
    running_loss = 0.0

    print(f"\n--- Training on 1,200 Synthetic Binding Samples ({max_steps} steps) ---")
    for epoch in range(2):
        for batch in train_loader:
            step += 1
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits, _ = model(input_ids)
            shift_logits = logits[..., :-1, :].contiguous().view(-1, 4096)
            shift_labels = labels[..., 1:].contiguous().view(-1)
            loss = loss_fn(shift_logits, shift_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            if step % 30 == 0:
                print(f"Step {step:3d} | Train Loss: {running_loss/30:.4f}")
                running_loss = 0.0
            if step >= max_steps:
                break
        if step >= max_steps:
            break

    print("\n--- Post-training Evaluation on OOD Val Set (Unseen Variables & Ranges) ---")
    post_c, post_tot, post_acc = eval_accuracy(model)
    print(f"Post-training OOD Val Exact Accuracy: {post_c}/{post_tot} ({post_acc:.1f}%)")

    results = {
        "pre_train_acc": init_acc,
        "post_train_acc": post_acc,
        "pre_correct": init_c,
        "post_correct": post_c,
        "total_val": init_tot,
        "steps": max_steps
    }
    with open("data/synthetic_binding_015/pilot_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("Pilot experiment results saved successfully!")

if __name__ == '__main__':
    train_and_eval_pilot()
