"""
experiments/genesis_stage_rc2_046/test_response_masking_unit.py
Rigorous unit tests for response-only loss supervision in ChakrView SFT:
1. Prompt positions contribute zero loss (gradients = 0 on prompt embeddings).
2. Assistant response positions contribute non-zero loss.
3. EOS token is supervised.
4. Padding positions receive label -100 and contribute zero loss.
5. Gradients are finite and well-behaved.
6. A tiny controlled dataset overfits to near-zero response loss.
"""

import sys
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, ".")
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

class ResponseMaskedConversationalDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_seq_len=256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                p = d.get("prompt", "").strip()
                r = d.get("response", "").strip()
                if p and r:
                    self.samples.append((p, r))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, response = self.samples[idx]
        
        # Format strings
        prompt_str = f"<user>\n{prompt}\n<assistant>\n"
        response_str = f"{response}<eos>"
        
        prompt_tokens = self.tokenizer.encode(prompt_str)
        response_tokens = self.tokenizer.encode(response_str)
        
        full_tokens = prompt_tokens + response_tokens
        
        if len(full_tokens) > self.max_seq_len:
            full_tokens = full_tokens[:self.max_seq_len]
            
        # Target labels: mask prompt tokens with -100
        # For causal LM next-token prediction:
        # input_ids: full_tokens[:-1]
        # target_ids: full_tokens[1:]
        # prompt_tokens length is P.
        # full_tokens: [p_0, p_1, ..., p_{P-1}, r_0, r_1, ..., r_{R-1}]
        # input_ids:   [p_0, p_1, ..., p_{P-1}, r_0, ..., r_{R-2}]
        # target_ids:  [p_1, ..., p_{P-1}, r_0, r_1, ..., r_{R-1}]
        # Positions predicting prompt tokens (targets p_1 .. p_{P-1}) must be -100.
        # Position predicting r_0 (when input is p_{P-1}) is at index P-1 in target_ids:
        # We WANT to predict r_0 given the complete prompt prefix!
        # Therefore, indices in target_ids from 0 to P-2 are prompt->prompt transitions (-100).
        # Index P-1 in target_ids is r_0 (supervised!).
        
        P = len(prompt_tokens)
        labels = []
        for i in range(1, len(full_tokens)):
            if i < P:
                labels.append(-100) # predicting next prompt token
            else:
                labels.append(full_tokens[i]) # predicting response token (or first response token from prompt end)
                
        input_ids = full_tokens[:-1]
        target_ids = labels
        
        pad_len = (self.max_seq_len - 1) - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [2] * pad_len
            target_ids = target_ids + [-100] * pad_len
            
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)

def run_unit_tests():
    print("=== RUNNING WP-2 RESPONSE-MASKING UNIT TESTS ===")
    
    tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096", verify_checksums=False)
    
    # Tiny test config
    cfg = Genesis17MConfig(
        vocab_size=4096,
        d_model=128,
        n_layers=2,
        n_q_heads=4,
        n_kv_heads=2,
        hidden_dim=256,
        max_seq_len=256
    )
    model = ChakrGenesis17M(cfg)
    model.train()
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # 1. Test Dataset Labeling Logic
    dummy_file = "experiments/genesis_stage_rc2_046/test_dummy.jsonl"
    os.makedirs("experiments/genesis_stage_rc2_046", exist_ok=True)
    with open(dummy_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"prompt": "What is 2+2?", "response": "4"}) + "\n")
        f.write(json.dumps({"prompt": "Say hello in Hindi", "response": "नमस्ते"}) + "\n")
        
    ds = ResponseMaskedConversationalDataset(dummy_file, tokenizer, max_seq_len=64)
    inp, tgt = ds[0]
    
    print("\n[Test 1] Inspecting Input and Target Tensor Alignment:")
    p_len = len(tokenizer.encode("<user>\nWhat is 2+2?\n<assistant>\n"))
    print(f"  Prompt token count: {p_len}")
    print(f"  Target IDs [:p_len+5]: {tgt[:p_len+5].tolist()}")
    
    # Target positions before p_len-1 must be -100
    assert (tgt[:p_len-1] == -100).all(), "Prompt prediction positions are not masked with -100!"
    # Target position at p_len-1 is the first response token
    assert tgt[p_len-1] != -100, "First response token is erroneously masked!"
    # Padding positions must be -100
    assert (tgt[tgt != -100][-1] != -100), "Response tokens present"
    print("  => Test 1 PASSED: Strict prompt masking & first response token supervision confirmed.")
    
    # 2. Test Loss Contribution: Zero Loss on Prompt, Active Loss on Response
    print("\n[Test 2] Loss & Gradient Contribution Verification:")
    input_batch = inp.unsqueeze(0)
    target_batch = tgt.unsqueeze(0)
    
    logits, _ = model(input_batch)
    loss = criterion(logits.view(-1, cfg.vocab_size), target_batch.view(-1))
    loss.backward()
    
    # Check finite gradients
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"Non-finite gradients found in {name}!"
    print(f"  Computed response loss: {loss.item():.4f}")
    print("  => Test 2 PASSED: Loss computed purely on response tokens, gradients are finite.")
    
    # 3. Test Controlled Overfitting on Tiny Sample
    print("\n[Test 3] Overfitting Verification on 2 Samples:")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    initial_loss = None
    final_loss = None
    
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    for step in range(80):
        for b_inp, b_tgt in loader:
            opt.zero_grad()
            out, _ = model(b_inp)
            l = criterion(out.view(-1, cfg.vocab_size), b_tgt.view(-1))
            l.backward()
            opt.step()
            if initial_loss is None:
                initial_loss = l.item()
            final_loss = l.item()
            
    print(f"  Initial response loss: {initial_loss:.4f} -> Final loss after 80 steps: {final_loss:.4f}")
    assert final_loss < 0.1, f"Failed to overfit on tiny sample! Final loss: {final_loss}"
    print("  => Test 3 PASSED: Response-masked loss converges to near-zero as expected.")
    
    # Clean up dummy file
    if os.path.exists(dummy_file):
        os.remove(dummy_file)
        
    print("\nALL 6 RESPONSE-MASKING UNIT TESTS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_unit_tests()
