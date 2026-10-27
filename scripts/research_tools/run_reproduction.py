import sys
import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath("."))
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.brain.pointer_generator import ChakrPointerMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

VOCAB_DIR = "data/experiments/vocab_4096"
EXP25_DATA_DIR = "experiments/overnight_runs/exp25_vocabulary_generalization_diagnostic_20261024_232014/data"
VAL_DIR = "experiments/overnight_runs/exp27_independent_validation_20261025_211125"
CANONICAL_CKPT = "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt"

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_data(tok, split):
    path = os.path.join(EXP25_DATA_DIR, f"{split}.jsonl")
    recs = [json.loads(line) for line in open(path, encoding="utf-8")]
    data = []
    for r in recs:
        p_ids = tok.encode(r["prompt"])
        t_ids = tok.encode(r["target"])
        seq = p_ids + t_ids + [1]
        data.append({"record": r, "prompt_len": len(p_ids), "seq": seq})
    return data

def train_run(model, train_data, is_pointer=False, steps=300):
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    model.train()
    n = len(train_data)
    for step in range(1, steps + 1):
        item = train_data[(step - 1) % n]
        seq = torch.tensor([item["seq"]], dtype=torch.long)
        p_len = item["prompt_len"]
        inputs = seq[:, :-1]
        targets = seq[:, 1:].clone()
        targets[:, :p_len - 1] = -100

        opt.zero_grad()
        if is_pointer:
            _, _, _, probs = model(inputs, prompt_len=p_len)
            log_p = torch.log(probs + 1e-9)
            loss = F.nll_loss(log_p.view(-1, 4096), targets.view(-1), ignore_index=-100)
        else:
            logits, _ = model(inputs, return_hidden_states=True)
            loss = F.cross_entropy(logits.view(-1, 4096), targets.view(-1), ignore_index=-100)
        loss.backward()
        opt.step()

def eval_split(model, tok, split, is_pointer=False):
    model.eval()
    path = os.path.join(EXP25_DATA_DIR, f"{split}.jsonl")
    recs = [json.loads(line) for line in open(path, encoding="utf-8")]
    em, pla = 0, 0
    with torch.no_grad():
        for r in recs:
            p1 = tok.encode(r["prompt"])
            p2 = tok.encode(r["cf_prompt"])
            if is_pointer:
                _, _, _, pr1 = model(torch.tensor([p1]), prompt_len=len(p1))
                _, _, _, pr2 = model(torch.tensor([p2]), prompt_len=len(p2))
                tok1 = torch.argmax(pr1[0, -1, :]).item()
                tok2 = torch.argmax(pr2[0, -1, :]).item()
            else:
                out1 = model(torch.tensor([p1]))
                out2 = model(torch.tensor([p2]))
                lg1 = out1[0] if isinstance(out1, tuple) else out1
                lg2 = out2[0] if isinstance(out2, tuple) else out2
                tok1 = torch.argmax(lg1[0, -1, :]).item()
                tok2 = torch.argmax(lg2[0, -1, :]).item()
            t1 = tok.decode([tok1]).strip()
            t2 = tok.decode([tok2]).strip()
            is_em = (t1 == r["target"].strip())
            is_cf_em = (t2 == r["cf_target"].strip())
            if is_em: em += 1
            if is_em and is_cf_em: pla += 1
    return {"n": len(recs), "em": em, "em_rate": em / len(recs), "pla": pla, "pla_rate": pla / len(recs)}

def run_repro():
    print("=== REPRODUCING EXPERIMENT 27 (PILOT & PRETRAINED CONTROL) ===")
    tok, _ = load_tokenizer_artifacts(VOCAB_DIR)
    train_data = prepare_data(tok, "train")

    cfg = ModelConfig()
    splits = ["test_1", "test_2a", "test_2b", "test_2c"]

    # Run 1: Exact Exp27 Arm B Reproduction (Pointer from scratch)
    set_seed(42)
    ptr_model = ChakrPointerMicro(cfg)
    train_run(ptr_model, train_data, is_pointer=True, steps=300)
    ptr_results = {s: eval_split(ptr_model, tok, s, is_pointer=True) for s in splits}

    # Run 2: Exact Exp27 Arm A Reproduction (Causal LM from scratch)
    set_seed(42)
    lm_scratch = ChakrMicro(cfg)
    train_run(lm_scratch, train_data, is_pointer=False, steps=300)
    lm_scratch_results = {s: eval_split(lm_scratch, tok, s, is_pointer=False) for s in splits}

    # Run 3: Pretrained Base Control (Fine-tuned from canonical step 10,000)
    set_seed(42)
    ckpt = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
    lm_pretrained = ChakrMicro(cfg)
    lm_pretrained.load_state_dict(ckpt["model_state_dict"])
    train_run(lm_pretrained, train_data, is_pointer=False, steps=300)
    lm_pretrained_results = {s: eval_split(lm_pretrained, tok, s, is_pointer=False) for s in splits}

    repro = {
        "arm_b_pointer_from_scratch": ptr_results,
        "arm_a_causal_lm_from_scratch": lm_scratch_results,
        "arm_control_causal_lm_from_pretrained": lm_pretrained_results
    }

    out_file = os.path.join(VAL_DIR, "reproducibility_data.json")
    with open(out_file, "w") as f:
        json.dump(repro, f, indent=2)
    print("Saved reproduction data to", out_file)
    print("Arm B (Pointer):", {s: ptr_results[s]["em_rate"] for s in splits})
    print("Arm A (LM Scratch):", {s: lm_scratch_results[s]["em_rate"] for s in splits})
    print("Arm Control (Pretrained LM):", {s: lm_pretrained_results[s]["em_rate"] for s in splits})

if __name__ == "__main__":
    run_repro()
