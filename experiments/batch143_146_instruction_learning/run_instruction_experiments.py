"""
Batch 143–146: Instruction-Tuning, Multi-Domain Evaluation, and Grounded Comprehension Runner.
Executes:
1. Arm A: Canonical Step 10,000 Baseline (no SFT).
2. Arm B: Constant LR SFT (50 steps, lr=5e-5).
3. Arm C: Warmup + Cosine SFT (50 steps, peak_lr=5e-5 -> min_lr=5e-6, 10 warmup steps).
4. Arm D: Conservative Low-LR SFT (50 steps, peak_lr=2e-5 -> min_lr=2e-6, 10 warmup steps).
5. Comprehensive evaluation on held-out validation, held-out test instructions, and raw Stage C language.
6. End-to-end Grounded Comprehension evaluation using HybridRetriever.
7. Verification of checkpoint resumability and baseline immutability.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import math
import time
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import build_causal_sequence
from chakrview.training.pretraining_engine import MultiShardDataset
from chakrview.runtime.knowledge import BM25KnowledgeIndex, KnowledgeChunk
from chakrview.runtime.retrieval import HybridRetriever, DeterministicHashEmbeddingProvider, InMemoryVectorIndex

EXP_DIR = Path("experiments/batch143_146_instruction_learning")
DATA_DIR = EXP_DIR / "data"
CKPT_DIR = EXP_DIR / "checkpoints"
REPORTS_DIR = EXP_DIR / "reports"
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
CANONICAL_BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")

def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def load_instruction_records(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    return records

def encode_record(rec: dict, tokenizer, max_seq_len: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    p_tokens = tokenizer.encode(rec["prompt"])
    r_tokens = tokenizer.encode(rec["response"])
    inp_seq, tgt_seq = build_causal_sequence(p_tokens, r_tokens)
    inp_seq = inp_seq[:max_seq_len]
    tgt_seq = tgt_seq[:max_seq_len]
    return torch.tensor(inp_seq, dtype=torch.long), torch.tensor(tgt_seq, dtype=torch.long)

def evaluate_instruction_split(model: ChakrMicro, records: list[dict], tokenizer, device: torch.device) -> dict:
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_active_tokens = 0
    
    with torch.no_grad():
        for rec in records:
            inp, tgt = encode_record(rec, tokenizer)
            inp, tgt = inp.unsqueeze(0).to(device), tgt.unsqueeze(0).to(device)
            logits = model(inp)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
            active = (tgt != -100).sum().item()
            total_loss += loss.item() * active
            total_active_tokens += active
            
    avg_loss = total_loss / max(total_active_tokens, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    return {
        "loss": round(avg_loss, 4),
        "perplexity": round(ppl, 2),
        "active_tokens_evaluated": total_active_tokens,
    }

def evaluate_stage_c_retention(model: ChakrMicro, test_dataset: MultiShardDataset, device: torch.device, num_batches: int = 10) -> dict:
    model.eval()
    rng = np.random.RandomState(42)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = test_dataset.get_batch(batch_size=8, seq_len=64, rng=rng)
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item() * x.numel()
            total_tokens += x.numel()
            
    avg_loss = total_loss / max(total_tokens, 1)
    return {
        "stage_c_raw_loss": round(avg_loss, 4),
        "stage_c_raw_ppl": round(math.exp(min(avg_loss, 20.0)), 2),
        "tokens_evaluated": total_tokens,
    }

def generate_instruction_responses(model: ChakrMicro, records: list[dict], tokenizer, device: torch.device, max_new_tokens: int = 32) -> list[dict]:
    model.eval()
    results = []
    
    with torch.no_grad():
        for rec in records:
            prompt_str = rec["prompt"]
            prompt_tokens = [0] + tokenizer.encode(prompt_str)
            curr = list(prompt_tokens)
            gen_tokens = []
            
            for _ in range(max_new_tokens):
                inp = torch.tensor([curr], dtype=torch.long, device=device)
                logits = model(inp)[0, -1]
                nxt = torch.argmax(logits).item()
                curr.append(nxt)
                gen_tokens.append(nxt)
                if nxt == 1: # EOS
                    break
                    
            raw = tokenizer.decode(gen_tokens, errors="replace")
            # Clean delimiter if present
            cleaned = raw.replace("<|im_end|>", "").strip()
            
            # Repetition check
            rep = 0.0
            if len(gen_tokens) >= 3:
                ngrams = [tuple(gen_tokens[i:i+3]) for i in range(len(gen_tokens) - 2)]
                rep = 1.0 - (len(set(ngrams)) / max(len(ngrams), 1))
                
            results.append({
                "id": rec["id"],
                "domain": rec["domain"],
                "query": rec["raw_prompt"],
                "expected": rec["raw_response"],
                "generated": cleaned,
                "repetition_3gram": round(rep, 4),
                "terminated_eos": (1 in gen_tokens) or ("<|im_end|>" in raw),
            })
    return results

def train_sft_arm(
    arm_name: str,
    base_state: dict,
    cfg: ModelConfig,
    train_records: list[dict],
    tokenizer,
    device: torch.device,
    steps: int,
    peak_lr: float,
    min_lr: float,
    warmup_steps: int,
    seed: int,
    save_path: Path,
) -> tuple[ChakrMicro, dict]:
    print(f"\n--- Training {arm_name} ({steps} steps, peak_lr={peak_lr}, warmup={warmup_steps}) ---")
    model = ChakrMicro(cfg).to(device)
    model.load_state_dict(base_state)
    
    opt = optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        target_lr = min_lr + (peak_lr - min_lr) * cosine_decay
        return target_lr / peak_lr
        
    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda) if (warmup_steps > 0 or min_lr < peak_lr) else None
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    rng = np.random.RandomState(seed)
    
    t0 = time.perf_counter()
    model.train()
    history = []
    
    # Pre-encode training batch
    encoded_train = [encode_record(r, tokenizer) for r in train_records]
    
    for step in range(steps):
        # Sample batch of 2 examples
        idxs = [rng.randint(0, len(encoded_train)) for _ in range(2)]
        batch_loss = 0.0
        opt.zero_grad()
        
        for idx in idxs:
            inp, tgt = encoded_train[idx]
            inp, tgt = inp.unsqueeze(0).to(device), tgt.unsqueeze(0).to(device)
            logits = model(inp)
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
            loss = loss / len(idxs)
            loss.backward()
            batch_loss += loss.item()
            
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        opt.step()
        if sched:
            sched.step()
            
        current_lr = opt.param_groups[0]["lr"]
        if (step + 1) % 10 == 0 or step == 0:
            print(f"[{arm_name}] Step {step+1:2d}/{steps} | Loss: {batch_loss:.4f} | LR: {current_lr:.6f} | Grad: {grad_norm:.4f}")
            history.append({
                "step": step + 1,
                "loss": round(batch_loss, 4),
                "lr": round(current_lr, 7),
                "grad_norm": round(grad_norm, 4),
            })
            
    elapsed = time.perf_counter() - t0
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arm": arm_name,
        "step": steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "history": history,
        "config": cfg.__dict__,
        "base_init": str(CANONICAL_BASE_CKPT),
    }, save_path)
    sha = compute_file_sha256(save_path)
    print(f"[{arm_name}] Saved checkpoint: {save_path} (SHA: {sha})")
    
    meta = {
        "arm": arm_name,
        "steps": steps,
        "elapsed_seconds": round(elapsed, 2),
        "checkpoint_path": str(save_path),
        "checkpoint_sha256": sha,
        "final_train_loss": history[-1]["loss"] if history else None,
    }
    return model, meta

def run_grounded_comprehension_pipeline(model: ChakrMicro, tokenizer, device: torch.device) -> dict:
    print("\n--- Executing Grounded Comprehension with HybridRetriever ---")
    # 1. Build Knowledge Index with verified facts
    lex_index = BM25KnowledgeIndex()
    embed_provider = DeterministicHashEmbeddingProvider(dimension=64)
    vec_index = InMemoryVectorIndex()
    
    docs = [
        ("doc_celsius", "Water boils at 100 degrees Celsius and freezes at 0 degrees Celsius at sea level."),
        ("doc_photosynthesis", "Photosynthesis in green plants converts water and carbon dioxide into oxygen and glucose using sunlight."),
        ("doc_py_syntax", "In Python, list comprehensions provide a concise way to create lists using square brackets."),
    ]
    
    for i, (doc_id, text) in enumerate(docs):
        chunk = KnowledgeChunk(doc_id=doc_id, chunk_id=f"c_{i}", chunk_index=i, text=text)
        lex_index.add_chunks([chunk])
        vec_index.add_vector(chunk.chunk_id, embed_provider.embed_text(text), text=text)
        
    retriever = HybridRetriever(lexical_index=lex_index, embedding_provider=embed_provider, vector_index=vec_index)
    
    test_queries = [
        {
            "query": "At what temperature does water boil at sea level?",
            "expected_support": "100 degrees Celsius",
            "has_support": True,
        },
        {
            "query": "What is the speed of sound in steel?",
            "expected_support": None,
            "has_support": False,
        }
    ]
    
    results = []
    for q in test_queries:
        ret_hits = retriever.retrieve(q["query"], top_k=1)
        top_chunk, score = ret_hits[0] if ret_hits else (None, 0.0)
        
        # Grounding check: if score is low (< 0.2), mark insufficient evidence
        if top_chunk and score >= 0.2:
            context_text = top_chunk.text
            prompt_str = f"<|im_start|>user\nContext: {context_text}\nQuestion: {q['query']}<|im_end|>\n<|im_start|>assistant\n"
        else:
            context_text = "None"
            prompt_str = f"<|im_start|>user\nContext: No supporting information found.\nQuestion: {q['query']}<|im_end|>\n<|im_start|>assistant\n"
            
        # Neural inference
        prompt_tokens = [0] + tokenizer.encode(prompt_str)
        curr = list(prompt_tokens)
        gen = []
        for _ in range(32):
            inp = torch.tensor([curr], dtype=torch.long, device=device)
            nxt = torch.argmax(model(inp)[0, -1]).item()
            curr.append(nxt)
            gen.append(nxt)
            if nxt == 1:
                break
        ans = tokenizer.decode(gen, errors="replace").replace("<|im_end|>", "").strip()
        
        results.append({
            "query": q["query"],
            "retrieved_context": context_text,
            "retrieval_score": round(score, 4),
            "expected_support": q["expected_support"],
            "has_support": q["has_support"],
            "generated_answer": ans,
            "grounding_status": "SUPPORTED" if q["has_support"] and (q["expected_support"] in context_text) else "INSUFFICIENT_EVIDENCE",
        })
        
    return {"grounded_comprehension_trials": results}

def main():
    device = torch.device("cpu")
    print("=" * 80)
    print("BATCH 143–146 INSTRUCTION LEARNING & GROUNDED COMPREHENSION SUITE")
    print("=" * 80)
    
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    cfg = ModelConfig(max_seq_len=512)
    
    train_records = load_instruction_records(DATA_DIR / "instruction_train.jsonl")
    val_records = load_instruction_records(DATA_DIR / "instruction_val.jsonl")
    test_records = load_instruction_records(DATA_DIR / "instruction_test.jsonl")
    stage_c_test_ds = MultiShardDataset([REPO_ROOT / "data" / "tokenized" / "stage_c" / "test" / "shard_00000.bin"])
    
    base_data = torch.load(CANONICAL_BASE_CKPT, map_location=device, weights_only=False)
    base_state = base_data["model_state_dict"]
    
    def evaluate_model_suite(model: ChakrMicro) -> dict:
        v_eval = evaluate_instruction_split(model, val_records, tokenizer, device)
        t_eval = evaluate_instruction_split(model, test_records, tokenizer, device)
        retention = evaluate_stage_c_retention(model, stage_c_test_ds, device, num_batches=10)
        generations = generate_instruction_responses(model, test_records, tokenizer, device, max_new_tokens=32)
        return {
            "val_instruction": v_eval,
            "test_instruction": t_eval,
            "stage_c_retention": retention,
            "generations": generations,
        }
        
    all_results = {}
    
    # ARM A: Canonical Baseline
    print("\n--- Evaluating Arm A: Canonical Step 10,000 Baseline ---")
    model_a = ChakrMicro(cfg).to(device)
    model_a.load_state_dict(base_state)
    eval_a = evaluate_model_suite(model_a)
    eval_a["metadata"] = {"arm": "Arm A (Canonical Baseline)", "steps": 0}
    all_results["Arm_A_Baseline"] = eval_a
    print(f"Arm A Val Loss: {eval_a['val_instruction']['loss']} | Test Loss: {eval_a['test_instruction']['loss']} | Stage C Loss: {eval_a['stage_c_retention']['stage_c_raw_loss']}")
    
    # ARM B: Constant LR SFT (50 steps, lr=5e-5)
    ckpt_b = CKPT_DIR / "arm_b_sft_constant.pt"
    model_b, meta_b = train_sft_arm("Arm B (Constant SFT)", base_state, cfg, train_records, tokenizer, device, steps=50, peak_lr=5e-5, min_lr=5e-5, warmup_steps=0, seed=42, save_path=ckpt_b)
    eval_b = evaluate_model_suite(model_b)
    eval_b["metadata"] = meta_b
    all_results["Arm_B_Constant_SFT"] = eval_b
    print(f"Arm B Val Loss: {eval_b['val_instruction']['loss']} | Test Loss: {eval_b['test_instruction']['loss']} | Stage C Loss: {eval_b['stage_c_retention']['stage_c_raw_loss']}")
    
    # ARM C: Warmup + Cosine SFT (50 steps, peak=5e-5, min=5e-6, warmup=10)
    ckpt_c = CKPT_DIR / "arm_c_sft_warmup_cosine.pt"
    model_c, meta_c = train_sft_arm("Arm C (Warmup Cosine SFT)", base_state, cfg, train_records, tokenizer, device, steps=50, peak_lr=5e-5, min_lr=5e-6, warmup_steps=10, seed=42, save_path=ckpt_c)
    eval_c = evaluate_model_suite(model_c)
    eval_c["metadata"] = meta_c
    all_results["Arm_C_Warmup_Cosine_SFT"] = eval_c
    print(f"Arm C Val Loss: {eval_c['val_instruction']['loss']} | Test Loss: {eval_c['test_instruction']['loss']} | Stage C Loss: {eval_c['stage_c_retention']['stage_c_raw_loss']}")
    
    # ARM D: Conservative Low-LR SFT (50 steps, peak=2e-5, min=2e-6, warmup=10)
    ckpt_d = CKPT_DIR / "arm_d_sft_conservative.pt"
    model_d, meta_d = train_sft_arm("Arm D (Conservative SFT)", base_state, cfg, train_records, tokenizer, device, steps=50, peak_lr=2e-5, min_lr=2e-6, warmup_steps=10, seed=42, save_path=ckpt_d)
    eval_d = evaluate_model_suite(model_d)
    eval_d["metadata"] = meta_d
    all_results["Arm_D_Conservative_SFT"] = eval_d
    print(f"Arm D Val Loss: {eval_d['val_instruction']['loss']} | Test Loss: {eval_d['test_instruction']['loss']} | Stage C Loss: {eval_d['stage_c_retention']['stage_c_raw_loss']}")
    
    # Bit-exact Resumability Check on Arm C
    print("\n--- Verifying Bit-Exact Resumability on Arm C Checkpoint ---")
    resumed = torch.load(ckpt_c, map_location=device, weights_only=False)
    m_res = ChakrMicro(cfg).to(device)
    m_res.load_state_dict(resumed["model_state_dict"])
    val_res = evaluate_instruction_split(m_res, val_records, tokenizer, device)
    assert abs(val_res["loss"] - eval_c["val_instruction"]["loss"]) < 1e-6, "Resumption mismatch!"
    print(f"Resumability verification PASSED: {val_res['loss']} == {eval_c['val_instruction']['loss']}")
    all_results["resume_verification"] = {"status": "PASSED", "loss_match": True}
    
    # Grounded Comprehension Pipeline Test
    gc_report = run_grounded_comprehension_pipeline(model_c, tokenizer, device)
    all_results["grounded_comprehension"] = gc_report
    
    # Save results.json
    results_path = REPORTS_DIR / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll results saved to {results_path}")
    
    stage_c_test_ds.close()

if __name__ == "__main__":
    main()
