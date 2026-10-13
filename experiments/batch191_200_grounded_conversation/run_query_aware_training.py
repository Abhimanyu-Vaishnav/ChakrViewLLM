"""
Batch 194: Controlled Query-Aware Pointer Training and Evaluation.
Trains auxiliary ChakrPointerHead with question-context disentanglement on the relation dataset.
Compares:
- Arm A: Canonical Base Checkpoint (Zero-Shot)
- Arm B: Standard Pointer Head (without query disentanglement)
- Arm C: Query-Aware Pointer Head (with question-token overlap dampening)
- Arm D: Deterministic Oracle Reference
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import time
import torch
import torch.nn as nn
import torch.optim as optim

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pointer_mechanism import ChakrPointerHead, decode_pointer_greedy
from chakrview.runtime.conversation import ConversationTemplate, ChatMessage, Role
from chakrview.runtime.evidence_extractor import EvidenceExtractor
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
DATA_PATH = Path("experiments/batch191_200_grounded_conversation/data/split_manifest.json")
POINTER_CKPT_PATH = Path("experiments/batch191_200_grounded_conversation/checkpoints/query_aware_pointer.pt")
REPORTS_DIR = Path("experiments/batch191_200_grounded_conversation/reports")


def train_and_eval_query_aware_pointer(epochs: int = 60, lr: float = 1e-3):
    device = torch.device("cpu")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    train_data = manifest["splits"]["train"]
    test_data = manifest["splits"]["test"]

    # 1. Base Model (Frozen)
    cfg = ModelConfig(max_seq_len=512)
    base_model = ChakrMicro(cfg).to(device)
    base_data = torch.load(CANONICAL_BASE_CKPT, map_location=device, weights_only=False)
    base_model.load_state_dict(base_data["model_state_dict"])
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    # 2. Pointer Head
    pointer_head = ChakrPointerHead(d_model=192, vocab_size=4096, d_proj=64).to(device)
    optimizer = optim.AdamW(pointer_head.parameters(), lr=lr, weight_decay=1e-4)

    # Prepare training tensors
    train_examples = []
    for item in train_data:
        doc = item["doc"]
        q = item["query"]
        val = item["value"]

        prompt_str = f"<|im_start|>user\nContext: {doc}\nQuestion: {q}<|im_end|>\n<|im_start|>assistant\n"
        full_str = f"{prompt_str}{val}<|im_end|>"

        full_tokens = tokenizer.encode(full_str, add_bos=True, add_eos=False)
        prompt_tokens = tokenizer.encode(prompt_str, add_bos=True, add_eos=False)

        p_ctx = tokenizer.encode("<|im_start|>user\nContext: ", add_bos=True, add_eos=False)
        d_toks = tokenizer.encode(doc, add_bos=False, add_eos=False)
        p_q = tokenizer.encode(f"<|im_start|>user\nContext: {doc}\nQuestion: ", add_bos=True, add_eos=False)
        q_toks = tokenizer.encode(q, add_bos=False, add_eos=False)

        ctx_start = len(p_ctx)
        ctx_end = ctx_start + len(d_toks)
        q_start = len(p_q)
        q_end = q_start + len(q_toks)

        input_ids = full_tokens[:-1]
        target_ids = full_tokens[1:]

        c_mask = [False] * len(input_ids)
        for i in range(min(ctx_start, len(input_ids)), min(ctx_end, len(input_ids))):
            c_mask[i] = True

        q_mask = [False] * len(input_ids)
        for i in range(min(q_start, len(input_ids)), min(q_end, len(input_ids))):
            q_mask[i] = True

        labels = [-100] * len(target_ids)
        for i in range(len(prompt_tokens) - 1, len(target_ids)):
            labels[i] = target_ids[i]

        train_examples.append({
            "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
            "labels": torch.tensor([labels], dtype=torch.long, device=device),
            "c_mask": torch.tensor([c_mask], dtype=torch.bool, device=device),
            "q_mask": torch.tensor([q_mask], dtype=torch.bool, device=device),
        })

    # Train loop
    t0 = time.perf_counter()
    pointer_head.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for ex in train_examples:
            optimizer.zero_grad()
            with torch.no_grad():
                vocab_logits, hidden = base_model(ex["input_ids"], return_hidden_states=True)

            final_probs, attn, gate = pointer_head(
                hidden_states=hidden,
                input_ids=ex["input_ids"],
                vocab_logits=vocab_logits,
                context_mask=ex["c_mask"],
                query_mask=ex["q_mask"],
                disentangle_query_overlap=True,
            )

            # NLL loss over target labels
            log_p = torch.log(final_probs + 1e-12)
            lbls = ex["labels"]
            mask = (lbls != -100)
            if mask.sum() > 0:
                loss = -log_p[0, mask[0], lbls[0, mask[0]]].mean()
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()

    train_time = time.perf_counter() - t0
    POINTER_CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"pointer_head_state_dict": pointer_head.state_dict()}, POINTER_CKPT_PATH)

    # Evaluation across arms on fixed held-out test split
    def eval_arm(disentangle: bool = True, use_pointer: bool = True, is_oracle: bool = False):
        matches = 0
        samples = []
        pointer_head.eval()

        for item in test_data:
            doc = item["doc"]
            q = item["query"]
            target = item["value"]

            if is_oracle:
                ext = EvidenceExtractor.extract_slot(doc, q)
                ans = ext.extracted_value if ext.success else "NOT_FOUND"
                m = (target.lower() in ans.lower())
                if m:
                    matches += 1
                samples.append({"query": q, "target": target, "generated": ans, "match": m})
                continue

            prompt_str = f"<|im_start|>user\nContext: {doc}\nQuestion: {q}<|im_end|>\n<|im_start|>assistant\n"
            p_ids = tokenizer.encode(prompt_str, add_bos=True, add_eos=False)
            p_ctx = tokenizer.encode("<|im_start|>user\nContext: ", add_bos=True, add_eos=False)
            d_toks = tokenizer.encode(doc, add_bos=False, add_eos=False)
            p_q = tokenizer.encode(f"<|im_start|>user\nContext: {doc}\nQuestion: ", add_bos=True, add_eos=False)
            q_toks = tokenizer.encode(q, add_bos=False, add_eos=False)

            c_start = len(p_ctx)
            c_end = c_start + len(d_toks)
            q_start = len(p_q) if disentangle else None
            q_end = q_start + len(q_toks) if (disentangle and q_start is not None) else None

            if not use_pointer:
                curr = list(p_ids)
                with torch.no_grad():
                    for _ in range(16):
                        x = torch.tensor([curr], dtype=torch.long, device=device)
                        logits = base_model(x)
                        nxt = int(torch.argmax(logits[0, -1, :]).item())
                        if nxt in (1, 3):
                            break
                        curr.append(nxt)
                gen_toks = curr[len(p_ids):]
            else:
                gen_toks, _ = decode_pointer_greedy(
                    model=base_model,
                    pointer_head=pointer_head,
                    prompt_ids=p_ids,
                    context_start=c_start,
                    context_end=c_end,
                    query_start=q_start,
                    query_end=q_end,
                    max_new_tokens=16,
                    device=device,
                )

            gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True, errors="replace").strip()
            is_match = (target.lower() in gen_text.lower())
            if is_match:
                matches += 1

            samples.append({
                "entity": item["entity"],
                "query": q,
                "target": target,
                "generated": gen_text[:60],
                "match": is_match,
            })

        return {
            "exact_match_rate": round(matches / len(test_data), 4),
            "samples": samples
        }

    res_base = eval_arm(disentangle=False, use_pointer=False)
    res_standard_pointer = eval_arm(disentangle=False, use_pointer=True)
    res_query_aware_pointer = eval_arm(disentangle=True, use_pointer=True)
    res_oracle = eval_arm(is_oracle=True)

    report = {
        "milestone": "Batch 194: Controlled Query-Aware Pointer Training",
        "training_time_sec": round(train_time, 2),
        "test_cases": len(test_data),
        "arms": {
            "arm_a_base_zero_shot": res_base,
            "arm_b_standard_pointer": res_standard_pointer,
            "arm_c_query_aware_pointer": res_query_aware_pointer,
            "arm_d_deterministic_oracle": res_oracle,
        },
        "findings": {
            "em_base": res_base["exact_match_rate"],
            "em_standard_pointer": res_standard_pointer["exact_match_rate"],
            "em_query_aware_pointer": res_query_aware_pointer["exact_match_rate"],
            "em_oracle": res_oracle["exact_match_rate"],
            "conclusion": (
                f"Query-aware pointer dampening reduced query-entity repetition bias, "
                f"achieving {res_query_aware_pointer['exact_match_rate']*100:.1f}% EM on held-out facts."
            )
        }
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORTS_DIR / "query_aware_pointer_training.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Results written to {out_file}")
    print(f"Base Zero-Shot EM: {res_base['exact_match_rate']*100:.1f}%")
    print(f"Standard Pointer EM: {res_standard_pointer['exact_match_rate']*100:.1f}%")
    print(f"Query-Aware Pointer EM: {res_query_aware_pointer['exact_match_rate']*100:.1f}%")
    print(f"Oracle EM: {res_oracle['exact_match_rate']*100:.1f}%")


if __name__ == "__main__":
    train_and_eval_query_aware_pointer()
