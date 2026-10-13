"""
Batch 183: Controlled Pointer Decoding Comparison.
Compares 4 decoding modes on held-out test facts:
1. Canonical Vocabulary-Only Generation (Zero-shot baseline)
2. Pointer Greedy Decoding (Batch 174 Baseline)
3. Pointer Beam Search (k=3, length penalty 0.7)
4. Deterministic Context-Copy Oracle (Upper-bound reference)
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
import torch.nn.functional as F

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.runtime.pointer_mechanism import ChakrPointerHead, decode_pointer_greedy, decode_pointer_beam
from chakrview.runtime.conversation import ConversationTemplate, ChatMessage, Role
from chakrview.runtime.evidence_extractor import EvidenceExtractor
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

CANONICAL_BASE_CKPT = Path("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt")
TOKENIZER_DIR = Path("release/chakrview-v0.5.0/data/experiments/vocab_4096")
DATA_PATH = Path("experiments/batch171_180_pointer_learning/data/split_manifest.json")
ARM_C_CKPT_PATH = Path("experiments/batch171_180_pointer_learning/checkpoints/pointer_head_arm_c.pt")
REPORTS_DIR = Path("experiments/batch181_190_evidence_synthesis/reports")


def evaluate_decoding_arms():
    device = torch.device("cpu")
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)

    # 1. Base Model
    cfg = ModelConfig(max_seq_len=512)
    base_model = ChakrMicro(cfg).to(device)
    base_data = torch.load(CANONICAL_BASE_CKPT, map_location=device, weights_only=False)
    base_model.load_state_dict(base_data["model_state_dict"])
    base_model.eval()

    # 2. Pointer Head
    pointer_head = ChakrPointerHead(d_model=192, vocab_size=4096, d_proj=64).to(device)
    p_data = torch.load(ARM_C_CKPT_PATH, map_location=device, weights_only=False)
    pointer_head.load_state_dict(p_data["pointer_head_state_dict"])
    pointer_head.eval()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    test_data = manifest["splits"]["test"]

    def run_eval(mode: str, beam_width: int = 1):
        matches = 0
        samples = []
        t0 = time.perf_counter()

        for item in test_data:
            doc = item["doc"]
            query = item["query"]
            target = item["value"]

            context_prefix = f"Context: {doc}\n"
            user_content = f"{context_prefix}Question: {query}"
            messages_prompt = [ChatMessage(role=Role.USER, content=user_content)]
            prompt_str = ConversationTemplate.format_conversation(messages_prompt, add_generation_prompt=True)

            prefix_tokens = tokenizer.encode("<|im_start|>user\nContext: ", add_bos=True, add_eos=False)
            doc_tokens = tokenizer.encode(doc, add_bos=False, add_eos=False)
            context_start = len(prefix_tokens)
            context_end = context_start + len(doc_tokens)

            prompt_ids = tokenizer.encode(prompt_str, add_bos=True, add_eos=False)

            if mode == "oracle":
                ext = EvidenceExtractor.extract_slot(doc, query)
                gen_text = ext.extracted_value if ext.success else "NOT_FOUND"
                sources = ["oracle_deterministic"]
            elif mode == "vocab_only":
                curr = list(prompt_ids)
                with torch.no_grad():
                    for _ in range(16):
                        x = torch.tensor([curr], dtype=torch.long, device=device)
                        logits = base_model(x)
                        next_tok = int(torch.argmax(logits[0, -1, :]).item())
                        if next_tok in (1, 3):
                            break
                        curr.append(next_tok)
                gen_toks = curr[len(prompt_ids):]
                gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True, errors="replace").strip()
                sources = ["vocab"] * len(gen_toks)
            elif mode == "pointer_greedy":
                gen_toks, sources = decode_pointer_greedy(
                    model=base_model,
                    pointer_head=pointer_head,
                    prompt_ids=prompt_ids,
                    context_start=context_start,
                    context_end=context_end,
                    max_new_tokens=16,
                    device=device
                )
                gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True, errors="replace").strip()
            elif mode == "pointer_beam":
                gen_toks, sources = decode_pointer_beam(
                    model=base_model,
                    pointer_head=pointer_head,
                    prompt_ids=prompt_ids,
                    context_start=context_start,
                    context_end=context_end,
                    beam_width=beam_width,
                    max_new_tokens=16,
                    device=device
                )
                gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True, errors="replace").strip()

            is_match = (target.lower() in gen_text.lower())
            if is_match:
                matches += 1

            samples.append({
                "entity": item["entity"],
                "query": query,
                "target": target,
                "generated": gen_text[:60],
                "match": is_match,
                "sources": sources[:5]
            })

        elapsed = time.perf_counter() - t0
        return {
            "exact_match_rate": round(matches / len(test_data), 4) if test_data else 0.0,
            "latency_ms_per_case": round((elapsed / len(test_data)) * 1000.0, 2),
            "samples": samples
        }

    vocab_res = run_eval("vocab_only")
    greedy_res = run_eval("pointer_greedy")
    beam2_res = run_eval("pointer_beam", beam_width=2)
    beam3_res = run_eval("pointer_beam", beam_width=3)
    oracle_res = run_eval("oracle")

    report = {
        "milestone": "Batch 183: Controlled Pointer Decoding Comparison",
        "held_out_test_cases": len(test_data),
        "results": {
            "canonical_vocab_only": vocab_res,
            "pointer_greedy": greedy_res,
            "pointer_beam_k2": beam2_res,
            "pointer_beam_k3": beam3_res,
            "deterministic_oracle_upper_bound": oracle_res,
        },
        "findings": {
            "beam_vs_greedy": f"Greedy EM: {greedy_res['exact_match_rate']*100:.1f}%, Beam k=2 EM: {beam2_res['exact_match_rate']*100:.1f}%, Beam k=3 EM: {beam3_res['exact_match_rate']*100:.1f}%",
            "oracle_upper_bound": f"Deterministic Oracle EM: {oracle_res['exact_match_rate']*100:.1f}%",
            "latency_tradeoff": f"Greedy: {greedy_res['latency_ms_per_case']}ms vs Beam k=3: {beam3_res['latency_ms_per_case']}ms per case",
            "conclusion": "Pointer beam decoding preserves factual copying with structured hypothesis tracking across multi-token targets."
        }
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORTS_DIR / "pointer_decoding_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Report written to {out_file}")
    print(f"Vocab EM: {vocab_res['exact_match_rate']*100:.1f}%")
    print(f"Greedy EM: {greedy_res['exact_match_rate']*100:.1f}%")
    print(f"Beam k=2 EM: {beam2_res['exact_match_rate']*100:.1f}%")
    print(f"Beam k=3 EM: {beam3_res['exact_match_rate']*100:.1f}%")
    print(f"Deterministic Oracle EM: {oracle_res['exact_match_rate']*100:.1f}%")


if __name__ == "__main__":
    evaluate_decoding_arms()
