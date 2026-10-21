"""
hybrid_generator_pointer.py
ChakrView Experiment 11 — Hybrid Generator-Pointer Model Architecture

Combines:
1. Native Causal Language Model (`ChakrMicroForCausalLM`):
   - Generates structural JSON tokens (quotes, keys, delimiters, braces).
2. Span-Pointer Module (`ChakrSpanPointerModel` from Exp 10):
   - Extracts contiguous context spans for target slot values.
3. Explicit Routing Controller:
   - When generating JSON values inside string quotes following a recognized schema key
     (e.g. "entity": ", "status": ", "code": "), the pointer head is invoked with
     pure candidate eligibility masks on the prompt context.
   - The selected multi-token span is spliced into the generation stream, followed by
     the closing quote and structural JSON synthesis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.research_v08_json_sft.exp10_span_pointer.span_pointer_model import ChakrSpanPointerModel

class ChakrHybridJsonModel(nn.Module):
    """
    Hybrid Generator-Pointer for Structured JSON Extraction.
    """
    def __init__(
        self,
        config: ModelConfig,
        max_span_len: int = 8,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.config = config
        self.max_span_len = max_span_len

        # Shared Transformer backbone (with native tied LM head)
        self.backbone = ChakrMicro(config)

        # 2. Span Pointer Heads (matching Exp 10 architecture)
        self.query_start = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key_start = nn.Linear(config.d_model, config.d_model, bias=False)

        self.query_end = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key_end = nn.Linear(config.d_model, config.d_model, bias=False)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Computes both LM logits and Span Pointer scores over input_ids.
        """
        # Forward through native ChakrMicro
        lm_logits, hidden_states = self.backbone(input_ids, return_hidden_states=True) # logits: [B, T, V], hidden: [B, T, D]

        # Span Pointer scores using query at sequence end
        h_query = hidden_states[:, -1:, :] # [B, 1, D]
        q_s = self.query_start(h_query)    # [B, 1, D]
        k_s = self.key_start(hidden_states)  # [B, T, D]
        start_scores = torch.bmm(q_s, k_s.transpose(1, 2)).squeeze(1) / (self.config.d_model ** 0.5)

        q_e = self.query_end(h_query)
        k_e = self.key_end(hidden_states)
        end_scores = torch.bmm(q_e, k_e.transpose(1, 2)).squeeze(1) / (self.config.d_model ** 0.5)

        if candidate_mask is not None:
            start_scores = start_scores.masked_fill(~candidate_mask, float("-inf"))
            end_scores = end_scores.masked_fill(~candidate_mask, float("-inf"))

        return {
            "lm_logits": lm_logits,
            "start_scores": start_scores,
            "end_scores": end_scores,
            "hidden_states": hidden_states
        }

    def point_span(
        self,
        input_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
        query_pos: Optional[int] = None
    ) -> Tuple[int, int, List[int]]:
        """
        Predicts the best contiguous span [s, e] from eligible candidate positions.
        """
        _, hidden_states = self.backbone(input_ids, return_hidden_states=True) # [1, T, D]
        seq_len = input_ids.size(1)
        actual_pos = (seq_len - 1) if (query_pos is None or query_pos == -1) else query_pos
        h_q = hidden_states[:, actual_pos:actual_pos+1, :]

        q_s = self.query_start(h_q)
        k_s = self.key_start(hidden_states)
        s_scores = torch.bmm(q_s, k_s.transpose(1, 2)).squeeze(1) / (self.config.d_model ** 0.5)

        q_e = self.query_end(h_q)
        k_e = self.key_end(hidden_states)
        e_scores = torch.bmm(q_e, k_e.transpose(1, 2)).squeeze(1) / (self.config.d_model ** 0.5)

        s_scores = s_scores.masked_fill(~candidate_mask, float("-inf"))
        e_scores = e_scores.masked_fill(~candidate_mask, float("-inf"))

        best_score = float("-inf")
        best_s, best_e = 0, 0
        seq_len = input_ids.size(1)

        for s in range(seq_len):
            if not candidate_mask[0, s]:
                continue
            max_e = min(s + self.max_span_len, seq_len)
            for e in range(s, max_e):
                if not candidate_mask[0, e]:
                    continue
                score = (s_scores[0, s] + e_scores[0, e]).item()
                if score > best_score:
                    best_score = score
                    best_s, best_e = s, e

        pred_tokens = input_ids[0, best_s:best_e+1].tolist()
        return best_s, best_e, pred_tokens

    def generate_json_hybrid(
        self,
        prompt_ids: List[int],
        tokenizer: Any,
        candidate_mask: torch.Tensor,
        max_new_tokens: int = 64
    ) -> Dict[str, Any]:
        """
        Autoregressively generates structured JSON using the Hybrid routing contract:
        - Structural syntax is generated by the LM head.
        - When a target field value quote is opened (e.g. ': "'), control routes to the pointer head.
        - The pointer extracts the exact entity span from prompt context.
        """
        curr_ids = list(prompt_ids)
        prompt_len = len(prompt_ids)
        routing_decisions = []

        # Target field triggers that route to the pointer
        # Trigger strings: ': "'
        trigger_tokens = tokenizer.encode(': "')

        gen_tokens = []
        finished = False

        for step in range(max_new_tokens):
            inp = torch.tensor([curr_ids], dtype=torch.long, device=next(self.parameters()).device)
            with torch.no_grad():
                out = self.forward(inp)
                next_logits = out["lm_logits"][0, -1, :]
                pred_tok = next_logits.argmax().item()

            # Check if previous tokens match trigger ': "'
            # Inspect the recent decoded stream to determine routing condition
            recent_decoded = tokenizer.decode(curr_ids[-5:])
            if recent_decoded.endswith(': "') or recent_decoded.endswith(':"'):
                # Routing event: invoke pointer head on prompt context
                routing_decisions.append({
                    "step": step,
                    "action": "ROUTE_TO_POINTER",
                    "context_len": prompt_len
                })
                # Pointer queries the prompt context [0..prompt_len-1]
                prompt_inp = torch.tensor([curr_ids[:prompt_len]], dtype=torch.long, device=inp.device)
                p_cand_mask = candidate_mask[:, :prompt_len]

                s, e, copied_toks = self.point_span(prompt_inp, p_cand_mask, query_pos=-1)
                routing_decisions[-1]["copied_span"] = [s, e]
                routing_decisions[-1]["copied_tokens"] = copied_toks
                routing_decisions[-1]["copied_value"] = tokenizer.decode(copied_toks)

                # Append copied tokens and closing quote
                quote_tok = tokenizer.encode('"')[0]
                curr_ids.extend(copied_toks)
                curr_ids.append(quote_tok)
                gen_tokens.extend(copied_toks)
                gen_tokens.append(quote_tok)

                # Check if JSON closure reached
                full_gen_text = tokenizer.decode(gen_tokens)
                if full_gen_text.endswith("}"):
                    finished = True
                    break
                continue

            # Standard LM token generation
            routing_decisions.append({
                "step": step,
                "action": "GENERATE_LM",
                "token_id": pred_tok,
                "token_str": tokenizer.decode([pred_tok])
            })
            curr_ids.append(pred_tok)
            gen_tokens.append(pred_tok)

            # Stop conditions
            if pred_tok == 1:  # EOS
                finished = True
                break
            full_gen_text = tokenizer.decode(gen_tokens)
            if "}" in full_gen_text and full_gen_text.strip().endswith("}"):
                finished = True
                break

        full_output_str = tokenizer.decode(gen_tokens).strip()

        return {
            "generated_text": full_output_str,
            "generated_tokens": gen_tokens,
            "routing_decisions": routing_decisions,
            "finished_cleanly": finished
        }
