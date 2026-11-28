"""
ChakrView Inference Package: Conversational Generator & Interactive CLI
Provides a standalone, reproducible generator for ChakrView models.

Features:
- Configurable generation: greedy decoding, temperature sampling, top-k, top-p, repetition penalty
- Multi-turn conversation history management
- Truncation to maximum context window (256 tokens)
- Transparent attribution: purely neural generation without hidden deterministic fakes
- Device agnostic: CUDA or CPU
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.abspath("."))
from chakrview.brain.genesis.chakr_genesis_38m import ChakrGenesis38M, Genesis38MConfig
from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

class ChakrViewGenerator:
    def __init__(self, checkpoint_path: str, tokenizer_path: str = "data/experiments/vocab_4096", device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading ChakrView checkpoint from {checkpoint_path} onto {self.device}...")
        self.ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = self.ckpt["config"]

        # Select model class based on config dimensions
        if hasattr(self.config, "d_model") and self.config.d_model == 512:
            self.model = ChakrGenesis38M(self.config).to(self.device)
        else:
            self.model = ChakrGenesis17M(self.config).to(self.device)

        self.model.load_state_dict(self.ckpt["model_state_dict"])
        self.model.eval()

        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer, _ = load_tokenizer_artifacts(tokenizer_path)

        self.max_context = getattr(self.config, "max_seq_len", 256)
        self.eos_token_id = 2
        self.pad_token_id = 0

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15
    ) -> Tuple[str, Dict]:
        self.model.eval()
        tokens = self.tokenizer.encode(prompt)
        
        # Enforce context window headroom
        if len(tokens) > (self.max_context - max_new_tokens):
            tokens = tokens[-(self.max_context - max_new_tokens):]

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        generated_tokens = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if self.device.type == "cuda":
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        out = self.model(input_ids)
                else:
                    out = self.model(input_ids)

                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                next_token_logits = logits[0, -1, :].clone()

                # Repetition penalty
                if repetition_penalty != 1.0:
                    for t in set(input_ids[0].tolist()):
                        if next_token_logits[t] < 0:
                            next_token_logits[t] *= repetition_penalty
                        else:
                            next_token_logits[t] /= repetition_penalty

                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    
                    # Top-k filtering
                    if top_k > 0:
                        v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                        next_token_logits[next_token_logits < v[-1]] = -float('Inf')

                    # Top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = -float('Inf')

                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1).item()

                if next_token == self.eos_token_id or next_token == self.pad_token_id:
                    break

                generated_tokens.append(next_token)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self.device)], dim=1)

                if input_ids.shape[1] >= self.max_context:
                    break

        decoded = self.tokenizer.decode(generated_tokens)
        meta = {
            "prompt_tokens": len(tokens),
            "generated_tokens": len(generated_tokens),
            "total_tokens": len(tokens) + len(generated_tokens),
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "device": str(self.device)
        }
        return decoded, meta

class ChakrChatSession:
    def __init__(self, generator: ChakrViewGenerator):
        self.generator = generator
        self.history: List[Dict[str, str]] = []

    def format_history(self) -> str:
        prompt_parts = []
        for turn in self.history:
            role = "User" if turn["role"] == "user" else "Assistant"
            prompt_parts.append(f"{role}: {turn['content']}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    def chat(self, user_message: str, **kwargs) -> str:
        self.history.append({"role": "user", "content": user_message})
        prompt = self.format_history()
        response, meta = self.generator.generate(prompt, **kwargs)
        clean_resp = response.strip()
        self.history.append({"role": "assistant", "content": clean_resp})
        return clean_resp

    def reset(self):
        self.history.clear()
