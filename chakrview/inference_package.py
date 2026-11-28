"""
ChakrView Inference Package: Conversational Generator & Interactive CLI
Provides a standalone, reproducible generator for ChakrView models.

Special Token Contract & Generation Stopping Specification:
- Canonical Special Token IDs:
  * BOS = 0 (<BOS>)
  * EOS = 1 (<EOS>)
  * PAD = 2 (<PAD>)
- Training Delimiter Invariant:
  * Pretraining and SFT data pipelines appended token ID 2 at document & response boundaries.
  * In SFT, response padding also used token ID 0 or token ID 2 with target loss masking (-100).
- Robust Inference Stopping Rule:
  * Prompt Prefix: Input prompt text tokens (never truncated unless exceeding max_context - max_new_tokens).
  * Generation Loop: Only tokens produced *after* the prompt are evaluated for stopping.
  * Stopping Tokens: A generated token terminates generation if next_token in [1, 2].
  * Padding Invariance: Padding inside input tensors (whether 0 or 2) is part of prompt context and NEVER prematurely halts generation.
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union

sys.path.insert(0, os.path.abspath("."))
try:
    from modeling_chakrgenesis import ChakrGenesis38M, Genesis38MConfig
    from tokenization_chakrview import load_tokenizer_artifacts
except ImportError:
    from chakrview.brain.genesis.chakr_genesis_38m import ChakrGenesis38M, Genesis38MConfig
    from chakrview.tokenizer.serialization import load_tokenizer_artifacts

class ChakrViewGenerator:
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str = "data/experiments/vocab_4096",
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Determine loader based on file extension
        if checkpoint_path.endswith(".safetensors"):
            import safetensors.torch
            self.config = Genesis38MConfig()
            self.model = ChakrGenesis38M(self.config).to(self.device)
            safetensors.torch.load_model(self.model, checkpoint_path, strict=True)
        else:
            self.ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.config = self.ckpt.get("config", Genesis38MConfig())
            self.model = ChakrGenesis38M(self.config).to(self.device)
            self.model.load_state_dict(self.ckpt["model_state_dict"])

        self.model.eval()

        self.tokenizer, self.tok_config = load_tokenizer_artifacts(tokenizer_path)
        self.max_context = getattr(self.config, "max_seq_len", 256)
        
        # Explicit special token mapping
        self.bos_token_id = 0
        self.canonical_eos_token_id = 1
        self.delimiter_eos_token_id = 2  # Delimiter emitted by trained model
        self.pad_token_id = 2
        
        # Stop on canonical EOS (1) or trained delimiter (2)
        self.stop_token_ids = {self.canonical_eos_token_id, self.delimiter_eos_token_id}

    def generate(
        self,
        prompt: Union[str, List[int]],
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15
    ) -> Tuple[str, Dict]:
        self.model.eval()
        
        if isinstance(prompt, str):
            if not prompt.strip():
                # Empty prompt safe fallback
                tokens = [self.bos_token_id]
            else:
                tokens = self.tokenizer.encode(prompt)
        else:
            tokens = list(prompt)
            if len(tokens) == 0:
                tokens = [self.bos_token_id]

        # Enforce context window headroom
        if len(tokens) > (self.max_context - max_new_tokens):
            tokens = tokens[-(self.max_context - max_new_tokens):]

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        prompt_len = input_ids.shape[1]
        generated_tokens = []
        stopped_on = None

        with torch.no_grad():
            for step in range(max_new_tokens):
                if self.device.type == "cuda":
                    with torch.amp.autocast('cuda', dtype=torch.float16):
                        out = self.model(input_ids)
                else:
                    out = self.model(input_ids)

                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                next_token_logits = logits[0, -1, :].clone()

                # Repetition penalty applied to all generated + prompt tokens
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

                # STOPPING CRITERION:
                # Evaluated strictly on newly generated tokens.
                if next_token in self.stop_token_ids:
                    stopped_on = f"EOS/Delimiter (token {next_token})"
                    break

                generated_tokens.append(next_token)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self.device)], dim=1)

                if input_ids.shape[1] >= self.max_context:
                    stopped_on = f"Max Context ({self.max_context} tokens)"
                    break

        if stopped_on is None:
            stopped_on = f"Max New Tokens ({max_new_tokens})"

        decoded = self.tokenizer.decode(generated_tokens)
        meta = {
            "prompt_tokens_count": prompt_len,
            "generated_tokens_count": len(generated_tokens),
            "total_tokens_count": prompt_len + len(generated_tokens),
            "generated_token_ids": generated_tokens,
            "stopped_on": stopped_on,
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
