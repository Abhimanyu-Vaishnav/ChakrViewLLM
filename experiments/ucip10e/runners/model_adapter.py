"""
ChakrView UCIP-10E Model Adapter & Inference Runner.

Provides a unified interface to load any ChakrView checkpoint
(ChakrMicro, ChakrMedium, etc.) with verified parameter shapes, tied embeddings,
and exact deterministic autoregressive generation.
"""

from __future__ import annotations

import sys
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

REPO_ROOT = Path(r"D:\Project\ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from scripts.run_m3_evaluations import DirectNeuralRunner
from chakrview.runtime.sampling import apply_repetition_penalty, apply_no_repeat_ngram


class ModelAdapter:
    """Adapts ChakrView checkpoints for standardized deterministic evaluation."""

    def __init__(self, checkpoint_path: str | Path, vocab_dir: Optional[str | Path] = None):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        self.vocab_dir = Path(vocab_dir) if vocab_dir else REPO_ROOT / "data" / "experiments" / "vocab_4096"
        self.sha256 = self._compute_sha256()

        print(f"[ModelAdapter] Loading: {self.checkpoint_path.name}")
        print(f"[ModelAdapter] SHA-256: {self.sha256}")

        self.runner = DirectNeuralRunner(str(self.checkpoint_path))
        self.model = self.runner.model
        self.tokenizer = self.runner.tokenizer
        self.model.eval()

        self.architecture = getattr(self.runner, "architecture", "Unknown")
        self.param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[ModelAdapter] Architecture: {self.architecture} ({self.param_count:,} params)")

    def _compute_sha256(self) -> str:
        h = hashlib.sha256()
        with open(self.checkpoint_path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 48,
        temperature: float = 0.0,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 3,
        system_prefix: str = "User: ",
        assistant_prefix: str = "\nAssistant:"
    ) -> str:
        """
        Executes strict deterministic autoregressive generation.
        Matches canonical ChakrView CLI prompt formatting.
        """
        formatted_prompt = f"{system_prefix}{prompt}{assistant_prefix}"
        input_ids = self.tokenizer.encode(formatted_prompt, add_bos=True, add_eos=False)
        curr = list(input_ids)
        gen: List[int] = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                t_in = torch.tensor([curr], dtype=torch.long)
                if t_in.shape[1] > 512:
                    t_in = t_in[:, -512:]

                logits = self.model(t_in)[0, -1, :]

                if gen and repetition_penalty > 1.0:
                    logits = apply_repetition_penalty(logits, gen, repetition_penalty)
                if len(curr) >= no_repeat_ngram_size and no_repeat_ngram_size > 0:
                    logits = apply_no_repeat_ngram(logits, curr, no_repeat_ngram_size)

                if temperature > 0.0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    nxt = int(torch.multinomial(probs, num_samples=1).item())
                else:
                    nxt = int(torch.argmax(logits).item())

                if nxt == 1:  # EOS token
                    break

                gen.append(nxt)
                curr.append(nxt)

        return self.tokenizer.decode(gen, errors="replace").strip()
