"""
ChakrView Step 56: Memory-Conditioned Inference Bridge.

Bridges:
- Episodic Memory store records
- Diagnostic reasoning states
- CognitiveContext contract
- ChakrMicro forward/decoding pipeline

Enforces strict separation:
NEURAL CORE != MEMORY != CHAKRKSHETRA != RIL
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple
import torch

from chakrview.brain.model import ChakrMicro
from chakrview.runtime.cortex_context import CognitiveContext
from chakrview.runtime.inference import GenerationConfig
from chakrview.runtime.interactive import DEFAULT_TOKENIZER_DIR
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


class MemoryConditionedInferenceBridge:
    """
    Constructs bounded cognitive context envelopes from episodic memory and diagnostic state,
    and executes conditioned generation on ChakrMicro without modifying base weights.
    """

    def __init__(self, model: ChakrMicro, tokenizer: Optional[Any] = None) -> None:
        self.model = model
        if tokenizer is None:
            self.tokenizer, _ = load_tokenizer_artifacts(DEFAULT_TOKENIZER_DIR)
        else:
            self.tokenizer = tokenizer

    def format_conditioned_prompt(
        self,
        task_prompt: str,
        task_mode: str = "REASONING",
        goal: str = "",
        state: str = "",
        memory_record: Optional[Dict[str, Any]] = None,
        reasoning_diagnosis: str = "",
        constraints: str = "",
    ) -> str:
        """
        Assemble prompt with prepended token-bounded <CORTEX_CONTEXT>.
        """
        memory_str = ""
        if memory_record:
            # Extract key lessons from episodic memory record
            obs = memory_record.get("observation", "")
            diag = memory_record.get("diagnosis", "")
            memory_str = f"Observation: {obs}; Prior Diagnosis: {diag}"

        context = CognitiveContext(
            task_mode=task_mode,
            goal=goal,
            state=state,
            memory_context=memory_str,
            reasoning_state=reasoning_diagnosis,
            constraints=constraints,
        )
        context.validate()
        return f"{context.serialize()}{task_prompt}"

    def generate_conditioned(
        self,
        task_prompt: str,
        task_mode: str = "REASONING",
        goal: str = "",
        state: str = "",
        memory_record: Optional[Dict[str, Any]] = None,
        reasoning_diagnosis: str = "",
        constraints: str = "",
        max_new_tokens: int = 16,
    ) -> Dict[str, Any]:
        """
        Generate completion using conditioned context on ChakrMicro.
        """
        full_prompt = self.format_conditioned_prompt(
            task_prompt=task_prompt,
            task_mode=task_mode,
            goal=goal,
            state=state,
            memory_record=memory_record,
            reasoning_diagnosis=reasoning_diagnosis,
            constraints=constraints,
        )

        prompt_ids = self.tokenizer.encode(full_prompt)
        # Verify 512 context limit
        if len(prompt_ids) + max_new_tokens > 512:
            raise ValueError(
                f"Conditioned sequence exceeds context horizon: {len(prompt_ids)} + {max_new_tokens} > 512"
            )

        input_tensor = torch.tensor([prompt_ids], dtype=torch.long)
        self.model.eval()

        generated_ids = []
        with torch.no_grad():
            cur_tokens = input_tensor
            for _ in range(max_new_tokens):
                logits = self.model(cur_tokens)
                next_token_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                if next_token_id == 1:  # EOS
                    break
                generated_ids.append(next_token_id)
                next_tensor = torch.tensor([[next_token_id]], dtype=torch.long)
                cur_tokens = torch.cat([cur_tokens, next_tensor], dim=1)

        raw_output = self.tokenizer.decode(generated_ids)
        return {
            "conditioned_prompt": full_prompt,
            "raw_output": raw_output,
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated_ids),
        }
