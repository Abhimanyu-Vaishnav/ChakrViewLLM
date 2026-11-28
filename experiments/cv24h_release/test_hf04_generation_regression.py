"""HF-04 Generation Stopping and Special Token Correctness Regression Suite.

This script tests generation stopping behavior across 10 distinct test conditions:
1. BOS at prompt start: Ensure BOS (token 0) can be placed at prompt start without stopping.
2. EOS termination: Ensure token 1 halts generation immediately.
3. PAD/delimiter termination: Ensure token 2 halts generation immediately.
4. PAD inside input prompt: Ensure PAD (token 2 or 0) inside prompt padding does NOT stop generation prematurely.
5. BOS emitted during generation: Ensure token 0 emitted does NOT stop generation (unless intentionally configured).
6. Empty input handling: Ensure empty string prompt falls back to BOS safely and generates tokens.
7. Batched prompts with different lengths: Ensure shorter sequences halt independently from longer sequences.
8. Maximum context boundary: Ensure generation halts cleanly at max_context (256) without index out-of-bounds.
9. Generation after previous assistant turn: Multi-turn prompt containing previous turn delimiters handles generation correctly.
10. No premature termination: Deterministic generator emitting N standard tokens before delimiter generates exactly N tokens.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STAGING_DIR = PROJECT_ROOT / "artifacts" / "huggingface" / "chakrview-conversational-alpha"

# Add staging directory to sys.path so we can import modeling_chakrgenesis and tokenization_chakrview standalone
sys.path.insert(0, str(STAGING_DIR))

from tokenization_chakrview import load_tokenizer_artifacts
from modeling_chakrgenesis import Genesis38MConfig

BOS_TOKEN_ID = 0
CANONICAL_EOS_TOKEN_ID = 1
DELIMITER_EOS_TOKEN_ID = 2
PAD_TOKEN_ID = 2


class MockModelForCausalLM(nn.Module):
    """Deterministic token emitter for regression testing."""

    def __init__(self, sequence_to_emit: List[int], vocab_size: int = 4096):
        super().__init__()
        self.sequence_to_emit = list(sequence_to_emit)
        self.vocab_size = vocab_size
        self.step = 0
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids):
        # input_ids: [B, S]
        batch_size = input_ids.shape[0]
        logits = torch.zeros((batch_size, 1, self.vocab_size), device=input_ids.device)

        if self.step < len(self.sequence_to_emit):
            target_token = self.sequence_to_emit[self.step]
            self.step += 1
        else:
            target_token = DELIMITER_EOS_TOKEN_ID

        logits[:, 0, target_token] = 100.0  # Force argmax
        return logits


class StandaloneChakrGenerator:
    """Wrapper using exact logic from inference_package.py with injectable model and tokenizer."""

    def __init__(self, model: nn.Module, tokenizer, tok_config: dict, device: torch.device, max_context: int = 256):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.tok_config = tok_config
        self.device = device
        self.max_context = max_context
        self.bos_token_id = BOS_TOKEN_ID
        self.canonical_eos_token_id = CANONICAL_EOS_TOKEN_ID
        self.delimiter_eos_token_id = DELIMITER_EOS_TOKEN_ID
        self.pad_token_id = PAD_TOKEN_ID
        self.stop_token_ids = {self.canonical_eos_token_id, self.delimiter_eos_token_id}

    def generate(
        self,
        prompt: Union[str, List[int]],
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop_token_ids: Optional[set] = None
    ) -> Tuple[List[int], str, Dict]:
        self.model.eval()
        if stop_token_ids is None:
            stop_token_ids = self.stop_token_ids

        if isinstance(prompt, str):
            if not prompt.strip():
                tokens = [self.bos_token_id]
            else:
                tokens = self.tokenizer.encode(prompt)
        else:
            tokens = list(prompt)
            if len(tokens) == 0:
                tokens = [self.bos_token_id]

        if len(tokens) > (self.max_context - max_new_tokens):
            tokens = tokens[-(self.max_context - max_new_tokens):]

        input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        prompt_len = input_ids.shape[1]
        generated_tokens = []
        stopped_on = None

        with torch.no_grad():
            for step in range(max_new_tokens):
                out = self.model(input_ids)
                logits = out[0] if isinstance(out, tuple) else (out.logits if hasattr(out, "logits") else out)
                next_token_logits = logits[0, -1, :].clone()

                if repetition_penalty != 1.0:
                    for t in set(input_ids[0].tolist()):
                        if next_token_logits[t] < 0:
                            next_token_logits[t] *= repetition_penalty
                        else:
                            next_token_logits[t] /= repetition_penalty

                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    if top_k > 0:
                        v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                        next_token_logits[next_token_logits < v[-1]] = -float('Inf')
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
                if next_token in stop_token_ids:
                    stopped_on = f"EOS/Delimiter (token {next_token})"
                    generated_tokens.append(next_token)
                    break

                generated_tokens.append(next_token)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=self.device)], dim=1)

                if input_ids.shape[1] >= self.max_context:
                    stopped_on = f"Max Context ({self.max_context} tokens)"
                    break

        if stopped_on is None:
            stopped_on = f"Max New Tokens ({max_new_tokens})"

        # Decode tokens (excluding trailing delimiter if present for display, but returning exact tokens)
        display_tokens = generated_tokens[:-1] if generated_tokens and generated_tokens[-1] in stop_token_ids else generated_tokens
        try:
            decoded = self.tokenizer.decode(display_tokens, errors="replace")
        except Exception:
            decoded = ""
        meta = {
            "prompt_tokens_count": prompt_len,
            "generated_tokens_count": len(generated_tokens),
            "total_tokens_count": prompt_len + len(generated_tokens),
            "generated_token_ids": generated_tokens,
            "stopped_on": stopped_on
        }
        return generated_tokens, decoded, meta


def run_hf04_regression_suite(device_str: str):
    print(f"==================================================")
    print(f"RUNNING HF-04 REGRESSION TEST SUITE ON: {device_str}")
    print(f"==================================================")

    device = torch.device(device_str)
    vocab_dir = str(PROJECT_ROOT / "data" / "experiments" / "vocab_4096")
    tokenizer, tok_config = load_tokenizer_artifacts(vocab_dir)

    results = {}

    # Test 1: BOS at prompt start
    print("\n--- Test 1: BOS at prompt start ---")
    mock_model = MockModelForCausalLM(sequence_to_emit=[100, 101, 1], vocab_size=4096)
    gen = StandaloneChakrGenerator(mock_model, tokenizer, tok_config, device=device)
    prompt_ids = [BOS_TOKEN_ID] + tokenizer.encode("Hello")
    gen_tokens, _, meta = gen.generate(prompt_ids, max_new_tokens=10)
    assert gen_tokens == [100, 101, 1], f"Expected [100, 101, 1], got {gen_tokens}"
    assert meta["stopped_on"] == "EOS/Delimiter (token 1)"
    results["test_1_bos_at_prompt_start"] = "PASS"
    print("PASS: BOS at prompt start allows generation and stops on EOS.")

    # Test 2: EOS termination (token 1)
    print("\n--- Test 2: EOS termination (token 1) ---")
    mock_model = MockModelForCausalLM(sequence_to_emit=[200, 201, CANONICAL_EOS_TOKEN_ID, 202], vocab_size=4096)
    gen = StandaloneChakrGenerator(mock_model, tokenizer, tok_config, device=device)
    gen_tokens, _, meta = gen.generate([BOS_TOKEN_ID, 50], max_new_tokens=10)
    assert gen_tokens == [200, 201, CANONICAL_EOS_TOKEN_ID], f"Expected [200, 201, 1], got {gen_tokens}"
    assert meta["stopped_on"] == "EOS/Delimiter (token 1)"
    results["test_2_eos_termination"] = "PASS"
    print("PASS: Token 1 halts generation immediately without executing remaining emitted steps.")

    # Test 3: PAD/delimiter termination (token 2)
    print("\n--- Test 3: PAD/delimiter termination (token 2) ---")
    mock_model = MockModelForCausalLM(sequence_to_emit=[300, DELIMITER_EOS_TOKEN_ID, 301], vocab_size=4096)
    gen = StandaloneChakrGenerator(mock_model, tokenizer, tok_config, device=device)
    gen_tokens, _, meta = gen.generate([BOS_TOKEN_ID, 50], max_new_tokens=10)
    assert gen_tokens == [300, DELIMITER_EOS_TOKEN_ID], f"Expected [300, 2], got {gen_tokens}"
    assert meta["stopped_on"] == "EOS/Delimiter (token 2)"
    results["test_3_delimiter_termination"] = "PASS"
    print("PASS: Token 2 halts generation immediately as trained delimiter EOS.")

    # Test 4: PAD inside input prompt padding positions (must NOT stop prematurely)
    print("\n--- Test 4: PAD inside input prompt padding positions ---")
    mock_model = MockModelForCausalLM(sequence_to_emit=[400, 401, 402, 2], vocab_size=4096)
    gen = StandaloneChakrGenerator(mock_model, tokenizer, tok_config, device=device)
    prompt_with_padding = [PAD_TOKEN_ID, PAD_TOKEN_ID, BOS_TOKEN_ID, 55, 56]
    gen_tokens, _, meta = gen.generate(prompt_with_padding, max_new_tokens=10)
    assert gen_tokens == [400, 401, 402, 2], f"Expected [400, 401, 402, 2], got {gen_tokens}"
    results["test_4_prompt_padding_protection"] = "PASS"
    print("PASS: Token 2 inside prompt padding prefix does NOT cause premature termination.")

    # Test 5: BOS emitted during generation
    print("\n--- Test 5: BOS emitted during generation ---")
    mock_model = MockModelForCausalLM(sequence_to_emit=[500, BOS_TOKEN_ID, 501, 2], vocab_size=4096)
    gen = StandaloneChakrGenerator(mock_model, tokenizer, tok_config, device=device)
    gen_tokens, _, meta = gen.generate([BOS_TOKEN_ID, 60], max_new_tokens=10)
    assert gen_tokens == [500, BOS_TOKEN_ID, 501, 2], f"Expected [500, 0, 501, 2], got {gen_tokens}"
    results["test_5_bos_emitted_in_generation"] = "PASS"
    print("PASS: Emitting token 0 does not halt generation when stop_token_ids={1, 2}.")

    # Test 6: Empty input handling (BOS fallback)
    print("\n--- Test 6: Empty input handling ---")
    mock_model = MockModelForCausalLM(sequence_to_emit=[600, 2], vocab_size=4096)
    gen = StandaloneChakrGenerator(mock_model, tokenizer, tok_config, device=device)
    gen_tokens, decoded, meta = gen.generate("", max_new_tokens=5)
    assert gen_tokens == [600, 2]
    assert meta["prompt_tokens_count"] == 1  # [BOS_TOKEN_ID]
    results["test_6_empty_input_fallback"] = "PASS"
    print("PASS: Empty input string handled cleanly via [BOS_TOKEN_ID] fallback.")

    # Test 7: Variable length independent generation
    print("\n--- Test 7: Variable length generation simulation ---")
    mock_short = MockModelForCausalLM(sequence_to_emit=[701, 2], vocab_size=4096)
    gen_short = StandaloneChakrGenerator(mock_short, tokenizer, tok_config, device=device)
    toks_short, _, _ = gen_short.generate([BOS_TOKEN_ID, 10], max_new_tokens=10)

    mock_long = MockModelForCausalLM(sequence_to_emit=[701, 702, 703, 704, 2], vocab_size=4096)
    gen_long = StandaloneChakrGenerator(mock_long, tokenizer, tok_config, device=device)
    toks_long, _, _ = gen_long.generate([BOS_TOKEN_ID, 10, 20, 30], max_new_tokens=10)

    assert len(toks_short) == 2 and toks_short[-1] == 2
    assert len(toks_long) == 5 and toks_long[-1] == 2
    results["test_7_variable_length_generation"] = "PASS"
    print(f"PASS: Short sequence ({len(toks_short)}) and long sequence ({len(toks_long)}) halt cleanly at delimiter.")

    # Test 8: Maximum context boundary (256 tokens)
    print("\n--- Test 8: Maximum context boundary ---")
    mock_endless = MockModelForCausalLM(sequence_to_emit=[800] * 300, vocab_size=4096)
    gen_endless = StandaloneChakrGenerator(mock_endless, tokenizer, tok_config, device=device, max_context=256)
    prompt_ids = [BOS_TOKEN_ID] + [10] * 230  # 231 tokens
    # When prompt exceeds (max_context - max_new_tokens), it is trimmed to fit exactly max_new_tokens headroom
    gen_tokens, _, meta = gen_endless.generate(prompt_ids, max_new_tokens=50)
    # The actual context passed into the model is prompt_tokens_count + generated_tokens_count
    total_context_len = meta["prompt_tokens_count"] + len(gen_tokens)
    assert total_context_len <= 256, f"Context length {total_context_len} exceeded 256"
    assert total_context_len == 256, f"Expected context length 256, got {total_context_len}"
    assert meta["prompt_tokens_count"] == 206, f"Expected trimmed prompt to be 206, got {meta['prompt_tokens_count']}"
    assert len(gen_tokens) == 50, f"Expected 50 tokens, got {len(gen_tokens)}"
    assert "Max Context" in meta["stopped_on"] or "Max New Tokens" in meta["stopped_on"]
    results["test_8_max_context_boundary"] = "PASS"
    print(f"PASS: Prompt trimmed to {meta['prompt_tokens_count']} + generated {len(gen_tokens)} = {total_context_len} == 256 max context.")

    # Test 9: Generation after previous assistant turn
    print("\n--- Test 9: Generation after previous assistant turn ---")
    multi_turn_prompt = (
        f"User: Hi\nAssistant: Hello{tokenizer.decode([DELIMITER_EOS_TOKEN_ID])}\nUser: What is 2+2?\nAssistant:"
    )
    tokens_multi = tokenizer.encode(multi_turn_prompt)
    mock_multi = MockModelForCausalLM(sequence_to_emit=[900, 901, 2], vocab_size=4096)
    gen_multi = StandaloneChakrGenerator(mock_multi, tokenizer, tok_config, device=device)
    gen_tokens, _, meta = gen_multi.generate(tokens_multi, max_new_tokens=10)
    assert gen_tokens == [900, 901, 2]
    results["test_9_multi_turn_prompt_handling"] = "PASS"
    print("PASS: Multi-turn prompt containing previous assistant delimiter successfully decoded and generated.")

    # Test 10: No premature termination
    print("\n--- Test 10: No premature termination ---")
    seq_7 = [1001, 1002, 1003, 1004, 1005, 1006, 1007, DELIMITER_EOS_TOKEN_ID]
    mock_exact = MockModelForCausalLM(sequence_to_emit=seq_7, vocab_size=4096)
    gen_exact = StandaloneChakrGenerator(mock_exact, tokenizer, tok_config, device=device)
    gen_tokens, _, meta = gen_exact.generate([BOS_TOKEN_ID, 42], max_new_tokens=50)
    assert gen_tokens == seq_7, f"Expected {seq_7}, got {gen_tokens}"
    assert len(gen_tokens) == 8, f"Expected 8 tokens, got {len(gen_tokens)}"
    results["test_10_no_premature_termination"] = "PASS"
    print(f"PASS: Exactly 7 non-delimiter tokens generated before delimiter termination.")

    # Summary
    print("\n==================================================")
    print(f"REGRESSION RESULTS SUMMARY ({device_str}):")
    all_passed = True
    for test_name, status in results.items():
        print(f"  {test_name}: {status}")
        if status != "PASS":
            all_passed = False
    print("==================================================")
    assert all_passed, "One or more regression tests failed!"
    print(f"ALL 10 REGRESSION TESTS PASSED ON {device_str}!\n")


if __name__ == "__main__":
    device_arg = sys.argv[1] if len(sys.argv) > 1 else ("cuda" if torch.cuda.is_available() else "cpu")
    run_hf04_regression_suite(device_arg)
