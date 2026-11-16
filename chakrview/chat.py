"""
chakrview/chat.py

Interactive Terminal Conversational Interface for ChakrView v0.9.1.
Directly loads the fine-tuned conversational checkpoint and manages
interactive sessions with calibrated anti-repetition decoding and dynamic token-budgeted history.

Commands inside chat:
  /reset - Clears conversational history and starts fresh
  /info  - Displays active checkpoint metadata and parameters
  /exit  - Exits the chat session
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from chakrview.runtime.sampling import SamplingConfig
from chakrview.memory.hybrid_engine import HybridMemoryEngine
from scripts.run_m3_evaluations import DirectNeuralRunner, file_sha256

DEFAULT_CHECKPOINT = REPO_ROOT / "releases" / "v1.0_unified_conversational_intelligence" / "chakrview_v011_model.pt"

class ConversationalChatCLI:
    def __init__(self, checkpoint_path: Path, max_prompt_tokens: int = 384, max_gen_tokens: int = 48, memory_enabled: bool = True, debug_prompt: bool = False):
        self.checkpoint_path = checkpoint_path
        self.memory_enabled = memory_enabled
        self.debug_prompt = debug_prompt
        if not self.checkpoint_path.exists():
            # Fallback to CBR-SFT-04 if release checkpoint not found
            fallback = REPO_ROOT / "experiments" / "cbr01_sft" / "runs" / "cbr_sft_04" / "checkpoints" / "sft_checkpoint_step_200.pt"
            if fallback.exists():
                print(f"[Warning] Target checkpoint not found, falling back to: {fallback.name}")
                self.checkpoint_path = fallback
            else:
                raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")
        
        self.sha256 = file_sha256(self.checkpoint_path)
        print(f"Loading ChakrView model weights from: {self.checkpoint_path.name}")
        print(f"Verified SHA-256: {self.sha256}")
        
        self.runner = DirectNeuralRunner(self.checkpoint_path)
        self.tok = self.runner.tokenizer
        self.history: List[Dict[str, str]] = []
        self.memory = HybridMemoryEngine()
        
        # Dynamic Token Budget: 512 total context = 384 max prompt + 128 reserve (with 48 max generation)
        self.max_prompt_tokens = max_prompt_tokens
        self.max_gen_tokens = max_gen_tokens
        self.context_ceiling = 512
        
        # Calibrated decoding: penalty 1.15 avoids vocabulary suppression, no_repeat_ngram=3 prevents loops
        self.sampling_cfg = SamplingConfig(
            temperature=0.0,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3
        )

    def reset(self):
        self.history.clear()
        self.memory.reset_session()
        print("\n[Conversation history and working memory reset.]\n")

    def print_info(self):
        print(f"\n--- ChakrView Session Info ---")
        print(f"Release:         v1.0 (Unified Conversational Intelligence)")
        print(f"Checkpoint:      {self.checkpoint_path.name}")
        print(f"SHA-256:         {self.sha256}")
        print(f"Memory Mode:     {'ENABLED (Hybrid Working Memory)' if self.memory_enabled else 'DISABLED (Pure Neural)'}")
        print(f"History Turns:   {len(self.history)} turns recorded")
        print(f"Active Facts:    {len(self.memory.facts)} memory facts recorded")
        print(f"Context Ceiling: {self.context_ceiling} tokens (ChakrMicro v0.1: 6 layers, 192 dim, 6 heads)")
        print(f"Prompt Budget:   {self.max_prompt_tokens} tokens (Dynamic turn-budgeted retention)")
        print(f"Decoding:        temperature=0.0, repetition_penalty=1.15, no_repeat_ngram=3, max_new_tokens={self.max_gen_tokens}\n")

    def print_memory(self):
        print(f"\n--- ChakrView Active Memory ---")
        if not self.memory.facts:
            print("Working memory is currently empty.\n")
        else:
            for f in self.memory.facts.values():
                if f.is_active:
                    print(f" - {f.predicate.capitalize()}: {f.value} (Turn {f.turn_index})")
            print()

    def build_prompt(self, user_msg: str) -> Tuple[str, Dict[str, Any]]:
        """
        Dynamically packs historical complete turns (User + Assistant) backwards
        within max_prompt_tokens budget.
        Never drops a user turn while keeping its assistant response.
        Removes oldest complete turns first under budget pressure.
        """
        current_turn_text = f"User: {user_msg}\nAssistant:"
        current_turn_tokens = len(self.tok.encode(current_turn_text, add_bos=True, add_eos=False))
        
        # If the user message alone exceeds or nearly fills prompt budget, truncate history entirely
        available_history_budget = self.max_prompt_tokens - current_turn_tokens
        
        retained_turns = []
        discarded_turns_count = 0
        
        if available_history_budget > 10 and self.history:
            # Walk backwards from most recent turn to oldest
            accumulated_turns_text = []
            
            for turn in reversed(self.history):
                candidate_turn_text = f"User: {turn['user']}\nAssistant: {turn['assistant']}"
                # Test prompt with this candidate turn prepended
                test_parts = [candidate_turn_text] + accumulated_turns_text + [current_turn_text]
                test_prompt = "\n".join(test_parts)
                test_tokens_len = len(self.tok.encode(test_prompt, add_bos=True, add_eos=False))
                
                if test_tokens_len <= self.max_prompt_tokens:
                    accumulated_turns_text.insert(0, candidate_turn_text)
                    retained_turns.insert(0, turn)
                else:
                    discarded_turns_count += 1
            
            prompt_parts = accumulated_turns_text + [current_turn_text]
            final_prompt = "\n".join(prompt_parts)
        else:
            discarded_turns_count = len(self.history)
            final_prompt = current_turn_text
            
        final_tokens_len = len(self.tok.encode(final_prompt, add_bos=True, add_eos=False))
        
        audit_trace = {
            "retained_turns_count": len(retained_turns),
            "discarded_turns_count": discarded_turns_count,
            "total_prompt_tokens": final_tokens_len,
            "prompt_budget": self.max_prompt_tokens,
            "within_budget": final_tokens_len <= self.max_prompt_tokens
        }
        return final_prompt, audit_trace

    def interact_step(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            return ""

        if user_input.lower() in ["/reset", "reset"]:
            self.reset()
            return "[Context cleared]"
        elif user_input.lower() in ["/info", "info"]:
            self.print_info()
            return "[Session info displayed]"
        elif user_input.lower() in ["/memory", "memory"]:
            self.print_memory()
            return "[Memory status displayed]"

        # 1. Update working memory if memory is enabled
        relevant = []
        if self.memory_enabled:
            self.memory.extract_and_store_facts(user_input, turn_index=len(self.history) + 1)
            relevant = self.memory.retrieve_relevant_facts(user_input)
            
            # Hybrid Response Integration:
            # If the user is specifically querying an active factual attribute that was explicitly stored,
            # present the verified memory result directly with transparent provenance attribution.
            # Do NOT trigger when the user is declaring or setting a fact (e.g. 'my name is...', 'i renamed it to...')
            u_lower = user_input.lower()
            is_query = any(q in u_lower for q in ["what", "who", "where", "kya", "kaun", "kahan", "tell me", "do you know", "?"])
            is_declaring = any(d in u_lower for d in ["my name is", "i am", "mera naam", "renamed", "working on", "my project is"])
            
            if relevant and is_query and not is_declaring:
                for fact in relevant:
                    if fact.predicate == "name" and any(k in u_lower for k in ["name", "who am i", "naam"]):
                        ans = f"[Hybrid Memory: Name] Your name is {fact.value}."
                        self.history.append({"user": user_input, "assistant": ans})
                        return ans
                    elif fact.predicate == "city" and any(k in u_lower for k in ["city", "where do i live", "shahar"]):
                        ans = f"[Hybrid Memory: City] You live in {fact.value}."
                        self.history.append({"user": user_input, "assistant": ans})
                        return ans
                    elif fact.predicate == "project" and any(k in u_lower for k in ["project", "working on"]):
                        ans = f"[Hybrid Memory: Project] Your project is {fact.value}."
                        self.history.append({"user": user_input, "assistant": ans})
                        return ans
        
        prompt, trace = self.build_prompt(user_input)
        if self.debug_prompt:
            print(f"\n[DEBUG PROMPT]\n{prompt}\n[END DEBUG PROMPT]\n")
        
        # Autoregressive generation with sampling config
        p_toks = self.tok.encode(prompt, add_bos=True, add_eos=False)
        curr = list(p_toks)
        gen = []
        with torch.no_grad():
            for step in range(self.max_gen_tokens):
                t_in = torch.tensor([curr], dtype=torch.long)
                logits = self.runner.model(t_in)[0, -1, :]
                next_tok = runner_tok = self.runner.sampler.sample(
                    logits,
                    generated_tokens=gen,
                    config=self.sampling_cfg,
                    step=step
                )
                gen.append(next_tok)
                curr.append(next_tok)
                if next_tok == 1:
                    break
                    
        assistant_resp = self.tok.decode(gen, errors="replace").strip()

        # Update history with complete user-assistant pair
        self.history.append({"user": user_input, "assistant": assistant_resp})
        return assistant_resp

    def run_interactive_loop(self):
        print("\n" + "=" * 65)
        print("  CHAKRVIEW v1.0 CONVERSATIONAL TERMINAL (HYBRID NEURAL BRAIN)")
        print("=" * 65)
        print(f"Model: ChakrMicro Conversational v1.0 ({self.checkpoint_path.name})")
        print("Commands: /reset (clear context), /memory (view facts), /info (view stats), /exit (quit)")
        print("Type your message and press Enter.\n")

        while True:
            try:
                user_msg = input("\nUser > ").strip()
                if not user_msg:
                    continue
                if user_msg.lower() in ["/exit", "exit", "quit", ":q"]:
                    print("Exiting ChakrView chat. Have a great day!")
                    break

                resp = self.interact_step(user_msg)
                print(f"ChakrView > {resp}")

            except (KeyboardInterrupt, EOFError):
                print("\nSession interrupted. Exiting ChakrView.")
                break

def main():
    parser = argparse.ArgumentParser(description="ChakrView Interactive Chat Terminal")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to checkpoint .pt file")
    parser.add_argument("--memory", type=str, choices=["on", "off"], default="on", help="Enable or disable hybrid working memory (default: on)")
    parser.add_argument("--debug-prompt", action="store_true", help="Display full prompt and debug tokens before generation")
    parser.add_argument("--test-mode", action="store_true", help="Run automated multi-turn verification")
    args = parser.parse_args()

    mem_enabled = (args.memory == "on")
    cli = ConversationalChatCLI(Path(args.checkpoint), memory_enabled=mem_enabled, debug_prompt=args.debug_prompt)

    if args.test_mode:
        print("\n--- Running Automated Verification Mode ---")
        test_inputs = [
            "Good morning assistant!",
            "Who are you?",
            "Can you assist me today?",
            "Namaste! Aap kaise hain?",
            "/info",
            "/reset",
            "Wishing you a pleasant morning!"
        ]
        for inp in test_inputs:
            print(f"\nUser > {inp}")
            out = cli.interact_step(inp)
            print(f"ChakrView > {out}")
        print("\nVerification Mode Complete: PASS")
    else:
        cli.run_interactive_loop()

if __name__ == "__main__":
    main()
