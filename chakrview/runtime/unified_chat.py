"""
ChakrView Unified Conversational Runtime (UCIP-01).

Implements the complete local conversational runtime:
- Model & tokenizer loading (supporting both ChakrMicro 3.44M and expanded ChakrMedium architectures)
- Dynamic turn-budgeted conversation history packing
- Hybrid memory integration (automatic entity extraction, deterministic retrieval, memory context formatting)
- Explicit user correction handling and state updating
- Safe generation with temperature, top-k, and repetition penalties
- Session reset and memory inspection commands (/reset, /info, /memory)
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

REPO_ROOT = Path("D:/Project/ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from chakrview.brain.model import ModelConfig, ChakrMicro
from chakrview.memory.hybrid_engine import HybridMemoryEngine, StructuredFact
from scripts.run_m3_evaluations import load_tokenizer_artifacts, Sampler
from chakrview.runtime.sampling import SamplingConfig

class UnifiedChatSession:
    def __init__(
        self,
        checkpoint_path: Path,
        session_id: str = "default_session",
        user_id: str = "default_user",
        max_prompt_tokens: int = 384,
        max_gen_tokens: int = 48
    ):
        self.checkpoint_path = checkpoint_path
        self.session_id = session_id
        self.user_id = user_id
        self.max_prompt_tokens = max_prompt_tokens
        self.max_gen_tokens = max_gen_tokens
        
        # Load Tokenizer
        self.tokenizer, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
        
        # Load Checkpoint & Model
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "config" in ckpt:
            cfg_dict = ckpt["config"]
            # Filter valid keys for ModelConfig
            valid_keys = ModelConfig.__annotations__.keys() if hasattr(ModelConfig, '__annotations__') else cfg_dict.keys()
            filtered_cfg = {k: v for k, v in cfg_dict.items() if k in valid_keys}
            self.cfg = ModelConfig(**filtered_cfg)
        else:
            self.cfg = ModelConfig()
            
        self.model = ChakrMicro(self.cfg)
        sd = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
        self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        
        self.sampler = Sampler()
        self.memory = HybridMemoryEngine(session_id=session_id, user_id=user_id)
        self.history: List[Dict[str, str]] = []
        self.turn_counter = 0

    def chat_turn(self, user_text: str) -> str:
        self.turn_counter += 1
        
        # Handle special commands
        clean_text = user_text.strip()
        if clean_text == "/reset":
            self.history.clear()
            self.memory.reset_session()
            return "[Session and conversational memory reset.]"
        elif clean_text == "/memory":
            facts = self.memory.facts
            if not facts:
                return "[Conversational memory is currently empty.]"
            lines = [f"- {f.predicate.capitalize()}: {f.value} (Turn {f.turn_index})" for f in facts.values() if f.is_active]
            return "Active Memory Facts:\n" + "\n".join(lines)
        elif clean_text == "/info":
            param_count = sum(p.numel() for p in self.model.parameters())
            return (
                f"--- ChakrView Session Info ---\n"
                f"Checkpoint:      {self.checkpoint_path.name}\n"
                f"Architecture:    ChakrMicro/Medium ({param_count:,} params, {self.cfg.d_model} dim, {self.cfg.n_layers} layers)\n"
                f"Active Turns:    {len(self.history)}\n"
                f"Active Facts:    {len(self.memory.facts)}\n"
                f"Memory Engine:   Hybrid Deterministic KV + Neural Context"
            )

        # 1. Extract and store any explicit facts from user message
        extracted = self.memory.extract_and_store_facts(user_text, turn_index=self.turn_counter)
        
        # 2. Retrieve relevant facts for prompt conditioning
        relevant_facts = self.memory.retrieve_relevant_facts(user_text)
        mem_prefix = self.memory.format_memory_context(relevant_facts)
        
        # 3. Construct dialogue prompt with dynamic history packing
        prompt_parts = []
        if mem_prefix:
            prompt_parts.append(mem_prefix.strip())
            
        # Add conversation history
        for turn in self.history[-3:]: # retain last 3 turns
            prompt_parts.append(f"User: {turn['user']}\nAssistant: {turn['assistant']}")
            
        prompt_parts.append(f"User: {user_text}\nAssistant:")
        full_prompt = "\n".join(prompt_parts)
        
        # 4. Tokenize & truncate if needed
        tokens = self.tokenizer.encode(full_prompt, add_bos=True, add_eos=False)
        if len(tokens) > self.max_prompt_tokens:
            tokens = tokens[-self.max_prompt_tokens:]
            
        # 5. Autoregressive Neural Generation
        curr_tokens = list(tokens)
        gen_tokens = []
        sampling_cfg = SamplingConfig(temperature=0.0, top_k=0)
        
        with torch.no_grad():
            for step in range(self.max_gen_tokens):
                t_in = torch.tensor([curr_tokens], dtype=torch.long)
                logits = self.model(t_in)[0, -1, :]
                next_tok = self.sampler.sample(
                    logits,
                    generated_tokens=gen_tokens,
                    config=sampling_cfg,
                    step=step
                )
                if next_tok == 1: # EOS token
                    break
                gen_tokens.append(next_tok)
                curr_tokens.append(next_tok)
                
        response = self.tokenizer.decode(gen_tokens).strip()
        
        # Clean response string if prompt echoed
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        if "User:" in response:
            response = response.split("User:")[0].strip()
            
        # Record turn in history
        self.history.append({"user": user_text, "assistant": response})
        return response
