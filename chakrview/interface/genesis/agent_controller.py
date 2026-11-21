"""
ChakrGenesis Unified Agent Controller & Cognitive Orchestrator.
Integrates:
1. ChakrGenesis 5.18M Neural Model (pre-trained on 6.65M tokens + conversational SFT)
2. Conversational Intent Dispatch Router
3. Transparent Relational Persistent Memory
4. Multi-container arithmetic solver & verification
5. Principled Personal Abstention engine
6. Dual-Mode execution:
   - "pure_neural": Direct autoregressive neural generation (no tools, no templates).
   - "hybrid": Synergistic neural generation guided by cognitive tools and memory with transparent attribution.
"""

from __future__ import annotations
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from chakrview.brain.genesis.chakr_genesis import ChakrGenesis, GenesisConfig
from chakrview.memory.ultimate.persistent_memory import PersistentMemoryManager
from chakrview.reasoning.ultimate.intent_router import ConversationalIntentRouter
from chakrview.reasoning.ultimate.response_policy import ConversationalResponsePolicy
from chakrview.tokenizer.serialization import load_tokenizer_artifacts


class ChakrGenesisAgent:
    def __init__(
        self,
        checkpoint_path: str = "experiments/project_genesis/conversation_training/genesis_sft_best.pt",
        tokenizer_path: str = "experiments/tokenizer/candidates/v4096",
        memory_dir: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.tokenizer, _ = load_tokenizer_artifacts(tokenizer_path)

        ckpt_file = Path(checkpoint_path)
        if ckpt_file.exists():
            ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
            self.config = ckpt["config"]
            self.model = ChakrGenesis(self.config).to(device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            print(f"ChakrGenesis Neural Core loaded successfully from {checkpoint_path} on {device}.")
        else:
            self.config = GenesisConfig()
            self.model = ChakrGenesis(self.config).to(device)
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Initialized randomly.")

        self.memory = PersistentMemoryManager(storage_dir=memory_dir)
        self.router = ConversationalIntentRouter()
        self.policy = ConversationalResponsePolicy()

    def process_message(self, user_message: str, mode: str = "hybrid") -> Dict[str, Any]:
        t0 = time.time()
        msg_clean = user_message.strip()

        # ---------------- MODE A: Pure Neural Core ----------------
        if mode == "pure_neural":
            prompt = msg_clean
            if not prompt.endswith("Assistant:"):
                prompt = f"User: {prompt}\nAssistant:"
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
            output_ids = self.model.generate(input_ids, max_new_tokens=40, temperature=0.6, top_k=40)
            full_text = self.tokenizer.decode(output_ids[0].tolist(), errors="replace")
            # Extract reply after Assistant:
            if "Assistant:" in full_text:
                reply = full_text.split("Assistant:")[-1].strip()
            else:
                reply = full_text.strip()

            latency = (time.time() - t0) * 1000
            return {
                "reply": reply,
                "source": "pure_neural_core",
                "latency_ms": latency
            }

        # ---------------- MODE B: Full Cognitive Hybrid ----------------
        intent_info = self.router.classify_intent(msg_clean)
        intent = intent_info["intent"]
        payload = intent_info.get("payload", {})

        # 1. Operational Self-Model
        if intent == "self_model":
            reply = (
                "I am **ChakrView Genesis v1.0**, an independent, indigenous conversational AI system built from scratch.\n"
                "- **Neural Foundation:** ChakrGenesis (5,177,600 parameters, 6 layers, RoPE, Grouped-Query Attention, SwiGLU FFN).\n"
                "- **Pretraining:** Trained on the 6.65M token Stage-C multi-domain corpus + conversational instruction fine-tuning.\n"
                "- **Cognitive Architecture:** Persistent relational memory, intent router, and deterministic arithmetic verification."
            )
            source = "self_model"

        # 2. Natural Language Arithmetic
        elif intent == "natural_math":
            reply = payload.get("answer", "")
            source = "reasoning_engine"

        # 3. Persistent Memory Storage
        elif intent == "memory_storage":
            facts = payload.get("facts", [])
            for f in facts:
                self.memory.store_fact(f["entity"], f["relation"], f["value"])
            reply = self.policy.generate_memory_stored_response(facts)
            source = "memory_storage"

        # 4. Persistent Memory Retrieval
        elif intent == "memory_retrieval":
            reply = self.policy.generate_memory_retrieval_response(
                payload.get("target", ""),
                payload.get("relation", ""),
                self.memory
            )
            source = "memory_retrieval"

        # 5. User Self-Correction
        elif intent == "user_correction":
            corr = payload.get("correction", "")
            self.memory.store_fact("user_context", "last_correction", corr)
            reply = f"Acknowledged. I have updated my understanding: {corr}."
            source = "user_correction"

        # 6. Principled Personal Abstention
        elif intent == "unknown_personal_abstention":
            reply = self.policy.generate_unknown_personal_response(payload.get("query", ""))
            source = "principled_abstention"

        # 7. Greetings & Social
        elif intent == "greeting":
            reply = self.policy.generate_greeting(is_hinglish=payload.get("is_hinglish", False))
            source = "conversational_policy"

        # 8. Neural-Assisted Conversation
        else:
            prompt = f"User: {msg_clean}\nAssistant:"
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
            output_ids = self.model.generate(input_ids, max_new_tokens=40, temperature=0.6, top_k=40)
            full_text = self.tokenizer.decode(output_ids[0].tolist(), errors="replace")
            if "Assistant:" in full_text:
                neural_reply = full_text.split("Assistant:")[-1].strip()
            else:
                neural_reply = full_text.strip()

            if len(neural_reply) > 5:
                reply = neural_reply
                source = "neural_core"
            else:
                reply = "I understand. What specific topic would you like to explore or discuss further?"
                source = "neural_fallback"

        # Record turn in working memory
        self.memory.add_working_turn("user", msg_clean)
        self.memory.add_working_turn("assistant", reply)

        latency = (time.time() - t0) * 1000
        return {
            "reply": reply,
            "source": source,
            "latency_ms": latency
        }
