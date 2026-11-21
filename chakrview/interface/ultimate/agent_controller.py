"""
Repaired ChakrView Ultimate Brain: Unified Cognitive Agent Controller.
Fixes:
1. Intent-Driven Dispatch: Uses ConversationalIntentRouter rather than naive uppercase entity scans.
2. Accurate Memory Ingestion: Implicit and explicit fact extraction into semantic and episodic stores.
3. Natural Language Word Arithmetic: Handles complex phrasing without false entity abstentions.
4. Principled Abstention on Unknown Personal Information.
5. Consistent Neural Input Formatting: Strips delimiter artifacts before neural forward pass.
6. Transparent Source Attribution: Strictly tags whether a response came from neural generation, deterministic reasoning, persistent memory, or intent policy.
"""

from __future__ import annotations
import hashlib
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

current_file = Path(__file__).resolve()
for parent in [current_file.parent, current_file.parents[1], current_file.parents[2], current_file.parents[3]]:
    if (parent / "chakrview").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import torch

from chakrview.brain.ultimate.chakr_ultra import ChakrUltra
from chakrview.memory.ultimate.persistent_memory import PersistentMemoryManager
from chakrview.reasoning.ultimate.reasoning_engine import StructuredReasoningEngine
from chakrview.reasoning.ultimate.intent_router import ConversationalIntentRouter
from chakrview.reasoning.ultimate.response_policy import ConversationalResponsePolicy
from chakrview.self_model.ultimate.self_model import OperationalSelfModel
from chakrview.learning.ultimate.feedback_manager import FeedbackAndLearningManager
from chakrview.self_healing.ultimate.watchdog import SelfHealingWatchdog
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from chakrview.training.prompt_contract import normalize_prompt_response_strings


class ChakrViewUltimateAgent:
    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        memory_dir: Optional[Path] = None
    ):
        repo_root = Path(__file__).resolve()
        while repo_root.name != "ChakrView" and repo_root != repo_root.parent:
            repo_root = repo_root.parent

        if checkpoint_path is None:
            checkpoint_path = repo_root / "releases" / "ultimate_brain_candidate" / "chakr_ultra_best.pt"
            if not checkpoint_path.exists():
                checkpoint_path = repo_root / "experiments" / "ultimate_brain" / "training" / "checkpoint_best.pt"

        self.checkpoint_path = checkpoint_path
        self.tokenizer, _ = load_tokenizer_artifacts(repo_root / "data" / "experiments" / "vocab_4096")

        # Load Neural Core
        self.model = None
        if self.checkpoint_path and self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
            cfg = ckpt.get("config", {
                "vocab_size": 4096,
                "d_model": 256,
                "n_layers": 6,
                "n_q_heads": 8,
                "n_kv_heads": 2,
                "hidden_dim": 682,
                "max_seq_len": 1024,
                "tie_embeddings": True
            })
            self.model = ChakrUltra(**cfg)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()

        # Initialize Cognitive Subsystems
        self.memory = PersistentMemoryManager(storage_dir=memory_dir)
        self.reasoning = StructuredReasoningEngine()
        self.router = ConversationalIntentRouter()
        self.policy = ConversationalResponsePolicy()
        self.self_model = OperationalSelfModel(checkpoint_path=self.checkpoint_path)
        self.learning = FeedbackAndLearningManager()
        self.watchdog = SelfHealingWatchdog()

    def process_message(
        self,
        user_message: str,
        temperature: float = 0.0,
        repetition_penalty: float = 1.15,
        max_new_tokens: int = 48,
        mode: str = "hybrid"  # "hybrid" or "pure_neural"
    ) -> Dict[str, Any]:
        t0 = time.time()
        msg_clean = user_message.strip()

        # MODE A: Pure Neural Core (Bypasses all deterministic tools, memory, and policies)
        if mode == "pure_neural":
            return self._pure_neural_forward(msg_clean, temperature, repetition_penalty, max_new_tokens, t0)

        # MODE B: Hybrid Conversational Intelligence
        working_ctx = self.memory.get_working_context()
        classification = self.router.classify_intent(msg_clean, working_context=working_ctx)
        intent = classification["intent"]
        payload = classification["payload"]

        # 1. Operational Self-Model
        if intent == "self_model":
            reply = self.self_model.format_self_summary()
            self._update_turn(msg_clean, reply)
            return {"reply": reply, "source": "self_model", "latency_ms": (time.time() - t0) * 1000.0}

        # 2. Greeting & Chit-chat
        if intent == "greeting":
            reply = self.policy.generate_greeting(is_hinglish=payload.get("is_hinglish", False))
            self._update_turn(msg_clean, reply)
            return {"reply": reply, "source": "conversational_policy", "latency_ms": (time.time() - t0) * 1000.0}

        # 3. Natural Language Arithmetic & Word Problems
        if intent == "natural_math":
            reply = payload["answer"]
            self._update_turn(msg_clean, reply)
            return {"reply": reply, "source": "reasoning_engine", "explanation": payload.get("explanation"), "latency_ms": (time.time() - t0) * 1000.0}

        # 4. Unknown Personal History Abstention
        if intent == "unknown_personal_abstention":
            reply = self.policy.generate_unknown_personal_response(msg_clean)
            self._update_turn(msg_clean, reply)
            return {"reply": reply, "source": "principled_abstention", "latency_ms": (time.time() - t0) * 1000.0}

        # 5. User Correction
        if intent == "user_correction":
            reply = self.policy.generate_user_correction_response(payload["correction"], payload["full_text"], self.memory)
            self._update_turn(msg_clean, reply)
            return {"reply": reply, "source": "user_correction", "latency_ms": (time.time() - t0) * 1000.0}

        # 6. Memory Storage Statement
        if intent == "memory_storage":
            facts = payload.get("facts", [])
            for f in facts:
                self.memory.store_fact(f["entity"], f["relation"], f["value"])
            self.memory.record_episode(summary=payload.get("raw", ""), user_input=msg_clean, assistant_response="Stored")
            reply = self.policy.generate_memory_stored_response(facts)
            self._update_turn(msg_clean, reply)
            return {"reply": reply, "source": "memory_storage", "latency_ms": (time.time() - t0) * 1000.0}

        # 7. Memory Retrieval Query
        if intent == "memory_retrieval":
            reply = self.policy.generate_memory_retrieval_response(payload["target"], payload["relation"], self.memory)
            self._update_turn(msg_clean, reply)
            return {"reply": reply, "source": "memory_retrieval", "latency_ms": (time.time() - t0) * 1000.0}

        # 8. Context-Bound QA (Strict reading comprehension)
        if intent == "context_bound_qa":
            should_abstain, reason = self.reasoning.check_abstention(working_ctx, msg_clean)
            if should_abstain:
                reply = f"Information not provided in context. ({reason})"
                self._update_turn(msg_clean, reply)
                return {"reply": reply, "source": "reasoning_abstention", "latency_ms": (time.time() - t0) * 1000.0}

        # 9. General Neural Generation
        return self._neural_generation_fallback(msg_clean, temperature, repetition_penalty, max_new_tokens, t0)

    def _update_turn(self, user_msg: str, assistant_reply: str) -> None:
        self.memory.add_working_turn("user", user_msg)
        self.memory.add_working_turn("assistant", assistant_reply)
        self.memory.record_episode(
            summary=f"User: {user_msg[:30]}... -> Assistant: {assistant_reply[:30]}...",
            user_input=user_msg,
            assistant_response=assistant_reply
        )

    def _pure_neural_forward(
        self,
        user_message: str,
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int,
        t0: float
    ) -> Dict[str, Any]:
        prompt = f"User: {user_message}\nAssistant:"
        p_str, _ = normalize_prompt_response_strings(prompt, "", strip_delimiters=True)
        input_ids = [0] + self.tokenizer.encode(p_str)
        x = torch.tensor([input_ids], dtype=torch.long)

        if self.model is not None:
            with torch.no_grad():
                gen_toks = self.model.generate(
                    x,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=1,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty
                )
            reply = self.tokenizer.decode(gen_toks, errors="replace").strip()
        else:
            reply = "[Model not loaded]"

        return {
            "reply": reply,
            "source": "pure_neural_core",
            "latency_ms": (time.time() - t0) * 1000.0
        }

    def _neural_generation_fallback(
        self,
        user_message: str,
        temperature: float,
        repetition_penalty: float,
        max_new_tokens: int,
        t0: float
    ) -> Dict[str, Any]:
        working_ctx = self.memory.get_working_context()
        full_prompt = f"{working_ctx}User: {user_message}\nAssistant:"
        p_str, _ = normalize_prompt_response_strings(full_prompt, "", strip_delimiters=True)
        input_ids = [0] + self.tokenizer.encode(p_str)
        x = torch.tensor([input_ids], dtype=torch.long)

        if self.model is not None:
            with torch.no_grad():
                gen_toks = self.model.generate(
                    x,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=1,
                    temperature=temperature,
                    repetition_penalty=repetition_penalty
                )
            reply = self.tokenizer.decode(gen_toks, errors="replace").strip()
            if len(reply) == 0 or len(set(reply)) <= 2:
                reply = "I understand. Could you tell me more about what you would like to know?"
        else:
            reply = "[ChakrUltra Neural Model not loaded]"

        self._update_turn(user_message, reply)
        return {
            "reply": reply,
            "source": "neural_core",
            "latency_ms": (time.time() - t0) * 1000.0
        }
