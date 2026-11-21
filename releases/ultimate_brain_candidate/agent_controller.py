"""
ChakrView Ultimate Brain: Unified Cognitive Agent Controller.
Harmonizes:
- ChakrUltra Neural Core (GQA + Gated Residuals)
- Persistent 4-Tier Memory (Working, Episodic, Semantic, Preference)
- Structured Reasoning & Verification Engine
- Operational Self-Model
- Continuous Feedback & Learning Manager
- Self-Healing Watchdog
"""

from __future__ import annotations
import hashlib
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is in sys.path regardless of location
current_file = Path(__file__).resolve()
for parent in [current_file.parent, current_file.parents[1], current_file.parents[2]]:
    if (parent / "chakrview").exists():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break

import torch

from chakrview.brain.ultimate.chakr_ultra import ChakrUltra
from chakrview.memory.ultimate.persistent_memory import PersistentMemoryManager
from chakrview.reasoning.ultimate.reasoning_engine import StructuredReasoningEngine
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
        if checkpoint_path is None:
            checkpoint_path = ROOT / "experiments" / "ultimate_brain" / "training" / "checkpoint_best.pt"
            if not checkpoint_path.exists():
                checkpoint_path = ROOT / "releases" / "ultimate_brain_candidate" / "chakr_ultra_best.pt"

        self.checkpoint_path = checkpoint_path
        self.tokenizer, _ = load_tokenizer_artifacts(ROOT / "data" / "experiments" / "vocab_4096")

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
        self.self_model = OperationalSelfModel(checkpoint_path=self.checkpoint_path)
        self.learning = FeedbackAndLearningManager()
        self.watchdog = SelfHealingWatchdog()

    def process_message(
        self,
        user_message: str,
        temperature: float = 0.0,
        repetition_penalty: float = 1.15,
        max_new_tokens: int = 48
    ) -> Dict[str, Any]:
        t0 = time.time()
        msg_clean = user_message.strip()

        # 1. Operational Self-Model Queries
        if any(kw in msg_clean.lower() for kw in ["who are you", "what is your architecture", "what are your capabilities", "tell me about yourself"]):
            reply = self.self_model.format_self_summary()
            self.memory.add_working_turn("user", msg_clean)
            self.memory.add_working_turn("assistant", reply)
            return {
                "reply": reply,
                "source": "self_model",
                "latency_ms": (time.time() - t0) * 1000.0
            }

        # 2. Memory Inquiries / Semantic Fact Updates
        # Check if user is teaching a fact: "NovaTrack was founded in 1981" or "Remember that my name is Alice"
        if "remember that" in msg_clean.lower():
            content = msg_clean.lower().replace("remember that", "").strip()
            if "my name is" in content:
                name = content.split("my name is")[-1].strip().capitalize()
                self.memory.set_preference("user_name", name)
                reply = f"I have saved to persistent preference memory: your name is {name}."
            else:
                self.memory.record_episode(summary=content, user_input=msg_clean, assistant_response="Remembered")
                reply = f"I have recorded this to episodic memory: {content}."
            self.memory.add_working_turn("user", msg_clean)
            self.memory.add_working_turn("assistant", reply)
            return {"reply": reply, "source": "memory_write", "latency_ms": (time.time() - t0) * 1000.0}

        # 3. Deterministic Arithmetic & Math Verification
        math_sol = self.reasoning.solve_math(msg_clean)
        if math_sol is not None:
            self.memory.add_working_turn("user", msg_clean)
            self.memory.add_working_turn("assistant", math_sol)
            return {
                "reply": math_sol,
                "source": "reasoning_engine",
                "latency_ms": (time.time() - t0) * 1000.0
            }

        # 4. Contextual Abstention Check
        working_ctx = self.memory.get_working_context()
        should_abstain, reason = self.reasoning.check_abstention(working_ctx, msg_clean)
        if should_abstain:
            reply = f"Information not provided in context. ({reason})"
            self.memory.add_working_turn("user", msg_clean)
            self.memory.add_working_turn("assistant", reply)
            return {
                "reply": reply,
                "source": "reasoning_abstention",
                "latency_ms": (time.time() - t0) * 1000.0
            }

        # 5. Neural Generation with Context Enrichment
        memory_ctx = self.memory.build_relevant_context(msg_clean)
        full_prompt = f"{memory_ctx}{working_ctx}User: {msg_clean}\nAssistant: "

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
        else:
            reply = "[ChakrUltra Neural Model not loaded]"

        # Update Working Memory & Episodic Store
        self.memory.add_working_turn("user", msg_clean)
        self.memory.add_working_turn("assistant", reply)
        self.memory.record_episode(
            summary=f"Q: {msg_clean[:40]}... -> A: {reply[:40]}...",
            user_input=msg_clean,
            assistant_response=reply
        )

        return {
            "reply": reply,
            "source": "neural_core_with_memory",
            "latency_ms": (time.time() - t0) * 1000.0
        }
