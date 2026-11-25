"""
ChakrView Conversation Session Manager.
Mission CV-GENESIS-NR-012.

Features:
- Multi-turn conversation management with unique session IDs.
- Dual-Path cognitive routing: Pure Neural vs Hybrid Cognitive.
- Context window sliding / history summarization / turn trimming.
- Integration with EpisodicMemoryBuffer and DeterministicMathRouter.
- Full session persistence and export to JSON.
"""

from __future__ import annotations
import os
import sys
import uuid
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.cognitive_architecture import CognitiveController, EpisodicMemoryBuffer
from chakrview.runtime.inference_17m import ChakrViewInferenceEngine17M


class ConversationSession:
    """Manages an active conversational dialogue session."""

    def __init__(
        self,
        engine: ChakrViewInferenceEngine17M,
        session_id: Optional[str] = None,
        mode: str = "hybrid",  # "pure_neural" or "hybrid"
        max_history_turns: int = 6,
    ):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.engine = engine
        self.mode = mode
        self.max_history_turns = max_history_turns
        self.history: List[Dict[str, Any]] = []
        self.cognitive_controller = CognitiveController()

    def set_mode(self, mode: str):
        if mode not in ["pure_neural", "hybrid"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'pure_neural' or 'hybrid'.")
        self.mode = mode

    def clear_memory(self):
        """Clears working memory, facts, and conversation history."""
        self.history.clear()
        self.cognitive_controller.memory = EpisodicMemoryBuffer(max_turns=self.max_history_turns)

    def get_memory_state(self) -> Dict[str, Any]:
        """Returns inspected state of episodic memory and asserted facts."""
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "turns_count": len(self.history),
            "stored_facts": dict(self.cognitive_controller.memory.facts),
            "history_summary": [
                {"role": h["role"], "content": h["content"][:60] + "..." if len(h["content"]) > 60 else h["content"]}
                for h in self.history
            ]
        }

    def chat(
        self,
        user_message: str,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15,
        max_new_tokens: int = 128,
    ) -> Dict[str, Any]:
        """
        Executes a turn of dialogue.
        Respects active mode (pure_neural vs hybrid).
        """
        t0 = time.perf_counter()

        def neural_fn(prompt_str: str) -> str:
            res = self.engine.generate(
                prompt=prompt_str,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            return res["response"]

        if self.mode == "pure_neural":
            # Direct neural generation without cognitive tool interventions
            prompt = f"User: {user_message}\nAssistant:"
            neural_res = self.engine.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            response_text = neural_res["response"]
            result = {
                "response": response_text,
                "origin": "pure_neural",
                "resolved_via_tool": False,
                "tokens_generated": neural_res["tokens_generated"],
                "tokens_per_sec": neural_res["tokens_per_sec"],
                "first_token_ms": neural_res["first_token_ms"],
                "latency_sec": time.perf_counter() - t0,
            }
        else:
            # Hybrid cognitive processing
            cog_result = self.cognitive_controller.process(
                query=user_message,
                pure_neural_fn=neural_fn,
                mode="hybrid"
            )
            response_text = cog_result["response"]
            result = {
                "response": response_text,
                "origin": cog_result["origin"],
                "resolved_via_tool": cog_result["resolved_via_tool"],
                "latency_sec": time.perf_counter() - t0,
            }

        # Record in history
        self.history.append({"role": "user", "content": user_message, "timestamp": time.time()})
        self.history.append({
            "role": "assistant",
            "content": response_text,
            "origin": result["origin"],
            "resolved_via_tool": result.get("resolved_via_tool", False),
            "timestamp": time.time()
        })

        # Trim history if exceeding limit
        if len(self.history) > self.max_history_turns * 2:
            self.history = self.history[-self.max_history_turns * 2:]

        return result

    def export_session(self, output_path: str | Path):
        """Dumps session transcript and memory state to a JSON file."""
        data = {
            "session_id": self.session_id,
            "mode": self.mode,
            "memory_state": self.get_memory_state(),
            "full_history": self.history
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
