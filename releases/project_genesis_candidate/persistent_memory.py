"""
Persistent Memory Subsystem for ChakrView Ultimate Brain.
Implements a 4-Tier Memory Hierarchy:
1. Short-Term Working Memory: active turns and session context.
2. Episodic Memory: past conversation events, interaction records, and user corrections.
3. Semantic Memory: verified factual triples (entity-relation-value) and domain knowledge.
4. Preference Memory: communication styles, user identity, and personal preferences.

Features:
- Structured JSON-backed persistent store.
- Exact and keyword-ranked semantic retrieval.
- Memory deduplication and collision resolution.
- Source attribution and confidence timestamps.
- Explicit inspection, update, and deletion operations.
"""

from __future__ import annotations
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple


class PersistentMemoryManager:
    def __init__(self, storage_dir: Optional[Path] = None):
        if storage_dir is None:
            storage_dir = Path("data/memory_store")
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.episodic_file = self.storage_dir / "episodic_memory.json"
        self.semantic_file = self.storage_dir / "semantic_memory.json"
        self.preference_file = self.storage_dir / "preference_memory.json"

        self.episodic: List[Dict[str, Any]] = self._load(self.episodic_file, default=[])
        self.semantic: Dict[str, Dict[str, Any]] = self._load(self.semantic_file, default={})
        self.preferences: Dict[str, Any] = self._load(self.preference_file, default={})

        self.working_memory: List[Dict[str, str]] = []

    def _load(self, filepath: Path, default: Any) -> Any:
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def _save_all(self) -> None:
        with open(self.episodic_file, "w", encoding="utf-8") as f:
            json.dump(self.episodic, f, indent=2)
        with open(self.semantic_file, "w", encoding="utf-8") as f:
            json.dump(self.semantic, f, indent=2)
        with open(self.preference_file, "w", encoding="utf-8") as f:
            json.dump(self.preferences, f, indent=2)

    # --- Working Memory ---
    def add_working_turn(self, role: str, text: str) -> None:
        self.working_memory.append({"role": role, "text": text, "timestamp": time.time()})
        if len(self.working_memory) > 10:
            self.working_memory.pop(0)

    def get_working_context(self) -> str:
        ctx = ""
        for turn in self.working_memory:
            r = "User" if turn["role"] == "user" else "Assistant"
            ctx += f"{r}: {turn['text']}\n"
        return ctx

    def clear_working_memory(self) -> None:
        self.working_memory.clear()

    # --- Episodic Memory ---
    def record_episode(self, summary: str, user_input: str, assistant_response: str, tags: Optional[List[str]] = None) -> None:
        episode = {
            "id": f"ep_{len(self.episodic) + 1}",
            "timestamp": time.time(),
            "summary": summary,
            "user_input": user_input,
            "assistant_response": assistant_response,
            "tags": tags or []
        }
        self.episodic.append(episode)
        self._save_all()

    def search_episodes(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        tokens = query.lower().split()
        matches = []
        for ep in self.episodic:
            score = sum(1 for t in tokens if t in ep["summary"].lower() or t in ep["user_input"].lower())
            if score > 0:
                matches.append((score, ep))
        matches.sort(key=lambda x: x[0], reverse=True)
        return [m[1] for m in matches[:top_k]]

    # --- Semantic Memory ---
    def store_fact(self, entity: str, relation: str, value: str, confidence: float = 1.0, source: str = "user_input") -> None:
        key = f"{entity.strip().lower()}::{relation.strip().lower()}"
        self.semantic[key] = {
            "entity": entity.strip(),
            "relation": relation.strip(),
            "value": value.strip(),
            "confidence": confidence,
            "source": source,
            "updated_at": time.time()
        }
        self._save_all()

    def get_fact(self, entity: str, relation: str) -> Optional[str]:
        key = f"{entity.strip().lower()}::{relation.strip().lower()}"
        if key in self.semantic:
            return self.semantic[key]["value"]
        return None

    def query_semantic(self, text: str) -> List[Dict[str, Any]]:
        tokens = text.lower().split()
        facts = []
        for key, fact in self.semantic.items():
            if any(t in fact["entity"].lower() or t in fact["relation"].lower() for t in tokens):
                facts.append(fact)
        return facts

    def delete_fact(self, entity: str, relation: str) -> bool:
        key = f"{entity.strip().lower()}::{relation.strip().lower()}"
        if key in self.semantic:
            del self.semantic[key]
            self._save_all()
            return True
        return False

    # --- Preference Memory ---
    def set_preference(self, key: str, value: Any) -> None:
        self.preferences[key] = {
            "value": value,
            "updated_at": time.time()
        }
        self._save_all()

    def get_preference(self, key: str, default: Any = None) -> Any:
        if key in self.preferences:
            return self.preferences[key]["value"]
        return default

    # --- Formatted Context for LLM Ingestion ---
    def build_relevant_context(self, current_prompt: str) -> str:
        parts = []
        # Check preferences
        if self.preferences:
            pref_strs = [f"{k}: {v['value']}" for k, v in self.preferences.items()]
            parts.append("User Preferences: " + ", ".join(pref_strs))

        # Check semantic facts
        facts = self.query_semantic(current_prompt)
        if facts:
            fact_strs = [f"{f['entity']} {f['relation']} is {f['value']}" for f in facts[:4]]
            parts.append("Relevant Stored Facts: " + "; ".join(fact_strs))

        # Check past episodes
        episodes = self.search_episodes(current_prompt, top_k=2)
        if episodes:
            ep_strs = [f"Past event: {ep['summary']}" for ep in episodes]
            parts.append("Past Interactions: " + "; ".join(ep_strs))

        if not parts:
            return ""
        return "[Memory Context]\n" + "\n".join(parts) + "\n[/Memory Context]\n"
