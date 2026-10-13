"""
Persistent Multi-Turn Session Memory & Storage Interface for ChakrView.
Location: chakrview/runtime/persistent_memory.py

Provides:
1. Isolated persistent conversation state per session ID.
2. Safe JSON/JSONL serialization with automatic atomic file flushing.
3. Turn budget compaction and window pruning.
4. Error recovery on corrupted or missing session logs.
5. Strict isolation between tenant/session stores.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional


@dataclass
class PersistentTurn:
    role: str  # "system", "user", "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PersistentTurn:
        return cls(**data)


class PersistentSessionMemory:
    """
    Manages persistent on-disk storage and in-memory caching of multi-turn dialogues.
    """

    def __init__(self, storage_dir: Path, max_history_turns: int = 50) -> None:
        self.storage_dir = Path(storage_dir)
        self.max_history_turns = max_history_turns
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[PersistentTurn]] = {}

    def _get_session_path(self, session_id: str) -> Path:
        safe_id = "".join(c for c in session_id if c.isalnum() or c in ("-", "_"))
        if not safe_id:
            safe_id = "default_session"
        return self.storage_dir / f"session_{safe_id}.json"

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PersistentTurn:
        turns = self.get_history(session_id)
        turn = PersistentTurn(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=dict(metadata or {}),
        )
        turns.append(turn)
        if len(turns) > self.max_history_turns:
            turns = turns[-self.max_history_turns :]

        self._cache[session_id] = turns
        self._save(session_id, turns)
        return turn

    def get_history(self, session_id: str, reload: bool = False) -> List[PersistentTurn]:
        if not reload and session_id in self._cache:
            return list(self._cache[session_id])

        path = self._get_session_path(session_id)
        if not path.exists():
            self._cache[session_id] = []
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            turns = [PersistentTurn.from_dict(t) for t in data.get("turns", [])]
            self._cache[session_id] = turns
            return list(turns)
        except Exception:
            # Corrupted storage recovery fallback: rename corrupted file and reinitialize
            corrupted_path = path.with_suffix(".corrupted")
            try:
                if corrupted_path.exists():
                    corrupted_path.unlink()
                path.rename(corrupted_path)
            except Exception:
                pass
            self._cache[session_id] = []
            return []

    def _save(self, session_id: str, turns: List[PersistentTurn]) -> None:
        path = self._get_session_path(session_id)
        tmp_path = path.with_suffix(".tmp")
        payload = {
            "session_id": session_id,
            "turn_count": len(turns),
            "updated_at": time.time(),
            "turns": [t.to_dict() for t in turns],
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        tmp_path.replace(path)

    def clear(self, session_id: str) -> None:
        self._cache.pop(session_id, None)
        path = self._get_session_path(session_id)
        if path.exists():
            path.unlink(missing_ok=True)
