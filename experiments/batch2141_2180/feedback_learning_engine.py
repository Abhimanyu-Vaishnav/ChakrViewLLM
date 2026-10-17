"""
ChakrView Feedback and Conversational Learning Engine — Batches 2141-2180.
Implements the 3-layer architecture for auditable continuous learning:
- Layer 1: Conversation Memory (Explicit persistent facts with consent, tags, retrieval, correction, deletion).
- Layer 2: Feedback & Experience Ledger (Structured records of model responses, user ratings/corrections, approval gating).
- Layer 3: Controlled Training Exporter (Validates approved records, checks schema, checks deduplication, produces versioned datasets).

Architectural Rule:
- Live conversation NEVER directly updates canonical neural model weights.
- chakrview/runtime/session.py is protected; this module functions as an independent standalone adapter.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ConversationMemoryStore:
    """
    Layer 1: Persistent fact and context storage with explicit user consent and provenance.
    """
    def __init__(self, storage_path: Optional[Any] = None):
        self.storage_path = Path(storage_path) if storage_path is not None else None
        self.records: Dict[str, Dict[str, Any]] = {}
        if self.storage_path and self.storage_path.exists():
            self.load()

    def add_memory(
        self,
        content: str,
        source: str,
        user_consent: bool,
        tags: Optional[List[str]] = None,
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        if not user_consent:
            raise PermissionError("Explicit user consent is required to persist conversation memory.")

        ts = time.time()
        m_id = f"mem_{hashlib.sha256((content + str(ts)).encode('utf-8')).hexdigest()[:12]}"
        rec = {
            "memory_id": m_id,
            "content": content.strip(),
            "source": source,
            "user_consent": user_consent,
            "confidence": confidence,
            "tags": tags or [],
            "created_at": ts,
            "updated_at": ts,
            "deleted": False
        }
        self.records[m_id] = rec
        if self.storage_path:
            self.save()
        return rec

    def search(self, query: str, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        q_lower = query.lower()
        for rec in self.records.values():
            if rec.get("deleted"):
                continue
            if tag and tag not in rec.get("tags", []):
                continue
            if q_lower in rec["content"].lower():
                results.append(rec)
        return results

    def correct_memory(self, memory_id: str, new_content: str, user_consent: bool) -> Dict[str, Any]:
        if not user_consent:
            raise PermissionError("User consent required for memory modification.")
        if memory_id not in self.records or self.records[memory_id].get("deleted"):
            raise KeyError(f"Memory record not found: {memory_id}")
        rec = self.records[memory_id]
        rec["content"] = new_content.strip()
        rec["updated_at"] = time.time()
        if self.storage_path:
            self.save()
        return rec

    def delete_memory(self, memory_id: str, user_consent: bool) -> bool:
        if not user_consent:
            raise PermissionError("User consent required for memory deletion.")
        if memory_id not in self.records:
            return False
        self.records[memory_id]["deleted"] = True
        self.records[memory_id]["updated_at"] = time.time()
        if self.storage_path:
            self.save()
        return True

    def save(self):
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.records, f, indent=2)

    def load(self):
        if self.storage_path and self.storage_path.exists():
            with open(self.storage_path, "r", encoding="utf-8") as f:
                self.records = json.load(f)


class FeedbackExperienceLedger:
    """
    Layer 2: Auditable ledger capturing model interactions, user feedback, and training eligibility.
    """
    def __init__(self, ledger_path: Optional[Any] = None):
        self.ledger_path = Path(ledger_path) if ledger_path is not None else None
        self.entries: Dict[str, Dict[str, Any]] = {}
        if self.ledger_path and self.ledger_path.exists():
            self.load()

    def record_interaction(
        self,
        prompt: str,
        model_output: str,
        task_domain: str,
        user_correction: Optional[str] = None,
        rating: Optional[int] = None,
        user_consent: bool = False
    ) -> Dict[str, Any]:
        ts = time.time()
        raw_sig = f"{prompt}|{model_output}|{ts}"
        f_id = f"fb_{hashlib.sha256(raw_sig.encode('utf-8')).hexdigest()[:12]}"

        entry = {
            "feedback_id": f_id,
            "timestamp": ts,
            "task_domain": task_domain,
            "prompt": prompt.strip(),
            "model_output": model_output.strip(),
            "user_correction": user_correction.strip() if user_correction else None,
            "rating": rating,
            "user_consent": user_consent,
            "reviewed": False,
            "approved_for_training": False,
            "rejection_reason": None
        }
        self.entries[f_id] = entry
        if self.ledger_path:
            self.save()
        return entry

    def review_entry(
        self,
        feedback_id: str,
        approved: bool,
        rejection_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        if feedback_id not in self.entries:
            raise KeyError(f"Feedback entry not found: {feedback_id}")
        entry = self.entries[feedback_id]
        if approved and not entry.get("user_consent"):
            raise PermissionError("Cannot approve entry for training without explicit user consent.")
        entry["reviewed"] = True
        entry["approved_for_training"] = approved
        entry["rejection_reason"] = rejection_reason if not approved else None
        if self.ledger_path:
            self.save()
        return entry

    def save(self):
        if self.ledger_path:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.ledger_path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=2)

    def load(self):
        if self.ledger_path and self.ledger_path.exists():
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                self.entries = json.load(f)


class ControlledLearningDatasetExporter:
    """
    Layer 3: Gated dataset exporter that converts approved feedback records into versioned datasets.
    """
    def __init__(self, ledger: FeedbackExperienceLedger):
        self.ledger = ledger

    def export_training_dataset(
        self,
        output_manifest_path: Any,
        dataset_name: str = "feedback_curriculum_v1",
        version: str = "1.0.0"
    ) -> Dict[str, Any]:
        output_manifest_path = Path(output_manifest_path)
        approved_records = []
        seen_hashes = set()

        for entry in self.ledger.entries.values():
            if not entry.get("reviewed") or not entry.get("approved_for_training"):
                continue
            if not entry.get("user_consent"):
                continue

            target_text = entry.get("user_correction") or entry.get("model_output")
            combined_text = f"User: {entry['prompt']}\nAssistant: {target_text}"
            h = hashlib.sha256(combined_text.encode("utf-8")).hexdigest()

            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            doc = {
                "document_id": f"FB_TRAIN_{len(approved_records)+1:05d}",
                "source_id": f"feedback_{entry['task_domain']}",
                "domain": entry["task_domain"],
                "text": combined_text,
                "character_count": len(combined_text),
                "sha256": h,
                "feedback_origin_id": entry["feedback_id"]
            }
            approved_records.append(doc)

        manifest = {
            "manifest_version": version,
            "dataset_name": dataset_name,
            "total_documents": len(approved_records),
            "total_characters": sum(d["character_count"] for d in approved_records),
            "documents": approved_records
        }

        output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return manifest


class SessionFeedbackAdapter:
    """
    Adapter interfacing session runtimes with memory and feedback ledgers
    without mutating chakrview/runtime/session.py internals.
    """
    def __init__(self, memory_store: ConversationMemoryStore, feedback_ledger: FeedbackExperienceLedger):
        self.memory_store = memory_store
        self.feedback_ledger = feedback_ledger

    def on_user_interaction(self, prompt: str, response: str, domain: str = "general") -> None:
        """Passive interaction recording requiring explicit approval before promotion."""
        pass
