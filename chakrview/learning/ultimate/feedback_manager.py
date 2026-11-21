"""
Interactive Feedback & Continuous Learning Subsystem.
Implements:
1. User feedback capture (correct, incorrect, explicit corrections).
2. Provenance tagging and timestamping.
3. Automated regression test generation from user corrections.
4. Experience replay queuing for scheduled offline model fine-tuning.
"""

from __future__ import annotations
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional


class FeedbackAndLearningManager:
    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path("data/feedback_store")
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.feedback_log_file = self.data_dir / "user_feedback_log.jsonl"
        self.replay_queue_file = self.data_dir / "experience_replay_queue.jsonl"
        self.regression_tests_file = self.data_dir / "generated_regression_tests.json"

    def record_feedback(
        self,
        prompt: str,
        predicted_response: str,
        is_correct: bool,
        corrected_response: Optional[str] = None,
        feedback_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        record = {
            "id": f"fb_{int(time.time()*1000)}",
            "timestamp": time.time(),
            "prompt": prompt,
            "predicted_response": predicted_response,
            "is_correct": is_correct,
            "corrected_response": corrected_response,
            "feedback_notes": feedback_notes
        }

        with open(self.feedback_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # If a correction was provided, queue for experience replay and generate regression test
        if corrected_response and not is_correct:
            self._queue_for_replay(prompt, corrected_response)
            self._generate_regression_test(prompt, corrected_response)

        return record

    def _queue_for_replay(self, prompt: str, target: str) -> None:
        replay_item = {
            "prompt": prompt,
            "response": target,
            "category": "user_feedback_correction",
            "language": "en",
            "turns": 1,
            "queued_at": time.time()
        }
        with open(self.replay_queue_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(replay_item) + "\n")

    def _generate_regression_test(self, prompt: str, expected: str) -> None:
        tests = []
        if self.regression_tests_file.exists():
            try:
                with open(self.regression_tests_file, "r", encoding="utf-8") as f:
                    tests = json.load(f)
            except Exception:
                tests = []

        tests.append({
            "test_id": f"reg_{len(tests)+1}",
            "prompt": prompt,
            "expected_contains": expected,
            "created_at": time.time()
        })

        with open(self.regression_tests_file, "w", encoding="utf-8") as f:
            json.dump(tests, f, indent=2)

    def get_queued_replay_examples(self) -> List[Dict[str, Any]]:
        if not self.replay_queue_file.exists():
            return []
        with open(self.replay_queue_file, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def get_regression_tests(self) -> List[Dict[str, Any]]:
        if not self.regression_tests_file.exists():
            return []
        with open(self.regression_tests_file, "r", encoding="utf-8") as f:
            return json.load(f)
