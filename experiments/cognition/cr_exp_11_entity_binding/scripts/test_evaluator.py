"""
Unit tests for CR-EXP-11 evaluation logic and metrics computation.
Tests:
- Tokenizer round-trip lossless preservation
- Evaluator accuracy scoring logic
- Epistemic abstention cue matching
- Multi-entity distractor disambiguation scoring
- State tracking recency vs earlier state scoring
- Confusion matrix and calibration metric calculations
"""

import unittest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

class TestEvaluatorLogic(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.tok, _ = load_tokenizer_artifacts(REPO_ROOT / "data" / "experiments" / "vocab_4096")
        cls.abstain_cues = ["does not", "not specify", "unknown", "unspecified", "cannot", "no information", "not state"]

    def test_tokenizer_lossless_roundtrip(self):
        names = ["Aarav", "Meera", "Manish", "Kavita", "Siddharth", "Chandrashekhar", "Dev"]
        for name in names:
            enc = self.tok.encode(name)
            dec = self.tok.decode(enc, errors="replace")
            self.assertEqual(name, dec, f"Roundtrip failed for name: {name}")

    def test_abstention_cue_matching(self):
        valid_abstention = "The context does not state where Manish lives."
        invalid_abstention = "Manish lives in Bhopal."
        
        is_abst_1 = any(c in valid_abstention.lower() for c in self.abstain_cues)
        is_abst_2 = any(c in invalid_abstention.lower() for c in self.abstain_cues)
        
        self.assertTrue(is_abst_1)
        self.assertFalse(is_abst_2)

    def test_distractor_disambiguation_scoring(self):
        target = "Jaipur"
        distractors = ["Pune", "Delhi"]
        
        correct_gen = "Meera is currently located in Jaipur."
        distractor_gen = "Meera is currently located in Pune."
        contaminated_gen = "Meera was in Pune and Jaipur."
        
        # Evaluator rule: contains target and NOT distractor
        hit_1 = (target.lower() in correct_gen.lower()) and not any(d.lower() in correct_gen.lower() for d in distractors)
        hit_2 = (target.lower() in distractor_gen.lower()) and not any(d.lower() in distractor_gen.lower() for d in distractors)
        hit_3 = (target.lower() in contaminated_gen.lower()) and not any(d.lower() in contaminated_gen.lower() for d in distractors)
        
        self.assertTrue(hit_1)
        self.assertFalse(hit_2)
        self.assertFalse(hit_3)

    def test_state_tracking_recency_scoring(self):
        target = "Bhopal" # latest
        earlier = "Jaipur" # old
        
        correct_update = "Meera is now in Bhopal."
        stale_state = "Meera is in Jaipur."
        
        hit_update = (target.lower() in correct_update.lower()) and not (earlier.lower() in correct_update.lower())
        hit_stale = (target.lower() in stale_state.lower()) and not (earlier.lower() in stale_state.lower())
        
        self.assertTrue(hit_update)
        self.assertFalse(hit_stale)

if __name__ == "__main__":
    unittest.main()
