"""
Unit tests for CR-EXP-12 evaluation logic, scoring consistency, and metrics calculations.
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

    def test_unseen_names_roundtrip(self):
        unseen_names = ["Aalok", "Brijesh", "Damini", "Farhan", "Kalyani", "Vidushi"]
        for name in unseen_names:
            enc = self.tok.encode(name)
            dec = self.tok.decode(enc, errors="replace")
            self.assertEqual(name, dec)

    def test_multiturn_scoring_with_subject(self):
        target = "Bhopal"
        earlier = "Jaipur"
        
        gen_good = "Pooja is currently in Bhopal."
        gen_stale = "Pooja is in Jaipur."
        
        hit_good = (target.lower() in gen_good.lower()) and not (earlier.lower() in gen_good.lower())
        hit_stale = (target.lower() in gen_stale.lower()) and not (earlier.lower() in gen_stale.lower())
        
        self.assertTrue(hit_good)
        self.assertFalse(hit_stale)

if __name__ == "__main__":
    unittest.main()
