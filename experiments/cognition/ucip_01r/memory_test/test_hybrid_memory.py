import sys
import unittest
from pathlib import Path

REPO_ROOT = Path("D:/Project/ChakrView")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.memory.hybrid_engine import HybridMemoryEngine

class TestHybridMemory(unittest.TestCase):
    def setUp(self):
        self.mem1 = HybridMemoryEngine("session_a", "user_1")
        self.mem2 = HybridMemoryEngine("session_b", "user_2")

    def test_direct_fact_recall(self):
        self.mem1.extract_and_store_facts("My name is Rohan and I live in Pune.", turn_index=1)
        name_fact = self.mem1.get_fact("user", "name")
        city_fact = self.mem1.get_fact("user", "city")
        self.assertIsNotNone(name_fact)
        self.assertEqual(name_fact.value, "Rohan")
        self.assertEqual(city_fact.value, "Pune")

    def test_fact_reassignment_and_provenance(self):
        self.mem1.extract_and_store_facts("My project is Yatrajivi.", turn_index=1)
        self.assertEqual(self.mem1.get_fact("user", "project").value, "Yatrajivi")
        
        # User renames project
        self.mem1.extract_and_store_facts("Actually, I renamed it to ChakrView.", turn_index=2)
        fact = self.mem1.get_fact("user", "project")
        self.assertEqual(fact.value, "ChakrView")
        self.assertEqual(fact.previous_value, "Yatrajivi")

    def test_session_isolation(self):
        self.mem1.extract_and_store_facts("My name is Alice.", turn_index=1)
        self.mem2.extract_and_store_facts("My name is Bob.", turn_index=1)
        
        self.assertEqual(self.mem1.get_fact("user", "name").value, "Alice")
        self.assertEqual(self.mem2.get_fact("user", "name").value, "Bob")
        self.assertIsNone(self.mem1.get_fact("user", "project"))

    def test_reset_behavior(self):
        self.mem1.extract_and_store_facts("My city is Mumbai.", turn_index=1)
        self.assertIsNotNone(self.mem1.get_fact("user", "city"))
        self.mem1.reset_session()
        self.assertIsNone(self.mem1.get_fact("user", "city"))
        self.assertEqual(len(self.mem1.facts), 0)

    def test_deterministic_relevance_retrieval(self):
        self.mem1.extract_and_store_facts("My name is Priya and my project is ChakrView.", turn_index=1)
        rel = self.mem1.retrieve_relevant_facts("What is my name?")
        vals = [f.value for f in rel]
        self.assertIn("Priya", vals)
        self.assertNotIn("ChakrView", vals)

    def test_multilingual_extraction(self):
        self.mem1.extract_and_store_facts("Mera naam Kabir hai aur mera project Bodhi hai.", turn_index=1)
        self.assertEqual(self.mem1.get_fact("user", "name").value, "Kabir")
        self.assertEqual(self.mem1.get_fact("user", "project").value, "Bodhi")

if __name__ == "__main__":
    unittest.main()
