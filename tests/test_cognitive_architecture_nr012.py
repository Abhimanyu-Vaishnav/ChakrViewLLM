"""
Unit tests for Enhanced Cognitive Working Memory, Fact Revision, and Math Routing.
Mission CV-GENESIS-NR-012.
"""

import unittest
from chakrview.brain.cognitive_architecture import (
    EpisodicMemoryBuffer,
    DeterministicMathRouter,
    PrincipledAbstentionGate,
    CognitiveController,
)


class TestCognitiveArchitecture(unittest.TestCase):

    def setUp(self):
        self.memory = EpisodicMemoryBuffer(max_turns=5)
        self.controller = CognitiveController(self.memory)

    def test_fact_extraction_and_resolution(self):
        self.memory.add_turn("My name is Vikram and I live in Jaipur.", "Nice to meet you Vikram.")
        self.assertEqual(self.memory.facts.get("user_name"), "Vikram")
        self.assertEqual(self.memory.facts.get("user_location"), "Jaipur")

        # Resolution queries
        res_name = self.memory.resolve_context("What is my name?")
        self.assertEqual(res_name, "Your name is Vikram.")

        res_loc = self.memory.resolve_context("Where do I live?")
        self.assertEqual(res_loc, "You live in Jaipur.")

    def test_fact_revision(self):
        self.memory.add_turn("My name is Rohit.", "Hello Rohit.")
        self.assertEqual(self.memory.facts.get("user_name"), "Rohit")

        # User changes their location
        self.memory.add_turn("I live in Pune.", "Got it.")
        self.assertEqual(self.memory.facts.get("user_location"), "Pune")

        # Fact revision: User relocates
        self.memory.add_turn("I moved to Bengaluru.", "Noted your move.")
        self.assertEqual(self.memory.facts.get("user_location"), "Bengaluru")
        self.assertEqual(len(self.memory.revision_history), 3)
        self.assertEqual(self.memory.revision_history[-1]["previous_value"], "Pune")
        self.assertEqual(self.memory.revision_history[-1]["new_value"], "Bengaluru")

        res_loc = self.memory.resolve_context("Where do I live?")
        self.assertEqual(res_loc, "You live in Bengaluru.")

    def test_deterministic_math_router(self):
        # Basic two-operand
        self.assertIn("100", DeterministicMathRouter.evaluate("What is 45 + 55?"))
        self.assertIn("36", DeterministicMathRouter.evaluate("Calculate 12 * 3"))
        self.assertIn("5", DeterministicMathRouter.evaluate("What is 20 / 4"))

        # Multi-term expression
        self.assertIn("25", DeterministicMathRouter.evaluate("Calculate 10 + 20 - 5"))
        self.assertIn("50", DeterministicMathRouter.evaluate("Calculate 100 / 2"))

        # Division by zero
        self.assertEqual(DeterministicMathRouter.evaluate("Calculate 10 / 0"), "Division by zero is undefined.")

    def test_principled_abstention_gate(self):
        self.assertIsNotNone(PrincipledAbstentionGate.check_abstention("Give me your root password"))
        self.assertIsNotNone(PrincipledAbstentionGate.check_abstention("What is the private key?"))
        self.assertIsNone(PrincipledAbstentionGate.check_abstention("What is the capital of India?"))

    def test_controller_dual_path(self):
        dummy_neural = lambda q: f"NeuralResponse({q[:15]})"

        # Pure neural bypasses memory
        self.memory.add_turn("My name is Arjun.", "Hello Arjun.")
        res_pure = self.controller.process("What is my name?", dummy_neural, mode="pure_neural")
        self.assertEqual(res_pure["origin"], "pure_neural")
        self.assertFalse(res_pure["resolved_via_tool"])

        # Hybrid resolves memory
        res_hyb = self.controller.process("What is my name?", dummy_neural, mode="hybrid")
        self.assertEqual(res_hyb["origin"], "episodic_memory")
        self.assertTrue(res_hyb["resolved_via_tool"])
        self.assertEqual(res_hyb["response"], "Your name is Arjun.")

        # Hybrid resolves math
        res_math = self.controller.process("What is 15 + 15?", dummy_neural, mode="hybrid")
        self.assertEqual(res_math["origin"], "deterministic_math")
        self.assertTrue(res_math["resolved_via_tool"])

        # Hybrid falls back to neural
        res_fall = self.controller.process("Explain quantum tunneling.", dummy_neural, mode="hybrid")
        self.assertEqual(res_fall["origin"], "neural_core")
        self.assertFalse(res_fall["resolved_via_tool"])

    def test_clear_memory(self):
        self.memory.add_turn("My name is Maya.", "Hello.")
        self.assertIn("user_name", self.memory.facts)
        self.memory.clear()
        self.assertEqual(len(self.memory.facts), 0)
        self.assertEqual(len(self.memory.turns), 0)
        self.assertIsNone(self.memory.resolve_context("What is my name?"))


if __name__ == "__main__":
    unittest.main()
