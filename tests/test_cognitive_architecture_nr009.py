"""
Unit Test Suite for ChakrView Cognitive Working Memory & Dual-Path Architecture.
Mission CV-GENESIS-NR-009 Phase 6.

Tests:
1. EpisodicMemoryBuffer fact extraction & resolution.
2. Fact updates and overwrite handling.
3. Multiple users/entities resolution.
4. Memory expiration beyond maximum turn capacity.
5. DeterministicMathRouter exact operations (+, -, *, /).
6. Math edge cases: division by zero, float arithmetic, multi-digit operations.
7. CognitiveController routing guarantees: pure neural isolation vs hybrid execution.
"""

import pytest
from chakrview.brain.cognitive_architecture import (
    EpisodicMemoryBuffer,
    DeterministicMathRouter,
    CognitiveController
)


class TestEpisodicMemoryBuffer:
    def test_single_fact_extraction(self):
        mem = EpisodicMemoryBuffer()
        mem.add_turn("My name is Vikram.", "Nice to meet you Vikram.")
        assert mem.resolve_context("What is my name?") == "Your name is Vikram."

    def test_fact_update_overwrite(self):
        mem = EpisodicMemoryBuffer()
        mem.add_turn("My name is Vikram.", "Hello Vikram.")
        assert mem.resolve_context("What is my name?") == "Your name is Vikram."
        mem.add_turn("Actually, call me Siddharth.", "Understood, Siddharth.")
        assert mem.resolve_context("What is my name?") == "Your name is Siddharth."

    def test_location_and_subject_tracking(self):
        mem = EpisodicMemoryBuffer()
        mem.add_turn("Ram lives in Delhi.", "Got it, Ram lives in Delhi.")
        assert mem.resolve_context("Where does he live?") == "Ram lives in Delhi."
        assert mem.resolve_context("What about his city?") == "Ram lives in Delhi."

    def test_turn_expiration(self):
        mem = EpisodicMemoryBuffer(max_turns=2)
        mem.add_turn("Turn 1", "Resp 1")
        mem.add_turn("Turn 2", "Resp 2")
        mem.add_turn("Turn 3", "Resp 3")
        assert len(mem.turns) == 2
        assert "Turn 1" not in mem.get_conversation_context()
        assert "Turn 3" in mem.get_conversation_context()


class TestDeterministicMathRouter:
    def test_basic_addition(self):
        res = DeterministicMathRouter.evaluate("What is 45 + 17?")
        assert res == "The result of 45 + 17 is 62."

    def test_basic_subtraction(self):
        res = DeterministicMathRouter.evaluate("Calculate 100 - 37.")
        assert res == "The result of 100 - 37 is 63."

    def test_basic_multiplication(self):
        res = DeterministicMathRouter.evaluate("What is 12 * 8?")
        assert res == "The result of 12 * 8 is 96."

    def test_basic_division(self):
        res = DeterministicMathRouter.evaluate("Compute 144 / 12.")
        assert res == "The result of 144 / 12 is 12."

    def test_division_by_zero(self):
        res = DeterministicMathRouter.evaluate("What is 25 / 0?")
        assert res == "Division by zero is undefined."

    def test_non_math_queries_return_none(self):
        assert DeterministicMathRouter.evaluate("Who is the Prime Minister of India?") is None
        assert DeterministicMathRouter.evaluate("Explain why a computer needs RAM.") is None


class TestCognitiveController:
    def test_pure_neural_mode_isolation(self):
        called = False
        def mock_neural(prompt):
            nonlocal called
            called = True
            return "neural response"

        controller = CognitiveController()
        # Even on math query, pure_neural mode MUST bypass math router
        res = controller.process("What is 10 + 10?", mock_neural, mode="pure_neural")
        assert called is True
        assert res["origin"] == "pure_neural"
        assert res["response"] == "neural response"

    def test_hybrid_mode_routes_math(self):
        called = False
        def mock_neural(prompt):
            nonlocal called
            called = True
            return "neural response"

        controller = CognitiveController()
        res = controller.process("What is 10 + 10?", mock_neural, mode="hybrid")
        assert called is False
        assert res["origin"] == "deterministic_math"
        assert "20" in res["response"]

    def test_hybrid_mode_routes_memory(self):
        called = False
        def mock_neural(prompt):
            nonlocal called
            called = True
            return "neural response"

        controller = CognitiveController()
        # Seed memory
        controller.memory.facts["user_name"] = "Priya"
        res = controller.process("What is my name?", mock_neural, mode="hybrid")
        assert called is False
        assert res["origin"] == "episodic_memory"
        assert "Priya" in res["response"]
