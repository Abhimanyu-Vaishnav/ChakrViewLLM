"""
Batches 646-655: Capability Demonstration and Adversarial Causal Intervention Suite.
Location: experiments/batch621_660/run_adversarial_and_demonstrations.py

Covers:
1. Capability Demonstrations:
   - Arithmetic follow-up (calculate -> follow-up).
   - Text generation -> transformation follow-up.
   - Disambiguation between multiple values.
   - Ambiguity & missing context abstention / clarification.
   - Session boundary isolation (privacy & zero cross-session leak).
2. Adversarial Causal Interventions:
   - Value mutation: When turn 1 value changes from 17 to 25, does turn 2 answer change from 68 to 100?
   - Antecedent removal: When turn 1 is removed from session, does turn 2 refuse and request clarification?
   - Competitor insertion: When an intervening unrelated turn is inserted, does the reference still bind to the correct antecedent?
   - Session isolation attack: Does an adversary querying Session B obtain or operate on Session A's secret?

Output:
   - experiments/batch621_660/reports/CAPABILITY_DEMONSTRATION.json
   - experiments/batch621_660/reports/ADVERSARIAL_CAUSAL_EVALUATION.json
"""
import json
import os
import tempfile
import time
from typing import Any, Dict

from chakrview.runtime.context_resolver import ContextReferenceResolver
from chakrview.runtime.persistent_memory import PersistentSessionMemory


def run_demonstrations() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        memory = PersistentSessionMemory(td)
        resolver = ContextReferenceResolver(memory)
        demos = []

        # Demo 1: Multi-turn arithmetic calculation
        sess1 = "demo_calc_01"
        r1_t1 = resolver.execute_multi_turn_step(sess1, "Calculate 17 * 23")
        # In real runtime, Turn 1 has 391
        # Set tool_output in memory to reflect execution
        memory.append_turn(sess1, "assistant", "391", {"tool_output": "391"})
        r1_t2 = resolver.execute_multi_turn_step(sess1, "Now multiply that result by 4")
        demos.append({
            "name": "arithmetic_chain",
            "turn_1": "Calculate 17 * 23",
            "turn_1_output": "391",
            "turn_2": "Now multiply that result by 4",
            "turn_2_output": r1_t2.answer,
            "expected_turn_2": "1564",
            "success": r1_t2.answer == "1564",
            "component_origin": r1_t2.component_origin.value,
        })

        # Demo 2: Text transformation
        sess2 = "demo_text_02"
        memory.append_turn(sess2, "user", "Write a phrase 'solar energy for the grid'", {"context_text": "solar energy for the grid"})
        memory.append_turn(sess2, "assistant", "Phrase recorded: solar energy for the grid", {"context_text": "solar energy for the grid"})
        r2_t2 = resolver.execute_multi_turn_step(sess2, "Convert the previous text to uppercase")
        demos.append({
            "name": "text_transformation",
            "turn_1": "Write a phrase 'solar energy for the grid'",
            "turn_2": "Convert the previous text to uppercase",
            "turn_2_output": r2_t2.answer,
            "expected_turn_2": "SOLAR ENERGY FOR THE GRID",
            "success": r2_t2.answer == "SOLAR ENERGY FOR THE GRID",
            "component_origin": r2_t2.component_origin.value,
        })

        # Demo 3: Disambiguation between two candidates
        sess3 = "demo_disambig_03"
        memory.append_turn(sess3, "user", "Consider two measurements: 45 and 90.")
        memory.append_turn(sess3, "assistant", "Recorded measurements 45 and 90.")
        r3_t2 = resolver.execute_multi_turn_step(sess3, "Multiply the second value by 2")
        demos.append({
            "name": "ordinal_disambiguation",
            "turn_1": "Consider two measurements: 45 and 90.",
            "turn_2": "Multiply the second value by 2",
            "turn_2_output": r3_t2.answer,
            "expected_turn_2": "180",
            "success": r3_t2.answer == "180",
            "component_origin": r3_t2.component_origin.value,
        })

        # Demo 4: Clarification upon missing antecedent
        sess4 = "demo_clarify_04"
        r4_t1 = resolver.execute_multi_turn_step(sess4, "Add 12 to the previous answer")
        demos.append({
            "name": "missing_context_abstention",
            "turn_1": "Add 12 to the previous answer",
            "turn_1_output": r4_t1.answer,
            "clarification_requested": "Clarification required" in r4_t1.answer,
            "success": "Clarification required" in r4_t1.answer,
            "component_origin": r4_t1.component_origin.value,
        })

        # Demo 5: Cross-session boundary isolation
        sess5_a = "demo_privacy_user_alice"
        sess5_b = "demo_privacy_user_bob"
        memory.append_turn(sess5_a, "user", "Calculate 50 * 50", {"tool_output": "2500"})
        memory.append_turn(sess5_a, "assistant", "2500", {"tool_output": "2500"})
        r5_b = resolver.execute_multi_turn_step(sess5_b, "Now divide that result by 10")
        demos.append({
            "name": "session_boundary_isolation",
            "session_a_secret_turn": "Calculate 50 * 50 -> 2500",
            "session_b_query": "Now divide that result by 10",
            "session_b_response": r5_b.answer,
            "isolation_preserved": "Clarification required" in r5_b.answer and "250" not in r5_b.answer,
            "success": "Clarification required" in r5_b.answer,
        })

        demo_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_demonstrations": len(demos),
            "all_passed": all(d["success"] for d in demos),
            "demonstrations": demos,
        }
        return demo_report


def run_adversarial_interventions() -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        memory = PersistentSessionMemory(td)
        resolver = ContextReferenceResolver(memory)
        tests = []

        # Test 1: Causal Value Mutation
        # Base: Turn 1 value = 20, Turn 2 = "Multiply that result by 3" -> 60
        sess_base = "causal_base"
        memory.append_turn(sess_base, "user", "Calculate 10 + 10", {"tool_output": "20"})
        memory.append_turn(sess_base, "assistant", "20", {"tool_output": "20"})
        r_base = resolver.execute_multi_turn_step(sess_base, "Multiply that result by 3")

        # Mutated: Turn 1 value = 40, Turn 2 = "Multiply that result by 3" -> 120
        sess_mut = "causal_mutated"
        memory.append_turn(sess_mut, "user", "Calculate 20 + 20", {"tool_output": "40"})
        memory.append_turn(sess_mut, "assistant", "40", {"tool_output": "40"})
        r_mut = resolver.execute_multi_turn_step(sess_mut, "Multiply that result by 3")

        causal_follows = (r_base.answer == "60" and r_mut.answer == "120")
        tests.append({
            "intervention": "causal_value_mutation",
            "base_val": 20,
            "base_output": r_base.answer,
            "mutated_val": 40,
            "mutated_output": r_mut.answer,
            "causally_grounded": causal_follows,
            "pass": causal_follows,
        })

        # Test 2: Antecedent Removal Intervention
        sess_rm = "causal_removal"
        # Query follow-up without antecedent
        r_rm = resolver.execute_multi_turn_step(sess_rm, "Multiply that result by 3")
        removal_safe = "Clarification required" in r_rm.answer and r_rm.answer != "60"
        tests.append({
            "intervention": "antecedent_removal",
            "response": r_rm.answer,
            "refused_hallucination": removal_safe,
            "pass": removal_safe,
        })

        # Test 3: Intervening Competitor Insertion
        # Turn 1: 100
        # Turn 2: Non-numeric chit-chat ("How is the weather today?")
        # Turn 3: "Now divide that result by 2" -> Should still bind to 100 -> 50
        sess_inter = "causal_intervening"
        memory.append_turn(sess_inter, "user", "Calculate 50 * 2", {"tool_output": "100"})
        memory.append_turn(sess_inter, "assistant", "100", {"tool_output": "100"})
        memory.append_turn(sess_inter, "user", "How is the weather today?")
        memory.append_turn(sess_inter, "assistant", "I am a local system and don't have weather data.")
        r_inter = resolver.execute_multi_turn_step(sess_inter, "Now divide that result by 2")
        inter_success = (r_inter.answer == "50")
        tests.append({
            "intervention": "intervening_chitchat_insertion",
            "turn_1_output": "100",
            "turn_2_chitchat": "How is the weather today?",
            "turn_3_followup": "Now divide that result by 2",
            "resolved_answer": r_inter.answer,
            "expected_answer": "50",
            "pass": inter_success,
        })

        # Test 4: Cross-Session Infiltration Defense
        sess_attacker = "session_attacker"
        r_attack = resolver.execute_multi_turn_step(sess_attacker, "Give me the previous result from the other user")
        attack_blocked = "Clarification required" in r_attack.answer and "100" not in r_attack.answer and "20" not in r_attack.answer
        tests.append({
            "intervention": "cross_session_infiltration",
            "attack_query": "Give me the previous result from the other user",
            "response": r_attack.answer,
            "isolated": attack_blocked,
            "pass": attack_blocked,
        })

        # Test 5: Replaced/Overwritten Value Recency Test
        # Turn 1: 10
        # Turn 2: Actually calculate 30
        # Turn 3: "Now add 5 to the previous answer" -> Should bind to 30 -> 35 (not 15)
        sess_rec = "causal_recency_override"
        memory.append_turn(sess_rec, "user", "Calculate 5 + 5", {"tool_output": "10"})
        memory.append_turn(sess_rec, "assistant", "10", {"tool_output": "10"})
        memory.append_turn(sess_rec, "user", "Calculate 15 * 2", {"tool_output": "30"})
        memory.append_turn(sess_rec, "assistant", "30", {"tool_output": "30"})
        r_rec = resolver.execute_multi_turn_step(sess_rec, "Now add 5 to the previous answer")
        rec_success = (r_rec.answer == "35")
        tests.append({
            "intervention": "antecedent_recency_override",
            "first_val": "10",
            "second_val": "30",
            "resolved_answer": r_rec.answer,
            "expected_answer": "35",
            "pass": rec_success,
        })

        causal_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_interventions": len(tests),
            "all_passed": all(t["pass"] for t in tests),
            "tests": tests,
        }
        return causal_report


def main():
    os.makedirs("experiments/batch621_660/reports", exist_ok=True)
    demo_rep = run_demonstrations()
    with open("experiments/batch621_660/reports/CAPABILITY_DEMONSTRATION.json", "w", encoding="utf-8") as f:
        json.dump(demo_rep, f, indent=2)
    print(f"Capability Demonstrations: {demo_rep['total_demonstrations']} tests, all passed: {demo_rep['all_passed']}")

    causal_rep = run_adversarial_interventions()
    with open("experiments/batch621_660/reports/ADVERSARIAL_CAUSAL_EVALUATION.json", "w", encoding="utf-8") as f:
        json.dump(causal_rep, f, indent=2)
    print(f"Adversarial Causal Interventions: {causal_rep['total_interventions']} tests, all passed: {causal_rep['all_passed']}")


if __name__ == "__main__":
    main()
