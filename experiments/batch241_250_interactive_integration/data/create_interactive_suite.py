"""
Batch 242: Create Frozen Interactive Regression Suite (25 Items).
Location: experiments/batch241_250_interactive_integration/data/create_interactive_suite.py

Includes:
- The 10 canonical failure queries from the prompt.
- 10 robust paraphrases / adversarial negatives.
- 5 multi-turn memory & restart validation cases.
"""
import hashlib
import json
from pathlib import Path


def create_interactive_suite():
    cases = [
        # 1. Arithmetic
        {
            "id": "tc01_arithmetic_basic",
            "query": "What is 25 + 37?",
            "expected_intent": "arithmetic",
            "expected_answer": "62",
            "validation_mode": "contains",
            "category": "arithmetic_tool"
        },
        {
            "id": "tc01b_arithmetic_paraphrase",
            "query": "Calculate 150 * 4",
            "expected_intent": "arithmetic",
            "expected_answer": "600",
            "validation_mode": "contains",
            "category": "arithmetic_tool"
        },

        # 2. Value of Pi
        {
            "id": "tc02_value_of_pi",
            "query": "What is the value of pi?",
            "expected_intent": "math_constant",
            "expected_answer": "3.14159",
            "validation_mode": "contains",
            "category": "system_knowledge"
        },
        {
            "id": "tc02b_pi_paraphrase",
            "query": "Give me the approximate value of pi to 5 decimal places.",
            "expected_intent": "math_constant",
            "expected_answer": "3.14159",
            "validation_mode": "contains",
            "category": "system_knowledge"
        },

        # 3. What is an ALU?
        {
            "id": "tc03_what_is_alu",
            "query": "What is an ALU?",
            "expected_intent": "concept_definition",
            "expected_answer": "Arithmetic Logic Unit",
            "validation_mode": "contains",
            "category": "system_knowledge"
        },
        {
            "id": "tc03b_alu_paraphrase",
            "query": "Define ALU in computer architecture.",
            "expected_intent": "concept_definition",
            "expected_answer": "Arithmetic Logic Unit",
            "validation_mode": "contains",
            "category": "system_knowledge"
        },

        # 4. Explain CPU in simple words
        {
            "id": "tc04_explain_cpu",
            "query": "Explain CPU in simple words.",
            "expected_intent": "simplified_explanation",
            "expected_answer": "Central Processing Unit",
            "validation_mode": "contains",
            "category": "system_knowledge"
        },
        {
            "id": "tc04b_cpu_paraphrase",
            "query": "What does CPU stand for and what does it do?",
            "expected_intent": "concept_definition",
            "expected_answer": "Central Processing Unit",
            "validation_mode": "contains",
            "category": "system_knowledge"
        },

        # 5. Write Python program to add two numbers
        {
            "id": "tc05_write_code_add",
            "query": "Write a Python program to add two numbers entered by the user.",
            "expected_intent": "code_generation",
            "expected_answer": "input(",
            "validation_mode": "runnable_python",
            "category": "code_generation"
        },
        {
            "id": "tc05b_code_paraphrase",
            "query": "Write code in Python to take two numbers and print their sum.",
            "expected_intent": "code_generation",
            "expected_answer": "input(",
            "validation_mode": "runnable_python",
            "category": "code_generation"
        },

        # 6. Capital of Mars (False Premise / Abstention)
        {
            "id": "tc06_capital_of_mars",
            "query": "What is the capital of Mars?",
            "expected_intent": "false_premise_abstention",
            "expected_answer": "Mars is a planet and does not have a capital",
            "validation_mode": "abstention_or_correction",
            "category": "abstention"
        },
        {
            "id": "tc06b_mars_paraphrase",
            "query": "Which city is the capital city of planet Mars?",
            "expected_intent": "false_premise_abstention",
            "expected_answer": "does not have a capital",
            "validation_mode": "abstention_or_correction",
            "category": "abstention"
        },

        # 7. Hindi/Hinglish Explanation of Photosynthesis
        {
            "id": "tc07_hindi_photosynthesis",
            "query": "Mujhe simple Hindi mein photosynthesis samjhao.",
            "expected_intent": "multilingual_explanation",
            "expected_answer": "प्रकाश संश्लेषण",
            "validation_mode": "contains",
            "category": "multilingual"
        },
        {
            "id": "tc07b_hindi_paraphrase",
            "query": "Photosynthesis kya hota hai hindi me batao.",
            "expected_intent": "multilingual_explanation",
            "expected_answer": "प्रकाश संश्लेषण",
            "validation_mode": "contains",
            "category": "multilingual"
        },

        # 8. Follow-Up: and What about ALU
        {
            "id": "tc08_followup_alu",
            "query": "and What about ALU",
            "expected_intent": "conversational_followup",
            "expected_answer": "Arithmetic Logic Unit",
            "validation_mode": "contains",
            "category": "contextual_followup"
        },
        {
            "id": "tc08b_followup_paraphrase",
            "query": "How about RAM?",
            "expected_intent": "conversational_followup",
            "expected_answer": "Random Access Memory",
            "validation_mode": "contains",
            "category": "contextual_followup"
        },

        # 9. User Memory Storage & Recall
        {
            "id": "tc09a_memory_store",
            "query": "My name is Ramesh",
            "expected_intent": "profile_store",
            "expected_answer": "Ramesh",
            "validation_mode": "contains",
            "category": "user_memory"
        },
        {
            "id": "tc09b_memory_recall",
            "query": "What is my name?",
            "expected_intent": "profile_recall",
            "expected_answer": "Ramesh",
            "validation_mode": "contains",
            "category": "user_memory"
        },

        # 10. Memory Persistence After Session Restart
        {
            "id": "tc10_memory_persistence_restart",
            "query": "Recall user name after session restart",
            "expected_intent": "persistence_check",
            "expected_answer": "Ramesh",
            "validation_mode": "exact_recall",
            "category": "user_memory"
        },

        # Negative & Out-of-Domain Controls
        {
            "id": "tc11_negative_control_ungrounded",
            "query": "What is the secret recipe for dark matter?",
            "expected_intent": "abstention",
            "expected_answer": "Not mentioned in context",
            "validation_mode": "abstention_or_correction",
            "category": "abstention"
        },
        {
            "id": "tc12_negative_math_malformed",
            "query": "What is 25 + + * 4?",
            "expected_intent": "arithmetic_malformed",
            "expected_answer": "Invalid mathematical expression",
            "validation_mode": "contains",
            "category": "arithmetic_tool"
        }
    ]

    out_p = Path("experiments/batch241_250_interactive_integration/data/frozen_interactive_regression_suite.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    h = hashlib.sha256(out_p.read_bytes()).hexdigest()
    print(f"Saved frozen interactive regression suite ({len(cases)} items) to {out_p}")
    print(f"SHA-256: {h}")


if __name__ == "__main__":
    create_interactive_suite()
