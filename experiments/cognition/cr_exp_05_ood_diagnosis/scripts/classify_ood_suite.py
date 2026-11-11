"""
Phase 2: Fine-grained Audit and Taxonomy Classification of the 100-Case OOD Suite
Classifies every test case into failure modes, records token count, entities, expected answer,
and evaluates performance of parent vs candidate.
"""

import json
import sys
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from chakrview.tokenizer.serialization import load_tokenizer_artifacts

SUITE_PATH = REPO_ROOT / "experiments" / "cognition" / "cr_exp_04_validation" / "independent_conversation_suite.json"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"

# Ground truth expected criteria and entity metadata for all 100 cases
METADATA_100 = {
    # A. Natural Conversation
    "A01": {"expected": "Greeting / Offer of assistance", "entities": [], "category": "A_Natural_Conversation", "tax": "greeting"},
    "A02": {"expected": "Greeting / Good morning", "entities": [], "category": "A_Natural_Conversation", "tax": "greeting"},
    "A03": {"expected": "Greeting / Good evening", "entities": [], "category": "A_Natural_Conversation", "tax": "greeting"},
    "A04": {"expected": "I am ChakrView...", "entities": ["ChakrView"], "category": "A_Natural_Conversation", "tax": "persona"},
    "A05": {"expected": "ChakrView", "entities": ["ChakrView"], "category": "A_Natural_Conversation", "tax": "persona"},
    "A06": {"expected": "ChakrMicro v0.1 / Transformer details", "entities": ["ChakrMicro"], "category": "A_Natural_Conversation", "tax": "persona"},
    "A07": {"expected": "ChakrView developers / Indigenous research", "entities": ["ChakrView"], "category": "A_Natural_Conversation", "tax": "persona"},
    "A08": {"expected": "No, I am ChakrView...", "entities": ["ChatGPT", "ChakrView"], "category": "A_Natural_Conversation", "tax": "persona"},
    "A09": {"expected": "I am doing well / Ready to help", "entities": [], "category": "A_Natural_Conversation", "tax": "greeting"},
    "A10": {"expected": "You are welcome / Happy to help", "entities": [], "category": "A_Natural_Conversation", "tax": "greeting"},
    "A11": {"expected": "You too / Goodbye", "entities": [], "category": "A_Natural_Conversation", "tax": "greeting"},
    "A12": {"expected": "Yes, I can assist with language tasks", "entities": [], "category": "A_Natural_Conversation", "tax": "greeting"},
    "A13": {"expected": "Good night", "entities": [], "category": "A_Natural_Conversation", "tax": "greeting"},
    "A14": {"expected": "Assistance with language, questions", "entities": [], "category": "A_Natural_Conversation", "tax": "persona"},
    "A15": {"expected": "Greetings / Hello", "entities": [], "category": "A_Natural_Conversation", "tax": "greeting"},
    "A16": {"expected": "Nice to meet you too", "entities": [], "category": "A_Natural_Conversation", "tax": "greeting"},
    "A17": {"expected": "I am a compact indigenous language model", "entities": ["ChakrMicro"], "category": "A_Natural_Conversation", "tax": "persona"},
    "A18": {"expected": "How can I help you?", "entities": [], "category": "A_Natural_Conversation", "tax": "greeting"},
    "A19": {"expected": "Goodbye / Have a nice day", "entities": [], "category": "A_Natural_Conversation", "tax": "greeting"},
    "A20": {"expected": "To explore neural language modeling / Assist users", "entities": [], "category": "A_Natural_Conversation", "tax": "persona"},

    # B. Context Retrieval
    "B01": {"expected": "Seattle", "entities": ["Alice", "Seattle"], "category": "B_Context_Retrieval", "tax": "unseen_entity_retrieval"},
    "B02": {"expected": "8080", "entities": ["8080"], "category": "B_Context_Retrieval", "tax": "familiar_entity_unfamiliar_template"},
    "B03": {"expected": "Hyderabad", "entities": ["Vikram", "Hyderabad"], "category": "B_Context_Retrieval", "tax": "unseen_entity_retrieval"},
    "B04": {"expected": "PostgreSQL", "entities": ["PostgreSQL"], "category": "B_Context_Retrieval", "tax": "familiar_entity_unfamiliar_template"},
    "B05": {"expected": "Rohan", "entities": ["Project Alpha", "Priya", "Project Beta", "Rohan"], "category": "B_Context_Retrieval", "tax": "unseen_entity_and_unseen_template"},
    "B06": {"expected": "eu-west-1", "entities": ["eu-west-1"], "category": "B_Context_Retrieval", "tax": "familiar_entity_unfamiliar_template"},
    "B07": {"expected": "Chicago", "entities": ["John", "Boston", "Chicago"], "category": "B_Context_Retrieval", "tax": "dynamic_state_updates"},
    "B08": {"expected": "False", "entities": ["True", "False"], "category": "B_Context_Retrieval", "tax": "dynamic_state_updates"},
    "B09": {"expected": "/var/log/syslog", "entities": ["/var/log/syslog"], "category": "B_Context_Retrieval", "tax": "familiar_entity_unfamiliar_template"},
    "B10": {"expected": "red", "entities": ["Car A", "red", "Car B", "green", "Car C", "yellow"], "category": "B_Context_Retrieval", "tax": "unseen_entity_and_unseen_template"},
    "B11": {"expected": "secret_key_99", "entities": ["secret_key_99"], "category": "B_Context_Retrieval", "tax": "unseen_entity_retrieval"},
    "B12": {"expected": "Thursday", "entities": ["Monday", "Thursday"], "category": "B_Context_Retrieval", "tax": "dynamic_state_updates"},
    "B13": {"expected": "300 seconds / 300", "entities": ["300 seconds"], "category": "B_Context_Retrieval", "tax": "familiar_entity_unfamiliar_template"},
    "B14": {"expected": "physics", "entities": ["Ananya", "Bengaluru", "physics"], "category": "B_Context_Retrieval", "tax": "unseen_entity_retrieval"},
    "B15": {"expected": "Worker 1", "entities": ["Worker 1", "task X", "Worker 2", "task Y"], "category": "B_Context_Retrieval", "tax": "unseen_entity_and_unseen_template"},

    # C. Multi-Turn Dialogue
    "C01": {"expected": "Kabir", "entities": ["Kabir"], "category": "C_Multi_Turn", "tax": "multi_turn_variable_binding"},
    "C02": {"expected": "Sun (or Proxima Centauri)", "entities": ["Sun"], "category": "C_Multi_Turn", "tax": "multi_hop_reasoning"},
    "C03": {"expected": "falcon", "entities": ["falcon"], "category": "C_Multi_Turn", "tax": "multi_turn_variable_binding"},
    "C04": {"expected": "ChakrMicro / 6 layers / 3.44M", "entities": ["ChakrMicro"], "category": "C_Multi_Turn", "tax": "persona"},
    "C05": {"expected": "Preheat oven", "entities": ["Preheat oven"], "category": "C_Multi_Turn", "tax": "multi_turn_variable_binding"},
    "C06": {"expected": "Yes (passport required for international travel)", "entities": ["Tokyo"], "category": "C_Multi_Turn", "tax": "multi_hop_reasoning"},
    "C07": {"expected": "Dog", "entities": ["dog"], "category": "C_Multi_Turn", "tax": "multi_hop_reasoning"},
    "C08": {"expected": "oranges", "entities": ["mango", "oranges"], "category": "C_Multi_Turn", "tax": "dynamic_state_updates"},
    "C09": {"expected": "1, 2, 3, 4", "entities": ["1, 2, 3, 4"], "category": "C_Multi_Turn", "tax": "generation_degeneration"},
    "C10": {"expected": "Response 3", "entities": [], "category": "C_Multi_Turn", "tax": "multi_turn_variable_binding"},
    "C11": {"expected": "print('hello world')", "entities": ["Python"], "category": "C_Multi_Turn", "tax": "familiar_entity_unfamiliar_template"},
    "C12": {"expected": "10", "entities": ["A = 10", "B = 20"], "category": "C_Multi_Turn", "tax": "multi_turn_variable_binding"},
    "C13": {"expected": "New Delhi", "entities": ["Delhi", "India"], "category": "C_Multi_Turn", "tax": "multi_hop_reasoning"},
    "C14": {"expected": "Politely", "entities": ["politely"], "category": "C_Multi_Turn", "tax": "multi_turn_variable_binding"},
    "C15": {"expected": "Start", "entities": ["Start", "Proceed", "Complete"], "category": "C_Multi_Turn", "tax": "multi_turn_variable_binding"},

    # D. Instruction Following
    "D01": {"expected": "Paris", "entities": ["France", "Paris"], "category": "D_Instruction_Following", "tax": "familiar_entity_unfamiliar_template"},
    "D02": {"expected": "Yes", "entities": ["ice"], "category": "D_Instruction_Following", "tax": "familiar_entity_unfamiliar_template"},
    "D03": {"expected": "- Apple\n- Banana\n- Orange", "entities": ["fruits"], "category": "D_Instruction_Following", "tax": "generation_degeneration"},
    "D04": {"expected": "Plants convert sunlight into energy.", "entities": ["photosynthesis"], "category": "D_Instruction_Following", "tax": "generation_degeneration"},
    "D05": {"expected": "50", "entities": ["42", "8"], "category": "D_Instruction_Following", "tax": "arithmetic"},
    "D06": {"expected": "ChakrView Neural Core", "entities": ["ChakrView Neural Core"], "category": "D_Instruction_Following", "tax": "familiar_entity_unfamiliar_template"},
    "D07": {"expected": "STOP", "entities": ["STOP"], "category": "D_Instruction_Following", "tax": "familiar_entity_unfamiliar_template"},
    "D08": {"expected": "Hello there / Good day", "entities": [], "category": "D_Instruction_Following", "tax": "greeting"},
    "D09": {"expected": "D", "entities": ["A, B, C"], "category": "D_Instruction_Following", "tax": "generation_degeneration"},
    "D10": {"expected": "{\"status\": \"ok\"}", "entities": ["status", "ok"], "category": "D_Instruction_Following", "tax": "generation_degeneration"},
    "D11": {"expected": "Welcome", "entities": ["welcome"], "category": "D_Instruction_Following", "tax": "familiar_entity_unfamiliar_template"},
    "D12": {"expected": "Blue", "entities": ["sky", "blue"], "category": "D_Instruction_Following", "tax": "familiar_entity_unfamiliar_template"},
    "D13": {"expected": "1, 2, 3, 4, 5", "entities": ["1 to 5"], "category": "D_Instruction_Following", "tax": "generation_degeneration"},
    "D14": {"expected": "Confirmed", "entities": ["Confirmed"], "category": "D_Instruction_Following", "tax": "familiar_entity_unfamiliar_template"},
    "D15": {"expected": "An atom is the basic unit of a chemical element.", "entities": ["atom"], "category": "D_Instruction_Following", "tax": "generation_degeneration"},

    # E. Epistemic Behaviour
    "E01": {"expected": "I do not know / Not provided", "entities": ["breakfast"], "category": "E_Epistemic_Behaviour", "tax": "unseen_entity_and_unseen_template"},
    "E02": {"expected": "Cannot predict future / Unknown", "entities": ["Bitcoin"], "category": "E_Epistemic_Behaviour", "tax": "unseen_entity_and_unseen_template"},
    "E03": {"expected": "I do not have access to private info", "entities": ["password"], "category": "E_Epistemic_Behaviour", "tax": "unseen_entity_and_unseen_template"},
    "E04": {"expected": "Information not provided in context", "entities": ["Suresh", "Chennai"], "category": "E_Epistemic_Behaviour", "tax": "unseen_entity_retrieval"},
    "E05": {"expected": "Information not provided in context (or 0)", "entities": ["ruler", "box"], "category": "E_Epistemic_Behaviour", "tax": "unseen_entity_and_unseen_template"},
    "E06": {"expected": "Cannot predict future", "entities": ["World Cup"], "category": "E_Epistemic_Behaviour", "tax": "unseen_entity_and_unseen_template"},
    "E07": {"expected": "I do not have access to real-time sensors", "entities": ["temperature"], "category": "E_Epistemic_Behaviour", "tax": "unseen_entity_and_unseen_template"},
    "E08": {"expected": "Information not provided in context", "entities": ["Neil", "Mira"], "category": "E_Epistemic_Behaviour", "tax": "unseen_entity_retrieval"},
    "E09": {"expected": "No, I am an AI language model", "entities": [], "category": "E_Epistemic_Behaviour", "tax": "persona"},
    "E10": {"expected": "Science has not confirmed life in Andromeda / Unknown", "entities": ["Andromeda"], "category": "E_Epistemic_Behaviour", "tax": "unseen_entity_and_unseen_template"},
    "E11": {"expected": "I do not have access to device info", "entities": ["laptop"], "category": "E_Epistemic_Behaviour", "tax": "unseen_entity_and_unseen_template"},
    "E12": {"expected": "Information not provided in context", "entities": ["Train 101"], "category": "E_Epistemic_Behaviour", "tax": "unseen_entity_retrieval"},
    "E13": {"expected": "Cannot predict future", "entities": ["President"], "category": "E_Epistemic_Behaviour", "tax": "unseen_entity_and_unseen_template"},
    "E14": {"expected": "I do not know where your keys are", "entities": ["keys"], "category": "E_Epistemic_Behaviour", "tax": "unseen_entity_and_unseen_template"},
    "E15": {"expected": "Information not provided in context", "entities": ["app-prod-01"], "category": "E_Epistemic_Behaviour", "tax": "unseen_entity_retrieval"},

    # F. Arithmetic & Reasoning
    "F01": {"expected": "5", "entities": ["2 + 3"], "category": "F_Arithmetic_Reasoning", "tax": "arithmetic"},
    "F02": {"expected": "6", "entities": ["10 - 4"], "category": "F_Arithmetic_Reasoning", "tax": "arithmetic"},
    "F03": {"expected": "7", "entities": ["5 + 2"], "category": "F_Arithmetic_Reasoning", "tax": "arithmetic"},
    "F04": {"expected": "15", "entities": ["7 + 8"], "category": "F_Arithmetic_Reasoning", "tax": "arithmetic"},
    "F05": {"expected": "15", "entities": ["15", "8"], "category": "F_Arithmetic_Reasoning", "tax": "arithmetic"},
    "F06": {"expected": "8", "entities": ["4 * 2"], "category": "F_Arithmetic_Reasoning", "tax": "arithmetic"},
    "F07": {"expected": "15", "entities": ["20 - 5"], "category": "F_Arithmetic_Reasoning", "tax": "arithmetic"},
    "F08": {"expected": "Tuesday", "entities": ["Monday", "Tuesday"], "category": "F_Arithmetic_Reasoning", "tax": "multi_hop_reasoning"},
    "F09": {"expected": "7", "entities": ["10 - 3"], "category": "F_Arithmetic_Reasoning", "tax": "arithmetic"},
    "F10": {"expected": "10", "entities": ["6 + 4"], "category": "F_Arithmetic_Reasoning", "tax": "arithmetic"},

    # G. Multilingual
    "G01": {"expected": "नमस्ते! / Greeting in Hindi", "entities": [], "category": "G_Multilingual", "tax": "pure_hindi"},
    "G02": {"expected": "मैं ठीक हूँ... / I am well in Hindi", "entities": [], "category": "G_Multilingual", "tax": "pure_hindi"},
    "G03": {"expected": "मेरा नाम चक्रव्यूह है / My name is ChakrView", "entities": ["ChakrView"], "category": "G_Multilingual", "tax": "pure_hindi"},
    "G04": {"expected": "नई दिल्ली / New Delhi", "entities": ["Bharat", "Rajdhani"], "category": "G_Multilingual", "tax": "pure_hindi"},
    "G05": {"expected": "आपका स्वागत है / You are welcome in Hindi", "entities": [], "category": "G_Multilingual", "tax": "pure_hindi"},
    "G06": {"expected": "Main theek hoon! / Hinglish greeting", "entities": [], "category": "G_Multilingual", "tax": "hinglish"},
    "G07": {"expected": "Mera naam ChakrView hai", "entities": ["ChakrView"], "category": "G_Multilingual", "tax": "hinglish"},
    "G08": {"expected": "Simple Python code snippet", "entities": [], "category": "G_Multilingual", "tax": "hinglish"},
    "G09": {"expected": "Shubh sandhya! / Good evening in Hinglish", "entities": [], "category": "G_Multilingual", "tax": "hinglish"},
    "G10": {"expected": "Haan, main Hindi samajhta hoon / Yes in Hinglish", "entities": [], "category": "G_Multilingual", "tax": "hinglish"},
}

def evaluate_case(cid, cand_out, meta):
    tax = meta["tax"]
    exp = meta["expected"].lower()
    cand_lower = cand_out.lower().strip()
    
    # Heuristic pass/fail evaluation
    passed = False
    
    if tax in ["greeting", "persona"]:
        # Check if greeting or persona keyword exists
        if "chakrview" in cand_lower or "assist" in cand_lower or "morning" in cand_lower or "afternoon" in cand_lower or "evening" in cand_lower or "help" in cand_lower or "welcome" in cand_lower:
            passed = True
    elif tax == "unseen_entity_retrieval":
        # Check if expected target entity is present in candidate output
        target = meta["expected"].lower()
        if target in cand_lower and "celsia" not in cand_lower:
            passed = True
    elif tax == "dynamic_state_updates":
        target = meta["expected"].lower()
        if target in cand_lower:
            passed = True
    elif tax == "arithmetic":
        target = meta["expected"].lower()
        if target in cand_lower and len(cand_lower) < 20:
            passed = True
    elif tax in ["pure_hindi", "hinglish"]:
        # Check if coherent Hindi/Hinglish exists
        if any(c in cand_out for c in ["नमस्ते", "ठीक", "चक्रव्यूह", "theek", "hoon", "aap"]):
            passed = True
    elif tax == "multi_turn_variable_binding":
        target = meta["expected"].lower()
        if target in cand_lower:
            passed = True
    elif tax == "familiar_entity_unfamiliar_template":
        target = meta["expected"].lower()
        if target in cand_lower:
            passed = True

    return passed

def main():
    print("=== CR-EXP-05 Phase 2: Audit and Taxonomy Classification of 100-Case Suite ===")
    tok, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    with open(SUITE_PATH, "r", encoding="utf-8") as f:
        suite = json.load(f)
        
    print(f"Loaded {len(suite)} test cases from {SUITE_PATH}")

    classified_cases = []
    taxonomy_counts = Counter()
    taxonomy_passed = Counter()

    for item in suite:
        cid = item["id"]
        meta = METADATA_100.get(cid, {})
        prompt = item["prompt"]
        parent_out = item["parent_response"]
        cand_out = item["candidate_response"]

        tokens = tok.encode(prompt)
        input_token_count = len(tokens)
        num_turns = prompt.count("User:") + prompt.count("Assistant:")
        
        tax = meta.get("tax", "unknown")
        taxonomy_counts[tax] += 1
        
        cand_pass = evaluate_case(cid, cand_out, meta)
        parent_pass = evaluate_case(cid, parent_out, meta)
        
        if cand_pass:
            taxonomy_passed[tax] += 1

        # Classify specific failure reason if candidate failed
        failure_type = "NONE (PASSED)"
        if not cand_pass:
            if "celsia" in cand_out.lower():
                failure_type = "OUT_OF_CONTEXT_ATTRACTOR_CELSIA"
            elif ">>>" in cand_out or "link_" in cand_out:
                failure_type = "SYNTAX_AST_CODE_BLEED"
            elif tax == "unseen_entity_retrieval":
                failure_type = "UNSEEN_ENTITY_RETRIEVAL_FAILURE"
            elif tax == "dynamic_state_updates":
                failure_type = "DYNAMIC_STATE_UPDATE_FAILURE"
            elif tax == "multi_turn_variable_binding":
                failure_type = "MULTI_TURN_BINDING_FAILURE"
            elif tax == "arithmetic":
                failure_type = "ARITHMETIC_FAILURE"
            elif tax == "pure_hindi":
                failure_type = "DEVANAGARI_HINDI_FRAGMENTATION"
            elif tax == "hinglish":
                failure_type = "HINGLISH_DEGRADATION"
            elif tax == "generation_degeneration":
                failure_type = "FORMATTING_INSTRUCTION_FAILURE"
            else:
                failure_type = "TEMPLATE_MISMATCH_FAILURE"

        classified_cases.append({
            "case_id": cid,
            "capability_category": item["category"],
            "failure_taxonomy": tax,
            "prompt_template": prompt.split("\n")[0],
            "entities": meta.get("entities", []),
            "number_of_turns": num_turns,
            "input_token_count": input_token_count,
            "expected_criterion": meta.get("expected", ""),
            "parent_output": parent_out,
            "candidate_output": cand_out,
            "candidate_passed": cand_pass,
            "parent_passed": parent_pass,
            "failure_classification": failure_type
        })

    # Summary by taxonomy
    tax_summary = {}
    for tax, total in taxonomy_counts.items():
        passed = taxonomy_passed[tax]
        tax_summary[tax] = {
            "total_cases": total,
            "passed_cases": passed,
            "accuracy_pct": round(passed / total * 100.0, 2)
        }

    out_data = {
        "suite_total": len(classified_cases),
        "overall_candidate_accuracy_pct": round(sum(taxonomy_passed.values()) / len(classified_cases) * 100.0, 2),
        "taxonomy_summary": tax_summary,
        "cases": classified_cases
    }

    out_p = REPO_ROOT / "experiments/cognition/cr_exp_05_ood_diagnosis/ood_case_classification.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"Classification saved to {out_p}")
    print("\nTaxonomy Summary:")
    for tax, s in tax_summary.items():
        print(f"  {tax:35s}: {s['passed_cases']}/{s['total_cases']} ({s['accuracy_pct']}%)")

if __name__ == "__main__":
    main()

