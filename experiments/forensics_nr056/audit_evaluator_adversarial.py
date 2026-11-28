import os
import sys
import json
import re

sys.path.insert(0, os.path.abspath("."))
from experiments.forensics_nr055.robust_evaluator import score_item

# Comprehensive Adversarial Suite for Evaluator Reliability Testing (25 diverse cases)
test_suite = [
    # 1. Correct Answers
    {"name": "clean_factual_tokyo", "resp": "Tokyo", "item": {"expected_answer": "Tokyo", "rubric_type": "exact_or_substring", "constraints": ["factual"]}, "gold": 3},
    {"name": "clean_factual_in_sentence", "resp": "The capital is Tokyo.", "item": {"expected_answer": "Tokyo", "rubric_type": "exact_or_substring", "constraints": ["factual"]}, "gold": 3},
    {"name": "clean_number_single", "resp": "12", "item": {"expected_answer": "12", "rubric_type": "numerical_exact", "constraints": ["exact_number"]}, "gold": 3},
    {"name": "clean_number_in_equation", "resp": "5 + 7 = 12", "item": {"expected_answer": "12", "rubric_type": "numerical_exact", "constraints": ["exact_number"]}, "gold": 2},
    
    # 2. Substrings inside words (False Positive Tests)
    {"name": "sub_ice_in_service", "resp": "Customer service is great.", "item": {"expected_answer": "ice", "rubric_type": "exact_or_substring", "constraints": ["factual"]}, "gold": 0},
    {"name": "sub_red_in_hundred", "resp": "One hundred people attended.", "item": {"expected_answer": "red", "rubric_type": "exact_or_substring", "constraints": ["factual"]}, "gold": 0},
    {"name": "sub_cat_in_category", "resp": "This is a new category.", "item": {"expected_answer": "cat", "rubric_type": "exact_or_substring", "constraints": ["factual"]}, "gold": 0},
    
    # 3. Numerical Noise & Dates (False Positive Tests)
    {"name": "num_2_in_year_2024", "resp": "Published in 2024.", "item": {"expected_answer": "2", "rubric_type": "numerical_exact", "constraints": ["exact_number"]}, "gold": 0},
    {"name": "num_5_in_500", "resp": "There are 500 items.", "item": {"expected_answer": "5", "rubric_type": "numerical_exact", "constraints": ["exact_number"]}, "gold": 0},
    {"name": "num_conflicting_numbers", "resp": "The answer might be 12, or maybe 7, or 99.", "item": {"expected_answer": "12", "rubric_type": "numerical_exact", "constraints": ["exact_number"]}, "gold": 1},
    
    # 4. Strict Constraints
    {"name": "strict_yes_perfect", "resp": "YES", "item": {"expected_answer": "YES", "rubric_type": "strict_exact", "constraints": ["single_word"]}, "gold": 3},
    {"name": "strict_yes_lower", "resp": "yes", "item": {"expected_answer": "YES", "acceptable_alternatives": ["yes"], "rubric_type": "strict_exact", "constraints": ["single_word"]}, "gold": 3},
    {"name": "strict_yes_with_preamble", "resp": "Yes, absolutely.", "item": {"expected_answer": "YES", "rubric_type": "strict_exact", "constraints": ["single_word"]}, "gold": 2},
    {"name": "strict_single_char_violation", "resp": "The single digit is 2.", "item": {"expected_answer": "2", "rubric_type": "strict_exact", "constraints": ["single_char"]}, "gold": 0},
    
    # 5. Repetitive Babble (Loop Detector)
    {"name": "loop_repetition_target", "resp": "Tokyo Tokyo Tokyo Tokyo Tokyo Tokyo Tokyo", "item": {"expected_answer": "Tokyo", "rubric_type": "exact_or_substring", "constraints": ["factual"]}, "gold": 0},
    {"name": "loop_phrase_repetition", "resp": "the capital is the capital is the capital is Tokyo", "item": {"expected_answer": "Tokyo", "rubric_type": "exact_or_substring", "constraints": ["factual"]}, "gold": 0},
    
    # 6. Devanagari & Hindi Boundaries
    {"name": "hin_clean_delhi", "resp": "भारत की राजधानी नई दिल्ली है।", "item": {"expected_answer": "नई दिल्ली", "rubric_type": "exact_or_substring", "constraints": ["factual_hindi"]}, "gold": 3},
    {"name": "hin_sub_vowel", "resp": "अनुसार अधिक अच्छा", "item": {"expected_answer": "अ", "rubric_type": "exact_or_substring", "constraints": ["factual_hindi"]}, "gold": 0},
    {"name": "hin_number_clean", "resp": "7", "item": {"expected_answer": "7", "rubric_type": "numerical_exact", "constraints": ["math_hindi"]}, "gold": 3},
    
    # 7. Contradictory & Entity Mismatch
    {"name": "contradictory_negation", "resp": "No, it is definitely not Paris, it is London.", "item": {"expected_answer": "Paris", "rubric_type": "exact_or_substring", "constraints": ["factual"]}, "gold": 1},
    {"name": "empty_resp", "resp": "", "item": {"expected_answer": "Paris", "rubric_type": "exact_or_substring", "constraints": ["factual"]}, "gold": 0},
    {"name": "whitespace_resp", "resp": "   \n\t  ", "item": {"expected_answer": "Paris", "rubric_type": "exact_or_substring", "constraints": ["factual"]}, "gold": 0},
    {"name": "token_set_complete", "resp": "Red, yellow, green", "item": {"expected_answer": "Red, Yellow, Green", "rubric_type": "token_set_match", "constraints": ["set_inclusion"]}, "gold": 3},
    {"name": "token_set_partial", "resp": "Red and green only", "item": {"expected_answer": "Red, Yellow, Green", "rubric_type": "token_set_match", "constraints": ["set_inclusion"]}, "gold": 1},
    {"name": "token_set_none", "resp": "Blue and purple", "item": {"expected_answer": "Red, Yellow, Green", "rubric_type": "token_set_match", "constraints": ["set_inclusion"]}, "gold": 0}
]

eval_results = []
correct_matches = 0
false_positives = 0
false_negatives = 0

for tc in test_suite:
    out = score_item(tc["resp"], tc["item"])
    score = out["score"]
    gold = tc["gold"]
    
    # Classify binary match (0 vs non-zero or exact tier)
    is_exact_match = (score == gold)
    if is_exact_match:
        correct_matches += 1
    elif score > 0 and gold == 0:
        false_positives += 1
    elif score == 0 and gold > 0:
        false_negatives += 1
        
    eval_results.append({
        "name": tc["name"],
        "input": tc["resp"],
        "expected_score": gold,
        "evaluator_score": score,
        "is_exact_match": is_exact_match,
        "verdict": out["verdict"],
        "reason": out["reason"]
    })

accuracy = correct_matches / len(test_suite)
fp_rate = false_positives / len(test_suite)
fn_rate = false_negatives / len(test_suite)

confusion_matrix = {
    "total_tests": len(test_suite),
    "exact_tier_matches": correct_matches,
    "accuracy": accuracy,
    "false_positives": false_positives,
    "false_negatives": false_negatives,
    "false_positive_rate": fp_rate,
    "false_negative_rate": fn_rate
}

with open("experiments/forensics_nr056/evaluator_adversarial_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)

with open("experiments/forensics_nr056/evaluator_confusion_matrix.json", "w") as f:
    json.dump(confusion_matrix, f, indent=2)

print(f"EVALUATOR ADVERSARIAL AUDIT: {correct_matches}/{len(test_suite)} exact matches ({accuracy*100:.1f}%).")
print(f"False Positives: {false_positives}, False Negatives: {false_negatives}")
