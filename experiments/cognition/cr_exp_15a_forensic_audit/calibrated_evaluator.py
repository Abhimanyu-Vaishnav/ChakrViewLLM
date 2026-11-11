"""CR-EXP-15A Calibrated Evaluator & Synthetic Test Suite
Distinguishes subject, attribute, value, state updates, and abstention correctness.
Includes 12 synthetic calibration test cases.
"""
import re
from typing import Dict, Any, List, Optional

class CalibratedEvaluator:
    def __init__(self):
        self.abstention_cues = [
            "not mentioned", "unknown", "never mentioned", "no record", 
            "cannot be determined", "not specified", "unspecified", "i don't know",
            "not stated", "no information", "has not been provided", "not discussed"
        ]

    def evaluate(self, 
                 task_type: str,
                 prompt: str,
                 raw_output: str, 
                 expected_subject: Optional[str] = None,
                 expected_attribute: Optional[str] = "location",
                 expected_value: Optional[str] = None,
                 earlier_value: Optional[str] = None,
                 distractor_values: Optional[List[str]] = None,
                 is_abstention: bool = False) -> Dict[str, Any]:
        """
        Calibrated evaluation returning decomposed flags:
        - subject_correct
        - attribute_correct
        - value_correct
        - stale_value_retained
        - distractor_retained
        - abstention_correct
        - strict_correct
        - loose_correct
        """
        clean_out = raw_output.strip()
        out_lower = clean_out.lower()
        
        # 1. Abstention analysis
        abstained = any(cue in out_lower for cue in self.abstention_cues)
        
        if is_abstention:
            abstention_correct = abstained
            # Strict & loose for abstention: must abstain and not hallucinate a value
            return {
                "task_type": task_type,
                "is_abstention": True,
                "abstained": abstained,
                "abstention_correct": abstention_correct,
                "subject_correct": None,
                "attribute_correct": None,
                "value_correct": not abstained, # hallucinated a value if not abstained
                "stale_value_retained": False,
                "distractor_retained": False,
                "strict_correct": abstention_correct,
                "loose_correct": abstention_correct,
                "raw_output": clean_out
            }
            
        # 2. Answerable probes
        if abstained:
            # False abstention
            return {
                "task_type": task_type,
                "is_abstention": False,
                "abstained": True,
                "abstention_correct": False,
                "subject_correct": False,
                "attribute_correct": False,
                "value_correct": False,
                "stale_value_retained": False,
                "distractor_retained": False,
                "strict_correct": False,
                "loose_correct": False,
                "raw_output": clean_out
            }
            
        subj_lower = expected_subject.lower() if expected_subject else ""
        val_lower = expected_value.lower() if expected_value else ""
        attr_lower = expected_attribute.lower() if expected_attribute else "location"
        
        # Word boundary regex search for subject & value to prevent subword false matches
        subj_match = False
        if subj_lower:
            subj_pattern = r'\b' + re.escape(subj_lower) + r'\b'
            subj_match = bool(re.search(subj_pattern, out_lower))
            
        val_match = False
        if val_lower:
            val_pattern = r'\b' + re.escape(val_lower) + r'\b'
            val_match = bool(re.search(val_pattern, out_lower))
            
        # Attribute check (e.g. lives in / located in / is in / city / location)
        attr_cues = ["in ", "at ", "lives", "located", "moved to", "currently in", "is in"]
        attr_match = any(cue in out_lower for cue in attr_cues) if expected_attribute else True
        
        # Check stale value (earlier value before state update)
        stale_match = False
        if earlier_value:
            stale_pattern = r'\b' + re.escape(earlier_value.lower()) + r'\b'
            stale_match = bool(re.search(stale_pattern, out_lower))
            
        # Check distractor match
        distractor_match = False
        if distractor_values:
            for dist in distractor_values:
                if dist and re.search(r'\b' + re.escape(dist.lower()) + r'\b', out_lower):
                    distractor_match = True
                    break
                    
        # Strict correctness: requires subject AND current expected value
        # For pure literal copying, if task explicitly asks for name, name is both subject and value
        if task_type == "entity_copying":
            strict_correct = subj_match
            loose_correct = subj_match
        else:
            strict_correct = subj_match and val_match and (not stale_match)
            loose_correct = val_match and (not stale_match)
            
        return {
            "task_type": task_type,
            "is_abstention": False,
            "abstained": False,
            "abstention_correct": False,
            "subject_correct": subj_match,
            "attribute_correct": attr_match,
            "value_correct": val_match,
            "stale_value_retained": stale_match,
            "distractor_retained": distractor_match,
            "strict_correct": strict_correct,
            "loose_correct": loose_correct,
            "raw_output": clean_out
        }

def run_calibration_suite():
    evaluator = CalibratedEvaluator()
    
    test_cases = [
        {
            "id": "CASE_1",
            "desc": "Correct complete subject-attribute-value response",
            "task": "single_entity_updates",
            "prompt": "Where is Rajnath currently located?",
            "raw_out": "Rajnath is currently located in Saharanpur.",
            "subj": "Rajnath", "attr": "location", "val": "Saharanpur", "earlier": "Rourkela",
            "distractors": ["Vellore"], "abstain": False,
            "expected_strict": True, "expected_subj": True, "expected_val": True
        },
        {
            "id": "CASE_2",
            "desc": "Correct concise value-only response where task asks for value",
            "task": "single_entity_updates",
            "prompt": "Where is Rajnath currently located?",
            "raw_out": "Saharanpur.",
            "subj": "Rajnath", "attr": "location", "val": "Saharanpur", "earlier": "Rourkela",
            "distractors": ["Vellore"], "abstain": False,
            "expected_strict": False, "expected_subj": False, "expected_val": True # Value correct, but subject missing
        },
        {
            "id": "CASE_3",
            "desc": "Wrong subject with the correct value",
            "task": "single_entity_updates",
            "prompt": "Where is Rajnath currently located?",
            "raw_out": "Amitabh is currently in Saharanpur.",
            "subj": "Rajnath", "attr": "location", "val": "Saharanpur", "earlier": "Rourkela",
            "distractors": ["Vellore"], "abstain": False,
            "expected_strict": False, "expected_subj": False, "expected_val": True
        },
        {
            "id": "CASE_4",
            "desc": "Correct subject with the wrong value",
            "task": "single_entity_updates",
            "prompt": "Where is Rajnath currently located?",
            "raw_out": "Rajnath is currently in Aurangabad.",
            "subj": "Rajnath", "attr": "location", "val": "Saharanpur", "earlier": "Rourkela",
            "distractors": ["Vellore"], "abstain": False,
            "expected_strict": False, "expected_subj": True, "expected_val": False
        },
        {
            "id": "CASE_5",
            "desc": "Correct subject with a stale value",
            "task": "single_entity_updates",
            "prompt": "Where is Rajnath currently located?",
            "raw_out": "Rajnath is currently in Rourkela.",
            "subj": "Rajnath", "attr": "location", "val": "Saharanpur", "earlier": "Rourkela",
            "distractors": ["Vellore"], "abstain": False,
            "expected_strict": False, "expected_subj": True, "expected_val": False
        },
        {
            "id": "CASE_6",
            "desc": "Correct subject with the wrong attribute / irrelevant answer",
            "task": "single_entity_updates",
            "prompt": "Where is Rajnath currently located?",
            "raw_out": "Rajnath is a doctor.",
            "subj": "Rajnath", "attr": "location", "val": "Saharanpur", "earlier": "Rourkela",
            "distractors": ["Vellore"], "abstain": False,
            "expected_strict": False, "expected_subj": True, "expected_val": False
        },
        {
            "id": "CASE_7",
            "desc": "Valid abstention on an unanswerable probe",
            "task": "calibrated_abstention",
            "prompt": "What is the location of Vikas?",
            "raw_out": "The location of Vikas is unknown and not mentioned in the context.",
            "subj": "Vikas", "attr": "location", "val": "abstain", "earlier": None,
            "distractors": [], "abstain": True,
            "expected_strict": True, "expected_subj": None, "expected_val": False
        },
        {
            "id": "CASE_8",
            "desc": "Incorrect abstention on an answerable probe",
            "task": "single_entity_updates",
            "prompt": "Where is Rajnath currently located?",
            "raw_out": "I don't know where Rajnath is.",
            "subj": "Rajnath", "attr": "location", "val": "Saharanpur", "earlier": "Rourkela",
            "distractors": [], "abstain": False,
            "expected_strict": False, "expected_subj": False, "expected_val": False
        },
        {
            "id": "CASE_9",
            "desc": "Empty response",
            "task": "single_entity_updates",
            "prompt": "Where is Rajnath currently located?",
            "raw_out": "",
            "subj": "Rajnath", "attr": "location", "val": "Saharanpur", "earlier": "Rourkela",
            "distractors": [], "abstain": False,
            "expected_strict": False, "expected_subj": False, "expected_val": False
        },
        {
            "id": "CASE_10",
            "desc": "Repeated generic assistant response",
            "task": "single_entity_updates",
            "prompt": "Where is Rajnath currently located?",
            "raw_out": "Assistant: Assistant: Assistant: Assistant:",
            "subj": "Rajnath", "attr": "location", "val": "Saharanpur", "earlier": "Rourkela",
            "distractors": [], "abstain": False,
            "expected_strict": False, "expected_subj": False, "expected_val": False
        },
        {
            "id": "CASE_11",
            "desc": "Response containing multiple conflicting values (stale and current)",
            "task": "single_entity_updates",
            "prompt": "Where is Rajnath currently located?",
            "raw_out": "Rajnath moved from Rourkela to Saharanpur.",
            "subj": "Rajnath", "attr": "location", "val": "Saharanpur", "earlier": "Rourkela",
            "distractors": [], "abstain": False,
            # Because both are present, evaluator flags stale_value_retained=True
            "expected_strict": False, "expected_subj": True, "expected_val": True
        },
        {
            "id": "CASE_12",
            "desc": "Valid response with punctuation and capitalization differences",
            "task": "single_entity_updates",
            "prompt": "Where is Rajnath currently located?",
            "raw_out": "rajnath lives in saharanpur!",
            "subj": "Rajnath", "attr": "location", "val": "Saharanpur", "earlier": "Rourkela",
            "distractors": [], "abstain": False,
            "expected_strict": True, "expected_subj": True, "expected_val": True
        }
    ]
    
    results = []
    print("=" * 60)
    print("RUNNING INDEPENDENT EVALUATOR CALIBRATION SUITE (12 CASES)")
    print("=" * 60)
    all_passed = True
    
    for tc in test_cases:
        res = evaluator.evaluate(
            task_type=tc["task"],
            prompt=tc["prompt"],
            raw_output=tc["raw_out"],
            expected_subject=tc["subj"],
            expected_attribute=tc["attr"],
            expected_value=tc["val"],
            earlier_value=tc["earlier"],
            distractor_values=tc["distractors"],
            is_abstention=tc["abstain"]
        )
        passed_strict = (res["strict_correct"] == tc["expected_strict"])
        if tc["expected_subj"] is not None:
            passed_subj = (res["subject_correct"] == tc["expected_subj"])
        else:
            passed_subj = True
        passed_val = (res["value_correct"] == tc["expected_val"])
        
        case_passed = passed_strict and passed_subj and passed_val
        if not case_passed:
            all_passed = False
            
        print(f"[{'PASS' if case_passed else 'FAIL'}] {tc['id']}: {tc['desc']}")
        print(f"       Strict: got {res['strict_correct']}, expected {tc['expected_strict']}")
        print(f"       Subj:   got {res['subject_correct']}, expected {tc['expected_subj']}")
        print(f"       Val:    got {res['value_correct']}, expected {tc['expected_val']}")
        
        tc_report = dict(tc)
        tc_report["eval_result"] = res
        tc_report["passed"] = case_passed
        results.append(tc_report)
        
    print("=" * 60)
    print(f"CALIBRATION SUITE SUMMARY: {'ALL 12 CASES PASSED' if all_passed else 'FAILURES DETECTED'}")
    print("=" * 60)
    return results

if __name__ == "__main__":
    run_calibration_suite()

