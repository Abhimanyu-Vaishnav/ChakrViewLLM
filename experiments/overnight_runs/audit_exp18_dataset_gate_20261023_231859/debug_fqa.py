import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chakrview.evaluation.task_evaluator import TaskEvaluator, EvaluationOutcome
from experiments.research_v09_conversation_sft.exp18_scaled_instruction_format_v1.generate_exp18_datasets import Exp18DatasetGenerator

evaluator = TaskEvaluator()
gen = Exp18DatasetGenerator()
ds = gen.generate_all_splits()

for split in ["train", "val", "test"]:
    for r in ds[split]:
        if r["category"] == "factual_qa":
            res = evaluator.evaluate_record(r, r["response"], stop_reason="STOP_TOKEN")
            if res.outcome != EvaluationOutcome.CORRECT:
                print(f"FQA Incorrect [{split}]: {r['id']}")
                print("  Prompt:  ", r["prompt"])
                print("  Response:", r["response"])
                print("  Target:  ", res.expected_target)
                print("  Reason:  ", res.reason_codes)
