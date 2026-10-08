import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chakrview.runtime.unified_cognitive_runtime import UnifiedCognitiveRuntime
from chakrview.tokenizer import BPETokenizer
from pathlib import Path
from chakrview.tokenizer.trainer import load_experiment_artifacts

def load_tokenizer() -> BPETokenizer:
    exp_dir = Path(__file__).resolve().parent.parent / "data" / "tokenizer_experiments" / "v4096"
    if exp_dir.is_dir():
        return load_experiment_artifacts(exp_dir)
    return BPETokenizer()

def run_wave505_end_to_end_lifecycle():
    print("=" * 60)
    print("RUNNING WAVE 505: END-TO-END TASK LIFECYCLE")
    print("=" * 60)
    
    os.makedirs("artifacts/waves_504_511_execution", exist_ok=True)
    tok = load_tokenizer()
    runtime = UnifiedCognitiveRuntime()
    
    # 12 Controlled End-to-End Tasks
    tasks = [
        {"id": "task_01", "name": "1-hop direct", "prompt": "Alpha is above Beta. Who is above Beta?", "target": "Alpha"},
        {"id": "task_02", "name": "2-hop composition", "prompt": "Alpha is above Beta. Beta is above Gamma. Who is above Beta's target?", "target": "Gamma"},
        {"id": "task_03", "name": "3-hop chain", "prompt": "Alpha above Beta. Beta above Gamma. Gamma above Delta.", "target": "Delta"},
        {"id": "task_04", "name": "4-hop long chain", "prompt": "Alpha above Beta. Beta above Gamma. Gamma above Delta. Delta above Epsilon.", "target": "Epsilon"},
        {"id": "task_05", "name": "unknown proposition", "prompt": "Omega is above who?", "target": "UNKNOWN"},
        {"id": "task_06", "name": "conflicting proposition", "prompt": "Alpha is above Beta. Alpha is below Beta. CONFLICTING status.", "target": "CONFLICTING"},
        {"id": "task_07", "name": "distractor heavy", "prompt": "Theta near Zeta. Alpha above Beta. Beta left Gamma.", "target": "Gamma"},
        {"id": "task_08", "name": "unseen entity", "prompt": "Krypton is above Xenon. Xenon is above Radon.", "target": "Radon"},
        {"id": "task_09", "name": "unseen combination", "prompt": "Alpha is left of Beta. Beta is above Gamma.", "target": "Gamma"},
        {"id": "task_10", "name": "English natural", "prompt": "The primary entity Alpha lies above Beta.", "target": "Beta"},
        {"id": "task_11", "name": "Hindi natural", "prompt": "Alpha Beta ke upar hai.", "target": "Beta"},
        {"id": "task_12", "name": "mixed language", "prompt": "Alpha is above Beta aur Beta Gamma ke upar hai.", "target": "Gamma"},
    ]
    
    records = []
    print(f"{'Task ID':<10} | {'Task Name':<24} | {'Strategy':<26} | {'Confidence'} | {'Outcome'}")
    print("-" * 88)
    
    for t in tasks:
        ans, rec, exp = runtime.execute_task(
            task_id=t["id"],
            prompt=t["prompt"],
            expected_target=t["target"],
            tokenizer=tok
        )
        records.append(rec.to_dict())
        print(f"{t['id']:<10} | {t['name']:<24} | {rec.strategy_selected:<26} | {rec.confidence:<10.2f} | {'PASS' if ans == t['target'] else 'FAIL'} ({ans})")
        
    all_passed = all(r["final_result"] == tasks[i]["target"] for i, r in enumerate(records))
    print(f"\nEnd-to-End Task Lifecycle Verification: {'PASS' if all_passed else 'FAIL'} ({len(records)}/{len(tasks)})")
    
    out_file = "artifacts/waves_504_511_execution/wave505_task_lifecycle.json"
    with open(out_file, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Wave 505 complete. Saved to {out_file}")
    return records

if __name__ == "__main__":
    run_wave505_end_to_end_lifecycle()
