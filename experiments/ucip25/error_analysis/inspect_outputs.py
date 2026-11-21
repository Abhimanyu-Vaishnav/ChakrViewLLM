"""
UCIP-25 Error Analysis & Prediction Inspection:
Analyzes exact token outputs, error modes, repetition patterns,
and generation boundary behavior for Step800 and UCIP-20.
"""

from collections import Counter
import json


def inspect_predictions():
    for m in ["step800", "ucip20_best"]:
        pred_path = f"experiments/ucip25/predictions/{m}_predictions.json"
        with open(pred_path, "r", encoding="utf-8") as f:
            preds = json.load(f)

        print(f"\n==================== {m} PREDICTIONS ANALYSIS ====================")
        print(f"Total predictions: {len(preds)}")

        # Check prediction distribution
        pred_texts = [p["prediction"].strip() for p in preds]
        counts = Counter(pred_texts)
        print("Top 10 Most Common Output Strings:")
        for text, cnt in counts.most_common(10):
            print(f"  ({cnt}x): {repr(text)}")

        # Check by task
        print("\nRepresentative Output per Task:")
        seen_tasks = set()
        for p in preds:
            t = p["task_type"]
            if t not in seen_tasks:
                seen_tasks.add(t)
                print(f"  {t}:")
                print(f"    Target:     {repr(p['target'])}")
                print(f"    Prediction: {repr(p['prediction'])}")
                print(f"    Tokens:     {p['prediction_tokens']}")

    # Write summary to error_analysis
    analysis_data = {}
    for m in ["step800", "ucip20_best"]:
        pred_path = f"experiments/ucip25/predictions/{m}_predictions.json"
        with open(pred_path, "r", encoding="utf-8") as f:
            preds = json.load(f)

        pred_texts = [p["prediction"].strip() for p in preds]
        counts = Counter(pred_texts)
        analysis_data[m] = {
            "total": len(preds),
            "unique_outputs": len(counts),
            "top_outputs": counts.most_common(10),
            "empty_outputs": sum(1 for t in pred_texts if not t),
            "eos_immediate_outputs": sum(1 for p in preds if not p["prediction_tokens"]),
            "dominant_attractor": counts.most_common(1)[0] if counts else None
        }

    with open("experiments/ucip25/error_analysis/error_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, indent=2)


if __name__ == "__main__":
    inspect_predictions()
