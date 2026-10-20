"""
Run 10 conversational probes against the canonical ChakrView model.
Covers: greetings, identity, direct instructions, factual extraction,
summarization, basic reasoning, multi-turn in-context prompt, and follow-ups.
"""
import json
import time
from pathlib import Path
from chakrview.runtime.neural_runtime_adapter import NeuralRuntimeAdapter

prompts = [
    {"id": "CONV-01", "type": "greeting", "prompt": "Hello! Good day to you."},
    {"id": "CONV-02", "type": "identity", "prompt": "What is your name and purpose?"},
    {"id": "CONV-03", "type": "direct_instruction", "prompt": "Please write the letters: X Y Z"},
    {"id": "CONV-04", "type": "context_extraction", "prompt": "Context: Patient beta-44 was discharged at 14:00.\nQuestion: At what time was the patient discharged?\nAnswer:"},
    {"id": "CONV-05", "type": "summarization", "prompt": "Passage: Solar panels convert sunlight into electrical power using photovoltaic cells. In brief, solar panels generate:"},
    {"id": "CONV-06", "type": "basic_reasoning", "prompt": "If a box contains 3 red balls and 2 green balls, how many total balls are in the box?"},
    {"id": "CONV-07", "type": "multi_turn_context", "prompt": "User: My favorite color is purple.\nAssistant: Understood.\nUser: What is my favorite color?\nAssistant:"},
    {"id": "CONV-08", "type": "follow_up", "prompt": "User: I am building a website in Python.\nAssistant: Nice! Python has great frameworks.\nUser: Which framework do you recommend?\nAssistant:"},
    {"id": "CONV-09", "type": "logic", "prompt": "Is ice hot or cold? Answer:"},
    {"id": "CONV-10", "type": "constrained_output", "prompt": "Complete with exactly one word: The color of grass is"}
]

adapter = NeuralRuntimeAdapter(max_new_tokens=32, prompt_mode="raw_completion")
adapter.initialize()

results = []
for p in prompts:
    t0 = time.perf_counter()
    res = adapter.generate(p["prompt"], no_repeat_ngram_size=3)
    elapsed = (time.perf_counter() - t0) * 1000.0
    results.append({
        "id": p["id"],
        "type": p["type"],
        "prompt": p["prompt"],
        "generated_text": res["generated_text"],
        "tokens": res["output_token_count"],
        "latency_ms": round(elapsed, 2),
        "stop_reason": res["stop_reason"]
    })
    print(f"[{p['id']}] {p['type']}: {repr(res['generated_text'])}")

out_file = Path("evaluation/intelligence_suite/conversational_probes_canonical_step10000.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump({"conversational_probes": results}, f, indent=2)

print(f"Saved conversational probes to: {out_file}")