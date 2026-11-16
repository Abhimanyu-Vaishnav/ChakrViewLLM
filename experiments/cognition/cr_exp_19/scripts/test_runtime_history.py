import json
from pathlib import Path
import sys

REPO_ROOT = Path("D:/Project/ChakrView")
sys.path.insert(0, str(REPO_ROOT))

from chakrview.chat import ConversationalChatCLI

def run_runtime_history_tests():
    cli = ConversationalChatCLI(REPO_ROOT / "releases/v0.9.1_conversational_preview/chakrview_v091_model.pt")
    
    traces = []
    
    # Test 1: One-turn conversation
    cli.reset()
    p1, tr1 = cli.build_prompt("Hello, my name is Zaraph.")
    assert tr1["retained_turns_count"] == 0
    assert tr1["within_budget"]
    traces.append({"test": "1_turn", "trace": tr1, "prompt": p1})
    cli.history.append({"user": "Hello, my name is Zaraph.", "assistant": "Hello Zaraph!"})
    
    # Test 2: Two-turn conversation
    p2, tr2 = cli.build_prompt("Where do you recommend I travel?")
    assert tr2["retained_turns_count"] == 1
    assert "Zaraph" in p2
    assert tr2["within_budget"]
    traces.append({"test": "2_turns", "trace": tr2, "prompt": p2})
    cli.history.append({"user": "Where do you recommend I travel?", "assistant": "I recommend Kyoto."})
    
    # Test 3: Five-turn conversation (all short, all fit in 384 tokens)
    cli.history.append({"user": "What fruit is nice there?", "assistant": "Persimmons."})
    cli.history.append({"user": "How many days should I stay?", "assistant": "Five days."})
    p5, tr5 = cli.build_prompt("What was my name again?")
    assert tr5["retained_turns_count"] == 4
    assert "Zaraph" in p5, "Zaraph must be preserved in 5-turn history when within budget!"
    assert tr5["within_budget"]
    traces.append({"test": "5_turns", "trace": tr5, "prompt": p5})
    
    # Test 4: Eight-turn conversation
    cli.history.append({"user": "What was my name again?", "assistant": "Your name is Zaraph."})
    cli.history.append({"user": "Do you like tea?", "assistant": "Yes, green tea."})
    cli.history.append({"user": "What about coffee?", "assistant": "Coffee is great too."})
    p8, tr8 = cli.build_prompt("Can you summarize our talk?")
    assert tr8["within_budget"]
    traces.append({"test": "8_turns", "trace": tr8, "prompt": p8})
    
    # Test 5: Oldest turn removed under budget pressure
    # Add a synthetic turn with 300 tokens
    long_msg = "This is a detailed background statement with multiple explanations. " * 20
    cli.history.append({"user": long_msg, "assistant": "Understood your detailed background."})
    p_budget, tr_budget = cli.build_prompt("Quick question after the long context.")
    assert tr_budget["within_budget"]
    assert tr_budget["discarded_turns_count"] > 0, "Oldest turns must be dropped under pressure!"
    traces.append({"test": "budget_pressure", "trace": tr_budget, "prompt_preview": p_budget[:150]})
    
    # Test 6: Recent complete turns preserved (never split user/assistant)
    for part in p_budget.split("\n\n"):
        if "Assistant:" in part:
            assert "User:" in part or p_budget.endswith(part), "Assistant must never be orphan without its User turn!"
            
    # Test 7: Reset clears history
    cli.reset()
    assert len(cli.history) == 0
    p_reset, tr_reset = cli.build_prompt("Fresh hello")
    assert tr_reset["retained_turns_count"] == 0
    
    # Test 8: Determinism
    cli.history.append({"user": "Fact A", "assistant": "Ack A"})
    p_det1, _ = cli.build_prompt("Query")
    p_det2, _ = cli.build_prompt("Query")
    assert p_det1 == p_det2, "Assembled prompt must be deterministic!"
    
    # Save traces
    with open(REPO_ROOT / "experiments/cognition/cr_exp_19/results/runtime_context_traces.json", "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2)
        
    print("ALL RUNTIME HISTORY TESTS PASSED: 8/8")

if __name__ == "__main__":
    run_runtime_history_tests()

