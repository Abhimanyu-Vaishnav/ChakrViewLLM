import urllib.request
import json
import time

time.sleep(2)
url_status = "http://127.0.0.1:7865/api/status"
req = urllib.request.urlopen(url_status)
print("Status Response:", req.read().decode("utf-8"))

test_prompts = [
    "Hi there! How are you doing today?",
    "Why does a computer need RAM in addition to storage?",
    "नमस्ते, आप कैसे हैं?",
    "भारत की राजधानी क्या है?",
    "Bhai kaise ho, kya hal chal hai?",
    "RAM aur ROM me kya farak hota hai, aasan shabdon me samjhao.",
    "Write a Python function to find the maximum element in a list of numbers.",
    "Calculate 24 + 37.",
    "The project deadline is Friday.",
    "Who will win the cricket world cup in 2035?"
]

results = []
for p in test_prompts:
    req_data = json.dumps({"prompt": p, "pure_neural": True, "temperature": 0.7, "max_new_tokens": 40}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:7865/api/generate", data=req_data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)
    out = json.loads(resp.read().decode("utf-8"))
    print(f"P: {p}")
    print(f"R: {out['reply']}")
    print(f"Source: {out['source']} | Latency: {out['latency_ms']} ms\n")
    results.append(out)

with open("experiments/genesis_conversational_foundation_005/evaluations/live_application_verification_10.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print("Verification complete.")
