import urllib.request
import json
import time

time.sleep(2)
url_status = "http://127.0.0.1:7866/api/status"
req = urllib.request.urlopen(url_status)
status_data = json.loads(req.read().decode("utf-8"))
print("Status Response:", json.dumps(status_data, indent=2))

test_prompts = [
    ("en", "Good morning! How is everything going today?"),
    ("en", "Why does a computer need RAM in addition to storage?"),
    ("en", "Calculate 128 + 256."),
    ("hi", "नमस्ते! आज आपका दिन कैसा चल रहा है?"),
    ("hi", "कंप्यूटर में सीपीयू का क्या मुख्य कार्य होता है?"),
    ("hi", "भारत की राजधानी क्या है?"),
    ("hinglish", "Bhai kya chal raha hai aaj kal?"),
    ("hinglish", "RAM aur ROM me kya farak hota hai, aasan shabdon me samjhao."),
    ("hinglish", "Python me virtual environment kyun banana chahiye?"),
    ("uncertainty", "What will be the exact weather in New York on December 15, 2038?")
]

results = []
for lang, p in test_prompts:
    req_data = json.dumps({"prompt": p, "pure_neural": True, "temperature": 0.7, "max_new_tokens": 40}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:7866/api/generate", data=req_data, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req)
    out = json.loads(resp.read().decode("utf-8"))
    print(f"[{lang}] P: {p}")
    print(f"R: {out['reply']}")
    print(f"Source: {out['source']} | Latency: {out['latency_ms']} ms\n")
    results.append(out)

output_file = "experiments/genesis_intelligence_validation_006/application/app_verification_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump({"server_status": status_data, "test_responses": results}, f, indent=2, ensure_ascii=False)
print(f"Results written to {output_file}")
