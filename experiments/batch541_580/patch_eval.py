from pathlib import Path

p = Path("experiments/batch541_580/train_intent_candidate.py")
content = p.read_text(encoding="utf-8")
old_chunk = """        try:
            # Check for direct JSON or enclosed JSON
            m = re.search(r"(\\{.*\\})", raw_gen)
            if m:
                pred_obj = json.loads(m.group(1))
                is_valid_json = True
                pred_intent = pred_obj.get("intent")
                pred_args = pred_obj.get("args", {})
        except Exception:
            is_valid_json = False"""

new_chunk = """        try:
            # Check for direct JSON or enclosed JSON
            m = re.search(r"({.*})", raw_gen)
            if m:
                pred_obj = json.loads(m.group(1))
                is_valid_json = True
                pred_intent = pred_obj.get("intent")
                pred_args = pred_obj.get("args", {})
        except Exception:
            is_valid_json = False"""

if old_chunk in content:
    content = content.replace(old_chunk, new_chunk)
    p.write_text(content, encoding="utf-8")
    print("PATCH APPLIED SUCCESSFULLY")
else:
    print("CHUNK NOT FOUND")
