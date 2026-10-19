import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.test_fake_model_schema_decoder import *

tok = FakeTokenizer(['<pad>', '<eos>', '{', '}', '"', ': ', ', ', 'status', 'code', 'queued', '202', ' ', '\n', '__main__'])
script = {
    (0,): {2: 10.0, 13: 8.0},
    (0, 2): {13: 15.0, 4: 8.0},
    (0, 2, 4): {13: 15.0, 7: 10.0, 8: 6.0},
    (0, 2, 4, 7): {4: 10.0},
    (0, 2, 4, 7, 4): {5: 10.0},
    (0, 2, 4, 7, 4, 5): {4: 10.0},
    (0, 2, 4, 7, 4, 5, 4): {9: 10.0},
    (0, 2, 4, 7, 4, 5, 4, 9): {4: 10.0},
    (0, 2, 4, 7, 4, 5, 4, 9, 4): {6: 10.0},
    (0, 2, 4, 7, 4, 5, 4, 9, 4, 6): {4: 10.0},
    (0, 2, 4, 7, 4, 5, 4, 9, 4, 6, 4): {8: 10.0},
    (0, 2, 4, 7, 4, 5, 4, 9, 4, 6, 4, 8): {4: 10.0},
    (0, 2, 4, 7, 4, 5, 4, 9, 4, 6, 4, 8, 4): {5: 10.0},
    (0, 2, 4, 7, 4, 5, 4, 9, 4, 6, 4, 8, 4, 5): {10: 10.0},
    (0, 2, 4, 7, 4, 5, 4, 9, 4, 6, 4, 8, 4, 5, 10): {3: 10.0},
}
model = FakeModel(script, len(tok.id_to_token))
schema = JSONSchema.from_keys(required_keys=['status', 'code'], type_map={'status': JSONType.STRING, 'code': JSONType.INTEGER})
out = generate_schema_constrained(model, tok, [0], schema, max_new_tokens=30, eos_id=1)
print("OUT TEXT:", repr(out["text"]))
print("STOP REASON:", out["stop_reason"])
print("GEN TOKENS:", out["generated_tokens"])
print("TOKEN NAMES:", [tok.id_to_token[t] for t in out["generated_tokens"]])
