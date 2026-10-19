import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.batch3111_3150.schema_prefix_trie import JSONSchema, SchemaConstrainedJSONValidator

tok, _ = load_tokenizer_artifacts("release/chakrview-v0.5.0/data/experiments/vocab_4096")
schema = JSONSchema.from_keys(required_keys=["status", "code"])
val = SchemaConstrainedJSONValidator(schema)

# Feed '{"status"'
prefix = '{"status"'
for c in prefix:
    res = val.feed_char(c)
    assert res, f"Failed on char {repr(c)}"

print("Base validator state:", val.base_validator.state)
print("Is in key:", val.is_in_key)
print("Emitted keys:", val.emitted_keys)
print("Valid extension ': ':", val.is_valid_string_extension(": "))
print("Valid extension '\\n       ':", val.is_valid_string_extension("\n       "))
