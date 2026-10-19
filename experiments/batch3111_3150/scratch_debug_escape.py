import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.batch3111_3150.schema_prefix_trie import JSONSchema, SchemaConstrainedJSONValidator, JSONType

schema = JSONSchema.from_keys(required_keys=["msg", "count"], type_map={"msg": JSONType.STRING, "count": JSONType.INTEGER})
val = SchemaConstrainedJSONValidator(schema)

test_str = r'{"msg": "line1\nline2 \"quoted\" \\ slash", "count": 12}'
for idx, c in enumerate(test_str):
    res = val.feed_char(c)
    print(f"idx={idx:2d} char={repr(c):6s} in_str={val.base_validator.in_string} esc={val.base_validator.escape} state={val.base_validator.state} res={res}")
    if not res:
        print("FAILED AT CHAR", repr(c))
        break
