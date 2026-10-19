import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.batch3111_3150.schema_prefix_trie import JSONSchema, SchemaConstrainedJSONValidator

schema = JSONSchema.from_keys(required_keys=['status', 'code'])
val = SchemaConstrainedJSONValidator(schema)

# 1. Feed '{'
val.feed_char('{')
# 2. Feed '"'
val.feed_char('"')
# 3. Feed 'stat'
for c in 'stat':
    val.feed_char(c)

print("Current key buffer:", repr(val.current_key_buffer))
print("Is in key:", val.is_in_key)

# Test candidate tokens:
# Token 459: 'us'
print("Can extend 'us'?", val.is_valid_string_extension('us'))

# What if 'us' is fed?
val.feed_char('u')
val.feed_char('s')
print("Current key buffer after 'us':", repr(val.current_key_buffer))

# Now at '{"status'
# Token 1240: '": "'
print("Can extend '\": \"'?", val.is_valid_string_extension('": "'))
# Token 544: '": '
print("Can extend '\": '?", val.is_valid_string_extension('": '))
# Token 37: '"'
print("Can extend '\"'?", val.is_valid_string_extension('"'))
