import json, sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

TOK_DIR = REPO_ROOT / "release" / "chakrview-v0.5.0" / "data" / "experiments" / "vocab_4096"
CKPT_PATH = REPO_ROOT / "experiments" / "batch3111_3150" / "json_sft_v2_pilot_run" / "sft_best_checkpoint.pt"

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts
from experiments.batch3111_3150.schema_prefix_trie import JSONSchema, JSONType, SchemaConstrainedJSONValidator

tok, _ = load_tokenizer_artifacts(str(TOK_DIR))
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
model = ChakrMicro(ModelConfig(max_seq_len=512))
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()

prompt = "Instruction: Return a JSON object with fields 'status' and 'code' indicating server health status and return code.\nResponse:"
tokens = [0] + tok.encode(prompt)

schema = JSONSchema.from_keys(
    required_keys=["status", "code"],
    type_map={"status": JSONType.STRING, "code": JSONType.INTEGER}
)
validator = SchemaConstrainedJSONValidator(schema=schema)

decoded_vocab = {tid: tok.decode([tid], errors="replace") for tid in range(tok.vocab_size)}

print("Starting generation trace...")
gen_tokens = []
curr = list(tokens)

for step in range(30):
    x = torch.tensor([curr], dtype=torch.long)
    with torch.no_grad():
        logits = model(x)[0, -1, :].clone()
    
    # Check top 10 raw logits
    top10 = torch.topk(logits, 10)
    top10_info = [(tid, decoded_vocab[tid], round(logits[tid].item(), 3), validator.is_valid_string_extension(decoded_vocab[tid])) for tid in top10.indices.tolist()]
    
    # Pick first valid token
    chosen = None
    sorted_all = torch.argsort(logits, descending=True).tolist()
    for tid in sorted_all:
        s = decoded_vocab[tid]
        if not s:
            continue
        if validator.is_valid_string_extension(s):
            chosen = tid
            break
            
    print(f"\n--- STEP {step+1} ---")
    print("State:", validator.base_validator.state, "Is in key:", validator.is_in_key, "Emitted:", validator.emitted_keys)
    print("Top 5 unconstrained:")
    for tid, s, score, val_ok in top10_info[:5]:
        print(f"  ID {tid:4d} | {repr(s):15s} | score: {score:6.2f} | valid: {val_ok}")
    print(f"Chosen token: ID {chosen} | {repr(decoded_vocab[chosen])}")
    
    for ch in decoded_vocab[chosen]:
        validator.feed_char(ch)
    curr.append(chosen)
    gen_tokens.append(chosen)

print("\nFinal generated text prefix:", repr(tok.decode(gen_tokens)))
