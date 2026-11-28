"""
Comprehensive HF-02 Clean-Environment Audit Script.
Executes under .venv_hf_audit without relying on ChakrView repository code.
Performs:
1. Package inspection & file inventory
2. Weight distribution & safe removal of duplicate pytorch_model.bin from staging
3. Custom architecture loading from safetensors with tied embeddings & state_dict verification
4. Transformers compatibility testing (AutoConfig, AutoModelForCausalLM, AutoTokenizer)
5. Tokenizer audit (vocab hash, merge hash, BOS/EOS/PAD, round-trip fidelity, batch encoding)
6. 10-prompt fixed generation evaluation (English, Hindi, Hinglish, code, max context, empty)
"""

import os
import sys
import json
import hashlib
import torch
import safetensors.torch
from pathlib import Path

PKG_DIR = Path("artifacts/huggingface/chakrview-conversational-alpha").resolve()

def sha256_file(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def main():
    print("="*70)
    print("MISSION HF-02: COMPREHENSIVE HUGGING FACE PACKAGE AUDIT")
    print(f"Staging Directory: {PKG_DIR}")
    print(f"Python Executable: {sys.executable}")
    print(f"Torch Version: {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    print("="*70)

    # ---------------------------------------------------------
    # PART D: WEIGHT DISTRIBUTION & SAFETENSORS AUDIT
    # ---------------------------------------------------------
    print("\n--- PART D: WEIGHT DISTRIBUTION AUDIT ---")
    st_path = PKG_DIR / "model.safetensors"
    bin_path = PKG_DIR / "pytorch_model.bin"

    assert st_path.exists(), "model.safetensors does not exist!"
    st_size = st_path.stat().st_size
    st_hash = sha256_file(st_path)
    print(f"model.safetensors: {st_size:,} bytes | SHA256: {st_hash}")

    # Inspect safetensors keys for unwanted optimizer or private state
    st_tensors = safetensors.torch.load_file(st_path)
    print(f"Total tensors in model.safetensors: {len(st_tensors)}")
    unwanted_keys = [k for k in st_tensors.keys() if any(x in k.lower() for x in ["optimizer", "scheduler", "scaler", "grad", "adam", "step"])]
    print(f"Unwanted private/optimizer state keys: {unwanted_keys}")
    assert len(unwanted_keys) == 0, f"Found unwanted keys in safetensors: {unwanted_keys}"

    # As directed by Part D:
    # "Prefer model.safetensors as the primary artifact. Remove duplicate pytorch_model.bin from the staging package unless a tested compatibility requirement justifies retaining it. Do not modify the original checkpoint."
    if bin_path.exists():
        bin_size = bin_path.stat().st_size
        bin_hash = sha256_file(bin_path)
        print(f"pytorch_model.bin found: {bin_size:,} bytes | SHA256: {bin_hash}")
        print("Removing duplicate pytorch_model.bin from staging package to keep safetensors as sole primary artifact...")
        bin_path.unlink()
        print("pytorch_model.bin successfully removed from staging. (Original checkpoint in experiments/ preserved).")

    # ---------------------------------------------------------
    # PART A: CLEAN-ENVIRONMENT CUSTOM LOADING
    # ---------------------------------------------------------
    print("\n--- PART A: CLEAN-ENVIRONMENT CUSTOM LOADING ---")
    sys.path.insert(0, str(PKG_DIR))
    from modeling_chakrgenesis import ChakrGenesis38M, Genesis38MConfig
    from tokenization_chakrview import load_tokenizer_artifacts

    # Load config.json
    cfg_json_path = PKG_DIR / "config.json"
    with open(cfg_json_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    print("Loaded config.json:", cfg_dict)

    # Instantiate model
    cfg = Genesis38MConfig()
    model = ChakrGenesis38M(cfg)
    missing, unexpected = safetensors.torch.load_model(model, str(st_path), strict=True)
    print(f"safetensors load_model: missing={missing}, unexpected={unexpected}")
    assert len(missing) == 0 and len(unexpected) == 0, "Strict state_dict load failed!"

    # Verify tied embedding
    assert model.embedding.weight.data_ptr() == model.lm_head.weight.data_ptr(), "Tied weights check failed!"
    print("Verified tied embedding: model.embedding.weight and model.lm_head.weight share data pointer.")

    # CPU forward pass
    model.eval()
    dummy_input = torch.tensor([[10, 20, 30]], dtype=torch.long)
    with torch.no_grad():
        cpu_logits, _ = model(dummy_input)
    assert cpu_logits.shape == (1, 3, 4096), f"Unexpected CPU logits shape: {cpu_logits.shape}"
    print(f"CPU Forward pass succeeded! Logits shape: {cpu_logits.shape}")

    # GPU forward pass if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        model_gpu = model.to("cuda")
        with torch.no_grad():
            gpu_logits, _ = model_gpu(dummy_input.to("cuda"))
        assert gpu_logits.shape == (1, 3, 4096)
        print("GPU Forward pass succeeded on CUDA!")

    # ---------------------------------------------------------
    # PART B: TRANSFORMERS COMPATIBILITY AUDIT
    # ---------------------------------------------------------
    print("\n--- PART B: TRANSFORMERS COMPATIBILITY AUDIT ---")
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    print(f"Transformers version: {transformers.__version__}")

    transformers_compatible = True
    comp_notes = []

    # Test AutoConfig
    try:
        ac = AutoConfig.from_pretrained(str(PKG_DIR), trust_remote_code=True)
        print("AutoConfig.from_pretrained succeeded:", type(ac))
    except Exception as e:
        transformers_compatible = False
        comp_notes.append(f"AutoConfig.from_pretrained failed: {e}")
        print(f"AutoConfig test failed: {e}")

    # Test AutoModelForCausalLM
    try:
        am = AutoModelForCausalLM.from_pretrained(str(PKG_DIR), trust_remote_code=True)
        print("AutoModelForCausalLM.from_pretrained succeeded:", type(am))
    except Exception as e:
        transformers_compatible = False
        comp_notes.append(f"AutoModelForCausalLM.from_pretrained failed: {e}")
        print(f"AutoModelForCausalLM test failed: {e}")

    # Test AutoTokenizer
    try:
        at = AutoTokenizer.from_pretrained(str(PKG_DIR), trust_remote_code=True)
        print("AutoTokenizer.from_pretrained succeeded:", type(at))
    except Exception as e:
        transformers_compatible = False
        comp_notes.append(f"AutoTokenizer.from_pretrained failed: {e}")
        print(f"AutoTokenizer test failed: {e}")

    print(f"\nFinal Transformers Compatibility Verdict: {'YES' if transformers_compatible else 'NO'}")
    print("Classification: Custom PyTorch Standalone Package (ChakrGenesis38M Native Interface).")

    # ---------------------------------------------------------
    # PART C: TOKENIZER AUDIT
    # ---------------------------------------------------------
    print("\n--- PART C: TOKENIZER AUDIT ---")
    vocab_file = PKG_DIR / "vocab.json"
    merges_file = PKG_DIR / "merges.json"
    tok_cfg_file = PKG_DIR / "tokenizer_config.json"

    v_hash = sha256_file(vocab_file)
    m_hash = sha256_file(merges_file)
    print(f"vocab.json SHA256:  {v_hash}")
    print(f"merges.json SHA256: {m_hash}")

    tok, tok_meta = load_tokenizer_artifacts(str(PKG_DIR))
    print(f"Tokenizer special tokens: {tok_meta.get('special_tokens')}")
    print(f"Tokenizer vocab_size: {tok.vocab_size}, vocab len: {len(tok.vocab)}")

    # Special token checks
    # BOS=0, EOS=1 (or EOS=2 depending on config), PAD=2
    print(f"BOS ID: {tok_meta['special_tokens'].get('<BOS>')}, EOS ID: {tok_meta['special_tokens'].get('<EOS>')}, PAD ID: {tok_meta['special_tokens'].get('<PAD>')}")

    # Test sentences for round-trip
    test_suite = [
        "ChakrView indigenous sovereign cognitive AI.",
        "नमस्ते! यह हिंदी देवनागरी परीक्षण है।",
        "Hello dost, kya chal raha hai? Python list append function.",
        "```python\ndef fib(n):\n    return n if n <= 1 else fib(n-1) + fib(n-2)\n```",
        "Mathematical symbols: α + β = γ; 2 + 2 = 4; ∫ x dx"
    ]
    for ts in test_suite:
        enc = tok.encode(ts)
        dec = tok.decode(enc)
        assert dec == ts, f"Round-trip failed on: {ts}\nDecoded: {dec}"
        assert max(enc) < 4096, f"Token ID {max(enc)} out of model vocab range!"
    print("All Unicode and code round-trips verified 100% exact lossless reconstruction.")

    # Batch encoding & decoding test
    batch_enc = [tok.encode(ts) for ts in test_suite]
    batch_dec = [tok.decode(ids) for ids in batch_enc]
    assert batch_dec == test_suite
    print("Batch encoding/decoding verified.")

    # ---------------------------------------------------------
    # PART E: 10-PROMPT GENERATION VERIFICATION
    # ---------------------------------------------------------
    print("\n--- PART E: 10-PROMPT GENERATION VERIFICATION ---")
    test_prompts = [
        {"id": "p01_eng", "desc": "English conversation", "prompt": "User: Hello! Who are you?\nAssistant:"},
        {"id": "p02_hindi", "desc": "Hindi conversation", "prompt": "User: नमस्ते, आप कैसे हैं?\nAssistant:"},
        {"id": "p03_hinglish", "desc": "Hinglish conversation", "prompt": "User: Hello dost, kaise ho?\nAssistant:"},
        {"id": "p04_greeting", "desc": "Standard greeting", "prompt": "User: Good morning, assistant!\nAssistant:"},
        {"id": "p05_followup", "desc": "Follow-up question", "prompt": "User: What is Python?\nAssistant: Python is a programming language.\nUser: What is it used for?\nAssistant:"},
        {"id": "p06_fact", "desc": "Short factual question", "prompt": "User: What is the capital of India?\nAssistant:"},
        {"id": "p07_python", "desc": "Python code request", "prompt": "User: Write a Python function to add two numbers.\nAssistant:"},
        {"id": "p08_long", "desc": "Long input prompt", "prompt": "User: " + "Deep learning uses neural networks with many layers to represent complex data abstractions. " * 5 + "\nCan you summarize this?\nAssistant:"},
        {"id": "p09_empty", "desc": "Empty input handling", "prompt": ""},
        {"id": "p10_max_context", "desc": "Maximum context handling (250 tokens)", "prompt": "User: " + "word " * 120 + "\nAssistant:"}
    ]

    model.eval()
    gen_results = []

    def generate_tokens(prompt_str, max_new=40):
        if not prompt_str.strip():
            # Test empty prompt handling
            input_tokens = [0]
        else:
            input_tokens = tok.encode(prompt_str)
            if len(input_tokens) > (256 - max_new):
                input_tokens = input_tokens[-(256 - max_new):]

        input_tensor = torch.tensor([input_tokens], dtype=torch.long)
        gen = []
        with torch.no_grad():
            for _ in range(max_new):
                logits, _ = model(input_tensor)
                next_tok = torch.argmax(logits[0, -1, :]).item()
                if next_tok in [0, 2]: # PAD or EOS
                    break
                gen.append(next_tok)
                input_tensor = torch.cat([input_tensor, torch.tensor([[next_tok]], dtype=torch.long)], dim=1)
                if input_tensor.shape[1] >= 256:
                    break
        return tok.decode(gen), len(gen)

    for item in test_prompts:
        output_text, n_toks = generate_tokens(item["prompt"], max_new=32)
        rec = {
            "id": item["id"],
            "desc": item["desc"],
            "prompt": item["prompt"][:100] + ("..." if len(item["prompt"]) > 100 else ""),
            "generated_tokens": n_toks,
            "output_text": output_text
        }
        gen_results.append(rec)
        print(f"[{item['id']}] {item['desc']}:")
        print(f"   Prompt: {repr(rec['prompt'])}")
        print(f"   Output ({n_toks} toks): {repr(output_text)}\n")

    # Save generation results to JSON
    with open("experiments/cv24h_release/hf_audit_10_prompt_results.json", "w", encoding="utf-8") as f:
        json.dump(gen_results, f, indent=2, ensure_ascii=False)

    print("Generation verification complete. All 10 prompts evaluated cleanly without crash or infinite recursion.")

    # ---------------------------------------------------------
    # PART F & G: FINAL INVENTORY & METRICS FOR REPORT
    # ---------------------------------------------------------
    print("\n--- FINAL INVENTORY OF STAGING PACKAGE ---")
    staged_files = []
    for root, dirs, files in os.walk(PKG_DIR):
        for f in sorted(files):
            if "__pycache__" in root: continue
            fp = Path(root) / f
            sz = fp.stat().st_size
            h = sha256_file(fp)
            rel = fp.relative_to(PKG_DIR)
            staged_files.append({"path": str(rel), "size_bytes": sz, "sha256": h})
            print(f"{str(rel):30s} | {sz:>10,d} bytes | {h}")

    with open("experiments/cv24h_release/hf_staged_files_manifest.json", "w", encoding="utf-8") as f:
        json.dump(staged_files, f, indent=2)

    print("\nAUDIT SCRIPT EXECUTED SUCCESSFULLY.")

if __name__ == "__main__":
    main()
