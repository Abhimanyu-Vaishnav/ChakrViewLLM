"""
chakrview/export/export_chakrmicro_gguf.py

Non-destructive prototype converter exporting ChakrMicro PyTorch checkpoints
into compliant GGUF format for architecture and packaging feasibility verification.
Preserves canonical artifacts completely.
"""

import sys
import os
import json
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.export.gguf_writer import GGUFWriter, GGML_TYPE_F32

CANONICAL_CKPT = REPO_ROOT / "experiments" / "batch16_21_combined" / "main_training_10000" / "training_checkpoint_step_10000.pt"
TOKENIZER_DIR = REPO_ROOT / "data" / "experiments" / "vocab_4096"


def convert_chakrmicro_to_gguf(ckpt_path: Path, output_gguf_path: Path):
    print("=" * 65)
    print("CHAKRMICRO TO GGUF PROTOTYPE CONVERTER")
    print("=" * 65)
    print(f"Loading checkpoint from: {ckpt_path.relative_to(REPO_ROOT)}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    
    from chakrview.tokenizer.serialization import load_tokenizer_artifacts
    tokenizer, _ = load_tokenizer_artifacts(TOKENIZER_DIR)
    
    tokens = ["<unk>"] * tokenizer.vocab_size
    for tok_id, tok_bytes in tokenizer.vocab.items():
        if 0 <= tok_id < len(tokens):
            try:
                tokens[tok_id] = tok_bytes.decode("utf-8")
            except UnicodeDecodeError:
                tokens[tok_id] = tok_bytes.decode("latin-1")
    
    output_gguf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_gguf_path, "wb") as f:
        writer = GGUFWriter(f)
        
        # 1. Architecture metadata (using standard 'llama' compatible schema)
        writer.add_string("general.architecture", "llama")
        writer.add_string("general.name", "ChakrMicro-Canonical-Step10000")
        writer.add_string("general.description", "ChakrView compact 4.5M pretraining foundation model")
        writer.add_uint32("llama.embedding_length", 192)
        writer.add_uint32("llama.block_count", 6)
        writer.add_uint32("llama.feed_forward_length", 512)
        writer.add_uint32("llama.attention.head_count", 6)
        writer.add_uint32("llama.attention.head_count_kv", 6)
        writer.add_uint32("llama.context_length", 512)
        writer.add_float32("llama.rope.freq_base", 10000.0)
        writer.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-5)
        writer.add_bool("general.weight_tying", True)
        
        # 2. Tokenizer metadata
        writer.add_string("tokenizer.ggml.model", "gpt2")  # BPE tokenizer
        writer.add_array_string("tokenizer.ggml.tokens", tokens)
        writer.add_uint32("tokenizer.ggml.bos_token_id", 0)
        writer.add_uint32("tokenizer.ggml.eos_token_id", 1)
        
        # 3. Add and map tensors
        tensor_map = {
            "embedding.weight": "token_embd.weight",
            "final_norm.weight": "output_norm.weight",
        }
        for i in range(6):
            tensor_map[f"layers.{i}.norm_1.weight"] = f"blk.{i}.attn_norm.weight"
            tensor_map[f"layers.{i}.attn.q_proj.weight"] = f"blk.{i}.attn_q.weight"
            tensor_map[f"layers.{i}.attn.k_proj.weight"] = f"blk.{i}.attn_k.weight"
            tensor_map[f"layers.{i}.attn.v_proj.weight"] = f"blk.{i}.attn_v.weight"
            tensor_map[f"layers.{i}.attn.out_proj.weight"] = f"blk.{i}.attn_output.weight"
            tensor_map[f"layers.{i}.norm_2.weight"] = f"blk.{i}.ffn_norm.weight"
            tensor_map[f"layers.{i}.ffn.gate_proj.weight"] = f"blk.{i}.ffn_gate.weight"
            tensor_map[f"layers.{i}.ffn.up_proj.weight"] = f"blk.{i}.ffn_up.weight"
            tensor_map[f"layers.{i}.ffn.down_proj.weight"] = f"blk.{i}.ffn_down.weight"
            
        mapped_count = 0
        for py_name, gguf_name in tensor_map.items():
            if py_name in state_dict:
                writer.add_tensor(gguf_name, state_dict[py_name], raw_dtype=GGML_TYPE_F32)
                mapped_count += 1
            else:
                print(f"Warning: Expected tensor '{py_name}' missing from state_dict!")
                
        # Also add output.weight tied to token_embd.weight for runtimes that require explicit output weight
        writer.add_tensor("output.weight", state_dict["embedding.weight"], raw_dtype=GGML_TYPE_F32)
        mapped_count += 1
        
        print(f"Writing GGUF header and {mapped_count} aligned tensors...")
        writer.write_header_and_data()
        
    file_size_mb = output_gguf_path.stat().st_size / (1024 * 1024)
    print(f"[OK] Successfully exported GGUF artifact to: {output_gguf_path}")
    print(f"[OK] Total file size: {file_size_mb:.2f} MB")
    print("=" * 65)


if __name__ == "__main__":
    out_path = REPO_ROOT / "artifacts" / "gguf_prototype" / "chakrmicro_step10000.gguf"
    convert_chakrmicro_to_gguf(CANONICAL_CKPT, out_path)
