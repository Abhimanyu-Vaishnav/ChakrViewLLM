import os
import sys
import torch
import json
import hashlib

sys.path.insert(0, os.path.abspath("."))

def get_file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()

checkpoints = {
    "Stage-D Canonical": "experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt",
    "RC1 Release": "release/v1.0.0-rc1/chakr_1_0_rc1_best.pt",
    "RC2 Experimental": "experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt",
    "RC3 Baseline": "experiments/fast_track_047_049/chakr_1_0_rc3_best.pt",
    "NR051 17M Base": "experiments/pretraining_nr051/chakr_17m_pretrained_base.pt",
    "NR051 38M Base": "experiments/pretraining_nr051/chakr_38m_pretrained_base.pt",
    "NR051 17M SFT": "experiments/pretraining_nr051/chakr_17m_sft_candidate.pt",
    "NR051 38M SFT": "experiments/pretraining_nr051/chakr_38m_sft_candidate.pt",
    "NR052 17M Base": "experiments/pretraining_nr052/chakr_17m_v4_base.pt",
    "NR052 38M Base": "experiments/pretraining_nr052/chakr_38m_v4_base.pt",
    "NR052 17M SFT": "experiments/pretraining_nr052/chakr_17m_nr052_sft.pt",
    "NR052 38M SFT": "experiments/pretraining_nr052/chakr_38m_nr052_sft.pt"
}

results = {}

for name, path in checkpoints.items():
    if not os.path.exists(path):
        print(f"ERROR: Missing {path}")
        continue
    sha = get_file_sha256(path)
    data = torch.load(path, map_location="cpu", weights_only=False)
    
    meta = {}
    if isinstance(data, dict):
        if "model_state_dict" in data:
            state_dict = data["model_state_dict"]
            meta = {k: v for k, v in data.items() if k != "model_state_dict"}
        elif "state_dict" in data:
            state_dict = data["state_dict"]
            meta = {k: v for k, v in data.items() if k != "state_dict"}
        else:
            state_dict = data
    else:
        state_dict = data
        
    keys = list(state_dict.keys())
    tensor_elements = sum(p.numel() for p in state_dict.values())
    
    # Layer count
    layer_indices = set()
    for k in keys:
        if "layers." in k:
            parts = k.split("layers.")[1].split(".")
            layer_indices.add(int(parts[0]))
    n_layers = len(layer_indices)
    
    # Embedding
    embed_shape = None
    tied = False
    for k in keys:
        if "embed" in k and "weight" in k:
            embed_shape = list(state_dict[k].shape)
            break
            
    # Check LM Head
    has_lm_head = False
    lm_head_shape = None
    for k in keys:
        if "lm_head" in k and "weight" in k:
            has_lm_head = True
            lm_head_shape = list(state_dict[k].shape)
            break
            
    if not has_lm_head:
        tied = True
    elif embed_shape and lm_head_shape:
        for k_emb in keys:
            if "embed" in k_emb and "weight" in k_emb:
                for k_head in keys:
                    if "lm_head" in k_head and "weight" in k_head:
                        if state_dict[k_emb].data_ptr() == state_dict[k_head].data_ptr():
                            tied = True
                        elif torch.equal(state_dict[k_emb], state_dict[k_head]):
                            tied = "equal_values_separate_tensor"
                        else:
                            tied = False
                            
    # Attention shapes in layer 0
    q_shape, k_shape, v_shape, qkv_shape, out_proj_shape = None, None, None, None, None
    mlp_gate_shape, mlp_up_shape, mlp_down_shape = None, None, None
    norm_type = "RMSNorm"
    has_norm_bias = False
    
    for k in keys:
        if "layers.0." in k:
            shape = list(state_dict[k].shape)
            if "q_proj" in k or "wq" in k: q_shape = shape
            elif "k_proj" in k or "wk" in k: k_shape = shape
            elif "v_proj" in k or "wv" in k: v_shape = shape
            elif "qkv" in k: qkv_shape = shape
            elif "out_proj" in k or "wo" in k or "o_proj" in k: out_proj_shape = shape
            elif "gate" in k or "w1" in k: mlp_gate_shape = shape
            elif "up" in k or "w3" in k: mlp_up_shape = shape
            elif "down" in k or "w2" in k: mlp_down_shape = shape
            if "norm" in k and "bias" in k:
                has_norm_bias = True
                norm_type = "LayerNorm"
                
    d_model = embed_shape[1] if embed_shape else None
    vocab_size = embed_shape[0] if embed_shape else None
    
    # Calculate unique parameters
    unique_params = 0
    seen_pointers = set()
    for p in state_dict.values():
        ptr = p.data_ptr()
        if ptr not in seen_pointers:
            seen_pointers.add(ptr)
            unique_params += p.numel()
            
    # Serialize config / meta
    clean_meta = {}
    config_details = {}
    if "config" in meta:
        cfg = meta["config"]
        if hasattr(cfg, "__dict__"):
            config_details = {ck: cv for ck, cv in cfg.__dict__.items() if isinstance(cv, (int, float, str, bool))}
        elif isinstance(cfg, dict):
            config_details = {ck: cv for ck, cv in cfg.items() if isinstance(cv, (int, float, str, bool))}
            
    for mk, mv in meta.items():
        if mk == "config":
            clean_meta["config"] = config_details
        elif isinstance(mv, (int, float, str, bool, list, dict)):
            clean_meta[mk] = mv
        else:
            clean_meta[mk] = str(mv)
            
    results[name] = {
        "name": name,
        "path": path,
        "sha256": sha,
        "n_layers": n_layers,
        "d_model": d_model,
        "vocab_size": vocab_size,
        "embed_shape": embed_shape,
        "has_separate_lm_head_key": has_lm_head,
        "lm_head_shape": lm_head_shape,
        "tied_embedding": tied,
        "total_tensor_elements": tensor_elements,
        "unique_parameters": unique_params,
        "norm_type": norm_type,
        "q_shape": q_shape,
        "k_shape": k_shape,
        "v_shape": v_shape,
        "out_proj_shape": out_proj_shape,
        "mlp_gate_shape": mlp_gate_shape,
        "mlp_up_shape": mlp_up_shape,
        "mlp_down_shape": mlp_down_shape,
        "config_details": config_details,
        "meta_keys": list(meta.keys())
    }

with open("experiments/forensics_nr055/architecture_registry.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"SUCCESS: Audited {len(results)} checkpoints.")
for name, r in results.items():
    cfg = r['config_details']
    print(f"[{name}] layers={r['n_layers']} d_model={r['d_model']} q_shape={r['q_shape']} k_shape={r['k_shape']} unique_params={r['unique_parameters']:,} total_elements={r['total_tensor_elements']:,} tied={r['tied_embedding']}")
    if cfg:
        print(f"    Config: n_layers={cfg.get('n_layers')} d_model={cfg.get('d_model')} n_q_heads={cfg.get('n_q_heads')} n_kv_heads={cfg.get('n_kv_heads')} hidden_dim={cfg.get('hidden_dim')}")
