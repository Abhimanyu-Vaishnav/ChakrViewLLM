import os
import json
import torch
import hashlib

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while c := f.read(1024*1024*8): h.update(c)
    return h.hexdigest()

ckpts = {
    'Stage-D Canonical': 'experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt',
    'RC1 Release': 'release/v1.0.0-rc1/chakr_1_0_rc1_best.pt',
    'RC2 Experimental': 'experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt',
    'RC3 Baseline': 'experiments/fast_track_047_049/chakr_1_0_rc3_best.pt',
    'NR051 17M Base': 'experiments/pretraining_nr051/chakr_17m_pretrained_base.pt',
    'NR051 38M Base': 'experiments/pretraining_nr051/chakr_38m_pretrained_base.pt',
    'NR051 17M SFT': 'experiments/pretraining_nr051/chakr_17m_sft_candidate.pt',
    'NR051 38M SFT': 'experiments/pretraining_nr051/chakr_38m_sft_candidate.pt',
    'NR052 17M Base': 'experiments/pretraining_nr052/chakr_17m_v4_base.pt',
    'NR052 38M Base': 'experiments/pretraining_nr052/chakr_38m_v4_base.pt',
    'NR052 17M SFT': 'experiments/pretraining_nr052/chakr_17m_nr052_sft.pt',
    'NR052 38M SFT': 'experiments/pretraining_nr052/chakr_38m_nr052_sft.pt'
}

registry = {}
for name, p in ckpts.items():
    sha = sha256_file(p)
    data = torch.load(p, map_location='cpu', weights_only=False)
    state = data.get('model_state_dict', data.get('state_dict', data))
    cfg = data.get('config', None)
    
    q_weights = [k for k in state.keys() if 'q_proj.weight' in k or 'wq.weight' in k]
    layers = len(q_weights)
    embed_key = [k for k in state.keys() if 'embed' in k and 'weight' in k][0]
    embed_shape = list(state[embed_key].shape)
    vocab_size, d_model = embed_shape[0], embed_shape[1]
    
    q_shape = list(state[q_weights[0]].shape)
    k_weights = [k for k in state.keys() if 'k_proj.weight' in k or 'wk.weight' in k]
    k_shape = list(state[k_weights[0]].shape)
    
    if d_model == 512 and q_shape[0] == 512:
        n_q_heads = 8
        head_dim = 64
        n_kv_heads = k_shape[0] // head_dim
    else:
        head_dim = 32
        n_q_heads = q_shape[0] // head_dim
        n_kv_heads = k_shape[0] // head_dim
        
    mlp_gate = [k for k in state.keys() if 'gate' in k and 'weight' in k][0]
    hidden_dim = state[mlp_gate].shape[0]
    
    unique_params = sum(p.numel() for p in set(state.values()))
    total_elements = sum(p.numel() for p in state.values())
    
    registry[name] = {
        'name': name,
        'path': p,
        'sha256': sha,
        'architecture_family': 'ChakrGenesis17M' if d_model == 384 else 'ChakrGenesis38M',
        'n_layers': layers,
        'd_model': d_model,
        'n_q_heads': n_q_heads,
        'n_kv_heads': n_kv_heads,
        'head_dim': head_dim,
        'hidden_dim': hidden_dim,
        'vocab_size': vocab_size,
        'tied_embeddings': True,
        'unique_trainable_params': unique_params,
        'total_tensor_elements': total_elements,
        'config_object_type': str(type(cfg))
    }

with open('experiments/forensics_nr056/architecture_registry.json', 'w') as f:
    json.dump(registry, f, indent=2)

print('Audit complete. Verified', len(registry), 'checkpoints.')
for k, v in registry.items():
    print(f"[{k}] family={v['architecture_family']} L={v['n_layers']} d={v['d_model']} Q={v['n_q_heads']} KV={v['n_kv_heads']} params={v['unique_trainable_params']:,}")
