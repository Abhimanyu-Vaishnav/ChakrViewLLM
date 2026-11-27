import os
import sys
import torch

sys.path.insert(0, os.path.abspath("."))

from chakrview.brain.genesis.chakr_genesis_17m import ChakrGenesis17M, Genesis17MConfig
from chakrview.brain.genesis.chakr_genesis_38m import ChakrGenesis38M, Genesis38MConfig
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

tokenizer, _ = load_tokenizer_artifacts("data/experiments/vocab_4096")

checkpoints = {
    "Stage-D": ("experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt", ChakrGenesis17M, Genesis17MConfig),
    "RC1": ("release/v1.0.0-rc1/chakr_1_0_rc1_best.pt", ChakrGenesis17M, Genesis17MConfig),
    "RC2": ("experiments/genesis_stage_rc2_046/chakr_1_0_rc2_best.pt", ChakrGenesis17M, Genesis17MConfig),
    "RC3": ("experiments/fast_track_047_049/chakr_1_0_rc3_best.pt", ChakrGenesis17M, Genesis17MConfig),
    "NR051_17M_Base": ("experiments/pretraining_nr051/chakr_17m_pretrained_base.pt", ChakrGenesis17M, Genesis17MConfig),
    "NR051_38M_Base": ("experiments/pretraining_nr051/chakr_38m_pretrained_base.pt", ChakrGenesis38M, Genesis38MConfig),
    "NR051_17M_SFT": ("experiments/pretraining_nr051/chakr_17m_sft_candidate.pt", ChakrGenesis17M, Genesis17MConfig),
    "NR051_38M_SFT": ("experiments/pretraining_nr051/chakr_38m_sft_candidate.pt", ChakrGenesis38M, Genesis38MConfig),
    "NR052_17M_Base": ("experiments/pretraining_nr052/chakr_17m_v4_base.pt", ChakrGenesis17M, Genesis17MConfig),
    "NR052_38M_Base": ("experiments/pretraining_nr052/chakr_38m_v4_base.pt", ChakrGenesis38M, Genesis38MConfig),
    "NR052_17M_SFT": ("experiments/pretraining_nr052/chakr_17m_nr052_sft.pt", ChakrGenesis17M, Genesis17MConfig),
    "NR052_38M_SFT": ("experiments/pretraining_nr052/chakr_38m_nr052_sft.pt", ChakrGenesis38M, Genesis38MConfig),
}

prompts = [
    "What is the capital of Japan?",
    "Which is the largest ocean on Earth?",
    "What is 5 minus 3?"
]

results = {}

for name, (path, cls, cfg_cls) in checkpoints.items():
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    cfg = ckpt.get("config", cfg_cls())
    if isinstance(cfg, dict):
        cfg = cfg_cls(**cfg)
        
    model = cls(cfg)
    incompat = model.load_state_dict(state, strict=True)
    
    unique_params = sum(p.numel() for p in model.parameters())
    
    # Run deterministic greedy test
    model.eval()
    gen_samples = {}
    for p in prompts:
        # Prompt formatting: format as user query
        formatted = f"<|user|>\n{p}\n<|assistant|>\n"
        tokens = tokenizer.encode(formatted)
        input_ids = torch.tensor([[0] + tokens], dtype=torch.long)
        
        # Test greedy decoding
        generated = input_ids.clone()
        with torch.no_grad():
            for _ in range(15):
                out = model(generated)
                logits = out[0] if isinstance(out, tuple) else out
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == 1 or next_token.item() == 2:  # EOS
                    break
        
        resp_tokens = generated[0, input_ids.shape[1]:].tolist()
        gen_text = tokenizer.decode(resp_tokens).strip()
        gen_samples[p] = gen_text
        
    print(f"[{name}] LOADED STRICTLY. Unique params: {unique_params:,}. Missing: {len(incompat.missing_keys)}, Unexpected: {len(incompat.unexpected_keys)}")
    for p, ans in gen_samples.items():
        print(f"   Q: {p} -> A: {ans}")
    results[name] = {
        "unique_params": unique_params,
        "strict_load": True,
        "samples": gen_samples
    }

print("ALL 12 CHECKPOINTS LOADED AND VALIDATED WITH ZERO ERRORS.")
