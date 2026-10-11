import json
from pathlib import Path
import torch
import torch.nn.functional as F
from chakrview.brain.config import ModelConfig
from chakrview.brain.model import ChakrMicro
from chakrview.tokenizer.serialization import load_tokenizer_artifacts

repo_root = Path(".")
tok_dir = repo_root / "data" / "experiments" / "vocab_4096"
tokenizer, _ = load_tokenizer_artifacts(tok_dir)

# Models to evaluate
torch.manual_seed(42)
model_rand = ChakrMicro(ModelConfig(max_seq_len=512)).eval()

ckpt_b13 = torch.load("experiments/batch13_stage_c_convergence/training_checkpoint_step_200.pt", map_location="cpu", weights_only=False)
model_b13 = ChakrMicro(ModelConfig(max_seq_len=512)).eval()
model_b13.load_state_dict(ckpt_b13["model_state_dict"])

ckpt_b14 = torch.load("experiments/batch14_stage_c_pretraining/training_checkpoint_step_1000.pt", map_location="cpu", weights_only=False)
model_b14 = ChakrMicro(ModelConfig(max_seq_len=512)).eval()
model_b14.load_state_dict(ckpt_b14["model_state_dict"])

models = {
    "Random Init": model_rand,
    "Batch 13 (Step 200)": model_b13,
    "Batch 14 (Step 1000)": model_b14,
}

probes = [
    ("English: Instruction set", "RISC-V represents an open, modular instruction set architecture designed for modern", " workloads"),
    ("English: Architecture stability", "Key components include Pre-RMSNorm for numerical stability during deep forward", " passes"),
    ("English: Special tokens", "Special tokens include Beginning of Sequence, End of Sequence, and", " Padding"),
    ("Hindi: Model autonomy", "यह प्रणाली बिना किसी पूर्व-प्रशिक्षित मॉडल या विदेशी सॉफ्टवेयर भार के पूर्णतः स्वतंत्र रूप से बनाई जा रही", " है"),
    ("Hindi: Digital knowledge", "भारतीय भाषाओं में डिजिटल ज्ञान का विस्तार करना हमारे अनुसंधान का केंद्रीय", " लक्ष्य"),
    ("Hinglish: Low resource design", "haan bilkul, ChakrView ko low resource systems ke liye hi design kiya gaya", " hai"),
    ("Hinglish: Test verification", "terminal mein jaakar pytest command run karo aur saare tests verify", " karo"),
]

print("=== LANGUAGE CAPABILITY PROBES EVALUATION ===")
results = []
for cat_name, prompt, expected_suffix in probes:
    p_ids = tokenizer.encode(prompt)
    exp_ids = tokenizer.encode(expected_suffix)
    full_ids = p_ids + exp_ids

    print(f"\nPROBE: {cat_name}")
    print(f"Prompt: {prompt.encode('ascii', 'backslashreplace').decode('ascii')}")
    print(f"Target suffix: {expected_suffix.encode('ascii', 'backslashreplace').decode('ascii')} (token IDs: {exp_ids})")

    cat_res = {"probe": cat_name, "prompt": prompt, "target": expected_suffix, "losses": {}, "greedy_gen": {}}

    for m_name, m in models.items():
        inp = torch.tensor([full_ids[:-1]], dtype=torch.long)
        tgt = torch.tensor([full_ids[1:]], dtype=torch.long)
        with torch.no_grad():
            logits = m(inp)
            loss_all = F.cross_entropy(logits.reshape(-1, 4096), tgt.reshape(-1), reduction="none")
            suffix_loss = loss_all[-(len(exp_ids)):].mean().item()
        cat_res["losses"][m_name] = round(suffix_loss, 4)

        gen_ids = list(p_ids)
        with torch.no_grad():
            for _ in range(6):
                cur_t = torch.tensor([gen_ids], dtype=torch.long)
                nxt = int(m(cur_t)[0, -1, :].argmax().item())
                gen_ids.append(nxt)
        continuation = tokenizer.decode(gen_ids[len(p_ids):], errors="replace")
        cat_res["greedy_gen"][m_name] = continuation
        cont_safe = continuation.encode('ascii', 'backslashreplace').decode('ascii')
        print(f"  [{m_name}] Target Loss: {suffix_loss:.4f} | Continuation: '{cont_safe}'")
    results.append(cat_res)

with open("experiments/batch14_stage_c_pretraining/language_probes.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("\nLanguage probes successfully saved to experiments/batch14_stage_c_pretraining/language_probes.json")
