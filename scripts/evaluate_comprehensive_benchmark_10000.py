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

# Models to evaluate across all milestones:
# Step 0 (Random), Step 200 (Batch 13), Step 1000 (Batch 14), Step 5000 (Batch 15), Step 10000 (Batch 16-21)
torch.manual_seed(42)
model_rand = ChakrMicro(ModelConfig(max_seq_len=512)).eval()

ckpt_b13 = torch.load("experiments/batch13_stage_c_convergence/training_checkpoint_step_200.pt", map_location="cpu", weights_only=False)
model_b13 = ChakrMicro(ModelConfig(max_seq_len=512)).eval()
model_b13.load_state_dict(ckpt_b13["model_state_dict"])

ckpt_b14 = torch.load("experiments/batch14_stage_c_pretraining/training_checkpoint_step_1000.pt", map_location="cpu", weights_only=False)
model_b14 = ChakrMicro(ModelConfig(max_seq_len=512)).eval()
model_b14.load_state_dict(ckpt_b14["model_state_dict"])

ckpt_b15 = torch.load("experiments/batch15_stage_c_scaling/main_training_5000/training_checkpoint_step_5000.pt", map_location="cpu", weights_only=False)
model_b15 = ChakrMicro(ModelConfig(max_seq_len=512)).eval()
model_b15.load_state_dict(ckpt_b15["model_state_dict"])

ckpt_b21 = torch.load("experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt", map_location="cpu", weights_only=False)
model_b21 = ChakrMicro(ModelConfig(max_seq_len=512)).eval()
model_b21.load_state_dict(ckpt_b21["model_state_dict"])

models = {
    "Step 0 (Random Init)": model_rand,
    "Step 200 (B13)": model_b13,
    "Step 1000 (B14)": model_b14,
    "Step 5000 (B15)": model_b15,
    "Step 10000 (B16-21)": model_b21,
}

probes = [
    # English
    ("English: Instruction set", "RISC-V represents an open, modular instruction set architecture designed for modern", " workloads"),
    ("English: Architecture stability", "Key components include Pre-RMSNorm for numerical stability during deep forward", " passes"),
    ("English: Special tokens", "Special tokens include Beginning of Sequence, End of Sequence, and", " Padding"),
    # Hindi Grammar & Agreement
    ("Hindi: Model autonomy", "यह प्रणाली बिना किसी पूर्व-प्रशिक्षित मॉडल या विदेशी सॉफ्टवेयर भार के पूर्णतः स्वतंत्र रूप से बनाई जा रही", " है"),
    ("Hindi: Digital knowledge", "भारतीय भाषाओं में डिजिटल ज्ञान का विस्तार करना हमारे अनुसंधान का केंद्रीय", " लक्ष्य"),
    ("Hindi: Verb-copula agreement", "संस्कृत के श्लोक और सूक्तियाँ ज्ञान का अनंत भंडार प्रस्तुत करती", " हैं"),
    ("Hindi: Postposition marker", "ग्रामीण और दूरदराज के क्षेत्रों में कम बिजली पर चलने वाले उपकरण अत्यंत उपयोगी सिद्ध", " होंगे"),
    # Hinglish
    ("Hinglish: Low resource design", "haan bilkul, ChakrView ko low resource systems ke liye hi design kiya gaya", " hai"),
    ("Hinglish: Test verification", "terminal mein jaakar pytest command run karo aur saare tests verify", " karo"),
]

print("=== STEP 0 TO STEP 10,000 MULTI-MILESTONE BENCHMARK ===")
results = []
for cat_name, prompt, expected_suffix in probes:
    p_ids = tokenizer.encode(prompt)
    exp_ids = tokenizer.encode(expected_suffix)
    full_ids = p_ids + exp_ids

    cat_res = {"probe": cat_name, "prompt": prompt, "target": expected_suffix, "target_tokens": exp_ids, "losses": {}, "greedy_gen": {}}

    target_safe = expected_suffix.encode("ascii", "backslashreplace").decode("ascii")
    print(f"\nPROBE: {cat_name}")
    print(f"Target: '{target_safe}' (tokens: {exp_ids})")

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
            for _ in range(8):
                cur_t = torch.tensor([gen_ids], dtype=torch.long)
                nxt = int(m(cur_t)[0, -1, :].argmax().item())
                gen_ids.append(nxt)
        continuation = tokenizer.decode(gen_ids[len(p_ids):], errors="replace")
        cat_res["greedy_gen"][m_name] = continuation
        cont_safe = continuation.encode("ascii", "backslashreplace").decode("ascii")
        print(f"  [{m_name}] Target Loss: {suffix_loss:.4f} | Continuation: '{cont_safe}'")
    results.append(cat_res)

out_file = repo_root / "experiments" / "batch16_21_combined" / "comprehensive_language_benchmark.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved comprehensive benchmark to: {out_file}")
