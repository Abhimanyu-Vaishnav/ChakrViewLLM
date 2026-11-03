# ChakrView v0.9.0 — Training Lineage and Checkpoint Provenance

## 1. Lineage Graph

```
[Synthetic Pretraining Corpus (Stage C, 6.65M tokens)]
                        │
                        ▼ (Pretraining Engine: AdamW, lr=5e-4, 10k steps)
     [Canonical Baseline Step 10,000 Checkpoint]
     (SHA-256: 96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee)
                        │
        ┌───────────────┴───────────────┐
        ▼                               ▼
[Exp34 SFT Pilot Runs]          [M3 Conversational Pilot Runs]
(Attractor Suppression)         (Steps 50, 100, 250, 500)
        │                               │
        ▼                               ▼
[Forensic Diagnosis (M4)]       [Diagnostic Evaluations (M4)]
```

---

## 2. Checkpoint Registry

| Checkpoint Name | Rel Path | Step | Checkpoint SHA-256 | Description |
| :--- | :--- | :---: | :--- | :--- |
| **Canonical Step 10k** | `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt` | 10,000 | `96357705a683482e...` | Canonical baseline pretraining checkpoint |
| **M3 Step 50** | `experiments/m3_conversational_intelligence/checkpoints/m3_step_50.pt` | 50 | `df3bc1c5fee3a3b2...` | Early conversational fine-tuning checkpoint |
| **M3 Step 100** | `experiments/m3_conversational_intelligence/checkpoints/m3_step_100.pt` | 100 | `8cd534e1d13a1767...` | Mid-stage conversational fine-tuning checkpoint |
| **M3 Step 250** | `experiments/m3_conversational_intelligence/checkpoints/m3_step_250.pt` | 250 | `91b0bae2a8c737c8...` | Lowest attractor rate checkpoint (15.6%) |
| **M3 Step 500** | `experiments/m3_conversational_intelligence/checkpoints/m3_step_500.pt` | 500 | `36d5aa5ca98fb995...` | Full-epoch conversational fine-tuning checkpoint |
