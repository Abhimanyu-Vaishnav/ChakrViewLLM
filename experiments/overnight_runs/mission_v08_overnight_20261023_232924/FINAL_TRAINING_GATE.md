# Final Training Gate Report — Exp18 SFT Run
**Experiment ID**: `EXP18_SFT_V08_BOUNDED_120`
**Timestamp**: 2026-10-23T23:42:30.051666
**Total Steps Completed**: 120
**Elapsed Time**: 4.52 seconds
**Device**: CPU

## Loss Progression
- Initial Validation Loss: **4.6159**
- Final Training Loss: **3.3530**
- Final Validation Loss: **3.5017** (Perplexity: 33.17)
- Best Validation Loss: **3.3562** at Step **80**

## Checkpoints Produced
- Best Checkpoint: `D:\Project\ChakrView\experiments\overnight_runs\mission_v08_overnight_20261023_232924\checkpoints\sft_exp18_run\sft_best_checkpoint.pt`
- Step 60 Checkpoint: `D:\Project\ChakrView\experiments\overnight_runs\mission_v08_overnight_20261023_232924\checkpoints\sft_exp18_run\sft_checkpoint_step_60.pt`
- Step 120 Checkpoint: `D:\Project\ChakrView\experiments\overnight_runs\mission_v08_overnight_20261023_232924\checkpoints\sft_exp18_run\sft_checkpoint_step_120.pt`

## Verification
- Precondition gates passed before training.
- Zero mutations to canonical checkpoint `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`.
- Zero training on held-out test data.
- Checkpoint lineage fully preserved.
