# CBR-01 — IMMEDIATE NEXT ACTIONS & EXECUTION ROADMAP

---

## High-Priority Sequential Tasks

1. **Verify Tokenizer & Delimiter Contract:**
   - Execute regression test ensuring `strip_prompt_delimiters=False` in `SFTConfig` prevents token prefix misalignment between training and runtime generation.
2. **Execute Pilot Training Run (CBR-SFT-01):**
   - Run 100-step controlled SFT on `exp34_train_split_900.jsonl` using the canonical Step 10k baseline as parent checkpoint.
   - Use cosine decay learning rate ($5\times 10^{-5}$ to $5\times 10^{-6}$), batch size 2, gradient accumulation 2.
   - Save checkpoints at steps 25, 50, 75, 100 in an isolated experiment directory (`experiments/cbr01_sft/`).
3. **Run Independent Evaluation Suite:**
   - Benchmark candidate checkpoints against the canonical Step 10k baseline across all 60 M3 conversational tasks, 32 native intelligence tasks, and 12 coding syntax tasks.
4. **Statistical Hypothesis Testing:**
   - Apply two-way cluster bootstrap to confirm whether conversational and instruction-following improvements are statistically significant.
5. **Update Mission Documentation & Lineage:**
   - Log final results in `docs/missions/CBR_01/EXPERIMENT_REGISTRY.md` and report release readiness against the mandatory gates.
