# CBR-01 — WORKSTREAM D: EXPERIMENT REGISTRY

**Mission:** Operation Conversational Breakthrough (CBR-01)  
**Parent Model:** ChakrMicro v0.1 Canonical Step 10k (`96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`)  
**Base Architecture:** 3,443,136 parameters, 6L/6H/192D, tied embeddings, RoPE, SwiGLU  

---

## Registered Experiments

### Exp 0: Canonical Baseline Replication
- **Status:** COMPLETED
- **Hypothesis:** Canonical Step 10k baseline yields reproducible 0/60 on M3 Conversational benchmark and 53.1% attractor rate on native CPU.
- **Outcome:** VERIFIED. Exactly 0/60 pass rate (0.0%).

### Exp 1: Prompt Contract & Delimiter Parity Fix
- **Status:** READY FOR VERIFICATION
- **Hypothesis:** Setting `strip_prompt_delimiters=False` in training dataset collation eliminates subword prefix mismatches between SFT training batches and runtime autoregressive generation.
- **Success Criteria:** Unit tests demonstrate bit-exact prompt tokenization parity between training tensor inputs and runtime session prompts.

### Exp 2: Diverse Instruction & Dynamic Induction SFT (CBR-SFT-01)
- **Status:** REGISTERED & SCHEDULED
- **Hypothesis:** Training ChakrMicro on a balanced, 100% diverse instruction mixture combining `Exp34 v3` (900 unique records) and dynamic induction variable binding will achieve non-zero conversational accuracy ($\ge 10\%$) while suppressing attractor loops ($\le 20\%$) and improving syntax validity.
- **Dataset:** `data/sft/exp34_candidate_v3/splits/exp34_train_split_900.jsonl` (900 train, 100 val)
- **Configuration:** Cosine LR ($5\times 10^{-5} \to 5\times 10^{-6}$), batch size 2, gradient accumulation 2, 250 steps, seed 42.
- **Parent Checkpoint:** Canonical Step 10,000.
