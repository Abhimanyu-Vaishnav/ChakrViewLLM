# PREREGISTRATION UNRESOLVED ITEMS & HUMAN APPROVAL CHECKLIST
**Batch:** 2581–2620  
**Repository:** ChakrViewLLM (`dev/v0.6-v0.7-validation`)  
**Context:** Actions required before training may be initiated.

---

### 1. Items Resolved in Audit Batches (Complete)
- [x] Checkpoint lineage unconfounded: Canonical Step 10,000 checkpoint identified as starting point for Arms 1, 2, and 3.
- [x] Reference comparator role established for historical Seed 42 CPT candidate.
- [x] 3-tiered leakage screening protocol specified (exact 8-gram, NFKC normalized Jaccard, MinHash subword).
- [x] Held-out test set strictly quarantined.
- [x] Multiseed protocol specified (`seeds = [42, 43, 44]`) with variance-aware decision rules.
- [x] Early stopping guardrails defined for Hindi and global regression.

---

### 2. Remaining Unresolved Actions & Human Approval Decisions

The following technical actions cannot be performed during an audit mission and remain as explicit approval gates for the human reviewer prior to training authorization:

1. **Candidate Corpus Selection & License Verification:**
   - *Action:* Review and approve the specific external Classical Sanskrit texts proposed for the 15% Sanskrit mix (e.g., Vedic/Classical public-domain corpora, licenses, provenance).
   - *Status:* Pending human approval.
2. **Empirical Tokenizer Preflight Execution:**
   - *Action:* Once candidate text files are provided, run the preflight script measuring characters per token against the provisional $\ge 1.80\text{ chars/token}$ threshold.
   - *Status:* Pending corpus availability.
3. **Leakage Screening Execution & Artifact Freezing:**
   - *Action:* Execute the 3-tier screening against all 290 held-out evaluation documents; review any matches and compute immutable SHA-256 manifests for the training mix.
   - *Status:* Pending corpus availability.
4. **Formal Authorization of GPU Compute:**
   - *Action:* Explicit human authorization to launch the 3-seed training comparison ($500 \text{ steps} \times 3 \text{ seeds} \times 3 \text{ arms} \approx 4,500 \text{ training steps}$).
   - *Status:* Strictly pending.
