# CHAKRVIEW ACTIVE BLOCKERS & RISKS

| Blocker ID | Description | Severity | Mitigation Strategy | Owner Phase | Status |
| :---: | :--- | :---: | :--- | :---: | :---: |
| **BLK-001** | Small General Conversational Corpus | HIGH | Construct high-quality, legally clean, multilingually balanced corpus (English/Hindi/Hinglish) across 22 capability areas. | Phase 3 (NR-023) | OPEN |
| **BLK-002** | Tokenizer Hindi Compression Efficiency | MEDIUM | Audit byte fallback rate on Devanagari script and evaluate vocabulary expansion without breaking canonical compatibility. | Phase 4 (NR-024) | OPEN |
| **BLK-003** | 6GB VRAM Constraint During Training | HIGH | Employ mixed precision (FP16/BF16), gradient accumulation, dynamic batch sizing, and data streaming. | Phase 5 & 7 | MITIGATED |
| **BLK-004** | State Continuity Across Agent Context Resets | CRITICAL | Maintain persistent ledger files (`CHAKRVIEW_CURRENT_STATE.md`, `CHAKRVIEW_MISSION_LEDGER.md`, `CHAKRVIEW_ARTIFACT_REGISTRY.json`) updated after every phase. | Phase 0-24 | RESOLVED |
