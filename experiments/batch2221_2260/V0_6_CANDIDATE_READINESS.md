# V0.6 RESEARCH CANDIDATE READINESS REPORT

**Batch:** 2221–2260  
**Candidate Name:** `chakrmicro-v0.6-candidate-cpt-seed42-step125`  
**Base Architecture:** ChakrMicro v0.1 (3,443,136 parameters, hash: `c5571c9c...`)  
**Tokenizer:** Canonical `vocab_4096` (`release/chakrview-v0.5.0/data/experiments/vocab_4096/`)  
**Candidate Checkpoint:** `experiments/batch2141_2180/cpt/seed_42/checkpoints/checkpoint_step_125.pt`  
**Checkpoint SHA-256:** `d3268fc49378f8c5b9a30e71d4bab58ce7c4db6bcaaa1f32932ccfd8694b07c3`  
**Candidate Bundle Directory:** `experiments/batch2221_2260/v0_6_candidate_bundle/`  

---

## 1. Candidate Packaging Specification

In accordance with project rules against duplicating large checkpoint files unnecessarily, the candidate bundle references canonical and candidate artifacts through verified relative paths and SHA-256 checksums:

```json
{
  "release_candidate": "chakrmicro-v0.6-candidate-cpt-seed42-step125",
  "version": "0.6.0-rc1",
  "date_prepared": "2026-10-17",
  "status": "RESEARCH_CANDIDATE_VALIDATED",
  "model_config": {
    "vocab_size": 4096,
    "d_model": 192,
    "n_layers": 6,
    "n_heads": 6,
    "hidden_dim": 512,
    "max_seq_len": 512,
    "tie_embeddings": true,
    "rope_theta": 10000.0,
    "parameter_count": 3443136,
    "architecture_hash": "c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da"
  },
  "artifacts": {
    "candidate_checkpoint_path": "experiments/batch2141_2180/cpt/seed_42/checkpoints/checkpoint_step_125.pt",
    "candidate_checkpoint_sha256": "d3268fc49378f8c5b9a30e71d4bab58ce7c4db6bcaaa1f32932ccfd8694b07c3",
    "canonical_baseline_checkpoint_path": "experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt",
    "canonical_baseline_checkpoint_sha256": "96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee",
    "tokenizer_dir": "release/chakrview-v0.5.0/data/experiments/vocab_4096",
    "tokenizer_vocab_sha256": "1d59ee38bbafe5bb7e279a0ec49d4f0d4a9ec032549a9539d91fbb984fa54fae"
  },
  "runtime_capabilities": {
    "adaptive_hardware_runtime": "Integrated (CPU for B=1-2, GPU for B>=4)",
    "verified_tool_router": "Integrated (SafeCalculator AST allowlist)",
    "consent_gated_feedback": "Integrated (--consent flag)",
    "citation_grounded_qa": "Integrated (LocalKnowledgeAssistant)"
  }
}
```

---

## 2. Release Gate Assessment

| Criteria / Dimension | Requirement | Candidate Status | Verdict |
| :--- | :--- | :--- | :---: |
| **Statistical Generalization** | Held-out Test CE lower than baseline across seeds | -0.1928 token CE delta; 76.55% doc win rate ($p < 10^{-45}$) | **PASS** |
| **Reproducibility** | Fresh independent reproduction matches | Evaluated 290 docs, 64,769 tokens; bit-identical replication | **PASS** |
| **Hardware Runtime** | Adaptive CPU/GPU execution without crashes | Benchmarked; 5.5x batched GPU speedup; CPU fallback verified | **PASS** |
| **Neuro-Symbolic Tooling** | Grounded arithmetic without eval/exec | 100% on 20 arithmetic probes via SafeCalculator | **PASS** |
| **User Privacy & Consent** | Consent required for conversation persistence | Enforced by default; no data stored unless opted-in | **PASS** |
| **Baseline Immutability** | Protected artifacts unaltered | Checkpoint `96357705...` and tokenizer verified unaltered | **PASS** |
| **Test Verification** | Focused integration suites pass cleanly | All tests passing across test suites | **PASS** |

### Release Decision:
The candidate is certified as a **Validated Research Candidate (`v0.6.0-rc1`)**. In accordance with project instructions, **ChakrView v0.6 is NOT declared publicly released by this batch**. The research candidate remains isolated in `experiments/batch2221_2260/` for continued evaluation.
