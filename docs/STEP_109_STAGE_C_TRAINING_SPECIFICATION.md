# ChakrView Step 109: Stage-C Foundation Pre-Training Curriculum & Specification

- **Milestone Designation**: Step 109 (Controlled Stage-C Curriculum, Phased Training Protocol & Multi-Domain Evaluation)
- **Status**: COMPLETE SPECIFICATION & READY FOR AUTHORIZED EXECUTION
- **Date**: October 5, 2026
- **Corpus**: Stage C.1 Pre-Training Corpus (`7,703,067` tokens, 27 train shards)
- **Model Invariant**: `ChakrMicro` (6 layers, $d_{\text{model}}=192$, $d_{\text{ff}}=512$, 6 heads, head dim 32, Pre-RMSNorm, RoPE, SwiGLU, 3,443,136 parameters)
- **Starting Checkpoint**: Step 102 Candidate `artifacts/step102_stage_b_curriculum/checkpoints/checkpoint_0000500.pt` (Hash: `d5886e02e62df29fc88379ed31b37239c85fcda9648ffaf0cf92191730df9a01`)
- **Canonical Baseline**: `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da` ($\Delta W_{\text{baseline}} \equiv 0$ strictly immutable)

---

## 1. Scientific Hypotheses & Target Metrics

1. **Capacity Saturation Resolution Hypothesis**:
   - Step 103 proved that the 405k-token Stage B corpus suffered from data saturation and cross-domain oscillation when trained beyond 500 steps.
   - Transitioning to the 6.65M-token Stage C training corpus ($16.4\times$ token volume) will enable continued cross-entropy loss reduction without domain collapse.
2. **Quantitative Target Milestones for Stage C**:
   - **Validation Cross-Entropy Loss**: $< 4.00$ (down from Step 102's $4.6951$).
   - **Validation Perplexity**: $< 50.0$ (down from Step 102's $109.41$).
   - **Syntactic Probe Top-5 Accuracy**: $\ge 30.0\%$ (up from Step 102's $15.0\%$).
   - **Syntactic Probe Top-1 Accuracy**: Emergence of $\ge 5.0\%$ exact next-token prediction (up from $0.0\%$).
   - **Numerical Stability**: 0 NaNs, 0 Infs, max gradient L2 norm clipped at $1.0$.

---

## 2. Phased Curriculum Training Strategy

Stage C is structured into 5 controlled, sequential phases:

```
[ PHASE A: Warm Start & Stabilization ]
- Steps 0 -> 250 (Warmup from Step 102 candidate)
- Effective batch size = 4 (sequence length 256)
- Learning rate: 1e-4 -> 3e-4 (linear warmup)
                   |
                   v
[ PHASE B: Broad Foundation Exposure ]
- Steps 250 -> 1,500 (Full streaming across English & Hindi Wikipedia shards)
- Learning rate: 3e-4 (cosine decay towards 1e-4)
                   |
                   v
[ PHASE C: Multi-Domain Balancing ]
- Steps 1,500 -> 3,000 (Interleaved code algorithms, Sanskrit, and structured reasoning)
- Domain loss tracking: separate evaluation across code vs prose
                   |
                   v
[ PHASE D: Validation & Checkpoint Selection ]
- Steps 3,000 -> 4,000 (Decay to floor lr = 3e-5)
- Full split validation every 250 steps
- Atomic selection of peak validation perplexity checkpoint
                   |
                   v
[ PHASE E: Targeted Capability Probing ]
- Final checkpoint evaluated against frozen 40-probe syntactic benchmark
- Held-out Stage C test split evaluation
- Pre-Stage-D Decision Gate
```

---

## 3. Training Hyperparameter Configuration

```json
{
  "seed": 42,
  "sequence_length": 256,
  "batch_size": 2,
  "gradient_accumulation_steps": 2,
  "effective_batch_tokens": 1024,
  "optimizer": "adamw",
  "learning_rate": 3e-4,
  "min_learning_rate": 3e-5,
  "weight_decay": 0.01,
  "adam_beta1": 0.9,
  "adam_beta2": 0.95,
  "adam_eps": 1e-8,
  "gradient_clipping": 1.0,
  "warmup_steps": 100,
  "eval_interval": 250,
  "save_interval": 250,
  "keep_last_n": 5,
  "device": "cpu"
}
```

---

## 4. Rollback & Failure Protection Rules

1. **NaN / Inf Detection**: If loss or gradient norm evaluates to NaN or Inf, training halts immediately, state is preserved in `artifacts/failed_runs/`, and weights roll back to the preceding valid checkpoint.
2. **Divergence Threshold**: If validation loss exceeds $8.0$ after step 500, training is aborted.
3. **Identity Verification**: Trainer strictly verifies model config, tokenizer checksum (`7498d92adeef7c9e64d4d1bbd5b4b23f8774692d868742e3b98e30c584c4403f`), and parameter count (`3,443,136`) on every step and resume.
