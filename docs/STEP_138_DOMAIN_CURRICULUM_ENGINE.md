# Step 138: Domain Curriculum Engine

## 1. Overview
Step 138 introduces a multi-stage curriculum abstraction designed to train and evaluate domain capability without compromising general capability.

## 2. Key Architecture Components

- `CurriculumStage`: Formal progression pipeline:
  `FOUNDATION -> DOMAIN_INTRO -> DOMAIN_STRUCTURE -> DOMAIN_REASONING -> DOMAIN_APPLICATION -> EVALUATION -> GENERALIZATION`.
- `DomainCurriculumSample`: Structured training/evaluation sample bound to curriculum stage and domain identity.
- `CurriculumExecutionManifest`: Audited lineage record capturing curriculum ID, domain ID, dataset SHA-256 checksum, tokenizer checksum, random seed, candidate checkpoint ID, and baseline model hash.
- `DomainCurriculumEngine`: Manages staged batch generation, held-out sample isolation, and manifest compilation.

## 3. Empirical Verification
- Verified generation of stage-specific training batches.
- Verified strict held-out validation sample isolation.
- Verified deterministic SHA-256 dataset checksum generation and manifest recording.
