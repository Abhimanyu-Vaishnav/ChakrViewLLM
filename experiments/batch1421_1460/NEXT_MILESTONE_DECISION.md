# Next Milestone Decision

## Batches 1421–1460 -> Batches 1461+

### 1. Selected Outcome and Strategic Recommendation

In accordance with Phase 7 of the mission mandate, we evaluate the four outcome criteria:

- **Selected Outcome: Outcome A**
  > *"The candidate's improvement is reproduced on independent data, with no material measured regression in the available auxiliary evaluations."*

#### Scientific Grounds:
1. The domain-adapted candidate (`Step 10,080`) demonstrated a statistically robust **41.5% perplexity reduction** across 30 completely independent, uninspected technical documents spanning 5 major computer science disciplines.
2. The improvement was consistent across **100% of evaluated documents** (30/30 wins), with negligible multi-seed training variance ($\sigma = 0.21$).
3. Qualitative probes confirmed reduced degeneration and improved technical formatting, while regression suites confirmed 100% backward compatibility across all 52 unit tests.

---

### 2. Recommended Next Engineering Experiment (Batches 1461+)

Now that `ChakrMicro`'s technical domain generalization has been independently established:

**Primary Focus: Constrained Generative Evidence Synthesis for Local Knowledge Assistant**
- Currently, the Local Knowledge Assistant returns verbatim extracted sentence chunks from the retrieved document.
- **Objective:** Evaluate whether the domain-adapted `ChakrMicro` core can take a prompt containing the user question and the verified retrieved evidence snippet, and autoregressively synthesize a **fluent, concise, grounded 1-to-2 sentence answer**, while preserving:
  1. 100% citation and evidence-span fidelity.
  2. Sub-10 ms CPU inference latency.
  3. Zero hallucination of ungrounded facts or numbers.

---

### 3. Checkpoint & Artifact Lineage

- **Canonical Checkpoint:**
  - Path: `experiments/batch16_21_combined/main_training_10000/training_checkpoint_step_10000.pt`
  - SHA-256: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee` (Canonical baseline)
- **Experimental Domain-Adapted Checkpoint:**
  - Path: `experiments/batch1381_1420/checkpoints/chakrmicro_domain_adapted_experimental.pt`
  - SHA-256: `15d48176efd6941415cc446b0f0602573a266a49a17da56d72d365d445091139`
- **Replication Checkpoints:**
  - Seed 101: `experiments/batch1421_1460/checkpoints/chakrmicro_domain_adapted_seed_101.pt`
  - Seed 202: `experiments/batch1421_1460/checkpoints/chakrmicro_domain_adapted_seed_202.pt`
- **Independent Corpus Manifest:**
  - `experiments/batch1421_1460/data/independent_corpus_manifest.json`
- **Machine-Readable Benchmark Results:**
  - `experiments/batch1421_1460/results/INDEPENDENT_EVALUATION_RESULTS.json`
  - `experiments/batch1421_1460/results/SEED_REPLICATION_RESULTS.json`
- **Regression Unit Tests:**
  - `tests/test_batch1421_1460_independent_validation.py` (5 tests passing)
  - Total repository pass rate: 52 / 52 pytest unit tests passing.
