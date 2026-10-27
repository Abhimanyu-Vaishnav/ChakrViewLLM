# ChakrView Branching & Merge Policy

**Document Status**: Active Repository Governance Standard
**Target Audience**: Release Engineers, ML Maintainers, and Contributors
**Applies To**: `Abhimanyu-Vaishnav/ChakrViewLLM`

---

## 1. Branch Taxonomy & Purpose

ChakrView enforces a structured multi-tier branch taxonomy to guarantee that experimental research, exploratory models, and production runtime run in strict isolation without destabilizing the stable public distribution.

```
                    [feature/*, experiment/*, research/*]
                                     |
                          (Evidence & Gate Review)
                                     v
                        [dev/v0.6-v0.7-validation] (develop)
                                     |
                          (Formal Release Review)
                                     v
                                   [main] (stable public)
                                     |
                                  [v0.X.0] (immutable tag)
```

### 1.1 `main` — Stable Public Branch
- **Purpose**: Represents the latest tested, peer-reviewed, and published release of ChakrView.
- **Contents**: Supported installation guides, production source code (`chakrview/`), regression suites (`tests/`), release notes, and public reproducibility documents.
- **Invariants**:
  - Zero raw overnight run dumps, ephemeral virtual environments, or intermediate checkpoint files.
  - Zero speculative or unverified cognitive architectures.
  - Protected branch requiring PR review and verified passing test gates.
  - Releases tagged directly on or cherry-picked to `main`.

### 1.2 `dev/v0.6-v0.7-validation` — Historical & Development Baseline
- **Purpose**: Serves as the continuous integration and release validation workbench where multi-batch features, runtime hardening, and release preparation take place before release publication.
- **Policy**: Preserved for historical continuity. Never force-pushed or rewritten. Changes targeting `main` must pass through explicit, documented merge gates.

### 1.3 `feature/<feature-name>` — Scoped Runtime & Infrastructure Features
- **Purpose**: Isolated branches for specific runtime enhancements, tokenizer adjustments, test suite additions, or infrastructure scripts.
- **Lifecycle**: Branch from `main` or `dev/v0.6-v0.7-validation`, implement focused changes with full unit tests, and submit a pull request once complete.

### 1.4 `experiment/expNN-<name>` — Experimental Research & Hypotheses
- **Purpose**: Research investigations exploring architectural variations (e.g. pointer copy heads, scaling parameters, novel loss objectives).
- **Isolation Rule**: Experimental implementations MUST remain quarantined on experiment branches and isolated directories (`experiments/`). They are **never merged directly to `main`** without passing an independent comparative gate against the canonical baseline ($\Delta W = 0$).

### 1.5 `research/<topic-name>` — Evaluation & Reproducibility Studies
- **Purpose**: Systematic evaluation suites, diagnostic audits, and empirical benchmark studies (e.g. human explanation reviews, attractor collapse probes).

### 1.6 Release Tags (`v0.X.0`)
- **Immutability Invariant**: Tags are immutable cryptographic bookmarks pointing to exact release commits.
- **Rule**: Never move, retarget, overwrite, or delete an existing release tag.

---

## 2. Mandatory Merge Review Requirements for `main`

A branch may be merged into `main` **only** when an explicit Merge Review Report is documented, satisfying all of the following criteria:

1. **Source & Target Verification**: Exact commit SHAs of source and target branches recorded.
2. **Ancestry & Fast-Forward**: Merge is verified clean without unresolvable conflicts.
3. **Canonical Artifact Hash Invariance**: Bit-exact SHA-256 verified for:
   - Checkpoint: `96357705a683482efcbb3b74e0ba239deb89ad59996b472a3849df9ce9f471ee`
   - Vocabulary: `db4d6119feddbcc08d0ca8c580965bfc2b6d9729aed741e83044181307e7218a`
   - Merges: `ada4b1dcfbb3d962a2e60203089881872a5063de73b2fef9eeff81ce0016efa7`
   - Model Core (`chakrview/brain/model.py`): `1343d7d18224cc308465b7f07eb6d889532e7f17cee74b5f84f878e542f72361`
   - Session Core (`chakrview/runtime/session.py`): `9080ff51adba38d4017801fb09fd655e27a854d23b05af557843865e6de9cb67`
4. **Automated Test Suite**: Regression smoke suite (`tests/test_v060_release_smoke.py`) and affected unit tests run and passing 100%.
5. **No Secret or Sensitive Data**: Scanned clean of API keys, tokens, personal identifiers, and machine-specific local paths.
6. **Documentation & Limitations**: Release notes and known limitations updated with honest epistemic bounds.
7. **Explicit Rollback Plan**: Documented rollback commit SHA and recovery procedure in case of unexpected regression.
