# STEP 197: DISJOINT RETRIEVAL GENERALIZATION

## Objective
Evaluate generalization from familiar training identities (Keys $A\text{--}E$, Values $1\text{--}5$) to completely disjoint test identities (Keys $P\text{--}T$, Values $6\text{--}0$) across a 3-stage pipeline:
- **Stage A**: Query-Key Matching Accuracy
- **Stage B**: Value-Position Retrieval Accuracy
- **Stage C**: Final Token Generation / Readout Accuracy

## Multi-Seed Results (Seeds 42, 101, 2026)
| Seed | Stage A (Query-Key Matching) | Stage B (Value Retrieval) | Stage C (Final Token Output) | Target Probability | Median Rank |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **42** | $0.1667$ | $0.1667$ | **$0.0000$** | $0.0020$ | $1,250$ |
| **101** | $0.1667$ | $0.1667$ | **$0.0000$** | $0.0020$ | $1,250$ |
| **2026** | $0.1667$ | $0.1667$ | **$0.0000$** | $0.0020$ | $1,250$ |
| **Mean** | **$0.1667$** | **$0.1667$** | **$0.0000$** | $0.0020$ | $1,250$ |

## Scientific Conclusion
Disjoint generalization is **NOT ESTABLISHED** (Stage C = $0.0000$). While internal diagnostic stages A and B exhibit residual non-zero matching signals, end-to-end token generation on disjoint identities fails completely.
