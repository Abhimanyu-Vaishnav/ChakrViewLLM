# Step 61: Semantic Memory Revalidation Specification

## 1. Objective
A previously learned repository experience cannot be assumed to remain valid when a repository undergoes runtime modifications. The `RepositoryImpactAnalyzer` provides deterministic evaluation of semantic memory applicability.

## 2. Revalidation Decision Space

| Status | Meaning | Applicability |
| :--- | :--- | :--- |
| `VALID` | Changes do not touch memory target modules or dependencies, or changes are purely cosmetic. | Applicable (`True`) |
| `CONDITIONALLY_VALID` | Transitive consumers or tests modified; root pattern remains intact. | Applicable (`True` with caution) |
| `STALE` | Direct dependencies or structural signatures of memory target modules mutated. | Not Applicable (`False`) |
| `INVALID` | Required target modules deleted, language/framework mismatch, or boundary violated. | Not Applicable (`False`) |
| `ABSTAIN` | Candidate memories conflict or evidence is insufficient. | Hard Abstention |

## 3. Decision Pipeline

$$\text{Memory} + \text{RepositoryState} + \text{RepositoryDiff} \longrightarrow \text{ImpactAnalysis} \longrightarrow \text{MemoryRevalidationDecision}$$

1. **Version Lineage Gate**: If `active_version == False`, status is immediately `INVALID`.
2. **Environment Gate**: If `language` or `framework` mismatches, status is `INVALID`.
3. **Module Existence Gate**: If any module in `affected_modules` is missing, status is `INVALID`.
4. **Boundary Gate**: If any changed module matches a negative boundary constraint in `known_boundaries`, status is `INVALID`.
5. **Cosmetic Gate**: If diff highest category is `COSMETIC`, status is `VALID`.
6. **Dependency Mutation Gate**: If target modules have `DEPENDENCY`, `ARCHITECTURAL`, or `BEHAVIORAL` changes, status is `STALE`.
7. **Downstream Gate**: If only downstream consumers are modified, status is `CONDITIONALLY_VALID`.
