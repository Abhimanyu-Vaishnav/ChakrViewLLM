# Step 221: Vocabulary Projection Investigation

## 1. Scientific Precondition Rule
> ONLY execute the full experiment if Step 218/219 provides evidence that:
> correct value representation is retrieved (Stage B passes) BUT final token generation fails.
> If Stage B is NOT successful, mark this step as:
> `NOT APPLICABLE — PRECONDITION FAILED`

## 2. Precondition Evaluation
- Stage B Value Representation Retrieval on Unseen Entities:
  - Contrastive Margin: $-0.0460$ (Threshold: $> 0.0$)
  - Representation Rank: $2.70$ (Threshold: $< 2.0$)
  - Result: **FAILED**
- Precondition Status:
  $$\mathbf{NOT\ APPLICABLE\ —\ PRECONDITION\ FAILED}$$

## 3. Isolated Candidate Projections (Comparative Reference)
For controlled diagnostic evaluation, candidate projections were evaluated in isolation:
- **Candidate A: Canonical Tied Readout** (0 parameter overhead)
  - Tied input-output embeddings. Familiar: 1.00, Disjoint: 0.00, Language Loss: 6.9691.
- **Candidate B: Isolated Untied Readout** ($+786,432$ parameters)
  - Disjoint: 0.00. Slower convergence, parameter bloat, does not fix routing.
- **Candidate C: Controlled Value-Representation Projection** ($+36,864$ parameters)
  - Maps retrieved state to candidate value subspace. Disjoint: 0.00 if upstream routing fails.
- **Candidate D: Nearest Contextual Value Representation Diagnostic** (0 parameter overhead)
  - Matches nearest in-context representation. Operates at chance ($\approx 0.333$) when value representations are not retrieved.

## 4. Scientific Conclusion
- Vocabulary projection cannot be blamed for failure on unseen entities when the model fails to retrieve the correct value representation internally.
- Associative routing must precede any vocabulary projection modification.
