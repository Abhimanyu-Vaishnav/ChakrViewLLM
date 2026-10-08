# STEP 467: Language → Relational Representation Probe

## Objective
Probe the internal representations of the language-trained model using causal interventions to test whether representations preserve entity identity, relation direction, ordering, and subject/object roles.

## Causal Interventions
1. **Entity Swap Intervention**: Swap subject and object entities (`"A is above B"` vs `"B is above A"`).
2. **Relation Inversion Intervention**: Invert relation token (`"above"` vs `"below"`).
3. **Relation Removal Intervention**: Remove relational predicate.
4. **Unseen Entity Transfer**: Test representation shifts on synthetic unseen entities.

## Probing Metrics (Cosine Similarity & Divergence)
- **Cosine Similarity (Subject/Object Swap)**: `0.9291` (Role Sensitivity = `0.0709`)
- **Cosine Similarity (Relation Inversion)**: `0.9939` (Relation Sensitivity = `0.0061`)
- **Cosine Similarity (Relation Removal)**: `0.9822`
- **Cosine Similarity (Unseen Entities)**: `0.9705`

## Mechanistic Takeaway
The language-trained transformer layers produce representations with weak directional sensitivity (only ~0.6% divergence between opposite relational directions). This explains why pure next-token autoregressive models confuse relational order without an explicit cognitive routing substrate (`UnifiedCompositionalCore`).
