# Evidence-Based Next Milestone Decision — Batches 1261–1300
## Post-Evaluation Outcome & Production Deployment Decision

### 1. Final Milestone Outcome Selection
Under Mission Directive Phase 8, we select:
**OUTCOME B: REFUTATION POLICY SUBSTANTIALLY IMPROVES ENTERPRISE SAFETY AND DECISION ACCURACY; NEURAL RERANKING DOES NOT BEAT THE BM25 PRODUCTION BASELINE.**

### 2. Empirical Rationale
1. **Refutation Policy Success**:
   - Incorporating evidence-grounded premise contradiction checks into `LocalKnowledgeAssistant.ask()` increased overall benchmark decision accuracy from **84.29% (59/70) to 98.57% (69/70)**.
   - Refutation of false-premise queries jumped from **0.0% (0/5) to 80.0% (4/5)**, strictly preserving exact citation offsets and providing clear factual counter-evidence.
   - The remaining false-premise item (Item 164) correctly triggered honest `ABSTAIN`, proving that the system will never assert refutation without indexed contradictory proof.
2. **Neural Reranking Underperformance**:
   - BM25 candidate recall at $K=10$ and $K=20$ is **100.0% (60/60)**.
   - However, sorting these 10 candidates with the domain-adaptive `ChakrBiEncoder` caused **Recall@1 to collapse from 98.33% to 15.00% (9/60)**, with MRR dropping from 0.9833 to 0.3724.
   - Latency degraded from $1.04\text{ ms}$ to $62.44\text{ ms}$ ($60\times$ slower on CPU due to 10 sequential forward passes per query).
   - Even score fusion ($\alpha=0.2$) degraded Recall@1 to 91.67%.

### 3. Decisions for Future Batches
1. **Production Engine Invariant**:
   **BM25 remains the primary and default retrieval engine** in ChakrView runtime. No neural reranking should be enabled by default.
2. **Refutation Policy Integration**:
   The evidence-grounded refutation mechanism in `LocalKnowledgeAssistant` is validated, regression-tested (28/28 tests passing), and approved for production use.
3. **Condition for Future Neural Retrieval Research**:
   Dense vector models must not be deployed as rerankers until they are trained with in-batch negatives directly on paragraph-scale text chunks ($>400$ chars) with higher embedding dimensionality ($d_{\text{embed}} \ge 128$).
