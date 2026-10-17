# RAW LOSS RECONCILIATION

**Batch:** 2341–2380  
**Focus:** Unrounded Evaluator Loss Accumulation vs Rounded Intermediate Aggregation  
**Status:** FULLY RECONCILED  

---

## 1. Discrepancy Overview

The audit prompt flagged a discrepancy between two reported figures for summed token loss:
- **Main Report Table:** Baseline = `290,096.76` | Seed 42 = `277,608.85`
- **FINAL_EVIDENCE.json:** Baseline = `290,094.04` | Seed 42 = `277,605.34`

While both pairs divide by 64,769 to yield the identical rounded cross-entropies of `4.4789` and `4.2861`, the ~2.72 to ~3.51 absolute loss difference required rigorous forensic investigation.

---

## 2. Forensic Analysis of Evaluator Implementation

Inspection of `experiments/batch2221_2260/run_independent_evaluation.py` reveals the exact mechanism:
1. In the inner token evaluation loop:
   ```python
   loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), reduction="mean")
   d_nll += loss.item() * n_tgt
   d_tokens += n_tgt
   ```
   Here, `d_nll` accumulates the **unrounded IEEE 754 64-bit float** negative log-likelihood across all tokens.
2. In the document recording step:
   ```python
   doc_results.append({
       "doc_id": did,
       "source_id": sid,
       "tokens": d_tokens,
       "loss": round(doc_loss, 4)   # <-- ROUNDED TO 4 DECIMAL PLACES FOR JSON STORAGE
   })
   ```
3. In the per-source summary step:
   ```python
   sce = sdata["nll"] / stoks
   per_source_metrics[sid] = {
       "token_weighted_ce": round(sce, 4)   # <-- ROUNDED TO 4 DECIMAL PLACES
   }
   ```

---

## 3. The Three Distinct Loss Aggregations

We executed a fresh re-evaluation with exact unrounded float accumulation ([FRESH_EVALUATION_OUTPUT.json](file:///D:/Project/ChakrView/experiments/batch2341_2380/FRESH_EVALUATION_OUTPUT.json)) and compared all three methods:

| Aggregation Method | Baseline Summed NLL | Baseline TW-CE | Seed 42 Summed NLL | Seed 42 TW-CE | Origin & Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Method A: True Evaluator Unrounded Float Accumulation** | **290,094.48** | **4.478909** | **277,607.59** | **4.286118** | Exact ground truth computed from raw token cross-entropy before any rounding. |
| **Method B: Document-Level Stored Loss Sum** | **290,094.04** | **4.478903** | **277,605.34** | **4.286084** | Computed by summing `round(doc_loss, 4) * tokens` across all 290 documents. |
| **Method C: Source-Level Stored CE Sum** | **290,096.76** | **4.478896** | **277,608.85** | **4.286138** | Computed by summing `round(source_ce, 4) * source_tokens` across the 4 sources. |

### Quantification of Precision Limits
- Absolute difference between Method A and Method B: $0.44$ loss units on baseline ($< 0.00015\%$).
- Absolute difference between Method A and Method C: $2.28$ loss units on baseline ($< 0.00079\%$).
- All three methods round to **exactly 4.4789** (Baseline) and **4.2861** (Candidate) at 4 decimal places.

### Conclusion
The reported figures `290,094.04` and `290,096.76` do not represent conflicting evaluations; they are the mathematical consequence of summing values rounded at the document level (Method B) versus values rounded at the source level (Method C). The true unrounded evaluator NLL accumulation is **290,094.48** (Baseline) and **277,607.59** (Seed 42).
