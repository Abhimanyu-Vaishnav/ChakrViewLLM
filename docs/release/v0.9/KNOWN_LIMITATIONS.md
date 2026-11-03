# ChakrView v0.9.0 — Known Limitations & Epistemic Boundaries

## 1. Parameter and Knowledge Limits
ChakrMicro is a 3.44M parameter model. It cannot compete with models orders of magnitude larger (e.g. 7B, 70B). It possesses no ungrounded factual knowledge beyond patterns present in its small synthetic pretraining corpus.

## 2. Repetition & Greedy Degeneracy
When decoding long sequences (>64 tokens) without temperature or repetition penalties, the model can enter phrase-repetition loops. Applying repetition penalties ($1.1 \le \text{rep} \le 1.25$) or n-gram blocking ($N=3$) mitigates this behavior.

## 3. Complex Arithmetic and Logic
The model cannot reliably perform multi-digit addition, subtraction, or logic word problems. Evaluated accuracy on arithmetic benchmarks is <10%.

## 4. Multi-Turn History Drift
In multi-turn sessions exceeding 300 tokens, attention heads struggle to track distant entity assignments due to microscopic model capacity.
