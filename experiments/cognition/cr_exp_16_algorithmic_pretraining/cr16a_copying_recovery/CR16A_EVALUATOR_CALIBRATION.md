# CR-EXP-16A Evaluator Calibration

## 1. Metrics Defined
1. **Exact Match (EM):**
   The generated text stripped of leading/trailing whitespace exactly equals the canonical expected target string:
   $$\text{EM} = \mathbb{I}(\text{pred.strip()} == \text{target.strip()})$$
2. **Token Presence (TP):**
   Every whitespace-separated word in the expected target is present within the set of whitespace-separated words in the generated prediction:
   $$\text{TP} = \mathbb{I}(\text{words}(\text{target}) \subseteq \text{words}(\text{pred}))$$
3. **Teacher-Forced Target Token Accuracy:**
   The proportion of target tokens where the argmax of the model's logits matches the ground truth token given the full preceding ground truth prefix.
4. **First Target Token Accuracy:**
   The argmax token at the prompt boundary position matches the very first token of the target sequence.

## 2. Calibration Findings
- On exact sequence copying (Greek letters), EM and TP closely correlate (EM: 66.0%, TP: 74.0%).
- On single and multi-token names, EM is 0.0% despite first-token accuracy being 78–88%. This divergence exposed the critical difference between **in-context pointer induction** (which works) and **subword spelling completion** (which fails in free autoregressive generation).

