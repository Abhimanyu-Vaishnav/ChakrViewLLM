# CBR-01 — CONVERSATIONAL MODEL RELEASE GATES & ACCEPTANCE CRITERIA

**Target Release:** ChakrView v0.9.1 (or Candidate v1.0)  
**Baseline Reference:** Canonical Step 10k (0.0% Conversational Accuracy, 53.1% Attractor Rate)  

---

## 1. Quantitative Capability Acceptance Gates

Before any candidate model can be promoted or published as a conversational milestone, it must satisfy all of the following numeric thresholds on independent, held-out evaluation benchmarks:

| Capability Dimension | Baseline Score | Mandatory Release Threshold | Target Benchmark Suite | Verification Method |
| :--- | :---: | :---: | :--- | :--- |
| **Conversational Benchmark (M3)** | **0.0%** (0/60) | **$\ge 10.0\%$** ($\ge 6/60$) | `m3_conversational_tasks.json` | Word-boundary / exact-match scoring |
| **Natural Greetings & Dialogue** | **0.0%** (0/5) | **$\ge 40.0\%$** ($\ge 2/5$) | `greetings_and_natural_dialogue` | Politeness & coherence check |
| **Direct Instruction Following** | **0.0%** (0/5) | **$\ge 20.0\%$** ($\ge 1/5$) | `direct_instruction_following` | Format & instruction compliance |
| **Attractor Degeneracy Rate** | **53.1%** | **$\le 20.0\%$** | 32 standard probes | Repetitive 4-gram ratio $< 0.05$ |
| **Clean `<EOS>` Turn Termination** | **6.7%** | **$\ge 80.0\%$** | M3 benchmark runs | Proper sequence termination before max tokens |
| **Native Intelligence Suite** | **3.1%** (1/32) | **$\ge 6.2\%$** ($\ge 2/32$) | `intelligence_tasks.json` | Non-regression check on reasoning |
| **Coding Syntax Validity** | **25.0%** (3/12) | **$\ge 33.3\%$** ($\ge 4/12$) | `coding_tasks.json` | Python AST parse validity |

---

## 2. Engineering & Operational Gates

1. **CPU Inference Latency:** Model cold-start $\le 600\text{ ms}$; generation speed $\ge 50\text{ tokens/sec}$ on standard laptop CPU.
2. **Reproducibility:** Training script reproduces candidate checkpoint within $\pm 1\%$ evaluation variance across identical seeds.
3. **Packaging Completeness:** Release package includes model card, evaluation report, dataset provenance, installation verification, and known limitations.
4. **Epistemic Honesty:** All negative results, out-of-distribution failures, and capacity bounds are published transparently.
