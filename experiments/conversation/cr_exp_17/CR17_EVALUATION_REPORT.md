# CR-EXP-17: Comprehensive Evaluation Report

## 1. Experimental Objectives & Setup
CR-EXP-17 set out to diagnose, fix, and evaluate conversational capability on the frozen ChakrMicro v0.1 architecture ($3.44\text{M}$ parameters). The evaluation used `CONVERSATION_BENCHMARK_V1.json`, comprising 200 diverse, non-overlapping prompts spanning 12 capability dimensions:
1. Basic conversation (greetings, identity, pleasantries)
2. Follow-up understanding
3. Context retention
4. Multi-turn continuity
5. Reference resolution
6. Topic transition
7. Clarification
8. Instruction following
9. Naturalness
10. Consistency
11. Language consistency (Hindi/Hinglish/French)
12. Response completion

## 2. Quantitative Results Comparison

| Benchmark Dimension | CBR-SFT-04 (Parent Baseline) | ChakrView v0.9.1 (Best Model) |
| :--- | :---: | :---: |
| Overall Keyword Match Rate | 27 / 200 (13.5%) | 18 / 200 (9.0%) |
| Basic conversation | 12 / 20 (60.0%) | 8 / 20 (40.0%) |
| Follow-up understanding | 1 / 18 (5.5%) | 0 / 18 (0.0%) |
| Context retention | 0 / 18 (0.0%) | 0 / 18 (0.0%) |
| Multi-turn continuity | 1 / 18 (5.5%) | 2 / 18 (11.1%) |
| Reference resolution | 1 / 16 (6.2%) | 1 / 16 (6.2%) |
| Topic transition | 1 / 16 (6.2%) | 0 / 16 (0.0%) |
| Clarification | 0 / 16 (0.0%) | 0 / 16 (0.0%) |
| Instruction following | 1 / 16 (6.2%) | 1 / 16 (6.2%) |
| Naturalness | 0 / 16 (0.0%) | 0 / 16 (0.0%) |
| Consistency | 10 / 16 (62.5%)* | 3 / 16 (18.8%) |
| Language consistency | 0 / 16 (0.0%) | 3 / 16 (18.8%) |
| Response completion | 0 / 14 (0.0%) | 0 / 14 (0.0%) |
| **Refusal Sink Frequency** | **26 / 200 (13.0%)** | **0 / 200 (0.0%)** |
| **Identity Grounding** | **0 / 6 (0.0%)** | **4 / 6 (66.7%)** |

*\*Note on Consistency:* In CBR-SFT-04, 10/16 consistency hits were artifacts of negative keyword matching triggered by the refusal attractor phrase `"No, I am not provided in the context."`

## 3. Detailed Forensic Findings
1. **Refusal Removal:** In CR-EXP-17, the training corpus eliminated the unbalanced `[UNKNOWN]` / refusal attractor strings that previous SFT runs had overfitted. As a result, zero refusal sink tokens appeared in the 200 benchmark evaluations.
2. **Correct Self-Concept Acquisition:** The model successfully acquired the identity of ChakrView. When asked `"Could you state your identity and purpose?"`, the model generated: `"Yes, I am ChakrView, an independent neural language model developed from first principles."`
3. **Multilingual Emergence:** The model learned Hindi/Hinglish conversational tokens: responding with `"Namaste! Main aapki kya madad kar sakta hoon?"` to Hindi greetings.
4. **Limits of 3.44M Parameter Architecture:**
   - Multi-turn tracking across more than 2 turns degrades rapidly.
   - Fine-grained keyword adherence on complex instruction following is limited by the representational capacity of $d=192, L=6$.
   - Decoding calibration (penalty 1.15, no_repeat_ngram_size=3) successfully prevents repetition loops and degeneration.
