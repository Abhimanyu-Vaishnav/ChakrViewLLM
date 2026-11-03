# ChakrView Milestone M3 — Conversational Benchmark Specification

**Suite Name:** `CHAKRVIEW_CONVERSATIONAL_BENCHMARK_v0.9`  
**Version:** `1.0.0`  
**File:** `evaluation/conversational_suite/m3_conversational_tasks.json`  
**Task Count:** Exactly 60 tasks  
**SHA-256:** `010906a46556363bcaeec6de0f0c324341264febd52fe0b950aa58e67300c20a`  

---

## Category Distribution (11 Required Categories)

| Category | Task Count | Target Capabilities |
|---|:---:|---|
| `greetings_and_natural_dialogue` | 5 | Natural greeting, assistant identity, polite response, goodbye, gratitude. |
| `direct_instruction_following` | 5 | Single-word responses, comma-separated outputs, phrase repeats, opposites. |
| `follow_up_understanding` | 5 | Answering queries conditioned on immediate previous turn context. |
| `multi_turn_context_retention` | 10 | 3-to-4 turn conversations with stored facts (names, secret codes, duration, colors, teams, allergies). |
| `grounded_question_answering` | 5 | Reading short supplied context passages and extracting exact facts without hallucination. |
| `basic_reasoning` | 5 | Syllogisms, temporal days of week, travel arrival time, states of matter, transitivity. |
| `arithmetic_and_logic` | 5 | Addition, subtraction/currency, multiplication, division, word problem count. |
| `summarization` | 5 | 1-sentence distillations of science, history, ecology, computing passages. |
| `format_adherence` | 5 | Valid JSON, markdown bullet list, bracket enclosing, ISO date, table headers. |
| `uncertainty_and_clarification` | 5 | Admitting lack of information for future events, private passwords, unstated facts. |
| `topic_changes_and_coherence` | 5 | Shifting abruptly from one domain to another without carrying over previous context. |

**Leakage Protection:** None of these 60 prompts or answers are present in any M3 training or validation split.

