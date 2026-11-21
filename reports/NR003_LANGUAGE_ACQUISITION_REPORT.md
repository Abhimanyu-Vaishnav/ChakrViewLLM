# NR003: Language Acquisition Report

**Date:** 21 November 2026  
**Mission ID:** CV-GENESIS-NR-003  
**Project:** ChakrView  

---

## 1. Controlled Language Probes

Evaluated with identical prompts across the three checkpoints:

| Probe Task | Prompt | 1. Pretraining-Only | 2. Genesis SFT | 3. Balanced Recovery SFT |
| :--- | :--- | :--- | :--- | :--- |
| **English Sentence Continuation** | `The sun is a` | `ftkenansीुनpi gi.lo...` | `ke dakloga RAM sist...` | `wcomponedeatwint...` |
| **Factual Continuation** | `Water is composed of` | `verurorptiorrangye...` | `जलल जलririririri...` | `ft ree! What kind ythyal...` |
| **Domain Explanation** | `In computer science, a function`| `o treses gena lo...` | `.` | `, wleकlphave a S...` |
| **Factual Knowledge** | `Bharat ki rajdhani` | `और SucorPmore...` | `iahanablurka lanations...` | `aInbhantdsachehat...` |
| **Conversational Greeting** | `User: Hello! Who are you?\nAssistant:` | `ImकaatINvalariaat...` | `Hello! I am doing well and ready to assist you.` | `You wत.` |

---

## 2. Findings on True Language Acquisition

1. **Pretraining Checkpoint:** Completely ungrounded because of the 3,677 permuted token IDs between `data/experiments/vocab_4096` and `experiments/tokenizer/candidates/v4096`. It did not acquire true semantic language representations.
2. **Genesis SFT Checkpoint:** Learned English syntax exclusively from the 1,476 SFT pairs. For conversational greetings that matched training examples (e.g. `"Hello! Who are you?"`), it produced fluent, coherent English text. However, for out-of-distribution prompts, it lacked foundational knowledge and outputted fragments of memorized response templates.
3. **Conclusion:** The model currently lacks general language acquisition. True conversational intelligence requires pretraining against properly aligned token IDs.
