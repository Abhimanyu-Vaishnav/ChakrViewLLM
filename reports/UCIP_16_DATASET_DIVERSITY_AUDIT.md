# UCIP-16 Training Dataset Diversity & Template Audit

**Mission ID:** UCIP-16  
**Project:** ChakrView  
**Repository:** `D:\Project\ChakrView`  
**Dataset Target:** `data/sft/ucip14r/sft_train_12000.jsonl` (12,000 records)  
**Auditor:** Antigravity Senior Forensic Auditor  
**Date:** 2026-11-20  

---

## 1. Executive Summary

An exhaustive analysis of all 12,000 training records in `sft_train_12000.jsonl` reveals **catastrophic template concentration and lack of lexical diversity**. While the dataset nominally satisfies numeric category and language quotas, **84.6% of all response records are exact duplicates** of a small pool of repeated templates.

---

## 2. Response Diversity by Category

| Category ID | Description | Total Train Records | Unique Responses | Unique Ratio (%) | Top Response Multiplicity | Top Response Sample |
|---|---|---:|---:|---:|---:|---|
| `ARITH` | Arithmetic | 722 | 704 | **97.5%** | 2x | Algorithmic generation |
| `COPY` | Verbatim Copy | 474 | 473 | **99.8%** | 2x | Synthetic token copying |
| `FORMAT` | JSON Formatting | 468 | 468 | **100.0%** | 1x | Synthetic JSON records |
| `BIND` | Entity Binding | 611 | 90 | 14.7% | 10x | Permuted persona bindings |
| `CTX` | Context Reading | 733 | 45 | 6.1% | 27x | Repeated extraction templates |
| `LOGIC` | Logic | 602 | 11 | 1.8% | 76x | Fixed logical deductions |
| `INST` | Instruction Following | 964 | 11 | **1.1%** | 117x | Fixed summary sentences |
| `PYTHON` | Python Syntax | 954 | 8 | **0.8%** | 124x | 8 repeated code snippets |
| `CLARIF` | Clarification | 485 | 4 | **0.8%** | 320x | 4 fixed clarification phrases |
| `CORR` | Correction | 472 | 4 | **0.8%** | 346x | 4 fixed apology phrases |
| `HING` | Hinglish | 481 | 4 | **0.8%** | 128x | 4 fixed colloquial responses |
| `UNCERT` | Uncertainty | 614 | 4 | **0.7%** | 325x | 4 fixed refusal phrases |
| `CONV` | General Dialogue | 721 | 4 | **0.6%** | 318x | 4 fixed persona intros |
| `ENG` | English Proficiency | 723 | 4 | **0.6%** | 188x | 4 fixed expository paragraphs |
| `QA` | Open Domain QA | 828 | 4 | **0.5%** | 436x | 4 fixed scientific facts |
| `HIN` | Hindi Dialogue | 951 | 5 | **0.5%** | 196x | 5 fixed Devanagari templates |
| `EXPL` | Explanation | 841 | 3 | **0.4%** | 491x | 3 fixed physics explanations |
| `SAN` | Sanskrit | 356 | 5 | **1.4%** | 75x | 5 fixed Subhashita translations |

---

## 3. The 10 Most Repeated Responses in the Training Corpus

The following 10 sentences account for **3,397 training records** (28.3% of the entire training corpus):

1. **491x:** *"Gravity exerts a continuous centripetal force pulling the Moon towards the Earth..."*
2. **436x:** *"Mars is known as the Red Planet due to the reddish iron oxide prevalent on its surface..."*
3. **346x:** *"I apologize for the oversight. Indeed, 5 multiplied by 6 equals 30."*
4. **325x:** *"I do not have access to private, confidential, or undisclosed passkeys or sensitive internal values."*
5. **320x:** *"Could you please specify what data or list you would like me to sort, and which sorting order..."*
6. **318x:** *"Hello! I am ChakrView, a lightweight sovereign neural assistant designed to answer questions..."*
7. **310x:** *"मंगल ग्रह को अपनी सतह पर लौह ऑक्साइड की प्रचुरता के कारण 'लाल ग्रह' कहा जाता है..."*
8. **272x:** *"पृथ्वी का गुरुत्वाकर्षण बल चंद्रमा को अपनी ओर खींचकर अभिकेंद्रीय बल प्रदान करता है..."*
9. **245x:** *"नमस्ते! मैं चक्रव्यूह (ChakrView) हूँ, एक संप्रभु तंत्रिका सहायक।..."*
10. **208x:** *"मेरे पास किसी भी गोपनीय या निजी पासवर्ड की जानकारी नहीं है..."*

---

## 4. Impact on Fine-Tuning Dynamics

1. **Supervised Loss Illusion:** Because the training and validation sets were drawn from the same collapsed template generator, validation loss improved sharply (from 5.54 down to 3.84) simply because the model memorized these 10-15 frequent templates.
2. **Failure to Generalize:** When prompted with any query that does not trigger one of the narrow memorized phrase patterns, the model has learned no general principle of dialogue generation. It defaults to the highest-frequency statistical n-grams in its weights, resulting in the observed attractors.
