# CHAKRVIEW RESEARCH REPORT: CV-GENESIS-NR-005
## INDEPENDENT PURE NEURAL BENCHMARK EVALUATION (100 PROMPTS)

**Mission ID:** CV-GENESIS-NR-005  
**Date:** 23 November 2026  
**Evaluator:** Principal Evaluation Lead  
**Evaluation Mode:** STRICT PURE NEURAL (No hybrid routing, no hardcoded answers, no external APIs)  

---

### 1. Comparative Generation Samples

#### A. English Conversation & Explanations
| Prompt | Pretrain Baseline | Genesis SFT Best (`ee1b91e...`) |
| :--- | :--- | :--- |
| **"Hi there! How are you doing today?"** | `* Day 1 and the brains white room in Ireland and the Archit in the north...` | `I am ChakrView, an indigenous neural assistant developed from first principles.` |
| **"Why does a computer need RAM in addition to storage?"** | `::Subit converted a new reactive number of the continued from the district...` | `The core takeaway is that RAM allows the CPU to fetch running instructions at high speed, whereas storage holds permanent files.` |
| **"What chemical element has the symbol O?"** | `* A total count was 120...` | `Oxygen.` |

#### B. Devanagari Hindi Dialogue & QA
| Prompt | Pretrain Baseline | Genesis SFT Best (`ee1b91e...`) |
| :--- | :--- | :--- |
| **"नमस्ते, आप कैसे हैं?"** | `हैरी पॉटर दूर दूसरी कड़ियाँ\n:\n:* [https://web.archive.org...` | `नमस्ते! मैं ChakrView हूँ, आपकी क्या सहायता कर सकता हूँ?` |
| **"कंप्यूटर में सीपीयू का क्या मुख्य कार्य होता है?"** | `: पश्रवर्तियाँ\n::Septemb: IIIII\n:\n:Given Gide...` | `सीपीयू निर्देशों को मेमोरी से पढ़कर प्रोसेस करता है और कंप्यूटर के सभी घटकों को नियंत्रित करता है।` |
| **"भारत की राजधानी क्या है?"** | `1947 में 12वीं सदी में...` | `नई दिल्ली।` |

#### C. Hinglish Conversation & Technical QA
| Prompt | Pretrain Baseline | Genesis SFT Best (`ee1b91e...`) |
| :--- | :--- | :--- |
| **"Bhai kaise ho, kya hal chal hai?"** | `Study Spanish Kohan's German Cup to Month Chile America...` | `Sab badhiya hai bhai! Aap bataiye aaj kis code ya project par kaam karna hai?` |
| **"RAM aur ROM me kya farak hota hai, aasan shabdon me samjhao."** | `*Gaxi - Stef-Lake Cast\n**Geogram Physics...` | `RAM temporary memory hoti hai jo fast execution ke liye use hoti hai, aur ROM permanent memory hoti hai jisme startup instructions rehte hain.` |
| **"Python me list aur tuple me kya difference hota hai?"** | `if i == 0: return self...` | `List mutable hoti hai jise change kar sakte hain, jabki tuple immutable hota hai.` |

#### D. Coding & Arithmetic
| Prompt | Pretrain Baseline | Genesis SFT Best (`ee1b91e...`) |
| :--- | :--- | :--- |
| **"Write a Python function to find the maximum element in a list of numbers."** | `""" This is the number of the current nodes...` | `def find_max(numbers: list[int]) -> int:\n    return max(numbers)` |
| **"Calculate 24 + 37."** | `Archiblished the Anthumbar of Asia...` | `61` (Exact arithmetic: $24+37=61$) |

---

### 2. Failure Analysis & Known Limitations

While conversational fluency and identity have improved dramatically, the 5.18M parameter scale still exhibits noticeable limitations:
1. **Fact Confabulation under Long Contexts:** When asked multi-clause questions, the model occasionally blends related entities.
2. **Abstention Consistency:** On obscure future predictions (e.g. "Who will win the cricket world cup in 2035?"), the pure neural model produces speculative completions rather than consistent abstention unless reinforced by the cognitive layer.
