# CR-EXP-18: Phase H — Tokenization & Position Investigation Report

## 1. Subword Tokenization Granularity
Inspection of the 4,096-vocabulary SentencePiece BPE tokenizer on test entities:

| Entity Text | Subword Token Count | Subword Token Pieces | Category |
| :--- | :---: | :--- | :--- |
| `Alice` | 2 | `['Al', 'ice']` | Familiar English Name |
| `Bob` | 2 | `['B', 'ob']` | Familiar English Name |
| `Zaraph` | 3 | `['Z', 'ara', 'ph']` | Unseen Fictional Name |
| `Kaelen` | 4 | `['K', 'a', 'el', 'en']` | Unseen Fictional Name |
| `Selene` | 3 | `['S', 'el', 'ene']` | Unseen Fictional Name |
| `Pranav` | 4 | `['P', 'ran', 'a', 'v']` | Training Name (Indian) |
| `Dev` | 2 | `['D', 'ev']` | Training Name (Indian) |
| `चक्रव्यूह` | 5 | `['च', 'क्र', 'व्य', 'ू', 'ह']` | Hindi Devanagari |
| `नमस्ते` | 4 | `['न', 'म', 'स्', 'ते']` | Hindi Devanagari |
| `London` | 2 | `['Lond', 'on']` | Familiar City |
| `Zephyria` | 5 | `['Z', 'e', 'phy', 'ri', 'a']` | Unseen Fictional City |
| `Laptop-1` | 4 | `['L', 'apt', 'op-', '1']` | Alphanumeric Device |

## 2. Tokenization and Retrieval Correlation
1. **Multi-Token Name Drift:** While single-token words (like `" Au"` or `" 45"`) can occasionally be retrieved when placed adjacent to fixed triggers in pretraining, **zero names in natural conversations are single-token in the 4,096 vocabulary**. Every name spans 2 to 5 subword tokens.
2. **Sequential Dependency Requirement:** Generating an unseen multi-token entity requires the autoregressive model to attend to token 1, emit piece 1, and then successfully condition on both piece 1 and the earlier prompt pieces to complete piece 2 and 3 without drifting into familiar words.
3. **Capacity Breakdown:** Because the model has only 6 layers and 6 heads, attention across multi-turn conversational spans (where the fact was introduced 30 to 80 tokens earlier) fails to sustain the positional induction circuit, resulting in instantaneous drift upon generating the first character.
