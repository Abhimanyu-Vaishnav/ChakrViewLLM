# CR-EXP-16B Tokenization Analysis

## 1. Tokenizer Specification
- Model: Byte-Pair Encoding (`BPETokenizer`, 4,096 vocabulary size).
- Special Tokens: `[BOS: 0, EOS: 1, PAD: 2, '\n': 13, ' ': 35]`.
- Encoding Mode: Subword piece segmentation with UTF-8 byte fallback.

## 2. Token Breakdown of Representative Probes

| String | Bare Token IDs | Decoded Subword Pieces | Space-Prefixed IDs | Decoded Space Pieces |
| :--- | :--- | :--- | :--- | :--- |
| **Dev** | `[1302]` | `['Dev']` | `[35, 1302]` | `[' ', 'Dev']` |
| **Roy** | `[1118, 124]` | `['Ro', 'y']` | `[35, 1118, 124]` | `[' ', 'Ro', 'y']` |
| **Prem** | `[2269, 112]` | `['Pre', 'm']` | `[35, 2269, 112]` | `[' ', 'Pre', 'm']` |
| **Amit** | `[68, 112, 390]` | `['A', 'm', 'it']` | `[1606, 112, 390]` | `[' A', 'm', 'it']` |
| **Gautam** | `[74, 4011, 1314]` | `['G', 'aut', 'am']` | `[35, 74, 4011, 1314]` | `[' ', 'G', 'aut', 'am']` |
| **Mukesh** | `[80, 120, 110, 363, 107]` | `['M', 'u', 'k', 'es', 'h']` | `[35, 80, 120, 110, 363, 107]` | `[' ', 'M', 'u', 'k', 'es', 'h']` |
| **Siddharth** | `[86, 409, 103, 588, 310]` | `['S', 'id', 'd', 'har', 'th']` | `[3458, 409, 103, 588, 310]` | `[' S', 'id', 'd', 'har', 'th']` |
| **Pankhuri** | `[83, 286, 1138, 326, 108]` | `['P', 'an', 'kh', 'ur', 'i']` | `[35, 83, 286, 1138, 326, 108]` | `[' ', 'P', 'an', 'kh', 'ur', 'i']` |
| **John** | `[2543]` | `['John']` | `[35, 2543]` | `[' ', 'John']` |
| **Victoria** | `[1323, 108, 100]` | `['Vict', 'i', 'a']` | `[35, 1323, 108, 100]` | `[' ', 'Vict', 'i', 'a']` |

## 3. The Whitespace and Fragment Boundary Effect
In subword BPE, prepending a space frequently fuses into a completely different token:
- `'A'` is ID `68`, but `' A'` is ID `1606`.
- `'S'` is ID `86`, but `' S'` is ID `3458`.

When prompts end with a space delimiter (`'Repeat the name: Gautam\nName: '`), the prompt ends with token ID `324` (`': '`). The first target token is therefore bare (`'G'` = `74`), not space-prefixed. The model successfully matches this boundary token ($P=0.9455$). However, because the remainder of the word is split into low-frequency subword pieces (`4011`, `1314`), the model lacks sufficient sequence-level binding to suppress high-probability common words.

