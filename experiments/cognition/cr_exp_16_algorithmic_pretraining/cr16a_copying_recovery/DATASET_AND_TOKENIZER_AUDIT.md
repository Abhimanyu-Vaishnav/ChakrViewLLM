# Dataset and Tokenizer Audit

## 1. Tokenizer Specifications & Integrity
- Architecture: Custom BPE Tokenizer (`BPETokenizer`)
- Artifact Path: `data/experiments/vocab_4096/`
- Vocabulary Size: $4,096$ tokens
- Special Tokens:
  - BOS ID: `0`
  - EOS ID: `1`
  - PAD ID: `2`
- Byte Encoding: UTF-8 with fallback replacement.

## 2. Tokenization Discrepancies on Entity Probes
A critical vulnerability in algorithmic sequence copying is subword token fragmentation:
1. **Single Words vs Multi-Piece Names:**
   - Familiar common words (`banana`, `cat`, `stone`) tokenize into single tokens.
   - Unseen entity names tokenize into fragmented character n-grams:
     - `Dev` $\to$ `[1302]` (1 token)
     - `Sonu` $\to$ `[86, 279, 120]` (`'S'`, `'on'`, `'u'`) (3 tokens)
     - `Gautam` $\to$ `[74, 4011, 1314]` (`'G'`, `'aut'`, `'am'`) (3 tokens)
     - `Sarvesh` $\to$ `[86, 280, 121, 363, 107]` (`'S'`, `'ar'`, `'v'`, `'es'`, `'h'`) (5 tokens)
     - `Yashwant` $\to$ `[92, 427, 107, 122, 1807]` (`'Y'`, `'as'`, `'h'`, `'w'`, `'ant'`) (5 tokens)

2. **Prefix Whitespace Sensitivity:**
   - In BPE tokenizers, a leading space creates completely different token IDs:
     - `' Alok'` $\to$ `[1606, 317, 110]` (`' A'`, `'lo'`, `'k'`)
     - `'Alok'` $\to$ `[68, 317, 110]` (`'A'`, `'lo'`, `'k'`)
   - If the prompt ends with `'Name: '` (trailing space) vs `'Name:'`, the first generated target token must NOT have a leading space, matching the exact bare string.

3. **Evaluation Protocol Alignment:**
   - Training targets were formatted without extraneous leading whitespace after the delimiter:
     `'Repeat the name: Dev\nName: '` $\implies$ `'Dev\n'`
   - This ensures that prompt and target token sequences align precisely across the attention boundary.

