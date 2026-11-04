# CBR-SFT-03A — GENERATION DRIFT & ATTRACTOR COLLAPSE ANALYSIS

---

## 1. Divergence Taxonomy & Attractor Basins

In free autoregressive generation, once Position 0 emits an off-target token, the model enters one of four distinct pretraining attractor basins:

| Attractor Basin | Trigger Tokens | Characteristic Output Sequence | Root Pretraining Domain |
| :--- | :--- | :--- | :--- |
| **Wiki Article Header Basin** | `Token 68 ('A')` | `"Androne = [http://www.google.com/books..."` or `"Andromed (b. 1964)..."` | Wikipedia biographical and geographic listings |
| **Definitional Phrase Loop** | `Token 898 ('The ')` | `"The first of the word of the word of the world..."` | Pretraining linguistic definitions |
| **Demographic Classification** | `Token 70 ('C')` | `"Chinese is a list of the world."` / `"California (b. 1994)..."` | Wikipedia census and regional tables |
| **Wiki Table Markup Basin** | `Token 13 ('\n')` | `"\n| scope=&quot;row&quot; | \n| style=..."` | HTML/MediaWiki table syntax |

---

## 2. EOS Behavior vs. Horizon Exhaustion

- **Turn Termination Rate:**
  In CBR-SFT-02, **91.2%** of generated outputs terminated with `<|eos|>` before reaching the 32 max-token cutoff.
- **The Paradox:** The model has successfully learned *when* to end a turn (`<|eos|>` supervision was effective), but *what* it generates during that turn is dominated by pretraining attractors rather than dialogue logic.
- **Conclusion:** EOS supervision succeeded, but prompt-conditioned token routing failed.
