# CR-EXP-01: VARIABLE BINDING AND IN-CONTEXT LEARNING EXPERIMENT DESIGN (REVISED)

## 1. Scientific Weakness Audit & Design Corrections

An independent scientific audit of the preliminary CR-EXP-01 proposal revealed critical vulnerabilities that would have confounded evaluation:

1. **Weakness 1: Accidental Tokenizer Shortcuts**:
   - *Problem*: In BPE tokenizers, certain common keys (e.g. `Alpha`, `Port`) or digits have single-token representations, while novel entities split into 2–4 subwords. A model might achieve high accuracy simply by exploiting token length cues or leading space boundaries rather than semantic binding.
   - *Correction*: All keys and values in the revised dataset are verified for subword parity. The evaluation suite normalizes token segmentation across familiar and novel pools, testing both single-token and multi-token entities.
2. **Weakness 2: Template Memorization**:
   - *Problem*: If all prompts use `Assignment: Key = Value. Query: Key? Answer:`, the model might memorize the exact distance between the colon and the value token.
   - *Correction*: Introduced **10 syntactically distinct prompt templates** (e.g., `State: Key has Value`, `Record: Value was assigned to Key`, `Attribute Value belongs to Key`), forcing attention heads to extract meaning across variable syntax.
3. **Weakness 3: Accidental Positional Patterns in Distractors**:
   - *Problem*: If target keys consistently appear at the beginning or end of distractor lists, positional bias masquerades as retrieval capability.
   - *Correction*: Target position is systematically counterbalanced across the prompt sequence ($P \in \{0.1, 0.3, 0.5, 0.7, 0.9\}$ relative depth).
4. **Weakness 4: Ambiguity in Variable Reassignment**:
   - *Problem*: Prompts like `X was 5 then X was 8` can trigger ambiguous recency vs primacy interpretations.
   - *Correction*: Formal temporal ordering markers are explicitly enforced: `Initial state: X = 5. Later update: X was modified to 8. Current state: What is the current value of X?`.
5. **Weakness 5: Over-claiming on Output Accuracy**:
   - *Problem*: High accuracy on a single test seed does not prove an induction circuit formed.
   - *Correction*: Output accuracy is separated from mechanistic circuit analysis. Generalization requires validation across 3 independent random seeds.

---

## 2. Experimental Conditions (The 10-Condition Battery)

The revised CR-EXP-01 tests the candidate model across 10 granular conditions:

| Condition | Objective | Description | Success Target |
|---|---|---|---|
| **Cond A: Memorized Associations** | Baseline Learning Check | Train pairs tested under train templates. | $EM \ge 90\%$ |
| **Cond B: Novel In-Context Associations** | Out-of-Distribution Binding | Unseen keys and values from held-out lexicon pool. | $EM \ge 60\%$ |
| **Cond C: Compositional Combinations** | Combinatorial Generalization | Familiar keys paired with novel values in novel arrangements. | $EM \ge 65\%$ |
| **Cond D: Distractor Resistance Sweep** | Noise Immunity | Tested with $k \in \{1, 2, 3, 4\}$ distractors in context. | $\ge 50\%$ at $k=3$ |
| **Cond E: Position Sensitivity** | Positional Invariance | Target key placed at Beginning, Middle, or End. | $\Delta Acc < 15\%$ across positions |
| **Cond F: Variable Reassignment** | Temporal State Tracking | Prior value updated to new value; model must emit current. | $EM \ge 55\%$ |
| **Cond G: Multi-Turn Dialogue Recall** | Multi-Turn Memory | Entity stated in Turn 1, queried in Turn 2. | $EM \ge 70\%$ |
| **Cond H: Held-Out Lexicon Generalization** | Zero-Leakage Lexicon | Tested on completely disjoint vocabulary partitions. | $EM \ge 60\%$ |
| **Cond I: Unseen Prompt Templates** | Syntactic Robustness | Tested on 5 prompt formulations never present in training. | $EM \ge 55\%$ |
| **Cond J: Multi-Seed Stability** | Reproducibility | Evaluated across Seeds 42, 101, 202. | Variance $\sigma^2 < 0.05$ |

---

## 3. Baseline & Candidate Identity

- **Parent Baseline**: `experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt` (SHA-256: `327118ce...`).
- **Candidate Under Test**: Isolated checkpoint saved to `experiments/cognition/cr_exp_01/checkpoints/candidate_step_150.pt`.
- **Inference Parity**: Both parent and candidate are evaluated using identical decoding parameters (`temperature=0.0`, `repetition_penalty=1.0`, greedy argmax) on identical frozen evaluation files.
