# ChakrView Step 55: Multi-Task Generalization Curriculum Specification

- **Document Version**: 1.0.0
- **Status**: Ratified Curriculum Specification
- **Focus**: Unified Multi-Task Domain Balancing & Anti-Forgetting Replay

---

## 1. Objective & Architecture

Step 55 shifts ChakrMicro from isolated curriculum stages to an **interleaved, multi-task representation**. To prevent catastrophic forgetting while expanding capability, seven distinct curriculum domains are formally defined and combined with configurable sampling weights:

```
+------------------------------------------------------------------------------------+
|                       CHAKRMICRO UNIFIED MULTI-TASK DATASET                        |
+------------------------------------------------------------------------------------+
  |               |               |              |            |          |         |
  v               v               v              v            v          v         v
[FOUNDATION] [COMPUTATION] [PROGRAMMING] [REASONING] [DIAGNOSIS] [TRAJ] [INSTRUCTION]
   (15%)         (15%)           (20%)         (20%)       (10%)   (10%)     (10%)
```

---

## 2. Seven Core Curriculum Domains

### 1. Domain: FOUNDATION (Weight: 0.15)
- **Delimiters & Pairs**: Multi-bracket balancing (`()`, `[]`, `{}`, `<>`).
- **Structured Data**: JSON objects, YAML sequences, key-value mappings.
- **Sequence Continuity**: Factual sentence completions and natural language structure.

### 2. Domain: COMPUTATION (Weight: 0.15)
- **Arithmetic**: Single and multi-operand addition, subtraction, multiplication, integer division.
- **Relational Comparisons**: Strict inequalities, equality, boundary comparisons (`>`, `<`, `==`, `!=`, `>=`, `<=`).
- **Parity & Sign**: Odd/even determination, positive/negative/zero evaluation.
- **String Transforms**: Uppercase, lowercase, reverse, antonym pairs.

### 3. Domain: PROGRAMMING (Weight: 0.20)
- **Function Signatures**: Typed Python definitions (`def func(a: int, b: int) -> int:`).
- **Return Expressions**: Arithmetic return values, boolean checks, clamping utilities.
- **Conditionals & Control Flow**: `if/elif/else` blocks, list comprehensions.
- **Assertions & Unit Tests**: `assert func(...) == expected` constructs.

### 4. Domain: REASONING (Weight: 0.20)
- **Single-Step Deduction**: Cause and effect, direct implications.
- **Multi-Step Arithmetic**: Multi-step word problems with explicit intermediate calculations.
- **Selection from Alternatives**: Multiple-choice format with explicit elimination reasoning.
- **Deterministic Planning**: Ordered steps to reach target states.
- **State Machine Transitions**: Pre-condition -> Action -> Post-condition updates.

### 5. Domain: DIAGNOSIS (Weight: 0.10)
- **Syntax Error Localization**: Identifying missing colons, mismatched brackets, indentation errors.
- **Runtime Error Localization**: Off-by-one errors, wrong operators, zero division traps.
- **Corrective Code Selection**: Emitting targeted patches based on diagnostic observations.

### 6. Domain: TRAJECTORY (Weight: 0.10)
- Canonical XML-tagged action/observation sequences strictly bounded within 512 tokens:
  ```text
  <TRAJECTORY>
  <SPEC>...</SPEC>
  <STATE>...</STATE>
  <ACTION>...</ACTION>
  <OBSERVATION>...</OBSERVATION>
  <DIAGNOSIS>...</DIAGNOSIS>
  <NEXT_ACTION>...</NEXT_ACTION>
  <RESULT>...</RESULT>
  </TRAJECTORY>
  ```

### 7. Domain: INSTRUCTION (Weight: 0.10)
- **Exact Output Following**: Producing verbatim target strings when explicitly requested.
- **Constrained Generation**: Output formatted strictly as JSON, single word, or boolean.
- **Deterministic Prompt-Response Alignment**: Responding strictly to direct commands without extraneous prelude.

---

## 3. Anti-Forgetting Interleaved Replay Strategy

1. **Uniform Token Window Ceiling**: Every sample is capped to fit easily within the 512-token context window.
2. **Deterministic Domain Stratification**:
   - Instead of block-concatenating domains sequentially (which induces forgetting of earlier domains), samples from all seven domains are interleaved into each shard.
3. **Reproducible Shuffling**:
   - Shuffling uses a seeded PRNG (`random.Random(seed)`), ensuring bit-exact dataset reconstruction.
4. **Metadata Provenance**:
   - Each sample retains its source domain tag (`foundation`, `computation`, `programming`, `reasoning`, `diagnosis`, `trajectory`, `instruction`), allowing granular retention tracking.
