# Steps 74-77: Experiment Protocol

## Objective
Demonstrate the complete path from Knowledge Acquisition (Step 74) to Resource-Adaptive Strategy evaluation (Step 77) without violating existing invariants.

## Execution
Run the experiment script:
```bash
python scripts/experiment_step74_77_research_evaluation.py
```

## Expected Observations

1. **Hardware Detection**: The system successfully profiles the local machine, printing core count, RAM, and GPU status.
2. **Strategy Selection**: A `RuntimeStrategy` (e.g., `LOW_RESOURCE` or `STANDARD`) is mapped appropriately to the hardware, determining the `max_context_size` and parallelism rules.
3. **Investigation Creation**: An `InvestigationRequest` is instantiated requesting the capital of France.
4. **Mock Evidence Ingestion**: Three distinct sources (Wikipedia, travel blog, low-reliability forum) are processed.
5. **Verification**: 
   - The forum source is marked `STALE` and untrusted.
   - The Wikipedia and blog sources corroborate one another.
   - The overall status evaluates to `VERIFIED`.
6. **Cognitive Integration**: The output is successfully transformed into a neural-safe string and structured dictionary namespace (`external_evidence`), preserving epistemic boundaries.

## Invariant Checks
- Parameter count remains 3,443,136.
- Model hash remains `c5571c9c5cb7738625c885481ab2c026a00fa65dfb65e9761ef7eebb00a282da`.
- 10/10 dedicated tests pass.
- Full regression suite passes.
