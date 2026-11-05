# Recommended Next Experiment: CBR-SFT-08 (Smallest Safe Corrective Experiment)

## 1. Executive Summary

Based on the forensic audit of CBR-SFT-07 Step 50, the primary failure modes are:
1. **Context Sink Bias**: Over-representation of `"not provided in the context"` template phrases.
2. **Multi-Turn Representation Scarcity**: Only 10.1% multi-turn examples in training data.
3. **Greedy Repetition Loop Vulnerability**: High logits on self-transitions (`neural neural...`).

To systematically solve these issues without architecture bloat, we propose **CBR-SFT-08: Balanced Induction & Multi-Turn Alignment**.

---

## 2. Minimal Safe Experiment Specification

### A. Parent Checkpoint
- **Parent**: `experiments/cbr01_sft/runs/cbr_sft_04/checkpoints/sft_checkpoint_step_200.pt` (or `cbr_sft_07/sft_checkpoint_step_50.pt` if fine-tuning continuation).
- **Recommendation**: Resume from **CBR-SFT-04 Step 200** (SHA-256: `327118ce...`), as it possesses the most stable language base before the severe context-attractor collapse occurred.

### B. Dataset Design: `cbr_sft_08`
Total size: **300 high-quality, balanced pairs**
1. **Multi-Turn Grounding & Memory (35%, ~105 records)**:
   - 2-turn and 3-turn interactions where the user states an entity (e.g. name, tool, city, project) and asks for it in the subsequent turn.
   - Format: `User: My name is X.\nAssistant: Hello X!\nUser: What is my name?\nAssistant: Your name is X.`
2. **Context-to-Output Entity Copying (20%, ~60 records)**:
   - Explicit single-turn extraction without the word `"context"`:
     - `User: The service port is 8080. What port is the service using?\nAssistant: The service is using port 8080.`
3. **Natural Conversational Greetings & Social (15%, ~45 records)**:
   - Diverse greetings that DO NOT collapse into `"ready to assist"` or `"context"`.
   - Variations: Casual greetings, polite closures, acknowledgments.
4. **Natural Abstention Without Cliché Sinks (15%, ~45 records)**:
   - Replace `"That information is not provided in the context"` with varied natural responses:
     - `"I do not have access to real-time information."`
     - `"I don't know that detail."`
     - `"I cannot answer without more details."`
5. **Core Technical Explanations & Bilingual/Hinglish (15%, ~45 records)**:
   - Clear, concise explanations of transformers, attention, and modularity in English and Hinglish.

### C. Training Hyperparameters
- **Steps**: 60 steps
- **Batch Size**: 4 (or 2 with `grad_accum_steps=2` for effective batch size 4)
- **Learning Rate**: $3 \times 10^{-5}$ with Cosine decay to $3 \times 10^{-6}$ (slightly lower than SFT-07 to prevent disruption of pretrained language representations)
- **Loss Masking**: Prompt masked (`mask_prompt=True`, ignore index `-100`)
- **Weight Decay**: 0.01
- **Grad Clip**: 1.0

### D. Inference Configuration
- **Temperature**: 0.0 (or 0.2 for conversational flow)
- **Repetition Penalty**: 1.15 (reduced from 1.25 to prevent grammatical disruption while suppressing token loops)
- **No Repeat N-Gram**: 3

---

## 3. Explicit Acceptance Gates

Before declaring CBR-SFT-08 successful:
1. **Identity Gate**: Correctly outputs `"ChakrView"` for `"What is your name?"` and `"Who are you?"` without repetition loops.
2. **Multi-Turn Recall Gate**: When told `"My name is Abhimanyu."` in turn 1, answering `"What is my name?"` in turn 2 must contain `"Abhimanyu"` (>= 80% accuracy across 10 evaluation trials).
3. **Context Grounding Gate**: Accurately reproduces the exact entity given in single-turn context queries without appending `"provided in the context"`.
4. **Zero UNK / Degeneracy Gate**: Generates coherent EOS-terminated sentences without infinite loops under greedy decoding.

