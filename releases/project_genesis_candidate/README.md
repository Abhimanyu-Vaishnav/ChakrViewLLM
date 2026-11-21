# ChakrView Project Genesis Candidate (v1.0)
## Independent Foundation Neural Model & Autonomous Cognitive Architecture

### Overview
Project Genesis delivers the first independent neural model in the ChakrView project trained with autoregressive language pretraining over multi-million tokens, coupled with conversational instruction fine-tuning and a full cognitive architecture.

### Key Capabilities
1. **ChakrGenesis Foundation Neural Model (`genesis_sft_best.pt`):**
   - 5,177,600 parameters.
   - Pre-trained on 6.65M tokens of the multi-domain Stage-C corpus using CUDA hardware acceleration.
   - Fine-tuned on 1,476 structured dialogue and reasoning instructions.
   - Autonomously generates coherent English dialogue completions (`"Hello! I am doing well and ready to assist you. How can I help you today?"`).
2. **Autonomous Cognitive Architecture (System B):**
   - **Intent Dispatch Router:** Intelligently identifies greetings, natural language arithmetic, memory storage, retrieval, and principled abstentions.
   - **Persistent Relational Memory:** Transparent storage and cross-session retrieval of user facts and system state.
   - **Multi-Container Arithmetic:** Solves multi-container and multi-step word problems deterministically.
   - **Principled Abstention:** Abstains on unrecorded private personal history without false-positive entity blocks.

### Quick Start

1. **Launch Web Application:**
```powershell
.\.venv_gpu\Scripts\python.exe releases\project_genesis_candidate\web_app.py
```
Open your browser at `http://127.0.0.1:7860`.

2. **Run Acceptance Evaluation:**
```powershell
.\.venv_gpu\Scripts\python.exe experiments\project_genesis\run_genesis_evaluation.py
```
