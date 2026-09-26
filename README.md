# ChakrView (चक्रव्यूह)

ChakrView is an indigenous AI research project being developed strictly incrementally from first principles.

## Vision

ChakrView is envisioned as an indigenous, hardware-efficient AI brain designed to execute efficiently on modern systems as well as older, low-resource computers and embedded/edge devices.

The long-term research roadmap encompasses:
1. **Indigenous Neural Language Core**: Designed from foundational principles without relying on third-party pretrained weights.
2. **Efficient Inference Runtime**: Optimized execution tailored to diverse and constrained target hardware.
3. **Memory System**: Contextual, persistent, and structured memory representations.
4. **Skill-Learning System**: Dynamic acquisition and consolidation of operational skills.
5. **Tool/Action System**: Extensible interfaces for environmental interaction and tool use.
6. **Adaptive Computation**: Dynamic allocation of compute based on task complexity.
7. **Reinforcement-Learning Based Skill Optimization**: Fine-tuning cognitive pathways and skills through environmental feedback.
8. **Hardware-Aware Execution**: Dynamic adaptation to heterogeneous compute resources (CPU, low-end GPU, memory constraints).

## Current Status (Step 0)

> **NOTICE**: Currently at **Step 0 — Environment Setup**. No model architecture, neural weights, or pretrained components (e.g., Llama, Qwen, Mistral, Gemma, GPT) have been implemented, imported, or downloaded.

The repository currently provides the initial project skeleton, reproducible Python virtual environment, hardware diagnostics, and module directories.

## Project Structure

```text
ChakrView/
├── brain/          # Future neural core and representation logic
├── data/           # Data pipelines and partitions
│   ├── raw/
│   ├── processed/
│   └── validation/
├── training/       # Training loops and optimization routines
├── inference/      # Inference engine and decoding mechanisms
├── memory/         # Context and long-term memory systems
├── skills/         # Modular skill definitions and learned capabilities
├── tools/          # Tool and action integrations
├── runtime/        # Hardware execution abstractions
├── evaluation/     # Benchmarks and evaluation harnesses
├── configs/        # System and experiment configurations
├── tests/          # Unit and integration test suites
├── checkpoints/    # Model checkpoint storage (empty)
├── scripts/        # Utility and automation scripts
└── docs/           # Documentation and status tracking
```

## Environment Setup & Activation

### Prerequisites
- Python 3.14 (or compatible 64-bit Python 3.x)
- Git

### Activation

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
.\.venv\Scripts\activate.bat
```

### Dependency Installation
```bash
pip install -r requirements.txt
```
