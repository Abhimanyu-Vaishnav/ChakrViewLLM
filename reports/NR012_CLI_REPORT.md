# CHAKRVIEW RESEARCH REPORT: NR-012 INTERACTIVE CLI REPORT
**Mission ID:** CV-GENESIS-NR-012  
**Date:** 2026-11-25  
**Component:** `chakrview/chat_cli.py`  

---

## 1. Overview
`chakrview/chat_cli.py` provides an interactive terminal REPL for local multi-turn interaction with ChakrGenesis-17M.

---

## 2. Interactive Commands
- `/mode <neural|hybrid>`: Switches active execution mode in real time.
- `/memory`: Dumps current facts stored in working memory.
- `/clear`: Wipes working memory and clears history turns.
- `/export <filename.json>`: Exports complete dialogue session with full provenance metadata.
- `/quit` or `exit`: Exits application.

---

## 3. CLI Demonstration
```
============================================================
  CHAKRVIEW INTERACTIVE CONVERSATIONAL RUNTIME
  Mission: CV-GENESIS-NR-012 | Model: ChakrGenesis-17M
============================================================
Loading checkpoint: experiments/genesis_stage_d_011/checkpoints/genesis_17m_staged_best.pt
Model loaded successfully on device: CUDA
Session started: 63c9165d | Active Mode: [HYBRID]

User> My name is Abhimanyu and I live in Jaipur.
Assistant [HYBRID]> Aapke dost ya network...
  [Origin: neural_core | Latency: 0.521s | Speed: 69.0 tok/s]

User> What is my name?
Assistant [HYBRID]> Your name is Abhimanyu.
  [Origin: episodic_memory | Latency: 0.0001s]

User> Calculate 56 + 44
Assistant [HYBRID]> The result of 56 + 44 is 100.
  [Origin: deterministic_math | Latency: 0.0001s]
```
