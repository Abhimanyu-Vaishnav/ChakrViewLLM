# UPDATED v1.0 CODING MODEL ROADMAP
**Roadmap Updated:** 2026-11-07T22:57:30+05:30  
**Target Milestone:** ChakrView v1.0 Coding Specialized Intelligence  
**Foundation Dependency:** Working Memory & Variable Binding Layer (v0.9.1 / v0.9.2)  

---

## 1. STRATEGIC POSITIONING
Neural code generation requires exact variable scope resolution and mutable state tracking across AST blocks. Our findings in CR-EXP-12—specifically the discovery that recency gating and variable-binding can be trained up from $2.3\% \to 43.0\%$ without architecture changes—validate our staged strategy.

Specializing into coding before variable binding reaches $\ge 70\%$ would yield broken programs with undeclared or substituted identifiers.

---

## 2. REVISED STAGES TO v1.0
1. **Stage 1: Working Memory Consolidation ($\ge 70\%$ Multi-Turn State Tracking)** (Current Frontier).
2. **Stage 2: Python / C Syntax AST Foundation & Token Masking** (Milestone v0.9.5).
3. **Stage 3: Function Infilling & Docstring-to-Code Generation** (Milestone v0.9.8).
4. **Stage 4: Autonomous Coding Agent with Functional Execution Verification** (v1.0 Production Release).
