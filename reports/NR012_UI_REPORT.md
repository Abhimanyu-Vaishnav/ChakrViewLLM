# CHAKRVIEW RESEARCH REPORT: NR-012 LOCAL WEB UI REPORT
**Mission ID:** CV-GENESIS-NR-012  
**Date:** 2026-11-25  
**Component:** `chakrview/ui/app.py`  

---

## 1. Overview & UI Design
The ChakrView Local Web Console is implemented with zero external heavy framework dependencies, utilizing Python's robust built-in `http.server`. It delivers an interactive, dark-themed responsive console for local neural conversation.

---

## 2. Key Interface Features
1. **Dual-Path Execution Switch:**
   - Interactive toggle between **Hybrid Cognitive** mode (memory + math tools) and **Pure Neural Core** mode.
2. **Cognitive Memory Inspector:**
   - Side panel displays all asserted facts currently held in episodic working memory.
3. **Session Management:**
   - One-click buttons to reset working memory and refresh state.
4. **Transparency & Performance Tags:**
   - Every assistant response displays execution origin (`pure_neural`, `episodic_memory`, `deterministic_math`, `principled_abstention`), generation latency in milliseconds, and token throughput speed.

---

## 3. Endpoints Implemented
- `GET /` — Serves responsive single-page web console.
- `GET /api/health` — Reports model status, active device, and checkpoint path.
- `GET /api/memory` — Returns current working memory facts and conversation history.
- `POST /api/mode` — Toggles active execution mode (`pure_neural` vs `hybrid`).
- `POST /api/memory/clear` — Erases episodic working memory facts.
- `POST /api/chat` — Executes dialogue turn and returns structured JSON response.
