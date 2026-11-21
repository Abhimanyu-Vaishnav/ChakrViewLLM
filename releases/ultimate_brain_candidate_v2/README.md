# ChakrView Ultimate Brain Candidate v2.0
## Repaired Real-World Release Candidate

### Overview
ChakrView Ultimate Brain v2.0 resolves the conversational, memory, and cognitive failures identified during real-world user testing in `ULTIMATE-BRAIN-002`.

### Core Fixes Included
1. **Conversational Intent Router**: Handles greetings, inquiries, and chit-chat in English, Hindi, and Hinglish. Eliminates gibberish generation loops for natural human messages.
2. **Natural Language Arithmetic**: Parses multi-container word problems (e.g. boxes/crates/bags/apples/books) and multi-step subtractions without triggering false entity abstentions.
3. **Transparent Persistent Memory**: Enables cross-session fact storage, entity relationship mapping, and robust retrieval (e.g., retrieving project name, language, or missing attributes).
4. **Principled Abstention**: Correctly abstains on private unrecorded personal history while eliminating false-positive abstentions on general capitalized words.
5. **Dual-Mode Architecture**: Supports transparent attribution between pure neural execution (`mode="pure_neural"`) and full cognitive hybrid execution (`mode="hybrid"`).

### How to Run

1. Launch Web Application (Default port 7860 to avoid common background scanner probes):
```powershell
.\.venv\Scripts\python.exe releases\ultimate_brain_candidate_v2\web_app.py
```
Open your browser at `http://127.0.0.1:7860`.

2. Interactive CLI Chat:
```powershell
.\.venv\Scripts\python.exe -m chakrview.interface.ultimate.chat
```

3. Run Dual-Mode Independent Evaluation:
```powershell
.\.venv\Scripts\python.exe experiments\ultimate_brain_002\eval\run_dual_mode_eval.py
```
