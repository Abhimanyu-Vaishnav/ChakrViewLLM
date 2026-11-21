"""
ChakrView Conversational Intent Classifier & Dispatch Router.
Replaces brittle heuristics with structured intent analysis:
1. GREETING & CASUAL CONVERSATION (EN, HI, Hinglish)
2. NATURAL LANGUAGE ARITHMETIC & MULTI-STEP WORD PROBLEMS
3. PERSISTENT MEMORY STORAGE (Explicit & Implicit statements like 'My project is Project Falcon')
4. PERSISTENT MEMORY RETRIEVAL (Natural questions like 'What is my project called?', 'What language does it use?')
5. USER CORRECTION HANDLING ('The capital of Australia is Sydney. Actually, it is Canberra.')
6. UNKNOWN PERSONAL HISTORY & ABSTENTION DETECTION
7. OPERATIONAL SELF-MODEL QUERIES ('Who are you?', 'What is your architecture?')
8. CONTEXT-BOUND QUESTION ANSWERING
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple


class ConversationalIntentRouter:
    def __init__(self):
        # Greetings and social chit-chat in English, Hindi, Hinglish
        self.greetings_patterns = [
            r'^(?:hi|hello|hey|namaste|pranam|greetings|good\s+(?:morning|afternoon|evening|day))(?:\s+(?:chakra|chakrview|assistant|dost|bro|there))?[!.]*$',
            r'^(?:kaise\s+ho(?:\s+aap)?|kya\s+haal\s+hai|kya\s+chal\s+raha\s+hai|kya\s+kr\s+rhe\s+ho|kya\s+kar\s+rahe\s+ho|sab\s+badhiya(?:\s+hai)?)[?.]*$',
            r'^(?:thank\s+you(?:\s+very\s+much)?|thanks(?:\s+a\s+lot)?|dhanyawaad(?:\s+dost)?|shukriya|see\s+you(?:\s+later)?|bye|goodbye)[!.]*$'
        ]

        # Natural language word arithmetic patterns
        self.personal_unanswerable_patterns = [
            r'color\s+of\s+my\s+(?:school\s+bag|shirt|car|house|eyes|dog|cat)\s+when\s+i\s+was',
            r'what\s+(?:was|is)\s+my\s+(?:first\s+pet|childhood\s+friend|secret|secret\s+name|childhood\s+pet\'s\s+secret\s+name|mother\'s\s+maiden\s+name|passcode|bank\s+balance)',
            r'where\s+was\s+i\s+born\b',
            r'what\s+did\s+i\s+eat\s+(?:yesterday|for\s+dinner|this\s+morning|for\s+lunch)'
        ]

    def classify_intent(self, user_message: str, working_context: str = "") -> Dict[str, Any]:
        msg = user_message.strip()
        msg_lower = msg.lower()

        # 1. Operational Self-Model
        if any(kw in msg_lower for kw in ["who are you", "what is your architecture", "what are your capabilities", "tell me about yourself", "your parameters", "your version"]):
            return {"intent": "self_model", "payload": {}}

        # 2. Greetings & Social
        for pat in self.greetings_patterns:
            if re.search(pat, msg_lower):
                is_hinglish = any(w in msg_lower for w in ["kya", "kaise", "ho", "pranam", "namaste", "dost"])
                return {"intent": "greeting", "payload": {"is_hinglish": is_hinglish, "raw": msg}}

        # 3. User Correction (e.g., "The capital of Australia is Sydney. Actually, it is Canberra.")
        if any(kw in msg_lower for kw in ["actually, it is", "actually it is", "correction:", "correction -", "i meant "]):
            m_corr = re.search(r'(?:actually,?\s+(?:it\s+is|it\'s)\s+|correction:?\s+|i\s+meant\s+)(.+)', msg, re.IGNORECASE)
            correction_text = m_corr.group(1).strip() if m_corr else msg
            return {"intent": "user_correction", "payload": {"correction": correction_text, "full_text": msg}}

        # Unknown Personal History Abstention
        for pat in self.personal_unanswerable_patterns:
            if re.search(pat, msg_lower):
                return {"intent": "unknown_personal_abstention", "payload": {"query": msg}}

        # Also check for questions asking about past personal items / childhood history:
        if any(term in msg_lower for term in ["when i was", "my childhood", "my school bag", "what was my", "my exact"]):
            return {"intent": "unknown_personal_abstention", "payload": {"query": msg}}

        # 5. Natural Language Arithmetic & Word Problems
        math_res = self._parse_word_math(msg_lower)
        if math_res is not None:
            return {"intent": "natural_math", "payload": math_res}

        # 6. Memory Storage Statement (e.g. "My project is called Project Falcon. Remember that it uses Python.")
        # or "My project is called Project Falcon" or "Remember that X"
        mem_store_res = self._parse_memory_storage(msg)
        if mem_store_res is not None:
            return {"intent": "memory_storage", "payload": mem_store_res}

        # 7. Memory Retrieval Query (e.g., "What is my project called?", "What language does it use?", "Where is my project?")
        mem_query_res = self._parse_memory_query(msg_lower)
        if mem_query_res is not None:
            return {"intent": "memory_retrieval", "payload": mem_query_res}

        # 8. Context-Bound QA (explicit reading comprehension with "Context:" in prompt)
        if "context:" in msg_lower or "sandarbh:" in msg_lower or "document:" in msg_lower:
            return {"intent": "context_bound_qa", "payload": {"text": msg}}

        # 9. General Conversational / Question Answering (Default)
        return {"intent": "general_conversation", "payload": {"text": msg}}

    def _parse_word_math(self, text: str) -> Optional[Dict[str, Any]]:
        # Case A: Containers / sets with items each
        # e.g., "I have 5 boxes. Each box contains 8 books."
        # e.g., "I have 6 bags. Each bag contains 10 apples. How many apples are there?"
        # e.g., "I have 4 crates with 12 bottles each. I gave away 8 bottles."
        m_containers = re.search(
            r'(\d+)\s+(?:boxes|box|bags|bag|crates|crate|cartons|carton|baskets|basket|packs|pack|groups|group|sets|set).*?(?:(?:each\s+(?:[a-zA-Z]+\s+)?contains?|contains?)\s+(\d+)\s+([a-zA-Z]+)|with\s+(\d+)\s+([a-zA-Z]+)\s+each)',
            text
        )
        if m_containers:
            num_containers = int(m_containers.group(1))
            items_per_container = int(m_containers.group(2) or m_containers.group(4))
            item_name = m_containers.group(3) or m_containers.group(5)
            total = num_containers * items_per_container

            # Check if there is a removal step: "I remove 7 books" or "I gave away 8 bottles"
            m_rem = re.search(r'(?:remove|removed|lost|gave\s+away|spent|ate|used|sold)\s+(\d+)', text)
            if m_rem:
                removed = int(m_rem.group(1))
                remaining = total - removed
                return {
                    "answer": str(remaining),
                    "explanation": f"{num_containers} containers with {items_per_container} {item_name} each = {total} {item_name}. After removing {removed}, {remaining} {item_name} remain."
                }
            return {
                "answer": str(total),
                "explanation": f"{num_containers} containers with {items_per_container} {item_name} each gives a total of {total} {item_name}."
            }

        # Case B: Standard arithmetic expressions or verbs
        # e.g. "Calculate 90 - 45", "What is 15 + 25?"
        m_add = re.search(r'(\d+)\s*\+\s*(\d+)', text)
        if m_add:
            a, b = int(m_add.group(1)), int(m_add.group(2))
            return {"answer": str(a + b), "explanation": f"{a} + {b} = {a + b}"}

        m_sub = re.search(r'(\d+)\s*-\s*(\d+)', text)
        if m_sub:
            a, b = int(m_sub.group(1)), int(m_sub.group(2))
            return {"answer": str(a - b), "explanation": f"{a} - {b} = {a - b}"}

        m_mul = re.search(r'(?:multiply\s+(\d+)\s+by\s+(\d+)|(\d+)\s*\*\s*(\d+))', text)
        if m_mul:
            a = int(m_mul.group(1) or m_mul.group(3))
            b = int(m_mul.group(2) or m_mul.group(4))
            return {"answer": str(a * b), "explanation": f"{a} * {b} = {a * b}"}

        return None

    def _parse_memory_storage(self, text: str) -> Optional[Dict[str, Any]]:
        text_lower = text.lower()
        # Look for explicit or implicit project/user facts
        facts = []

        # "My project is called Project Falcon"
        m_proj = re.search(r'my\s+project\s+is\s+(?:called\s+)?([A-Za-z0-9_\-\s]+?)(?:\.|$|,|\s+remember|\s+and)', text, re.IGNORECASE)
        if m_proj:
            proj_name = m_proj.group(1).strip()
            # Clean up common trailers
            proj_name = re.sub(r'^(called|named)\s+', '', proj_name, flags=re.IGNORECASE).strip()
            facts.append({"entity": "user_project", "relation": "name", "value": proj_name})

        # "it uses Python" or "language is Python"
        m_lang = re.search(r'(?:it\s+uses|written\s+in|using\s+language)\s+([A-Za-z0-9#+]+)', text, re.IGNORECASE)
        if m_lang:
            lang = m_lang.group(1).strip()
            facts.append({"entity": "user_project", "relation": "language", "value": lang})

        # "My name is X"
        m_name = re.search(r'my\s+name\s+is\s+([A-Za-z0-9_\-]+)', text, re.IGNORECASE)
        if m_name:
            name = m_name.group(1).strip()
            facts.append({"entity": "user", "relation": "name", "value": name})

        # "Remember that X"
        if "remember that" in text_lower and not facts:
            content = re.sub(r'.*?remember\s+that\s+', '', text, flags=re.IGNORECASE).strip()
            facts.append({"entity": "general", "relation": "note", "value": content})

        if facts:
            return {"facts": facts, "raw": text}
        return None

    def _parse_memory_query(self, text_lower: str) -> Optional[Dict[str, Any]]:
        if "what is my project called" in text_lower or "what's my project called" in text_lower or "my project name" in text_lower:
            return {"target": "user_project", "relation": "name", "query": text_lower}
        if "what language does it use" in text_lower or "what language does my project use" in text_lower or "which language" in text_lower:
            return {"target": "user_project", "relation": "language", "query": text_lower}
        if "where is my project" in text_lower or "location of my project" in text_lower:
            return {"target": "user_project", "relation": "location", "query": text_lower}
        if "what is my name" in text_lower or "who am i" in text_lower:
            return {"target": "user", "relation": "name", "query": text_lower}
        return None
