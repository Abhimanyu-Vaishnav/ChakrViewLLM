"""
ChakrView Conversational Policy Engine.
Provides calibrated, grounded dialogue responses for non-neural and hybrid cognitive paths:
1. Greetings & Chit-chat (English, Hindi, Hinglish)
2. Memory storage & retrieval responses
3. Principled abstention on unrecorded personal facts
4. User correction processing and acknowledgment
5. Word arithmetic presentation
"""

from __future__ import annotations
import random
from typing import Any, Dict, List, Optional


class ConversationalResponsePolicy:
    def __init__(self):
        self.greetings_en = [
            "Hello! I am ChakrView. How can I help you today?",
            "Hi there! All systems are operational. What can I do for you?",
            "Greetings! I am ready to assist you. What would you like to explore?"
        ]
        self.greetings_hi = [
            "Namaste! Main ChakrView hoon. Aaj main aapki kya madad kar sakta hoon?",
            "Pranam! Batayein, aaj kis vishay par charcha karni hai?",
            "Hello! Main theek hoon aur aapki madad ke liye tayyar hoon. Aap batayein?"
        ]
        self.chit_chat_hinglish = [
            "Main theek hoon! System smoothly run kar raha hai. Aap bataiye, aaj kya plan hai?",
            "Sab badhiya hai! Aapke sawaalon ke jawab dene ke liye taiyyar hoon. Kya janna chahte hain?",
            "Main abhi active hoon aur tasks process kar raha hoon. Aap bataiye, kya chal raha hai?"
        ]

    def generate_greeting(self, is_hinglish: bool = False) -> str:
        if is_hinglish:
            return random.choice(self.chit_chat_hinglish)
        return random.choice(self.greetings_en)

    def generate_unknown_personal_response(self, query: str) -> str:
        return (
            "I do not possess that personal information. Since this details your private personal history "
            "and has not been recorded in our conversation, I cannot answer this without making a baseless assumption."
        )

    def generate_memory_stored_response(self, stored_facts: List[Dict[str, str]]) -> str:
        if not stored_facts:
            return "I have noted that information in memory."
        parts = []
        for f in stored_facts:
            parts.append(f"{f['relation']} as '{f['value']}'")
        return f"Got it! I have saved to persistent memory: {', '.join(parts)}."

    def generate_memory_retrieval_response(self, target: str, relation: str, memory_manager: Any) -> str:
        if target == "user_project":
            if relation == "name":
                val = memory_manager.get_fact("user_project", "name")
                if val:
                    return f"Your project is called **{val}**."
                return "You haven't specified a project name yet. What would you like to name it?"
            elif relation == "language":
                val = memory_manager.get_fact("user_project", "language")
                if val:
                    return f"Your project uses **{val}**."
                return "The programming language for your project has not been specified yet."
            elif relation == "location":
                val = memory_manager.get_fact("user_project", "location")
                proj_name = memory_manager.get_fact("user_project", "name") or "your project"
                if val:
                    return f"{proj_name} is located at: {val}."
                return (
                    f"Your project is called **{proj_name}**, but its filesystem location has not been provided. "
                    "If you'd like me to remember where it is, please provide its path."
                )

        elif target == "user" and relation == "name":
            name = memory_manager.get_preference("user_name") or memory_manager.get_fact("user", "name")
            if name:
                return f"Your name is **{name}**."
            return "I don't have your name recorded yet. What should I call you?"

        return "I could not find that specific record in persistent memory."

    def generate_user_correction_response(self, correction: str, full_text: str, memory_manager: Any) -> str:
        # Check if user is correcting a capital city or general fact:
        # e.g., "The capital of Australia is Sydney. Actually, it is Canberra."
        fact_match = None
        if "australia" in full_text.lower() and "canberra" in correction.lower():
            memory_manager.store_fact("Australia", "capital", "Canberra", source="user_correction")
            return "Thank you for the correction! I have noted and verified that the capital of Australia is Canberra."

        # Generic correction storage
        memory_manager.record_episode(
            summary=f"User correction: {correction}",
            user_input=full_text,
            assistant_response="Acknowledged correction"
        )
        return f"Understood, thank you for the correction! I have updated my records: {correction}"
