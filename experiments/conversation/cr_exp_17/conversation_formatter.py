"""
Phase D: Conversation Formatter and Sequence Builder (conversation_formatter.py)
Provides standard turn formatting, prompt masking, and sequence encoding
for ChakrView conversational training and inference.
"""

from typing import List, Dict, Tuple, Optional

class ConversationFormatter:
    """
    Formats multi-turn dialogue with strict User: / Assistant: boundaries
    and constructs masked training tensors where prompt tokens are ignored (-100).
    """
    USER_TAG = "User: "
    ASSISTANT_TAG = "\nAssistant: "
    EOS_TOKEN_ID = 1
    IGNORE_INDEX = -100

    def __init__(self, tokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def format_turns(self, turns: List[Dict[str, str]]) -> str:
        """
        Formats a list of turn dicts [{'role': 'user', 'content': '...'}, ...]
        into a canonical conversational prompt string.
        """
        formatted = []
        for i, turn in enumerate(turns):
            role = turn["role"].capitalize()
            content = turn["content"].strip()
            if i == 0:
                formatted.append(f"{role}: {content}")
            else:
                formatted.append(f"\n{role}: {content}")
        return "".join(formatted)

    def prepare_training_pair(self, prompt: str, response: str) -> Tuple[List[int], List[int]]:
        """
        Encodes prompt and response into input_ids and labels.
        Labels for prompt tokens are masked with IGNORE_INDEX (-100).
        Labels for response tokens predict the target sequence and end with EOS.
        """
        prompt_ids = self.tokenizer.encode(prompt.strip())
        resp_ids = self.tokenizer.encode(response.strip()) + [self.EOS_TOKEN_ID]
        
        full_ids = prompt_ids + resp_ids
        
        # Truncate if exceeds max_seq_len
        if len(full_ids) > self.max_seq_len:
            # Keep full prompt if possible, or truncate from beginning
            excess = len(full_ids) - self.max_seq_len
            if len(prompt_ids) > excess + 10:
                prompt_ids = prompt_ids[excess:]
                full_ids = prompt_ids + resp_ids
            else:
                full_ids = full_ids[:self.max_seq_len]
                
        labels = [self.IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids):]
        
        return full_ids, labels
