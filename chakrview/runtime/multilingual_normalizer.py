"""
chakrview/runtime/multilingual_normalizer.py
Multilingual Numeral and Text Normalization Engine for English, Hindi, and Hinglish.

Capabilities:
1. Devanagari digit normalization (०-९ -> 0-9)
2. Hindi number-word parsing (एक -> 1, दो -> 2, पंद्रह -> 15, सौ -> 100, etc.)
3. English number-word parsing (one -> 1, fifteen -> 15, twenty-five -> 25, etc.)
4. Hinglish transliterated number parsing (ek, do, teen, pandrah, bees, etc.)
5. Negative and decimal number extraction
6. Unicode normalization (NFC) and whitespace cleansing
7. Preservation of original character offsets / source spans
"""

import re
import unicodedata
from typing import Dict, Any, List, Optional, Tuple, Union

# Mapping of Devanagari digits to ASCII digits
DEV_DIGIT_MAP = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

# Hindi number words dictionary
HINDI_NUMBER_WORDS = {
    'शून्य': 0, 'एक': 1, 'दो': 2, 'तीन': 3, 'चार': 4, 'पाँच': 5, 'पांच': 5,
    'छह': 6, 'छः': 6, 'सात': 7, 'आठ': 8, 'नौ': 9, 'दस': 10,
    'ग्यारह': 11, 'बारह': 12, 'तेरह': 13, 'चौदह': 14, 'पंद्रह': 15,
    'सोलह': 16, 'सत्रह': 17, 'अठारह': 18, 'उन्नीस': 19, 'बीस': 20,
    'इक्कीस': 21, 'बाईस': 22, 'तेईस': 23, 'चौबीस': 24, 'पच्चीस': 25,
    'छब्बीस': 26, 'सत्ताईस': 27, 'अट्ठाईस': 28, 'उनतीस': 29, 'तीस': 30,
    'इकतीस': 31, 'बत्तीस': 32, 'तैंतीस': 33, 'चौंतीस': 34, 'पैंतीस': 35,
    'छत्तीस': 36, 'सैंतीस': 37, 'अड़तीस': 38, 'उनतालीस': 39, 'चालीस': 40,
    'पचास': 50, 'साठ': 60, 'सत्तर': 70, 'अस्सी': 80, 'नब्बे': 90, 'सौ': 100
}

# Hinglish (Romanized Hindi) words
HINGLISH_NUMBER_WORDS = {
    'shunya': 0, 'zero': 0, 'ek': 1, 'do': 2, 'teen': 3, 'char': 4, 'chaar': 4,
    'paanch': 5, 'panch': 5, 'chhah': 6, 'che': 6, 'chhe': 6, 'saat': 7,
    'aath': 8, 'ath': 8, 'nau': 9, 'das': 10, 'gyarah': 11, 'barah': 12,
    'baarah': 12, 'terah': 13, 'chaudah': 14, 'pandrah': 15, 'solah': 16,
    'satrah': 17, 'atharah': 18, 'unnees': 19, 'bees': 20, 'pachees': 25,
    'tees': 30, 'chaalis': 40, 'chalis': 40, 'pachaas': 50, 'pachas': 50,
    'saath': 60, 'sattar': 70, 'assi': 80, 'nabbe': 90, 'sau': 100
}

# English number words
ENGLISH_NUMBER_WORDS = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
    'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
    'ninety': 90, 'hundred': 100
}

class MultilingualNormalizer:
    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Converts Unicode text to NFC normalized form."""
        return unicodedata.normalize('NFC', text)

    @classmethod
    def convert_devanagari_digits(cls, text: str) -> str:
        """Converts Devanagari digits (०-९) to ASCII digits (0-9)."""
        res = []
        for ch in text:
            res.append(DEV_DIGIT_MAP.get(ch, ch))
        return "".join(res)

    @classmethod
    def parse_numeric_token(cls, token: str) -> Optional[Union[int, float]]:
        """Parses a single token string (ASCII, Devanagari, or word) to int or float."""
        clean = token.strip().lower()
        if not clean:
            return None

        # Try converting Devanagari digits first
        ascii_candidate = cls.convert_devanagari_digits(clean)

        # 1. Direct int conversion
        try:
            return int(ascii_candidate)
        except ValueError:
            pass

        # 2. Direct float conversion
        try:
            return float(ascii_candidate)
        except ValueError:
            pass

        # 3. English word match
        if clean in ENGLISH_NUMBER_WORDS:
            return ENGLISH_NUMBER_WORDS[clean]

        # 4. Hindi Devanagari word match
        if clean in HINDI_NUMBER_WORDS:
            return HINDI_NUMBER_WORDS[clean]

        # 5. Hinglish word match
        if clean in HINGLISH_NUMBER_WORDS:
            return HINGLISH_NUMBER_WORDS[clean]

        return None

    @classmethod
    def extract_numbers_with_spans(cls, text: str) -> List[Dict[str, Any]]:
        """
        Finds all numbers (digits, Devanagari digits, and number words) in text
        along with exact (start, end) character offsets in the original text.
        """
        normalized_nfc = cls.normalize_unicode(text)
        results = []

        # Pattern for numeric expressions (including decimals and negatives)
        # Matches ASCII digits, Devanagari digits, or decimal points
        num_pattern = re.compile(r'[-+]?[0-9०-९]+(?:\.[0-9०-९]+)?')
        for match in num_pattern.finditer(normalized_nfc):
            raw_substr = match.group(0)
            val = cls.parse_numeric_token(raw_substr)
            if val is not None:
                results.append({
                    "raw_text": raw_substr,
                    "value": val,
                    "start": match.start(),
                    "end": match.end(),
                    "type": "numeric_literal"
                })

        # Pattern for number words (English, Hindi Devanagari, Hinglish)
        # Tokenize by non-word boundaries
        word_pattern = re.compile(r'[\w]+', re.UNICODE)
        for match in word_pattern.finditer(normalized_nfc):
            word = match.group(0)
            # If this span overlaps with an already matched numeric literal, skip
            if any(r["start"] <= match.start() and match.end() <= r["end"] for r in results):
                continue
            val = cls.parse_numeric_token(word)
            if val is not None:
                results.append({
                    "raw_text": word,
                    "value": val,
                    "start": match.start(),
                    "end": match.end(),
                    "type": "number_word"
                })

        # Sort results by start position
        results.sort(key=lambda x: x["start"])
        return results

    @classmethod
    def extract_array_with_span(cls, text: str) -> Optional[Dict[str, Any]]:
        """
        Extracts JSON-like or Python-like lists, e.g. [1, 2, 3] or [१, २, ३]
        returning parsed list, raw substring, and character span.
        """
        pattern = re.compile(r'\[([^\]]+)\]')
        match = pattern.search(text)
        if not match:
            return None

        raw_content = match.group(1)
        # Split by comma or whitespace
        parts = re.split(r'[,，\s]+', raw_content.strip())
        parsed_items = []
        for p in parts:
            if not p:
                continue
            val = cls.parse_numeric_token(p)
            if val is not None:
                parsed_items.append(val)
            else:
                # keep as string if non-numeric
                parsed_items.append(p.strip('\'"'))

        return {
            "raw_text": match.group(0),
            "value": parsed_items,
            "start": match.start(),
            "end": match.end(),
            "type": "array_literal"
        }
