from src.exception.exception import FeedException
from src.logging.logger import logging
import re
import logging
from typing import Optional

class TextNormalizer:
    def __init__(self):
        self.suffixes = [
            "ment", "ness", "ful", "less", "ious", "ive", "able", "ible",
            "ing", "ly", "ed", "es", "s", "ies", "y"  # Added more common ones
        ]
        
    def _reduce_repeated_chars(self, word):
        return re.sub(r"(.)\1{3,}", r"\1\1\1", word)
    
    def _simple_sten(self, word):
        original = word
        for suffix in self.suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 3:
                word = word[: -len(suffix)]
                if suffix in ["ing","ed"] and word.endswith(("nn", "ll", "ss", "ff")):
                    word = word[:-1]
                    
                break
        if word == original:
            irregulars = {"was": "be", "were": "be", "been": "be", "went": "go"}
            word = irregulars.get(word, word)
        return word
    
    def normalize(self, text):
        if not text or not isinstance(text, str):
            logging.warning("Invalid text input provided for normalization.")
            return ""
        
        words = re.findall(r"\b\w+(?:['’-]\w+)?\b", text)
        normalized = []
        for w in words:
            if len(w) <= 2:
                continue
            
            w_lower = w.lower()
            w_reduced = self._reduce_repeated_chars(w_lower)
            w_stemmed = self._simple_sten(w_reduced)
            normalized.append(w_stemmed)
            
        normalized_text = " ".join(normalized)
        normalized_text = re.sub(r"\s+", " ", normalized_text).strip()
        
        logging.debug(f"Normalized '{text[:50]}...' to '{normalized_text[:50]}...'")
        return normalized_text