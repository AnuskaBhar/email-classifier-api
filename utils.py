import re
import spacy
from typing import List, Dict, Tuple

nlp = spacy.load("en_core_web_sm")

class PIIMasker:
    def __init__(self):
        # Regex patterns for PII detection
        self.patterns = {
            "full_name": r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)',
            "email": r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
            "phone_number": r'(\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
            "dob": r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            "aadhar_num": r'(\d{4}\s?\d{4}\s?\d{4})',
            "credit_debit_no": r'(\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4})',
            "cvv_no": r'(\b\d{3,4}\b)',
            "expiry_no": r'((0[1-9]|1[0-2])[/-]?\d{2,4})'
        }
    
    def mask_pii(self, text: str) -> Tuple[str, List[Dict]]:
        masked_text = text
        entities = []
        
        # Detect and mask using regex patterns
        for entity_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                start, end = match.span()
                original = match.group()
                masked_text = masked_text.replace(original, f"[{entity_type}]")
                entities.append({
                    "position": [start, end],
                    "classification": entity_type,
                    "entity": original
                })
        
        # Additional NER using spaCy
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                masked_text = masked_text.replace(ent.text, "[full_name]")
                entities.append({
                    "position": [ent.start_char, ent.end_char],
                    "classification": "full_name",
                    "entity": ent.text
                })
        
        return masked_text, entities