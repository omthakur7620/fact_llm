"""
Claim and entity extraction from input text
"""
import spacy
import re
from typing import List, Dict

class ClaimExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise ImportError("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
    
    def extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from input text
        """
        doc = self.nlp(text)
        
        claims = []
        
        # Extract named entities that might be part of claims
        entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "LAW", "EVENT"]]
        
        # Use sentence segmentation and pattern matching
        for sent in doc.sents:
            sentence_text = sent.text.strip()
            
            # Filter for factual statements (remove questions, opinions)
            if self._is_factual_claim(sentence_text):
                claims.append(sentence_text)
        
        # If no claims found with NLP, use the whole text as a single claim
        if not claims:
            claims = [text]
        
        return claims[:3]  # Return max 3 claims
    
    def _is_factual_claim(self, text: str) -> bool:
        """
        Determine if text contains a factual claim that can be verified
        """
        # Patterns that indicate factual claims
        factual_patterns = [
            r'\b(announced|declared|stated|confirmed|revealed|launched|introduced)\b',
            r'\b(will|shall|going to)\b',
            r'\b(government|minister|official|department)\b',
            r'\d{4}',  # Years
            r'\b(scheme|policy|program|initiative)\b'
        ]
        
        # Patterns that indicate non-factual content
        non_factual_patterns = [
            r'^\s*[?]',  # Questions
            r'\b(I think|I believe|in my opinion|probably|maybe)\b',  # Opinions
            r'^\s*[!]'  # Exclamations
        ]
        
        # Check for factual patterns
        has_factual = any(re.search(pattern, text, re.IGNORECASE) for pattern in factual_patterns)
        
        # Check against non-factual patterns
        has_non_factual = any(re.search(pattern, text, re.IGNORECASE) for pattern in non_factual_patterns)
        
        return has_factual and not has_non_factual and len(text.split()) >= 4
    
    def extract_entities(self, text: str) -> List[Dict]:
        """
        Extract named entities from text
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "description": spacy.explain(ent.label_)
            })
        
        return entities