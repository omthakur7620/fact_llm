"""
Main fact checking orchestrator
"""
from typing import Dict, Any, List
from .claim_extractor import ClaimExtractor
from .retriever import EnhancedFactRetriever as FactRetriever  # ✅ Use enhanced retriever
from .llm_client import GroqClient

class FactChecker:
    def __init__(self, claim_extractor: ClaimExtractor, retriever: FactRetriever, llm_client: GroqClient):
        self.claim_extractor = claim_extractor
        self.retriever = retriever
        self.llm_client = llm_client
    
    def check_claim(self, text: str) -> Dict[str, Any]:
        """
        Main method to check a claim
        """
        # Step 1: Extract claims
        claims = self.claim_extractor.extract_claims(text)
        entities = self.claim_extractor.extract_entities(text)
        
        if not claims:
            return self._get_no_claims_response(text)
        
        # Step 2: For each claim, retrieve similar facts and verify
        claim_results = []
        for claim in claims:
            retrieved_facts = self.retriever.retrieve_similar_facts(claim)
            verification_result = self.llm_client.fact_check_claim(claim, retrieved_facts)
            
            claim_results.append({
                "claim": claim,
                "verification_result": verification_result
            })
        
        # Step 3: Aggregate results
        return self._aggregate_results(text, claim_results, entities)
    
    def _get_no_claims_response(self, text: str) -> Dict[str, Any]:
        """Handle case where no factual claims are found"""
        return {
            "original_text": text,
            "verdict": "UNVERIFIABLE",
            "reasoning": "No verifiable factual claims could be extracted from the input.",
            "claims_analyzed": [],
            "entities_found": [],
            "confidence": "low",
            "error": "No factual claims detected"
        }
    
    def _aggregate_results(self, text: str, claim_results: List[Dict], entities: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from multiple claims"""
        
        # Use the first claim's result as primary (simplified approach)
        primary_result = claim_results[0]["verification_result"]
        
        return {
            "original_text": text,
            "verdict": primary_result.get("verdict", "UNVERIFIABLE"),
            "reasoning": primary_result.get("reasoning", "No reasoning provided"),
            "confidence": primary_result.get("confidence", "low"),
            "claims_analyzed": [result["claim"] for result in claim_results],
            "verification_details": claim_results,
            "entities_found": entities,
            "key_evidence": primary_result.get("key_evidence", []),
            "retrieved_facts_count": primary_result.get("retrieved_facts_count", 0)
        }