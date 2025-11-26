"""
Enhanced Fact Retriever with better context handling
"""
from typing import List, Dict
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from app_config import config

class FactRetriever:
    """Original Fact Retriever - keeping for compatibility"""
    def __init__(self, embedding_generator: EmbeddingGenerator, vector_store: VectorStore):
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
    
    def retrieve_similar_facts(self, claim: str, top_k: int = None, similarity_threshold: float = None) -> List[Dict]:
        """
        Retrieve similar facts for a given claim
        """
        if top_k is None:
            top_k = config.TOP_K_RESULTS
        if similarity_threshold is None:
            similarity_threshold = config.SIMILARITY_THRESHOLD
        
        # Generate embedding for the claim
        claim_embedding = self.embedding_generator.generate_embedding(claim)
        
        # Search in vector store
        results = self.vector_store.search(claim_embedding, k=top_k)
        
        # Filter by similarity threshold
        filtered_results = [
            result for result in results 
            if result["similarity"] >= similarity_threshold
        ]
        
        return filtered_results

class EnhancedFactRetriever:
    """Enhanced retriever with better context matching"""
    def __init__(self, embedding_generator: EmbeddingGenerator, vector_store: VectorStore):
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
    
    def retrieve_similar_facts(self, claim: str, top_k: int = None, similarity_threshold: float = None) -> List[Dict]:
        """
        Retrieve similar facts with enhanced context matching
        """
        if top_k is None:
            top_k = config.TOP_K_RESULTS
        if similarity_threshold is None:
            similarity_threshold = config.SIMILARITY_THRESHOLD
        
        # Generate embedding for the claim
        claim_embedding = self.embedding_generator.generate_embedding(claim)
        
        # Search in vector store with more results initially
        initial_results = self.vector_store.search(claim_embedding, k=top_k * 2)
        
        # Apply similarity threshold but be more lenient
        filtered_results = [
            result for result in initial_results 
            if result["similarity"] >= similarity_threshold
        ]
        
        # If no results above threshold, include some lower similarity ones for context
        if not filtered_results and initial_results:
            # Take top 2 even if below threshold for contextual understanding
            filtered_results = initial_results[:2]
            print(f"⚠️  Using lower similarity results: {filtered_results[0]['similarity']:.3f}")
        
        return filtered_results[:top_k]  # Return at most top_k results
    
    def expand_query_terms(self, claim: str) -> List[str]:
        """
        Generate expanded query terms for better retrieval
        """
        # Simple query expansion based on common government terminology
        expansions = []
        claim_lower = claim.lower()
        
        # Ministry-specific expansions
        if 'railway' in claim_lower or 'train' in claim_lower:
            expansions.extend(['railways', 'locomotive', 'heritage', 'tourism'])
        
        if 'rural' in claim_lower or 'development' in claim_lower:
            expansions.extend(['rural development', 'drinking water', 'sanitation', 'funds'])
        
        if 'steel' in claim_lower:
            expansions.extend(['steel plant', 'investment', 'crore', 'upgradation'])
        
        if 'fund' in claim_lower or 'money' in claim_lower or 'rs.' in claim_lower:
            expansions.extend(['allocation', 'released', 'lakh', 'crore'])
        
        # Remove duplicates and return
        return list(set(expansions))