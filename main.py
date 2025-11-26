#!/usr/bin/env python3
"""
Main entry point for LLM Fact Checker
"""
import argparse
import json
from app_config import config
from src.claim_extractor import ClaimExtractor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import FactRetriever
from src.llm_client import GroqClient
from src.fact_checker import FactChecker

def initialize_system():
    """Initialize all components of the fact checking system"""
    
    # Validate configuration
    config.validate_config()
    
    # Initialize components
    claim_extractor = ClaimExtractor()
    embedding_generator = EmbeddingGenerator(config.EMBEDDING_MODEL)
    vector_store = VectorStore(config.FAISS_INDEX_PATH, config.METADATA_PATH)
    
    # Load vector store
    vector_store.load()
    
    retriever = FactRetriever(embedding_generator, vector_store)
    llm_client = GroqClient(config.GROQ_API_KEY, config.GROQ_MODEL)
    fact_checker = FactChecker(claim_extractor, retriever, llm_client)
    
    print("Fact Checking System Initialized Successfully!")
    return fact_checker

def check_single_claim(fact_checker: FactChecker, claim: str):
    """Check a single claim and print results"""
    print(f"Checking Claim: {claim}")
    print("=" * 60)
    
    result = fact_checker.check_claim(claim)
    
    # Print results
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    print(f"Claims Analyzed: {len(result['claims_analyzed'])}")
    
    print(f"Reasoning: {result['reasoning']}")
    
    if result.get('key_evidence'):
        print(f"Key Evidence:")
        for evidence in result['key_evidence']:
            print(f"  - {evidence}")
    
    if result.get('entities_found'):
        print(f"Entities Found:")
        for entity in result['entities_found'][:5]:
            print(f"  - {entity['text']} ({entity['label']})")
    
    print(f"Retrieved Facts: {result['retrieved_facts_count']}")

def main():
    """Main function"""
    try:
        # Initialize system
        fact_checker = initialize_system()
        
        print("LLM Fact Checker System")
        print("=" * 40)
        
        # Sample claims for demonstration
        sample_claims = [
            "The Indian government has announced free electricity to all farmers starting July 2025.",
            "Indian Railways have earmarked Nilgiri Mountain Railways for promotion of heritage tourism by steam locomotive.",
            "Ministry of Rural Development has released Rs. 70.90 lakh to Assam for rural drinking water supply.",
        ]
        
        # Check sample claims
        for claim in sample_claims:
            check_single_claim(fact_checker, claim)
            print("\n" + "="*60 + "\n")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()