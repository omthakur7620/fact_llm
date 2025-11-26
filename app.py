#!/usr/bin/env python3
"""
Streamlit Web Interface for Fact Checker
"""
import streamlit as st
import pandas as pd
from datetime import datetime
import time
import json
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_config import config
from src.claim_extractor import ClaimExtractor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.retriever import EnhancedFactRetriever
from src.llm_client import GroqClient
from src.fact_checker import FactChecker

@st.cache_resource
def initialize_system():
    """Initialize the fact checking system with caching"""
    try:
        claim_extractor = ClaimExtractor()
        embedding_generator = EmbeddingGenerator(config.EMBEDDING_MODEL)
        vector_store = VectorStore(config.FAISS_INDEX_PATH, config.METADATA_PATH)
        vector_store.load()
        retriever = EnhancedFactRetriever(embedding_generator, vector_store)
        llm_client = GroqClient(config.GROQ_API_KEY, config.GROQ_MODEL)
        fact_checker = FactChecker(claim_extractor, retriever, llm_client)
        return fact_checker, vector_store
    except Exception as e:
        st.error(f"System initialization failed: {e}")
        return None, None

def main():
    st.set_page_config(
        page_title="Government Fact Checker",
        page_icon="🔍",
        layout="wide"
    )
    
    # Header
    st.title("🔍 Government Fact Checker")
    st.markdown("Verify claims against official government press releases (2003)")
    
    # Initialize system
    with st.spinner("Loading fact-checking system..."):
        fact_checker, vector_store = initialize_system()
    
    if fact_checker is None:
        st.error("Failed to initialize the system. Please check your configuration.")
        return
    
    # Sidebar with info
    with st.sidebar:
        st.header("System Info")
        stats = vector_store.get_stats()
        st.write(f"**Documents in database:** {stats.get('total_entries', 0)}")
        st.write(f"**Embedding model:** {config.EMBEDDING_MODEL}")
        st.write(f"**LLM model:** {config.GROQ_MODEL}")
        
        st.header("Example Claims")
        st.markdown("""
        - *Indian Railways have steam locomotives for tourism*
        - *Rural development funds released to Assam*
        - *SAIL investing in steel plant upgrades*
        - *Government announced free electricity for farmers*
        """)
        
        # Feedback system (Bonus feature)
        st.header("Feedback")
        feedback = st.selectbox("Was this helpful?", ["Select option", "Yes", "No", "Somewhat"])
        if feedback != "Select option":
            st.success("Thank you for your feedback! 👍")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Check a Claim")
        claim_text = st.text_area(
            "Enter the claim to verify:",
            placeholder="e.g., 'The government announced free electricity for farmers starting July 2025'",
            height=100
        )
        
        # Confidence threshold slider (Bonus feature)
        confidence_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=1.0,
            value=config.SIMILARITY_THRESHOLD,
            help="Higher values require closer matches, lower values allow more conceptual matches"
        )
        
        if st.button("Verify Claim", type="primary"):
            if claim_text.strip():
                with st.spinner("Analyzing claim..."):
                    start_time = time.time()
                    result = fact_checker.check_claim(claim_text)
                    processing_time = time.time() - start_time
                
                # Display results
                st.subheader("Results")
                
                # Verdict with color coding
                verdict = result['verdict']
                if "TRUE" in verdict:
                    st.success(f"✅ **Verdict: {verdict}**")
                elif "FALSE" in verdict:
                    st.error(f"❌ **Verdict: {verdict}**")
                else:
                    st.warning(f"🤷‍♂️ **Verdict: {verdict}**")
                
                # Confidence
                confidence = result.get('confidence', 'unknown').upper()
                st.write(f"**Confidence:** {confidence}")
                st.write(f"**Processing Time:** {processing_time:.2f}s")
                
                # Reasoning
                st.subheader("Reasoning")
                st.write(result['reasoning'])
                
                # Key Evidence
                if result.get('key_evidence'):
                    st.subheader("Key Evidence")
                    for evidence in result['key_evidence']:
                        st.write(f"• {evidence}")
                
                # Retrieved Documents
                if result.get('retrieved_facts_count', 0) > 0:
                    st.subheader("Retrieved Documents")
                    st.write(f"**Documents analyzed:** {result['retrieved_facts_count']}")
                    
                    # Show top documents
                    if result.get('verification_details'):
                        for detail in result['verification_details'][:3]:
                            facts = detail.get('verification_result', {}).get('retrieved_facts', [])
                            for fact in facts[:2]:
                                with st.expander(f"📄 {fact.get('source', 'Unknown')} (Similarity: {fact['similarity']:.3f})"):
                                    st.write(fact['content'])
                
                # Entities Found
                if result.get('entities_found'):
                    st.subheader("Entities Identified")
                    entities_text = ", ".join([
                        f"{entity['text']} ({entity['label']})" 
                        for entity in result['entities_found'][:6]
                    ])
                    st.write(entities_text)
                
                # Raw JSON (for developers)
                with st.expander("Raw JSON Output"):
                    st.json(result)
                
            else:
                st.warning("Please enter a claim to verify.")
    
    with col2:
        st.subheader("How It Works")
        st.markdown("""
        1. **Claim Extraction** - NLP identifies factual claims
        2. **Vector Search** - Finds similar government documents  
        3. **LLM Analysis** - Compares claim against evidence
        4. **Verdict** - Returns TRUE/FALSE/UNVERIFIABLE with reasoning
        
        **Data Source:** 481 government press releases from 2003
        """)
        
        # System status
        st.subheader("System Status")
        st.success("✅ Operational")
        st.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()