"""
Production Configuration for Government Press Release Fact Checker
"""
import os
import pandas as pd
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class AppConfig:
    # API Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # Model Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    GROQ_MODEL = "llama-3.3-70b-versatile"
    
    # Vector Database Configuration
    SIMILARITY_THRESHOLD = 0.6
    TOP_K_RESULTS = 8
    MAX_CONTENT_LENGTH = 1500
    
    # File Paths
    DATA_PATH = "data/processed/press_release_2003.csv"
    VECTOR_DB_DIR = "data/vector_db"
    FAISS_INDEX_PATH = f"{VECTOR_DB_DIR}/faiss_index"
    METADATA_PATH = f"{VECTOR_DB_DIR}/metadata.json"
    
    # Press Release Specific Settings
    PR_COLUMNS = ['pr_id', 'pr_datetime', 'pr_issued_by', 'pr_title', 'pr_content']
    IMPORTANT_MINISTRIES = [
        'Ministry of Railways', 'Ministry of Finance', 'Ministry of Health',
        'Ministry of Education', 'Ministry of Rural Development', 'Ministry of Steel'
    ]
    
    # Fact Checking Parameters
    CONFIDENCE_THRESHOLDS = {
        'high': 0.8,
        'medium': 0.6,
        'low': 0.4
    }
    
    @classmethod
    def validate_config(cls):
        """Validate all configuration parameters"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        if not os.path.exists(cls.DATA_PATH):
            raise FileNotFoundError(f"Data file not found: {cls.DATA_PATH}")
        
        # Create necessary directories
        os.makedirs(cls.VECTOR_DB_DIR, exist_ok=True)

# Global config instance
config = AppConfig()