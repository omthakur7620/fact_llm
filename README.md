# 🔍 LLM-Powered Government Fact Checker
Verify claims against official Government of India press releases (2003)

---

This project implements a lightweight, production-ready RAG (Retrieval-Augmented Generation) fact-checking system.  
Given any public claim, the system:

1. Extracts the key factual statement  
2. Embeds it using Sentence Transformers  
3. Retrieves relevant official press-release segments via FAISS  
4. Uses an LLM (Llama-3.3-70B via Groq) to compare claim vs evidence  
5. Classifies the claim into: **TRUE**, **FALSE**, or **UNVERIFIABLE**  
6. Returns evidence, reasoning, confidence score, and entities detected  

This system demonstrates practical LLM engineering, modular architecture, and real-world fact-verification workflow.

---

## ✨ Features

### ✔ Claim Understanding  
- spaCy sentence extraction  
- Optional LLM-based refinement for clean factual claims  

### ✔ Vector Search (FAISS)  
- Sentence-transformer embeddings  
- Automatic chunking of press releases  
- Fast cosine-similarity retrieval  

### ✔ LLM Verdict Generation  
- Uses Groq’s Llama-3.3-70B  
- Structured reasoning and JSON-safe outputs  

### ✔ Two Interfaces  
- **Interactive CLI**  
- **Streamlit Web App**

---

## 🏛 System Architecture

User Input
→ Claim Extractor
→ Embedding Model
→ FAISS Vector Store
→ Retriever (Top-K Similarity)
→ LLM Comparator
→ Verdict + Reasoning + Evidence

Project Architecture

FACT-LLM/
│
├── config.py                      # Global settings: file paths, thresholds, model names, constants
│
├── src/                           # Core backend modules (RAG + Fact Checking Pipeline)
│   ├── claim_extractor.py         # Extracts factual claims using spaCy + rule-based logic
│   ├── embeddings.py              # Loads SentenceTransformer model & generates embeddings
│   ├── vector_store.py            # Builds/loads FAISS index & handles text chunking
│   ├── retriever.py               # Retrieves top-K semantically similar chunks
│   ├── llm_client.py              # Manages Groq LLM calls with structured prompts
│   ├── fact_checker.py            # Main pipeline: claim extraction → retrieval → verification verdict
│   └── __init__.py                # Makes src/ a Python package
│
├── scripts/                       # Utility scripts for preprocessing & indexing
│   ├── build_vector_store.py      # Generates FAISS index from processed dataset
│
├── app.py                         # Web-based UI for fact checking (e.g., Streamlit/FastAPI)
├── interactive.py                 # Interactive CLI version for terminal usage
│
├── data/
│   ├── raw/                       # Original unprocessed files (government datasets, reports, etc.)
│   ├── processed/                 # Cleaned dataset (e.g., press_release_2003.csv)
│   └── vector_db/                 # FAISS index + metadata.json stored here
│
├── sample_inputs.json             # Example inputs for testing the fact-checking pipeline
├── sample_outputs.json            # Example outputs: True / False / Unverifiable verdicts
│
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (Groq API key, etc.) — NOT included in repo
└── README.md                      # Documentation & usage guide


## ⚙️ Installation

### 1. Create virtual environment (Python 3.11)
```bash
python3.11 -m venv venv

source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows


pip install -r requirements.txt


python -m spacy download en_core_web_sm

🏗 Build the Vector Store

Place your press release CSV in:
data/processed/press_release_2003.csv

Then run:
python scripts/build_vector_store.py

This will:

Load CSV

Clean and chunk text

Generate embeddings

Build FAISS index

Save index + metadata

Run Streamlit Web App
streamlit run app.py

🧱 Tech Stack
LLMs: Groq Llama-3.3-70B
Embeddings: MiniLM-L6-v2
Vector DB: FAISS
NLP: spaCy
Frontend: Streamlit
Language: Python 3.11

Limitations
Dataset contains only select 2003 press releases
Claims outside dataset → unverifiable by design
No cross-year or multi-source fact checking

🚀 Future Improvements
Expand dataset automatically using PIB RSS
Add multi-year fact checking
Confidence calibration model
Add caching layer for embeddings & LLM responses
Multi-document cross-verification


Author
Om Bramhakshatriya
Machine Learning Engineer
Passionate about AI, NLP, and real-world LLM systems.