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
├── config.py                 # Global configuration: paths, thresholds, model names
│
├── src/
│   ├── claim_extractor.py    # Extracts factual claims using spaCy + rule-based logic
│   ├── embeddings.py         # Loads SentenceTransformer model & generates embeddings
│   ├── vector_store.py       # Builds/loads FAISS index and handles chunking
│   ├── retriever.py          # Retrieves top-K similar chunks from vector DB
│   ├── llm_client.py         # Handles Groq LLM calls and structured responses
│   ├── fact_checker.py       # Main pipeline: extraction → retrieval → verdict
│   └── __init__.py           # Package initializer
│
├── scripts/
│   ├── build_vector_store.py # Generates FAISS index from press_release_2003.csv
│   
│
├
│── app.py      # Full web UI for fact checking
│── interactive.py            # Interactive terminal (CLI) version
│
├── data/
│   ├── raw/                  # Original CSVs or government raw data
│   ├── processed/            # Cleaned datasets (e.g., press_release_2003.csv)
│   └── vector_db/            # FAISS index + metadata.json
│
├── sample_inputs.json        # Example test inputs for evaluation
├── sample_outputs.json       # Example true/false/unverifiable outputs
├── requirements.txt          # Python dependencies
├── .env                      # API keys (Groq key, etc.)
└── README.md                 # Documentation

---

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