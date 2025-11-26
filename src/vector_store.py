"""
FAISS vector store for efficient similarity search with better error handling
"""
import faiss
import numpy as np
import json
import os
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, index_path: str, metadata_path: str):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = []
        
    def create_index(self, embedding_dim: int):
        """Create a new FAISS index"""
        self.index = faiss.IndexFlatIP(embedding_dim)
        print(f"✅ Created FAISS index with dimension: {embedding_dim}")
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict]):
        """Add embeddings and metadata to the index"""
        if self.index is None:
            raise ValueError("Index not initialized. Call create_index first.")
        
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        # Check embedding dimension matches index dimension
        if embeddings.shape[1] != self.index.d:
            raise ValueError(f"Embedding dimension mismatch: index expects {self.index.d}, got {embeddings.shape[1]}")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Ensure metadata is JSON serializable
        cleaned_metadata = []
        for meta in metadata:
            cleaned_meta = {}
            for key, value in meta.items():
                if isinstance(value, (np.integer, np.int64)):
                    cleaned_meta[key] = int(value)
                elif isinstance(value, (np.floating, np.float64)):
                    cleaned_meta[key] = float(value)
                elif isinstance(value, np.ndarray):
                    cleaned_meta[key] = value.tolist()
                else:
                    cleaned_meta[key] = value
            cleaned_metadata.append(cleaned_meta)
        
        self.metadata.extend(cleaned_metadata)
    
    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar vectors with dimension validation"""
        if self.index is None:
            print("❌ Index not initialized")
            return []
        
        if self.index.ntotal == 0:
            print("❌ Index is empty")
            return []
        
        # Validate query embedding dimension
        if query_embedding.shape[0] != self.index.d:
            print(f"❌ Query dimension mismatch: index expects {self.index.d}, got {query_embedding.shape[0]}")
            return []
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search
            similarities, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.metadata) and similarity > 0:
                    results.append({
                        "similarity": float(similarity),
                        "content": self.metadata[idx]["content"],
                        "source": self.metadata[idx].get("source", "unknown"),
                        "id": int(idx)
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def save(self):
        """Save index and metadata to disk"""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        
        # Save metadata with proper serialization
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Saved vector store with {len(self.metadata)} documents")
    
    def load(self):
        """Load index and metadata from disk"""
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"Index file not found: {self.index_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(self.index_path)
        print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors, dimension: {self.index.d}")
        
        # Load metadata
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"✅ Loaded metadata for {len(self.metadata)} documents")
        else:
            print("⚠️  Metadata file not found")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        if self.index is None:
            return {"status": "not_initialized"}
        
        return {
            "status": "loaded",
            "total_entries": int(self.index.ntotal),
            "embedding_dim": int(self.index.d),
            "metadata_count": len(self.metadata)
        }