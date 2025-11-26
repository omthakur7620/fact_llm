"""
Script to build vector store from CSV data with proper encoding handling
"""
import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_config import config
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore

def load_csv_properly(file_path):
    """Load CSV with proper encoding for this dataset"""
    try:
        # Use latin1 encoding which works for this file
        df = pd.read_csv(file_path, encoding='latin1')
        print(f"✅ Successfully loaded CSV with latin1 encoding")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        raise

def prepare_documents(df):
    """Prepare documents for embedding"""
    documents = []
    
    for _, row in df.iterrows():
        # Use pr_content as the main content
        content = str(row['pr_content'])
        title = str(row['pr_title'])
        source = str(row['pr_issued_by'])
        
        # Combine title and content for better context
        combined_content = f"{title}. {content}"
        
        documents.append({
            "content": combined_content,
            "source": source,
            "title": title,
            "pr_id": row['pr_id'],
            "pr_datetime": row['pr_datetime']
        })
    
    print(f"✅ Prepared {len(documents)} documents")
    return documents

def build_vector_store():
    """Build and save vector store from CSV data"""
    
    try:
        print("🚀 Starting vector store build process...")
        
        # Load data
        df = load_csv_properly(config.DATA_PATH)
        
        # Prepare documents
        documents = prepare_documents(df)
        
        # Extract texts and metadata
        texts = [doc["content"] for doc in documents]
        metadata = documents
        
        print(f"📊 Sample document:")
        print(f"   Source: {documents[0]['source']}")
        print(f"   Title: {documents[0]['title']}")
        print(f"   Content preview: {texts[0][:100]}...")
        
        # Initialize components
        print("🔄 Initializing embedding generator...")
        embedding_generator = EmbeddingGenerator(config.EMBEDDING_MODEL)
        
        print("🔄 Creating vector store...")
        vector_store = VectorStore(config.FAISS_INDEX_PATH, config.METADATA_PATH)
        
        # Create index
        embedding_dim = embedding_generator.get_embedding_dimension()
        vector_store.create_index(embedding_dim)
        print(f"✅ Created FAISS index with dimension: {embedding_dim}")
        
        # Generate embeddings in batches
        print("🔄 Generating embeddings...")
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embedding_generator.generate_embeddings_batch(batch_texts)
            all_embeddings.append(batch_embeddings)
            progress = min(i + batch_size, len(texts))
            print(f"   Processed {progress}/{len(texts)} documents")
        
        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        print(f"✅ Generated {len(embeddings)} embeddings")
        
        # Add to vector store
        print("🔄 Adding embeddings to vector store...")
        vector_store.add_embeddings(embeddings, metadata)
        
        # Save vector store
        print("💾 Saving vector store...")
        vector_store.save()
        
        # Print statistics
        stats = vector_store.get_stats()
        print("\n" + "="*60)
        print("🎉 VECTOR STORE BUILD COMPLETE!")
        print("="*60)
        print(f"📄 Total documents: {stats['total_entries']}")
        print(f"🔢 Embedding dimension: {stats['embedding_dim']}")
        print(f"📝 Metadata entries: {stats['metadata_count']}")
        
        # Show ministry distribution
        ministries = {}
        for doc in documents:
            ministry = doc['source']
            ministries[ministry] = ministries.get(ministry, 0) + 1
        
        print(f"\n🏛️  Ministry Distribution:")
        for ministry, count in sorted(ministries.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   • {ministry}: {count} documents")
        
        print(f"\n✅ Vector store saved to:")
        print(f"   FAISS index: {config.FAISS_INDEX_PATH}")
        print(f"   Metadata: {config.METADATA_PATH}")
        
    except Exception as e:
        print(f"❌ Error building vector store: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    build_vector_store()