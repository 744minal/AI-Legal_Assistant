"""
Indexer: Load CSV data, create embeddings, build FAISS index
Run this once to create the vector database
"""
import pandas as pd
import numpy as np
import faiss
import pickle
import os

from sentence_transformers import SentenceTransformer

# Get the root directory (parent of src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths - Updated to match YOUR folder structure
DATA_PATH = os.path.join(ROOT_DIR, "dataset", "indian_legal_dataset.csv")
VECTOR_DB_DIR = os.path.join(ROOT_DIR, "vector_db")
INDEX_PATH = os.path.join(VECTOR_DB_DIR, "faiss.index")
METADATA_PATH = os.path.join(VECTOR_DB_DIR, "metadata.pkl")
EMBEDDINGS_PATH = os.path.join(VECTOR_DB_DIR, "embeddings.npy")

def load_data():
    """Load and prepare the legal cases dataset"""
    print(f"📂 Loading dataset from: {DATA_PATH}")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded {len(df)} cases")
    print(f"   Columns: {list(df.columns)}")
    return df

def create_embeddings(df, model_name="all-MiniLM-L6-v2"):
    """Create embeddings using sentence-transformers"""
    print(f"\n🔄 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("📝 Preparing text for embedding...")
    texts = []
    for _, row in df.iterrows():
        # Combine key fields for richer retrieval
        parts = []
        if pd.notna(row.get('summary', '')):
            parts.append(str(row['summary']))
        if pd.notna(row.get('legal_issue', '')):
            parts.append(f"Legal Issue: {row['legal_issue']}")
        if pd.notna(row.get('acts_involved', '')):
            parts.append(f"Acts: {row['acts_involved']}")
        
        combined = " ".join(parts) if parts else str(row.get('full_text', ''))[:1000]
        texts.append(combined)
    
    print(f"⚙️ Creating embeddings for {len(texts)} cases...")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    print(f"✅ Created embeddings of shape {embeddings.shape}")
    return embeddings.astype(np.float32)

def build_faiss_index(embeddings):
    """Build FAISS index for fast similarity search"""
    print("\n🔨 Building FAISS index...")
    dimension = embeddings.shape[1]
    
    # Using IndexFlatIP for cosine similarity (normalized embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    
    print(f"✅ FAISS index built with {index.ntotal} vectors")
    return index

def save_all(index, embeddings, df):
    """Save FAISS index, embeddings, and metadata"""
    # Create vector_db directory in ROOT
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    print(f"\n💾 Saving to: {VECTOR_DB_DIR}")
    
    # Save FAISS index
    faiss.write_index(index, INDEX_PATH)
    print(f"   ✅ Saved FAISS index: {INDEX_PATH}")
    
    # Save embeddings separately (backup)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"   ✅ Saved embeddings: {EMBEDDINGS_PATH}")
    
    # Save metadata (case details for retrieval)
    metadata = df.to_dict('records')
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"   ✅ Saved metadata: {METADATA_PATH}")

def main():
    """Main indexing pipeline"""
    print("\n" + "="*60)
    print("🏛️  LEGAL ASSISTANT - INDEXING PIPELINE")
    print("="*60)
    
    # Step 1: Load data
    df = load_data()
    
    # Step 2: Create embeddings
    embeddings = create_embeddings(df)
    
    # Step 3: Build FAISS index
    index = build_faiss_index(embeddings)
    
    # Step 4: Save everything
    save_all(index, embeddings, df)
    
    print("\n" + "="*60)
    print("✅ INDEXING COMPLETE!")
    print(f"   Index location: {VECTOR_DB_DIR}")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()