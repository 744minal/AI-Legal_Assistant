"""
Retriever: Search FAISS index for relevant cases
Includes query understanding and confidence scoring
"""
import faiss
import pickle
import numpy as np
import os
from sentence_transformers import SentenceTransformer

# Get the root directory (parent of src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths - matching your folder structure
VECTOR_DB_DIR = os.path.join(ROOT_DIR, "vector_db")
INDEX_PATH = os.path.join(VECTOR_DB_DIR, "faiss.index")
METADATA_PATH = os.path.join(VECTOR_DB_DIR, "metadata.pkl")

class LegalRetriever:
    def __init__(self):
        """Initialize retriever with FAISS index and embedding model"""
        print("🔄 Loading retriever...")
        
        # Check if index exists
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"❌ FAISS index not found at {INDEX_PATH}\n"
                f"   Run 'python src/index_builder.py' first!"
            )
        
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(INDEX_PATH)
        
        with open(METADATA_PATH, 'rb') as f:
            self.metadata = pickle.load(f)
        
        print(f"✅ Retriever loaded with {self.index.ntotal} cases")
    
    def understand_query(self, query):
        """Step 1: Query Understanding - Extract key legal concepts"""
        legal_keywords = {
            'property': ['tenant', 'landlord', 'rent', 'eviction', 'lease', 'property', 'premises'],
            'criminal': ['murder', 'theft', 'bail', 'arrest', 'fir', 'criminal', 'accused', 'conviction'],
            'constitutional': ['fundamental', 'rights', 'article', 'constitution', 'writ', 'petition'],
            'contract': ['agreement', 'breach', 'contract', 'damages', 'specific performance'],
            'family': ['divorce', 'custody', 'maintenance', 'marriage', 'alimony', 'matrimonial'],
            'labor': ['employee', 'employer', 'termination', 'wages', 'industrial', 'workman']
        }
        
        query_lower = query.lower()
        detected_areas = []
        
        for area, keywords in legal_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected_areas.append(area)
        
        return {
            'original_query': query,
            'detected_legal_areas': detected_areas if detected_areas else ['general'],
            'query_length': len(query.split())
        }
    
    def search(self, query, top_k=5):
        """Step 2: Retrieval - Search for relevant cases with confidence scores"""
        # Embed the query
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding.astype(np.float32), top_k)
        
        # Prepare results with confidence
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.metadata):
                continue
                
            case = self.metadata[idx].copy()
            
            # Confidence scoring (cosine similarity 0-1 for normalized embeddings)
            confidence = float(max(0, min(score, 1)))  # Clamp to 0-1
            confidence_pct = round(confidence * 100, 1)
            
            # Determine confidence level
            if confidence >= 0.5:
                conf_level = "HIGH"
            elif confidence >= 0.35:
                conf_level = "MEDIUM"
            else:
                conf_level = "LOW"
            
            case['similarity_score'] = confidence
            case['confidence_pct'] = confidence_pct
            case['confidence_level'] = conf_level
            case['rank'] = i + 1
            
            results.append(case)
        
        return results
    
    def retrieve(self, query, top_k=5):
        """Full retrieval pipeline with query understanding"""
        # Step 1: Understand query
        query_info = self.understand_query(query)
        
        # Step 2: Search
        results = self.search(query, top_k)
        
        # Calculate overall retrieval confidence
        if results:
            avg_conf = np.mean([r['similarity_score'] for r in results])
            max_conf = results[0]['similarity_score']
        else:
            avg_conf, max_conf = 0, 0
        
        return {
            'query_analysis': query_info,
            'cases': results,
            'retrieval_stats': {
                'total_retrieved': len(results),
                'avg_confidence': round(avg_conf * 100, 1),
                'top_confidence': round(max_conf * 100, 1),
                'has_strong_match': max_conf >= 0.5
            }
        }

def format_retrieved_cases(retrieval_result):
    """Format cases for display"""
    output = []
    for case in retrieval_result['cases']:
        summary = str(case.get('summary', 'N/A'))[:300]
        formatted = f"""
📋 Case {case['rank']}: {case.get('case_name', 'Unknown')}
   Court: {case.get('court', 'N/A')} | Year: {case.get('year', 'N/A')}
   Confidence: {case['confidence_pct']}% ({case['confidence_level']})
   Legal Issue: {case.get('legal_issue', 'N/A')}
   Acts: {case.get('acts_involved', 'N/A')}
   Summary: {summary}...
"""
        output.append(formatted)
    return "\n".join(output)

# Test
if __name__ == "__main__":
    retriever = LegalRetriever()
    test_query = "tenant eviction without notice"
    print(f"\n🔍 Testing: '{test_query}'\n")
    result = retriever.retrieve(test_query)
    print(f"Stats: {result['retrieval_stats']}")
    print(format_retrieved_cases(result))