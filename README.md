# AI Legal Research Assistant for Indian Case Law

A RAG (Retrieval-Augmented Generation) system that helps users find relevant Indian court judgements for their legal scenarios.

## Data Source & Size

- **Dataset**: 175 Indian legal cases (CSV format)
- **Fields Used**: case_id, case_name, court, year, bench, acts_involved, legal_issue, summary, full_text
- **Source**: Pre-cleaned dataset covering Supreme Court and High Court judgements
- **No preprocessing required** - dataset is clean and ready to use

## Retrieval Approach

**Vector Search using FAISS + Sentence Transformers**

| Component | Choice | Reason |
|-----------|--------|--------|
| Embedding Model | `all-MiniLM-L6-v2` | Fast, free, good quality for semantic search |
| Vector DB | FAISS (IndexFlatIP) | Simple, no setup, perfect for 175 docs |
| Similarity | Cosine Similarity | Normalized embeddings for accurate matching |
| Text Combined | summary + legal_issue + acts | Rich context for better retrieval |

**Confidence Scoring:**
- HIGH: ≥50% similarity
- MEDIUM: 35-50% similarity  
- LOW: <35% similarity

## LLM & Prompt Strategy

**Model**: Groq's `llama-3.1-8b-instant` (fast inference, free tier)

**Prompt Strategy (Strict Grounding):**
1. System prompt explicitly forbids hallucination
2. Only retrieved cases provided in context
3. Temperature set to 0.3 for factual accuracy
4. Response format enforces case citations
5. Confidence assessment required in output

## Preventing Hallucinated Case Names

1. **Context Limitation**: LLM only sees cases from FAISS retrieval
2. **Explicit Instructions**: Prompt states "ONLY cite cases from RETRIEVED CASES section"
3. **Citation Format**: Required format includes case name, court, year
4. **Low Confidence Handling**: If no good matches, system admits it rather than guessing
5. **Metadata Verification**: All case details come from original CSV, not LLM

## Installation & Usage

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add your Groq API key to .env
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Build the index (one-time)
python src/indexer.py

# 4. Run the assistant
python main.py
```

## 3-Step Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  USER QUERY: "Tenant eviction without notice in Mumbai"     │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: QUERY UNDERSTANDING                                │
│  • Extract legal keywords                                   │
│  • Detect legal area: [property, tenancy]                   │
│  • Prepare for embedding                                    │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: RETRIEVAL (FAISS)                                  │
│  • Embed query with sentence-transformers                   │
│  • Search top-5 similar cases                               │
│  • Return with confidence scores                            │
└─────────────────────┬───────────────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: RESPONSE GENERATION (Groq LLM)                     │
│  • Build context from retrieved cases                       │
│  • Generate grounded analysis                               │
│  • Cite specific cases with confidence                      │
└─────────────────────────────────────────────────────────────┘
```

## Limitations & Edge Cases

- **Small Dataset**: Only 175 cases - many scenarios won't have direct matches
- **No Real-time Updates**: Cannot access latest judgements
- **Jurisdiction Gaps**: May not cover all High Courts equally
- **Language**: Works best with English queries
- **No Legal Advice**: System explicitly disclaims it's not legal advice

## Future Improvements

1. **Hybrid Search**: Combine vector search with BM25 for keyword matching
2. **Metadata Filtering**: Filter by court, year, legal area before search
3. **Re-ranking**: Add cross-encoder for more accurate ranking
4. **Larger Dataset**: Expand to thousands of cases
5. **Query Expansion**: Use LLM to rephrase queries for better retrieval
6. **Evaluation Pipeline**: Add retrieval quality metrics (MRR, NDCG)

## Project Structure

```
legal-assistant/
├── data/
│   └── indian_legal_cases_175.csv
├── vector_db/
│   ├── faiss.index
│   └── metadata.pkl
├── src/
│   ├── __init__.py
│   ├── indexer.py      # Build index
│   ├── retriever.py    # Search cases
│   └── generator.py    # Generate responses
├── main.py             # Main entry point
├── .env                # API keys
├── requirements.txt    # Dependencies
└── README.md           # This file
```