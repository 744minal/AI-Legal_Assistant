# AI Legal Research Assistant for Indian Case Law

A Retrieval-Augmented Generation (RAG) system that helps you search through Indian legal judgments and get clear, understandable summaries based on real case law.

## What This Project Does

This assistant helps you:
- Search through a database of Indian legal cases
- Find the most relevant judgments for your question
- Get plain-language explanations of complex legal concepts
- Understand how case law applies to your situation

**Example:** Ask "What are tenant rights in eviction cases?" and get actual case references with simple explanations.

## How It Works

### 1. Indexing (One-time setup)
The system reads all legal cases from the dataset and converts them into searchable vectors using AI embeddings. This creates a "smart search index" that understands legal concepts, not just keywords.

### 2. Retrieval (When you ask a question)
When you type a question, the system:
- Converts your question into a vector
- Searches for the most similar cases in the database
- Returns the top 3-5 most relevant judgments

### 3. Response Generation
The system takes the retrieved cases and:
- Extracts key legal principles
- Summarizes the facts
- Explains how they apply to your question
- Provides actionable insights in simple language

## Project Structure

```
AI-LEGAL_ASSISTANT/
│
├── dataset/
│   └── indian_legal_dataset.csv        # Your legal case database
│
├── src/                                # Source code folder
│   ├── index_builder.py               # Builds the searchable index
│   ├── case_retriever.py              # Finds relevant cases
│   ├── response_generator.py          # Creates summaries
│   └── __init__.py                    # Package initializer
│
├── vector_db/                          # Generated files (auto-created)
│   ├── embeddings.npy                 # Case vectors
│   ├── faiss.index                    # Search index
│   └── metadata.pkl                   # Case information
│
├── main.py                            # Main application
├── requirements.txt                   # Python packages needed
└── README.md                          # This file
```

## Installation Guide

### Step 1: Create Virtual Environment

A virtual environment keeps this project's packages separate from your system Python.

```bash
python -m venv .venv
```

### Step 2: Activate Virtual Environment

**On Windows (PowerShell):**
```bash
.\.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```bash
.\.venv\Scripts\activate.bat
```

**On Mac/Linux:**
```bash
source .venv/bin/activate
```

You should see `(.venv)` at the start of your command line.

### Step 3: Install Required Packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- `sentence-transformers` - For creating embeddings
- `faiss-cpu` - For fast similarity search
- `pandas` - For handling the dataset
- `numpy` - For numerical operations
- Other dependencies

## First Time Setup

### Building the Index

The first time you run the application, it needs to process all cases and build the search index.

```bash
python main.py
```

You'll see:
```
Building index...
Processing cases... [may take 2-5 minutes]
Index built successfully.
Saved to vector_db/
```

**This only happens once.** After the index is built, the app starts instantly.

## Using the Assistant

### Starting the Application

```bash
python main.py
```

You'll see:
```
===============================================================
 AI LEGAL RESEARCH ASSISTANT FOR INDIAN CASE LAW
===============================================================
Choose mode:
1. Run demo queries
2. Interactive mode
3. Exit

Enter your choice (1-3):
```

### Option 1: Demo Mode

Runs pre-written sample queries to show you how the system works.

Example queries:
- "What are tenant rights in Mumbai?"
- "How to file harassment case?"
- "Property dispute between brothers"

### Option 2: Interactive Mode

Type your own legal questions and get instant results.

```
Enter your choice: 2

What's your legal question? tenant eviction without notice
```

### Example Output

```
===============================================================
YOUR QUERY:
tenant eviction without notice

CASES RETRIEVED: 5
===============================================================

Relevant Cases Found:

1. Ram Kumar vs Shyam Lal
   Court: Delhi High Court
   Year: 2019
   Excerpt: "A landlord cannot evict a tenant without proper legal notice
   as mandated under the Rent Control Act..."

2. Maharashtra Housing Board vs Tenants Association
   Court: Supreme Court
   Year: 2020
   Excerpt: "Eviction proceedings must follow due process including
   mandatory notice period of at least 3 months..."

[3 more cases listed...]

===============================================================
LEGAL ANALYSIS
===============================================================

Based on the retrieved cases, here's what you need to know:

Key Legal Principles:
- Landlords must provide written notice before eviction (Ram Kumar vs Shyam Lal)
- Minimum notice period is typically 3 months for residential tenants
- Eviction without notice is illegal and can be challenged in court

Your Situation:
If you're facing eviction without proper notice, you have strong legal
protections. The cases show that courts consistently rule in favor of
tenants who did not receive adequate notice.

Next Steps:
1. Document that you received no notice or inadequate notice
2. Consult with a local lawyer about filing for stay of eviction
3. Gather your rent payment records and tenancy agreement

===============================================================
IMPORTANT NOTE
===============================================================
This is for informational purposes only and not legal advice.
Please consult a qualified lawyer for your specific situation.
```

## Understanding the Results

### Case Information
Each retrieved case shows:
- **Case Name**: The official case title
- **Court**: Which court decided it (Supreme Court, High Court, etc.)
- **Year**: When it was decided
- **Excerpt**: The most relevant part of the judgment

### Legal Analysis
The system provides:
- **Key Principles**: What the law says based on these cases
- **Application**: How it applies to your question
- **Next Steps**: Practical advice on what to do

## Tips for Best Results

### Good Questions
✅ "What are grounds for divorce in India?"
✅ "Can landlord increase rent without agreement?"
✅ "Employee termination without notice period"

### Less Effective Questions
❌ "Help me" (too vague)
❌ "Is this legal?" (needs context)
❌ "What should I do?" (too broad)

**Tip:** Include relevant details like location, relationship (tenant/landlord, employer/employee), and the specific legal issue.

## Technical Details

### Dependencies

Main packages used:
- **sentence-transformers**: Creates AI embeddings of legal text
- **faiss-cpu**: Performs fast similarity search across thousands of cases
- **pandas**: Handles the legal dataset
- **groq** (optional): For advanced AI summaries

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: At least 4GB recommended
- **Storage**: 2GB free space for models and index
- **Internet**: Required for first-time model download

### Dataset Format

The `indian_legal_dataset.csv` should contain:
- Case names
- Court information
- Year of judgment
- Full case text or excerpts
- Legal principles/holdings

## Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "Index not found" error
Delete the `vector_db/` folder and run `python main.py` again to rebuild.

### Slow performance
First run is slow (building index). Subsequent runs are fast.

### No results found
Try rephrasing your question with different keywords or more detail.

## Future Enhancements

Planned features:
- Filter by court, year, or legal act
- Export results to PDF
- Web interface (browser-based)
- Multi-language support
- Citation formatting (for research papers)
- Case law visualization

## Important Legal Disclaimer

⚠️ **This tool is for research and educational purposes only.**

- Not a substitute for professional legal advice
- Always consult a qualified lawyer for your specific situation
- Case law interpretations may vary by jurisdiction
- Laws change over time - verify current applicability

## Contributing

Want to improve this project?
- Add more cases to the dataset
- Improve the response generation
- Create a web interface
- Add new features

## License

This project is for educational use. The legal dataset may have its own terms of use.

## Questions or Issues?

If you encounter problems:
1. Check that all files are in the correct folders
2. Verify Python version (3.8+)
3. Ensure all packages installed correctly
4. Try rebuilding the index

---
