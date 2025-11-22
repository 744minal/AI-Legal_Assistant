
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.case_retriever import LegalRetriever, format_retrieved_cases
from src.response_generator import LegalGenerator

DEMO_QUERIES = [
    {
        "query": "A tenant in Mumbai has been living in an apartment for 10 years. The landlord wants to evict without proper notice. What are the tenant's rights?",
        "area": "Property/Tenancy Law"
    },
    {
        "query": "An employee was terminated without any prior warning or notice period. The company claims poor performance but never gave written warnings. What legal recourse does the employee have?",
        "area": "Labor/Employment Law"
    },
    {
        "query": "A person was arrested without being informed of the grounds for arrest. What are their fundamental rights in this situation?",
        "area": "Constitutional/Criminal Law"
    }
]

class LegalAssistant:
    def __init__(self):
        """Initialize the Legal Research Assistant"""
        print("\n" + "="*60)
        print("🏛️  INITIALIZING AI LEGAL RESEARCH ASSISTANT")
        print("="*60 + "\n")
        
        self.retriever = LegalRetriever()
        self.generator = LegalGenerator()
        print("\n✅ Legal Assistant Ready!\n")
    
    def process_query(self, query, top_k=5):
        """Process a legal query through the RAG pipeline"""
        print("\n" + "-"*60)
        print(f"📌 YOUR QUERY:\n{query}")
        print("-"*60)
        
        print(f"\n🔍 CASES RETRIEVED: {top_k}")
        retrieval_result = self.retriever.retrieve(query, top_k)
        
        qa = retrieval_result['query_analysis']
        print(f"   Detected Legal Areas: {', '.join(qa['detected_legal_areas'])}")
        
        stats = retrieval_result['retrieval_stats']
        print(f"   Top Match Confidence: {stats['top_confidence']}%")
        print(f"   Average Confidence: {stats['avg_confidence']}%")
        
        print("\n" + "-"*60)
        print("📝 GENERATING ANALYSIS...")
        print("-"*60 + "\n")
        
        response = self.generator.generate(query, retrieval_result)
        print(response)
 
        print("\n" + "-"*60)
        print("📚 RETRIEVED CASES DETAIL:")
        print("-"*60)
        print(format_retrieved_cases(retrieval_result))
        
        return response, retrieval_result

def run_demo(assistant):
   
    print("\n" + "="*60)
    print("🎯 RUNNING DEMO QUERIES (3 Demos)")
    print("="*60)
    
    for i, demo in enumerate(DEMO_QUERIES, 1):
        print(f"\n\n{'#'*60}")
        print(f"# DEMO {i}: {demo['area']}")
        print('#'*60)
        
        assistant.process_query(demo['query'])
        
        if i < len(DEMO_QUERIES):
            input("\n⏎ Press Enter for next demo...")

def interactive_mode(assistant):
  
    print("\n" + "="*60)
    print("💬 INTERACTIVE MODE")
    print("   Type your legal query or 'quit' to exit")
    print("="*60)
    
    while True:
        print("\n")
        query = input("🔎 Your legal question: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Thank you for using AI Legal Research Assistant!")
            break
        
        if not query:
            print("⚠️ Please enter a valid query")
            continue
        
        assistant.process_query(query)

def main():
    
    print("AI LEGAL RESEARCH ASSISTANT FOR INDIAN CASE LAW")
    
    
    print("\nChoose mode:")
    print("1. Run demo queries (3 Demos)")
    print("2. Interactive mode (ask your own questions)")
    print("3. Exit")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '3':
        print("Goodbye!")
        return
 
    assistant = LegalAssistant()
    
    if choice == '1':
        run_demo(assistant)
    elif choice == '2':
        interactive_mode(assistant)
    else:
        print("Invalid choice. Running demo by default...")
        run_demo(assistant)
    
    print("\n" + "="*60)
    print("✅ Session Complete")
    print("="*60 + "\n")

if __name__ == "__main__":

    main()
