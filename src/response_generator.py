
import os
from groq import Groq
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

class LegalGenerator:
    def __init__(self):
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "❌ GROQ_API_KEY not found!\n"
                "   Add it to .env file: GROQ_API_KEY=your_key_here"
            )
        
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"
        print(f"✅ Generator initialized with {self.model}")
    
    def _build_context(self, cases):
       
        context_parts = []
        for case in cases:
            full_text = str(case.get('full_text', ''))[:800]
            ctx = f"""
CASE {case['rank']}: {case.get('case_name', 'Unknown')}
- Court: {case.get('court', 'N/A')}
- Year: {case.get('year', 'N/A')}
- Bench: {case.get('bench', 'N/A')}
- Legal Issue: {case.get('legal_issue', 'N/A')}
- Acts Involved: {case.get('acts_involved', 'N/A')}
- Confidence Match: {case['confidence_pct']}%
- Summary: {case.get('summary', 'N/A')}
- Excerpt: {full_text}...
"""
            context_parts.append(ctx)
        return "\n---\n".join(context_parts)
    
    def _build_prompt(self, query, cases, stats):
        """Build the prompt - strict grounding"""
        context = self._build_context(cases)
        
        return f"""You are an AI Legal Research Assistant for Indian law.
Analyze the query using ONLY the retrieved cases below.

STRICT RULES:
1. ONLY cite cases from "RETRIEVED CASES" - never invent cases
2. Always mention case name, court, and year when citing
3. Distinguish facts from cases vs your interpretation
4. If cases aren't relevant, say so honestly

RETRIEVED CASES:
{context}

RETRIEVAL STATS:
- Cases Found: {stats['total_retrieved']}
- Top Confidence: {stats['top_confidence']}%
- Average Confidence: {stats['avg_confidence']}%

USER QUERY: {query}

Respond in this format:

**Relevant Cases Found:**
[List most relevant cases with citations]

**Analysis:**
[Detailed analysis citing specific cases and their holdings]

**Key Takeaways:**
[Main legal principles as bullet points]

**Confidence Assessment:**
[How well do retrieved cases match the query - be honest]

**Important Note:**
[Disclaimer about needing professional legal advice]"""
    
    def generate(self, query, retrieval_result):
        """Step 3: Generate grounded legal analysis"""
        cases = retrieval_result['cases']
        stats = retrieval_result['retrieval_stats']
        
       
        if not cases or stats['top_confidence'] < 20:
            return self._no_cases_response(query, stats)
        
        prompt = self._build_prompt(query, cases, stats)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise legal research assistant. Only cite cases provided in context. Never hallucinate case names."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            return response.choices[0].message.content
        
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"
    
    def _no_cases_response(self, query, stats):
        """Handle when no relevant cases found"""
        return f"""
**No Directly Relevant Cases Found**

Your Query: "{query}"

The retrieved cases don't appear directly relevant to your question.

**Retrieval Stats:**
- Top Match Confidence: {stats['top_confidence']}% (below threshold)

**Note:** This assistant covers 175 Indian court cases. Your scenario may need cases not in our database.
"""

if __name__ == "__main__":
    gen = LegalGenerator()
    print("Generator ready for testing!")
