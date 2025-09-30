import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from tavily import TavilyClient

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in .env")
if not TAVILY_API_KEY:
    raise RuntimeError("Missing TAVILY_API_KEY in .env")

# --------------------------
# Create Tavily client
# --------------------------
def create_tavily_client():
    try:
        return TavilyClient(api_key=TAVILY_API_KEY)
    except ImportError:
        raise RuntimeError("tavily-python SDK is not installed. Run: pip install tavily-python")

def tavily_web_search(query, top_k=5):
    client = create_tavily_client()
    results = client.search(query=query, max_results=top_k)
    return [r["content"] for r in results.get("results", [])]

# --------------------------
# Create LLM client (Groq)
# --------------------------
def create_llm():
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model="llama-3.1-8b-instant"  # ✅ Updated to supported model
    )

# --------------------------
# Competitor Model Extraction
# --------------------------
def extract_competitor_models(search_results, model):
    template = """
    From the following search results, extract smartwatch competitor models (brand + model names).
    Ignore article/blog titles and only list smartwatch names.

    Search Results:
    {results}

    Return competitors as a bullet list.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()
    competitors = chain.invoke({"results": "\n".join(search_results)})
    return [c.strip("-• ") for c in competitors.split("\n") if c.strip()]

# --------------------------
# Features and Price Extraction
# --------------------------
def extract_features_and_prices(competitors, model):
    template = """
    For each of the following smartwatches, provide:
    - Key Features (3–5 points)
    - Approximate Price Range (in USD and PKR if available)

    Smartwatches:
    {competitors}

    Return as structured markdown text.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()
    details = chain.invoke({"competitors": "\n".join(competitors)})
    return details

# --------------------------
# Report Synthesis
# --------------------------
def synthesize_report(competitors, features_prices, model):
    template = """
    You are a market research analyst.

    Competitors:
    {competitors}

    Features & Prices:
    {features_prices}

    Based on this, write a final **Market Research Report** with:
    1. Competitor List
    2. Features & Price Comparison
    3. Strategic Suggestions for Nexus Smartwatch Pro 2
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | model | StrOutputParser()
    report = chain.invoke({"competitors": "\n".join(competitors),
                           "features_prices": features_prices})
    return report

# --------------------------
# Main Agent Run
# --------------------------
def main():
    model = create_llm()

    # Step 1: Web searches
    queries = [
        "best smartwatches 2025",
        "top smartwatches Pakistan price",
        "smartwatch competitors to Nexus Smartwatch Pro 2"
    ]

    all_results = []
    for q in queries:
        print(f"[search] {q}")
        results = tavily_web_search(q, top_k=6)
        all_results.extend(results)

    # Step 2: Extract competitor smartwatch models
    competitors = extract_competitor_models(all_results, model)
    print(f"[info] competitors found: {competitors}")

    # Step 3: Extract features and prices
    features_prices = extract_features_and_prices(competitors, model)

    # Step 4: Generate final report
    report = synthesize_report(competitors, features_prices, model)

    print("\n\n----- FINAL REPORT -----\n")
    print(report)

# --------------------------
# Run the script
# --------------------------
if __name__ == "__main__":
    main()
