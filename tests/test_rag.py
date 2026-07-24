"""
RAG Retrieval Evaluation
5 sample queries tested for retrieval quality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_retrieval_evaluation():
    """
    5 Sample Queries - Retrieval Quality Check
    """
    from rag.pipeline import get_retriever

    retriever = get_retriever()

    queries = [
        {
            "query": "How do I register my home food business in Sri Lanka?",
            "expect": ["registration", "sole proprietor", "divisional"],
            "label": "Business Registration"
        },
        {
            "query": "What allergens must I declare on my baked goods label?",
            "expect": ["gluten", "dairy", "eggs", "allergen"],
            "label": "Allergen Labeling"
        },
        {
            "query": "How do I calculate the price of my cupcakes?",
            "expect": ["cost", "ingredient", "margin", "price"],
            "label": "Pricing Guide"
        },
        {
            "query": "What loans are available for women food entrepreneurs?",
            "expect": ["sthree shakthiya", "women", "loan", "4 percent"],
            "label": "Women Entrepreneur Finance"
        },
        {
            "query": "What are the food labeling requirements in Sri Lanka?",
            "expect": ["label", "ingredient", "best before", "allergen"],
            "label": "Food Labeling"
        },
    ]

    print("\n" + "=" * 65)
    print("  BAKEWISE LK — RAG RETRIEVAL EVALUATION")
    print("=" * 65)

    relevant_count = 0
    total_score = 0.0

    for i, item in enumerate(queries, 1):
        docs, context = retriever.retrieve_and_format(item["query"], top_k=3)

        top_score = docs[0].metadata.get("retrieval_score", 0) if docs else 0
        top_source = docs[0].metadata.get("source", "N/A") if docs else "N/A"
        context_lower = context.lower()

        found = [kw for kw in item["expect"] if kw.lower() in context_lower]
        is_relevant = len(found) >= 1

        if is_relevant:
            relevant_count += 1
        total_score += top_score

        status = "RELEVANT" if is_relevant else "NOT RELEVANT"

        print(f"\n[{i}] {item['label']}")
        print(f"    Query   : {item['query']}")
        print(f"    Source  : {top_source}")
        print(f"    Score   : {top_score:.4f}")
        print(f"    Found   : {found}")
        print(f"    Status  : {status}")

    avg = total_score / len(queries)
    accuracy = relevant_count / len(queries) * 100

    print("\n" + "=" * 65)
    print(f"  Relevant   : {relevant_count}/5")
    print(f"  Avg Score  : {avg:.4f}")
    print(f"  Accuracy   : {accuracy:.0f}%")

    if relevant_count >= 4:
        print("  Verdict    : GOOD - RAG pipeline working well")
    elif relevant_count >= 3:
        print("  Verdict    : ACCEPTABLE - Consider adding more docs")
    else:
        print("  Verdict    : NEEDS IMPROVEMENT")
    print("=" * 65)

    assert relevant_count >= 3, f"RAG quality too low: {relevant_count}/5"


if __name__ == "__main__":
    test_retrieval_evaluation()