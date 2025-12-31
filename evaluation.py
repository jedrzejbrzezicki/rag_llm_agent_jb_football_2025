import re
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

def evaluate_rag_pipeline(chain, test_questions):
    """
    Evaluates the RAG pipeline on a set of test questions.

    Args:
        chain (Callable): The RAG pipeline.
        test_questions (list): List of test questions with expected keywords.

    Returns:
        list: Evaluation results.
    """
    results = []
    for test in test_questions:
        question = test["question"]
        expected_keywords = test["expected_keywords"]

        response = chain.invoke(question)
        hits = sum(1 for keyword in expected_keywords if keyword.lower() in response.lower())
        score = hits / len(expected_keywords)

        results.append({
            "question": question,
            "response": response,
            "score": score,
            "found_keywords": [kw for kw in expected_keywords if kw.lower() in response.lower()],
            "missing_keywords": [kw for kw in expected_keywords if kw.lower() not in response.lower()]
        })
    return results