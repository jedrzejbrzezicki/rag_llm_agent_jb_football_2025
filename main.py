from chroma_setup import setup_chroma_db
from rag_pipeline import create_rag_pipeline
from evaluation import evaluate_rag_pipeline

# Constants
CHROMA_PATH = "./data/chroma"
FILE_PATHS = [
    "./Sports-Essentials-Football-Coaching-Guide-2021.pdf",
    "./PL_Handbook_25_26_07_10.pdf",
    "./Basic_Football_Tactics-1.pdf"
]
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
TOKENIZER_NAME = "meta-llama/Meta-Llama-3-8B"
PROMPT_TEMPLATE = """You are a football tactics expert. Use only the information provided in the context below to answer the question. If the answer is not in the context, say \"I don't have enough information to answer that.\"\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""

# Setup Chroma DB
db = setup_chroma_db(CHROMA_PATH, FILE_PATHS, EMBEDDING_MODEL_NAME)

# Create RAG Pipeline
rag_pipeline = create_rag_pipeline(db, MODEL_NAME, TOKENIZER_NAME, PROMPT_TEMPLATE)

# Test Questions
test_questions = [
    {
        "question": "What are the four key components of football described in the guide?",
        "expected_keywords": ["attacking", "defending", "transition to defence", "transition to attack"]
    },
    {
        "question": "What is the purpose of maintaining compactness in team defending?",
        "expected_keywords": ["reduce space", "limit passing options", "stay close together", "defensive organization"]
    }
]

# Evaluate RAG Pipeline
results = evaluate_rag_pipeline(rag_pipeline, test_questions)

# Print Results
for result in results:
    print(f"Question: {result['question']}")
    print(f"Response: {result['response']}")
    print(f"Score: {result['score']:.2f}")
    print(f"Found Keywords: {result['found_keywords']}")
    print(f"Missing Keywords: {result['missing_keywords']}")
    print("-")