from chroma_setup import setup_chroma_db
from rag_pipeline import create_rag_pipeline
from evaluation import evaluate_rag_pipeline
from config import Config

# Constants
# this one is not in config cause I may need to experiment with that frequently
PROMPT_TEMPLATE = """You are a football tactics expert.

You MUST follow these rules:
1. Use ONLY the information explicitly stated in the Context.
2. If the Context does NOT contain the answer, output EXACTLY:
   "I don't have enough information to answer that."
3. If you output the sentence above, STOP. Do not add anything else.
4. Do NOT use prior knowledge.
5. Do NOT explain unless the answer is found in the Context.

Context:
{context}

Question: {question}

Answer:
"""

# Load configuration
config = Config()

# Setup Chroma DB
db = setup_chroma_db(config.CHROMA_PATH, config.FILE_PATHS, config.EMBEDDING_MODEL_NAME)

# Create RAG Pipeline
rag_pipeline = create_rag_pipeline(db, config.MODEL_NAME, config.TOKENIZER_NAME, PROMPT_TEMPLATE)

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