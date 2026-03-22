from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model (lightweight + fast)
model = SentenceTransformer("all-MiniLM-L6-v2")


def evaluate_rag(query, answer, contexts):
    """
    Evaluates the RAG system using cosine similarity.

    Parameters:
    query (str): User query
    answer (str): LLM generated answer
    contexts (list): Retrieved context chunks

    Returns:
    dict: Evaluation scores
    """

    # Convert text to embeddings
    query_emb = model.encode([query])
    answer_emb = model.encode([answer])
    context_emb = model.encode(contexts)

    # 1️⃣ Relevance Score (Query vs Answer)
    relevance = cosine_similarity(query_emb, answer_emb)[0][0]

    # 2️⃣ Faithfulness Score (Answer vs Context)
    faithfulness = np.mean(cosine_similarity(answer_emb, context_emb))
    faithfullnes= (faithfullness + 1) /2

    return {
        "Relevance Score": round(float(relevance), 3),
        "Faithfulness Score": round(float(faithfulness), 3)
    }
