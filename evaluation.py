from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def evaluate_rag(query, answer, contexts):

    # Safety check
    if not contexts:
        return {
            "Relevance Score": 0,
            "Faithfulness Score": 0
        }

    # Embeddings
    query_emb = model.encode([query])
    answer_emb = model.encode([answer])
    context_emb = model.encode(contexts)

    # Relevance
    relevance = cosine_similarity(query_emb, answer_emb)[0][0]

    # Faithfulness (mean similarity)
    faithfulness = np.mean(cosine_similarity(answer_emb, context_emb))

    # 🔥 IMPORTANT: Convert to float BEFORE normalization
    relevance = float(relevance)
    faithfulness = float(faithfulness)

    # 🔥 Normalize (-1 to 1 → 0 to 1)
    relevance = (relevance + 1) / 2
    faithfulness = (faithfulness + 1) / 2

    return {
        "Relevance Score": round(relevance, 3),
        "Faithfulness Score": round(faithfulness, 3)
    }
