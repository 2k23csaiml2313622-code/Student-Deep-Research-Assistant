from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def evaluate_rag(query, answer, contexts):

    try:
        # 🔥 Clean contexts properly
        contexts = [c.strip() for c in contexts if c.strip()]

        if len(contexts) == 0:
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

        # Faithfulness
        faithfulness = np.mean(cosine_similarity(answer_emb, context_emb))

        # Convert to float
        relevance = float(relevance)
        faithfulness = float(faithfulness)

        # Normalize
        relevance = (relevance + 1) / 2
        faithfulness = (faithfulness + 1) / 2

        return {
            "Relevance Score": round(relevance, 3),
            "Faithfulness Score": round(faithfulness, 3)
        }

    except Exception as e:
        return {
            "Relevance Score": 0,
            "Faithfulness Score": 0,
            "Error": str(e)
        }                            
