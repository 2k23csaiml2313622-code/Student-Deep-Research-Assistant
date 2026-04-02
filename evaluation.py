from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def evaluate_rag(query, answer, contexts):

    try:
        # Clean contexts
        contexts = [c.strip() for c in contexts if c.strip()]

        if len(contexts) == 0:
            return {
                "Relevance Score": 0,
                "Faithfulness Score": 0,
                "Precision Score": 0,
                "Recall Score": 0,
                "Groundedness Score": 0
            }

        # Embeddings
        query_emb = model.encode([query])
        answer_emb = model.encode([answer])
        context_emb = model.encode(contexts)

        # =========================
        # 1. RELEVANCE
        # =========================
        relevance = cosine_similarity(query_emb, answer_emb)[0][0]

        # =========================
        # 2. FAITHFULNESS
        # =========================
        faithfulness = np.mean(cosine_similarity(answer_emb, context_emb))

        # =========================
        # 3. PRECISION
        # (Answer vs Context similarity)
        # =========================
        precision = np.max(cosine_similarity(answer_emb, context_emb))

        # =========================
        # 4. RECALL
        # (Context vs Answer coverage)
        # =========================
        recall = np.mean(cosine_similarity(context_emb, answer_emb))

        # =========================
        # 5. GROUNDEDNESS
        # (How grounded answer is in context)
        # =========================
        groundedness = (faithfulness + precision) / 2

        # =========================
        # NORMALIZATION
        # =========================
        def normalize(x):
            return (float(x) + 1) / 2

        relevance = normalize(relevance)
        faithfulness = normalize(faithfulness)
        precision = normalize(precision)
        recall = normalize(recall)
        groundedness = normalize(groundedness)

        return {
            "Relevance Score": round(relevance, 3),
            "Faithfulness Score": round(faithfulness, 3),
            "Precision Score": round(precision, 3),
            "Recall Score": round(recall, 3),
            "Groundedness Score": round(groundedness, 3)
        }

    except Exception as e:
        return {
            "Relevance Score": 0,
            "Faithfulness Score": 0,
            "Precision Score": 0,
            "Recall Score": 0,
            "Groundedness Score": 0,
            "Error": str(e)
        }
