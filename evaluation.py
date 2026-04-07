from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def evaluate_rag(query, answer, contexts):

    try:
        # 🔥 Clean contexts properly (same as your original code)
        contexts = [c.strip() for c in contexts if c.strip()]

        if len(contexts) == 0:
            return {
                "Relevance Score": 0,
                "Faithfulness Score": 0,
                "Precision Score": 0,
                "Recall Score": 0,
                "Groundedness Score": 0
            }

        # =========================
        # EMBEDDINGS (same logic)
        # =========================
        query_emb = model.encode([query])
        answer_emb = model.encode([answer])
        context_emb = model.encode(contexts)

        # =========================
        # EXISTING METRICS
        # =========================

        # Relevance → Query vs Answer
        relevance = cosine_similarity(query_emb, answer_emb)[0][0]

        # Faithfulness → Answer vs Context (average)
        faithfulness = np.mean(cosine_similarity(answer_emb, context_emb))

        # =========================
        # NEW METRICS (ADDED)
        # =========================

        # Precision → how much answer is supported by best matching context
        precision = np.max(cosine_similarity(answer_emb, context_emb))

        # Recall → how much of context is covered in answer
        recall = np.mean(cosine_similarity(context_emb, answer_emb))

        # Groundedness → strictest alignment (worst-case similarity)
        groundedness = np.min(cosine_similarity(answer_emb, context_emb))

        # =========================
        # NORMALIZATION (same style)
        # =========================

        relevance = (relevance + 1) / 2
        faithfulness = (faithfulness + 1) / 2
        precision = (precision + 1) / 2
        recall = (recall + 1) / 2
        groundedness = (groundedness + 1) / 2

        return {
            "Relevance Score": round(float(relevance), 3),
            "Faithfulness Score": round(float(faithfulness), 3),
            "Precision Score": round(float(precision), 3),
            "Recall Score": round(float(recall), 3),
            "Groundedness Score": round(float(groundedness), 3)
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
