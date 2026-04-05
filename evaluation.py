from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

# Load lightweight embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


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

        # =========================
        # EMBEDDINGS (UPDATED)
        # =========================

        query_emb = embedding_model.embed_query(query)
        answer_emb = embedding_model.embed_query(answer)
        context_emb = embedding_model.embed_documents(contexts)

        # Convert to numpy arrays
        query_emb = np.array(query_emb).reshape(1, -1)
        answer_emb = np.array(answer_emb).reshape(1, -1)
        context_emb = np.array(context_emb)

        # =========================
        # EXISTING METRICS
        # =========================

        relevance = cosine_similarity(query_emb, answer_emb)[0][0]
        faithfulness = np.mean(cosine_similarity(answer_emb, context_emb))

        # =========================
        # NEW METRICS
        # =========================

        precision = np.max(cosine_similarity(answer_emb, context_emb))
        recall = np.mean(cosine_similarity(context_emb, answer_emb))
        groundedness = np.min(cosine_similarity(answer_emb, context_emb))

        # =========================
        # NORMALIZATION (0–1)
        # =========================

        def normalize(x):
            return (float(x) + 1) / 2

        return {
            "Relevance Score": round(normalize(relevance), 3),
            "Faithfulness Score": round(normalize(faithfulness), 3),
            "Precision Score": round(normalize(precision), 3),
            "Recall Score": round(normalize(recall), 3),
            "Groundedness Score": round(normalize(groundedness), 3)
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
