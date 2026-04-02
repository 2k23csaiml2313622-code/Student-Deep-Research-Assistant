import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# =========================
# CREATE PERSISTENT FOLDER
# =========================
os.makedirs("./chroma_db", exist_ok=True)

# =========================
# LOAD EMBEDDING MODEL
# =========================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# CREATE VECTOR STORE
# =========================
def create_vector_store(text):

    # 🔥 SAFETY CHECK
    if not text or not text.strip():
        raise ValueError("No valid text provided for vector store.")

    # =========================
    # TEXT SPLITTING
    # =========================
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = splitter.split_text(text)

    # =========================
    # CLEAN EMPTY CHUNKS
    # =========================
    docs = [d.strip() for d in docs if d.strip()]

    if len(docs) == 0:
        raise ValueError("Text splitting resulted in empty chunks.")

    # =========================
    # CREATE CHROMA DB (FIXED)
    # =========================
    vector_db = Chroma.from_texts(
        texts=docs,
        embedding=embedding_model,
        persist_directory="./chroma_db",   # ✅ CRITICAL FIX
        collection_name="rag_collection"   # ✅ STABILITY FIX
    )

    return vector_db


# =========================
# RETRIEVE CONTEXT
# =========================
def retrieve_context(vector_db, query):

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(query)

    # 🔥 SAFETY CHECK
    if not docs:
        return ""

    context = "\n".join(
        [doc.page_content for doc in docs if doc.page_content]
    )

    return context
