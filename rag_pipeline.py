from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load embedding model once
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_vector_store(text):

    # =========================
    # SAFETY CHECK (CRITICAL)
    # =========================
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
    # CREATE VECTOR DB
    # =========================
    vector_db = Chroma.from_texts(
        texts=docs,
        embedding=embedding_model
    )

    return vector_db


def retrieve_context(vector_db, query):

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(query)

    # EXTRA SAFETY
    if not docs:
        return ""

    context = "\n".join([doc.page_content for doc in docs if doc.page_content])

    return context
