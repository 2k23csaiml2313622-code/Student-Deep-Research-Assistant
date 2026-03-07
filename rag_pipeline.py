from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def create_vector_store(text):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = splitter.split_text(text)

    vector_db = Chroma.from_texts(
        docs,
        embedding=embedding_model
    )

    return vector_db


def retrieve_context(vector_db, query):

    retriever = vector_db.as_retriever(
        search_kwargs={"k":4}
    )

    docs = retriever.invoke(query)

    context = "\n".join([doc.page_content for doc in docs])

    return context