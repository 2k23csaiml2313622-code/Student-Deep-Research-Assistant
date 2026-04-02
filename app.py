import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from utils import web_search, scrape_website, read_pdf
from rag_pipeline import create_vector_store, retrieve_context
from report_generator import generate_report
from evaluation import evaluate_rag  # lightweight evaluation


# =========================
# ENV + LLM SETUP
# =========================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.1-8b-instant",
    temperature=0.3
)

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(
    page_title="Student Deep Research Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Deep Research Assistant")
st.markdown("AI powered research system using **RAG + Web + PDF knowledge**")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Enter your research topic")
pdf = st.file_uploader("Upload PDF for additional knowledge", type=["pdf"])

# Optional evaluation toggle
run_eval = st.checkbox("Enable Performance Evaluation")


# =========================
# MAIN LOGIC
# =========================

if st.button("Start Research"):

    progress = st.progress(0)

    text_data = ""

    # =========================
    # WEB SEARCH
    # =========================

    st.write("🔎 Searching web sources...")

    urls = web_search(query)

    progress.progress(20)

    # =========================
    # SCRAPING (IMPROVED)
    # =========================

    for url in urls:
        try:
            content = scrape_website(url)
            if content:
                text_data += content
        except Exception as e:
            st.warning(f"⚠️ Failed to scrape: {url}")

    progress.progress(40)

    # =========================
    # PDF PROCESSING
    # =========================

    if pdf:
        st.write("📄 Reading PDF...")
        text_data += read_pdf(pdf)

    progress.progress(60)

    # =========================
    # ❗ CRITICAL FIX (EMPTY DATA CHECK)
    # =========================

    if not text_data.strip():
        st.error("❌ No data could be retrieved. Try a different query or upload a PDF.")
        st.stop()

    # =========================
    # VECTOR DB
    # =========================

    st.write("🧠 Creating knowledge base...")

    vector_db = create_vector_store(text_data)

    progress.progress(80)

    context = retrieve_context(vector_db, query)

    # =========================
    # LLM ANSWER
    # =========================

    st.write("🤖 Generating research answer...")

    answer_prompt = f"""
You are a research assistant.

Answer the following query using the provided context.

Context:
{context}

Query:
{query}
"""

    answer = llm.invoke(answer_prompt).content

    progress.progress(100)

    # =========================
    # DISPLAY ANSWER
    # =========================

    st.subheader("📚 Research Answer")
    st.write(answer)

    # =========================
    # REPORT GENERATION
    # =========================

    report = generate_report(llm, context, query)

    st.subheader("📑 Structured Research Report")
    st.write(report)

    # =========================
    # PERFORMANCE EVALUATION
    # =========================

    if run_eval:
        try:
            contexts_list = context.split("\n")

            metrics = evaluate_rag(query, answer, contexts_list)

            st.subheader("📊 Performance Metrics")
            st.write(metrics)

        except Exception as e:
            st.warning("⚠️ Evaluation could not be computed.")

    # =========================
    # SOURCES
    # =========================

    st.subheader("🔗 Sources")

    for u in urls:
        st.write(u)

    # =========================
    # DOWNLOAD REPORT
    # =========================

    st.download_button(
        label="Download Research Report",
        data=report,
        file_name="research_report.txt"
    )   
