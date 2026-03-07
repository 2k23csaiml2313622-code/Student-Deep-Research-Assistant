🎓 Student Deep Research Assistant (RAG-Based AI Research Tool)

An AI-powered research assistant that automatically gathers, analyzes, and synthesizes information from web sources and uploaded documents to generate structured research answers using Retrieval Augmented Generation (RAG).

This system enables students and researchers to obtain accurate, context-aware explanations by combining LLMs, vector databases, document retrieval, and web search.

🚀 Features
🔎 Automatic Web Research

Uses DuckDuckGo Search to retrieve relevant web sources for the user’s query.

🌐 Website Content Extraction

Extracts useful textual information from web pages using BeautifulSoup.

📄 PDF Document Ingestion

Allows users to upload research papers or notes in PDF format.

Extracts text using PyPDF.

🧠 Retrieval Augmented Generation (RAG)

Creates embeddings from documents and web content.

Stores them in a Chroma Vector Database.

Retrieves the most relevant chunks to generate accurate responses.

✂️ Intelligent Text Chunking

Splits large documents into manageable chunks using RecursiveCharacterTextSplitter.

🔍 Semantic Search

Uses Sentence Transformer embeddings to find semantically similar information.

🤖 AI-Powered Answer Generation

Uses Groq LLM (LLaMA models) to generate structured research responses.

📑 Structured Research Reports

Generates detailed reports including:

Introduction

Key Concepts

Applications

Challenges

Future Scope

Conclusion

📊 Research Progress Visualization

Displays research pipeline stages to the user:

Web search

PDF processing

Vector database creation

Answer generation

🔗 Source Citation

Displays the sources used for generating the answer.

📥 Downloadable Research Report

Users can download the generated research report.

🧠 System Architecture
User Query
     ↓
Query Processing
     ↓
Web Search (DuckDuckGo)
     ↓
Website Scraping
     ↓
PDF Content Extraction
     ↓
Text Chunking
     ↓
Embedding Generation
     ↓
Chroma Vector Database
     ↓
Retriever
     ↓
Groq LLM
     ↓
Structured Research Answer + Sources
🛠 Tech Stack
Programming Language

Python

Frameworks & Libraries

Streamlit – Interactive web application

LangChain – LLM orchestration

LangChain Groq – LLM integration

ChromaDB – Vector database

Sentence Transformers – Embedding generation

BeautifulSoup4 – Website scraping

DuckDuckGo Search – Web search API

PyPDF – PDF text extraction

Requests – HTTP requests

Pandas – Data handling

AI Models

LLaMA 3.1 (via Groq)

📂 Project Structure
Student-Deep-Research-Assistant
│
├── app.py                # Main Streamlit application
├── rag_pipeline.py       # Vector DB creation and retrieval logic
├── utils.py              # Web search, scraping, PDF reading utilities
├── report_generator.py   # Structured report generation
├── requirements.txt      # Required Python libraries
├── README.md             # Project documentation
└── .env                  # API keys (not included in repo)
⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/2k23csaiml2313622-code/Student-Deep-Research-Assistant.git
cd student-deep-research-assistant
2️⃣ Create a virtual environment
python -m venv venv

Activate the environment:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Add your Groq API key

Create a .env file in the project folder:

GROQ_API_KEY=your_api_key_here

You can obtain a free API key from:

https://console.groq.com

5️⃣ Run the application
streamlit run app.py

The app will open at:

http://localhost:8501
🧪 Example Usage
Example Query
Explain Large Language Models and their applications
With PDF Upload
Explain the key concepts mentioned in this research paper
Output

The system will generate:

A detailed research explanation

A structured research report

Source references

📊 Project Highlights

✔ Combines LLMs, web search, and document retrieval
✔ Implements Retrieval Augmented Generation (RAG)
✔ Demonstrates modern GenAI architecture
✔ Supports multi-source research (web + PDFs)
✔ Built using industry-relevant AI tools

🎯 Future Improvements

Potential enhancements include:

Multi-agent research pipelines

Conversational memory for follow-up questions

Academic source prioritization (arXiv, research papers)

Citation with page numbers

Research graph visualization

Integration with academic APIs

👨‍💻 Author

Aishwarya Mishra

B.Tech Final Year Project
Artificial Intelligence / Machine Learning

⭐ If you found this project useful

Please consider starring the repository.

