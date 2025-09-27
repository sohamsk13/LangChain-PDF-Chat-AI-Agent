# 💬 Chat with Multiple PDFs using Agentic AI

Interact with your documents like never before! This AI-powered Streamlit app lets you upload multiple PDFs and ask questions directly—powered by Google Gemini, LangChain, HuggingFace embeddings, and FAISS vector search.

---

## 🚀 Features

- 📄 **Multi-PDF Upload** – Upload and process multiple PDFs at once.
- 🧠 **Semantic Search** – Uses HuggingFace embeddings + FAISS for fast, accurate retrieval.
- 🤖 **Google Gemini Integration** – Generates concise, context-aware answers using Gemini 2.5 Flash.
- 🔍 **Custom Prompting** – Tailored prompt template ensures grounded, document-based responses.
- 🧩 **Chunked Text Processing** – Efficiently splits large documents for scalable indexing.

---

## 🛠️ Tech Stack

| Layer              | Tools Used                                                                 |
|-------------------|-----------------------------------------------------------------------------|
| Frontend UI        | `Streamlit`                                                                |
| PDF Parsing        | `PyPDF2`                                                                   |
| Text Chunking      | `LangChain` `RecursiveCharacterTextSplitter`                              |
| Embeddings         | `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`)                              |
| Vector Store       | `FAISS`                                                                    |
| LLM                | `Google Gemini 2.5 Flash` via `ChatGoogleGenerativeAI`                     |
| Prompting          | `LangChain PromptTemplate` + `load_qa_chain`                              |
| Environment Config | `dotenv`                                                                   |

---

## 📦 Installation

```bash
git clone https://github.com/sohamsk13/LangChain-PDF-Chat-AI-Agent.git
cd LangChain-PDF-Chat-AI-Agent
pip install -r requirements.txt

.env
GOOGLE_API_KEY=your_google_api_key_here


streamlit run app.py


