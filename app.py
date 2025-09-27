import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


# Create and store FAISS vector index
def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# Build the conversational chain with custom prompt
def get_conversational_chain():
    prompt_template = """
    You are an AI assistant for answering questions from a set of documents.
    Use only the provided context to answer the question. Be concise and accurate. 
    If you don't know the answer, just say you don't know.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Streamlit UI
def main():
    st.set_page_config(page_title="Chat with PDF - Google GenAI", layout="wide")
    st.header("📄 Chat with your PDF using Google Gemini + FAISS")

    # Upload PDF(s)
    pdf_docs = st.file_uploader("Upload your PDF(s)", accept_multiple_files=True, type=["pdf"])
    if pdf_docs and st.button("Process"):
        with st.spinner("Processing documents..."):
            # Extract and index
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("PDF processed successfully! You can now ask questions.")

    # Question Answering
    # Question Answering
question = st.text_input("Ask a question from the PDF:")
if question:
    # ✅ Use HuggingFace instead of Google embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    docs = vector_store.similarity_search(question, k=3)
    chain = get_conversational_chain()

    with st.spinner("Generating answer..."):
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

    st.subheader("Answer:")
    st.write(response["output_text"])



if __name__ == "__main__":
    main()
