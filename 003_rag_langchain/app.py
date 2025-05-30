# code based on https://raw.githubusercontent.com/Ajmal0197/DeepseekOllamaRag/refs/heads/main/app.py
import os
curdir = os.path.dirname(os.path.abspath(__file__))

import env3
from langchain_community.document_loaders import PDFPlumberLoader
import streamlit as st

from rag import get_qa_model


# Get env
ENV = env3.read_env(os.path.join(curdir, ".env"))

# Define color palette with improved contrast
primary_color = "#007BFF"  # Bright blue for primary buttons
secondary_color = "#FFC107"  # Amber for secondary buttons
background_color = "#F8F9FA"  # Light gray for the main background
sidebar_background = "#2C2F33"  # Dark gray for sidebar (better contrast)
text_color = "#212529"  # Dark gray for content text
sidebar_text_color = "#FFFFFF"  # White text for sidebar
header_text_color = "#000000"  # Black headings for better visibility

st.markdown("""
    <style>
    /* Main Background */
    .stApp {{
        background-color: #F8F9FA;
        color: #212529;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: #2C2F33 !important;
        color: #FFFFFF !important;
    }}
    [data-testid="stSidebar"] * {{
        color: #FFFFFF !important;
        font-size: 16px !important;
    }}

    /* Headings */
    h1, h2, h3, h4, h5, h6 {{
        color: #000000 !important;
        font-weight: bold;
    }}

    /* Fix Text Visibility */
    p, span, div {{
        color: #212529 !important;
    }}

    /* File Uploader */
    .stFileUploader>div>div>div>button {{
        background-color: #FFC107;
        color: #000000;
        font-weight: bold;
        border-radius: 8px;
    }}

    /* Fix Navigation Bar (Top Bar) */
    header {{
        background-color: #1E1E1E !important;
    }}
    header * {{
        color: #FFFFFF !important;
    }}
    </style>
""", unsafe_allow_html=True)


# App title
st.title("📄 Build a RAG System with DeepSeek R1 & Ollama")

# Sidebar for instructions and settings
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Upload a PDF file using the uploader below.
    2. Ask questions related to the document.
    3. The system will retrieve relevant content and provide a concise answer.
    """)

    st.header("Settings")
    st.markdown("""
    - **Embedding Model**: embeddings-zh（https://pypi.org/project/embeddings-zh/）
    - **Retriever Type**: Similarity Search
    - **LLM**: DeepSeek R1 (Ollama)
    """)

# Main file uploader section
st.header("📁 Upload a PDF Document")
uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")

if uploaded_file is not None:
    st.success("PDF uploaded successfully! Processing...")

    # Save the uploaded file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load the PDF
    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    # Split the document into chunks
    st.subheader("📚 Splitting the document into chunks...")

    qa = get_qa_model(docs=docs, ollama_model=ENV.get("DEEKSEEK_MODEL"))

    # Question input and response display
    st.header("❓ Ask a Question")
    user_input = st.text_input("Type your question related to the document:")

    if user_input:
        with st.spinner("Processing your query..."):
            try:
                response = qa(user_input)["result"]
                st.success("✅ Response:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a PDF file to start.")

