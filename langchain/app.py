# code based on https://raw.githubusercontent.com/Ajmal0197/DeepseekOllamaRag/refs/heads/main/app.py

import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

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
    - **Embedding Model**: HuggingFace
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
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    # Instantiate the embedding model
    embedder = HuggingFaceEmbeddings()

    # Create vector store and retriever
    st.subheader("🔍 Creating embeddings and setting up the retriever...")
    vector = InMemoryVectorStore(embedder) # https://python.langchain.com/docs/concepts/vectorstores/
    print("documents", documents[0])
    vector.add_documents(documents=documents)

    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Define the LLM and the prompt
    llm = OllamaLLM(model="deepseek-r1:14b")
    prompt = """
    1. Use the following pieces of context to answer the question at the end.
    2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
    3. Keep the answer crisp and limited to 3,4 sentences.
    Context: {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    # Define the document and combination chains
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, verbose=True)
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="Context:\ncontent:{page_content}\nsource:{source}",
    )
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        verbose=True
    )

    qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        verbose=True,
        return_source_documents=True
    )

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

