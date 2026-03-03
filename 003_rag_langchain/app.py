# code based on https://raw.githubusercontent.com/Ajmal0197/DeepseekOllamaRag/refs/heads/main/app.py
import os
curdir = os.path.dirname(os.path.abspath(__file__))

import env3
from langchain_community.document_loaders import PDFPlumberLoader
import streamlit as st

# from rag_hf_emb import get_qa_model # HuggingFaceEmbeddings
from rag import get_qa_model # chatopera/Synonyms Embeddings

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
st.title("📄 RAG Service 使用 DeepSeek R1, Ollama, Synonyms, LangChain 和 Streamlit")

# Sidebar for instructions and settings
with st.sidebar:
    st.header("说明")
    st.markdown("""
    1. 上传 PDF 文件进行分析检索.
    2. 提出和 PDF 文件有关的问题.
    3. 本服务会利用 RAG 召回，并结合提示词，使用大语言模型推理答案.
    """)

    st.header("配置")
    st.markdown("""
    - **Embedding Model**: [Synonyms 中文近义词模型](https://github.com/chatopera/Synonyms/)
    - **Retriever Type**: Similarity Search
    - **LLM**: DeepSeek R1 (Ollama)
    """)

# Main file uploader section
st.header("📁 上传一个 PDF 文件")
uploaded_file = st.file_uploader("上传", type="pdf")

if uploaded_file is not None:
    st.success("上传成功! 正在处理该 PDF 文件 ...")

    # Save the uploaded file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Load the PDF
    loader = PDFPlumberLoader("temp.pdf")
    docs = loader.load()

    # Split the document into chunks
    st.subheader("📚 将该文件分片 ...")

    qa = get_qa_model(docs=docs, ollama_model=ENV.get("DEEKSEEK_MODEL", "deepseek-r1:14b"))

    # Question input and response display
    st.header("❓ 发送问题")
    user_input = st.text_input("输入一个和该 PDF 文件有关的问题:")

    if user_input:
        with st.spinner("正在推理 ..."):
            try:
                response = qa(user_input)["result"]
                st.success("✅ 返回结果:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.info("上传一个 PDF 文件，程序将完成分析，然后提供检索服务。")

