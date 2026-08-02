import os
import sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir)

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import gradio as gr

# Load environment variables and check for GROQ API key
load_dotenv()
OLLAMA_BASEURL = os.getenv("OLLAMA_BASEURL")
if not OLLAMA_BASEURL:
    raise EnvironmentError("OLLAMA_BASEURL is missing! Please add it.")

print("OLLAMA_BASEURL", OLLAMA_BASEURL)

from ollama import Client
client = Client(
    host=OLLAMA_BASEURL
)


# Global variable to store the processed PDF vector store persistently
global_vector_store = None

EXAMPLE_QUERIES = [
    "Provide a brief outline of this document.",
    "What is the summary points of this document?",
    "List the main topics discussed in this document.",
    "What are the key takeaways from this document?",
    "Explain the purpose of this document.",
    "What are the key findings of this document?",
    "What are the main conclusions of this document?",
    "What are the limitations of this document?",
    "Identify the challenges or problems discussed in this document.",
    "What methodology or approach is discussed in this document?",
]


def process_pdf(pdf_file):
    """
    Processes the uploaded PDF:
    - Checks if the input has a 'read' attribute.
    - Reads the PDF bytes (or opens the file if a string is provided).
    - Writes the bytes to a temporary file.
    - Loads and splits the document, generates embeddings, and builds a vector store.
    Returns a tuple: (status_message, vector_store).
    """
    global global_vector_store
    if pdf_file is None:
        return ("Please upload a valid PDF file.", None)
    try:
        # If pdf_file has a 'read' attribute, use it; otherwise, assume it's a file path
        if hasattr(pdf_file, "read"):
            pdf_bytes = pdf_file.read()
        else:
            with open(pdf_file, "rb") as f:
                pdf_bytes = f.read()

        # Write the bytes to a persistent temporary file
        temp_path = "uploaded_temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(pdf_bytes)

        # Process the temporary file
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5")
        vector_store = FAISS.from_documents(chunks, embedder)
        global_vector_store = vector_store

        return ("解析/向量化成功。向量化方案：智谱 AI - BAAI/bge-base-zh-v1.5。文档解析完毕，请提问！")
    except Exception as e:
        return (f"Error processing PDF: {str(e)}", None)


def query_pdf(query):
    """
    Uses the stored vector store to answer the user's query.
    """
    if global_vector_store is None:
        return "Please process a PDF first."
    try:
        retriever = global_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        retrieved_docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"You are a helpful assistant. Based on the following document:\n{context}\nAnswer the question: {query}"

        response = client.chat(
            model="deepseek-r1:14b",
            messages=[{"role": "user", "content": prompt}],
            stream=False  # Enable streaming
        )

        print("response", response)

        return response.message.content
    except Exception as e:
        return f"Error answering query: {str(e)}"


def rag_interface(pdf, query):
    """
    If a PDF is uploaded, process it and update the global vector store.
    Then use the stored vector store to answer the query.
    """
    global global_vector_store
    if pdf:
        status, store = process_pdf(pdf)
        if store is None:
            return status
    if not query:
        return "Please enter a query."
    return query_pdf(query)


# Gradio interface using two tabs to separate PDF upload from query
with gr.Blocks() as interface:
    gr.Markdown("## 基于 Gradio 的文档级 RAG 问答系统开发（DeepSeek + 智谱大模型实战）")

    with gr.Tab("Upload PDF"):
        pdf_input = gr.File(label="上传 PDF 文件", file_types=[".pdf"])
        process_button = gr.Button("解析文档")
        process_status = gr.Textbox(label="Status", interactive=False)
        process_button.click(process_pdf, inputs=[pdf_input], outputs=[process_status])

    with gr.Tab("Ask Query"):
        query_input = gr.Textbox(label="Ask a question", placeholder="Type your query here...")
        submit_button = gr.Button("Submit")
        answer_output = gr.Textbox(label="Answer")
        gr.Examples(
            examples=[[q] for q in EXAMPLE_QUERIES],
            inputs=query_input,
            label="Example Queries"
        )
        submit_button.click(query_pdf, inputs=[query_input], outputs=[answer_output])

if __name__ == "__main__":
    interface.launch(server_name="127.0.0.1", share=True)
