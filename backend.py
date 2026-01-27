import os
import tempfile
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_community.vectorstores import Chroma
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.memory.chat_memory import ConversationBufferMemory


# ---------------- ENV ----------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# ---------------- FILE EXTRACTION ----------------
def extract_pdf_pypdf(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


def extract_pdf_unstructured(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file.read())
        tmp_path = tmp.name

    loader = UnstructuredPDFLoader(tmp_path)
    docs = loader.load()
    return "\n".join(doc.page_content for doc in docs)


def extract_txt(file):
    return file.read().decode("utf-8")


# ---------------- TEXT SPLITTING ----------------
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_text(text)


# ---------------- VECTOR STORE ----------------
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------- SUMMARY LLM ----------------
def get_summary_llm():
    endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.3,
        max_new_tokens=300
    )
    return ChatHuggingFace(llm=endpoint)


# ---------------- RAG CHAIN WITH MEMORY ----------------
def get_chain(vectorstore, role="Student"):
    """
    Returns a RAG chain with conversation memory and role-based response style.
    """
    endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.3,
        max_new_tokens=400
    )

    llm = ChatHuggingFace(llm=endpoint)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""
            You are a helpful AI assistant.
            Answer as a {role}.
            Use ONLY the document context.
            If the answer is not in the documents, say:
            'The topic is not listed in the document.'
            """
        ),
        (
            "human",
            """
            Context:
            {context}

            Chat History:
            {chat_history}

            Question:
            {question}
            """
        )
    ])

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: memory.load_memory_variables({})["chat_history"]
        }
        | prompt
        | llm
    )

    return chain, memory



