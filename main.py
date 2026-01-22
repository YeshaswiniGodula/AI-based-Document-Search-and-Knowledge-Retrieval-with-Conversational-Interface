import streamlit as st 
import os 
import shutil 
import time
from dotenv import load_dotenv 
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader, TextLoader 
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_classic.prompts import ChatPromptTemplate 
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain 
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders import PyPDFLoader


load_dotenv()
DATA_DIR = "data"

# -----------------------------
# Load documents
# -----------------------------
def load_documents():
    pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    text_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)

    documents = pdf_loader.load() + text_loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    return splitter.split_documents(documents)


# -----------------------------
# Create vector DB
# -----------------------------
def create_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(docs, embedding=embeddings)
    return vectordb


# -----------------------------
# Ask Question (FIXED)
# -----------------------------
def ask_question(vectordb, question):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 8}
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI document assistant.

        Answer the question using the given context.
        If the answer exists indirectly, explain it clearly.
        Do NOT say the information is missing unless it is truly absent.

        Context:
        {context}

        Question:
        {input}
        """
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    response = retrieval_chain.invoke({"input": question})
    return response["answer"]



