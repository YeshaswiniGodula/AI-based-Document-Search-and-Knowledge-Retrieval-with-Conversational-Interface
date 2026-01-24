import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

DATA_DIR = "data"

# -----------------------------
# Load documents
# -----------------------------
def load_documents():
    documents = []

    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())

        elif file.endswith(".txt"):
            loader = TextLoader(path)
            documents.extend(loader.load())

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

    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )

    return vectordb


# -----------------------------
# Ask Question
# -----------------------------
def ask_question(vectordb, question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2
    )

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI document assistant.
        Answer using ONLY the given context.
        Explain clearly and simply.

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
