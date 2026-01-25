import os
import tempfile
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores.utils import filter_complex_metadata
# ENV



# PDF / TXT EXTRACTION

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




# SPLITTING text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_text(text)




# VECTOR STORE

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

def has_relevant_context(docs, min_chars=200):
    total_text = "".join(doc.page_content for doc in docs)
    return len(total_text.strip()) >= min_chars


# LLM initialization and CHAIN

def get_chain(vectorstore, hf_token):
    endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
        huggingfacehub_api_token=hf_token,
        temperature=0.3,
        max_new_tokens=400
    )

    llm = ChatHuggingFace(llm=endpoint)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    return llm, retriever

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a document-based assistant.

Rules:
1. Answer ONLY using the provided context.
2. If the answer is NOT found in the context:
   - Say clearly: "This topic is not present in the uploaded documents."
   - Ask the user: "Do you want a general answer instead?"
3. Do NOT use your own knowledge unless the user explicitly agrees.
"""
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])


    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return chain

