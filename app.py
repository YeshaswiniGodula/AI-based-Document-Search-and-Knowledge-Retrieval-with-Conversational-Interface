import os
import streamlit as st

from main import load_documents, create_vectorstore, ask_question

st.set_page_config(page_title="AI Document Copilot", layout="wide")
st.title("🤖 AI Document Copilot")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------------
# Sidebar - Upload files
# ----------------------------------
st.sidebar.header("Upload Documents")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        file_path = os.path.join(DATA_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

    st.sidebar.success("Files uploaded successfully")

# ----------------------------------
# Build Knowledge Base
# ----------------------------------
if st.sidebar.button("Build Knowledge Base"):
    with st.spinner("Processing documents..."):
        docs = load_documents()
        vectordb = create_vectorstore(docs)
        st.session_state.vectordb = vectordb

    st.sidebar.success(f"Processed {len(docs)} chunks successfully")

# ----------------------------------
# Chat Section
# ----------------------------------
st.subheader("Ask Questions from Documents")

user_query = st.text_input("Type your question")

if user_query:
    if "vectordb" not in st.session_state:
        st.warning("Please upload files and build the knowledge base first")
    else:
        answer = ask_question(st.session_state.vectordb, user_query)

        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(answer)

