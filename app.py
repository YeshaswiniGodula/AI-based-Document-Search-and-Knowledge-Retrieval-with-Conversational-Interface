import os
import streamlit as st
from dotenv import load_dotenv
from backend import extract_pdf_pypdf, extract_pdf_unstructured, extract_txt, split_text, get_vectorstore, get_chain, has_relevant_context

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")

st.set_page_config(
    page_title="AI-Based Document Retrieval Bot",
    layout="wide"
)

def main():
    st.title("📄 AI-Based Document Retrieval Bot")

    if not HF_TOKEN:
        st.error("HF_TOKEN not found")
        st.stop()

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "awaiting_permission" not in st.session_state:
        st.session_state.awaiting_permission = False

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Upload PDF / TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )

        if st.button("Process Documents"):
            all_chunks = []
            for file in uploaded_files:
                if file.name.endswith(".pdf"):
                    text = extract_pdf_pypdf(file)
                else:
                    text = extract_txt(file)
                all_chunks.extend(split_text(text))

            st.session_state.vectorstore = get_vectorstore(all_chunks)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.awaiting_permission:
        reply = st.chat_input("Type yes to allow general answer, or no")

        if reply:
            llm, _ = get_chain(st.session_state.vectorstore, HF_TOKEN)
            response = llm.invoke(st.session_state.pending_question)
            st.markdown(response.content)

            st.session_state.awaiting_permission = False
            st.session_state.pending_question = None

        return

    if question := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": question})

        llm, retriever = get_chain(st.session_state.vectorstore, HF_TOKEN)
        docs = retriever.get_relevant_documents(question)

        if not docs or not has_relevant_context(docs):
            msg = "This topic is not present in the documents. Do you want a general answer?"
            st.markdown(msg)
            st.session_state.awaiting_permission = True
            st.session_state.pending_question = question
            return

        context = "\n\n".join(doc.page_content for doc in docs)
        response = llm.invoke(context)
        st.markdown(response.content)

if __name__ == "__main__":
    main()

