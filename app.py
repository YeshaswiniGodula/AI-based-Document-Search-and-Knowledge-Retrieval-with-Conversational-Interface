import os
import streamlit as st
from dotenv import load_dotenv

from backend import extract_pdf_pypdf,extract_pdf_unstructured,extract_txt,split_text,get_vectorstore,get_chain,has_relevant_context


# -----------------------------
# ENV
# -----------------------------

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI-Based Document Retrieval Bot",
    layout="wide"
)

st.markdown("""
<style>
h1 { font-size: 1.7rem !important; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("📄 AI-Based Document Retrieval Bot")

    if not HF_TOKEN:
        st.error("HF_TOKEN not found")
        st.stop()

    # -----------------------------
    # SESSION STATE
    # -----------------------------

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "awaiting_permission" not in st.session_state:
        st.session_state.awaiting_permission = False

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None

    # -----------------------------
    # SIDEBAR: UPLOAD
    # -----------------------------

    with st.sidebar:
        st.header("📤 Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload PDF / TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )

        if st.button("Process Documents"):
            if not uploaded_files:
                st.error("Upload at least one document")
            else:
                all_chunks = []

                with st.spinner("Processing documents..."):
                    for file in uploaded_files:
                        text = ""

                        if file.name.endswith(".pdf"):
                            text = extract_pdf_pypdf(file)

                            if len(text.strip()) < 50:
                                file.seek(0)
                                text = extract_pdf_unstructured(file)

                        elif file.name.endswith(".txt"):
                            text = extract_txt(file)

                        if len(text.strip()) < 50:
                            continue

                        all_chunks.extend(split_text(text))

                    if not all_chunks:
                        st.error("No readable text found")
                        st.stop()

                    st.session_state.vectorstore = get_vectorstore(all_chunks)
                    st.success("✅ Documents processed successfully")

    # -----------------------------
    # CHAT HISTORY
    # -----------------------------

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # -----------------------------
    # PERMISSION HANDLING
    # -----------------------------

    if st.session_state.awaiting_permission:
        reply = st.chat_input("Type yes to allow general answer, or no to cancel")

        if reply:
            with st.chat_message("assistant"):
                if reply.lower() in ["yes", "y", "ok", "sure"]:
                    llm, _ = get_chain(st.session_state.vectorstore, HF_TOKEN)
                    response = llm.invoke(
                        f"Answer using general knowledge:\n{st.session_state.pending_question}"
                    )
                    st.markdown(response.content)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response.content}
                    )
                else:
                    msg = "Okay 👍 I will answer only from the uploaded documents."
                    st.markdown(msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": msg}
                    )

            st.session_state.awaiting_permission = False
            st.session_state.pending_question = None

        return

    # -----------------------------
    # CHAT INPUT
    # -----------------------------

    if question := st.chat_input("Ask a question from the documents"):
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("assistant"):
            if not st.session_state.vectorstore:
                st.warning("Upload and process documents first")
                return

            with st.spinner("Thinking..."):
                llm, retriever = get_chain(st.session_state.vectorstore, HF_TOKEN)
                docs = retriever.get_relevant_documents(question)

                if not docs or not has_relevant_context(docs):
                    warning = (
                        "⚠️ This topic is not present in the uploaded documents.\n\n"
                        "Do you want a general answer instead? (yes / no)"
                    )
                    st.markdown(warning)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": warning}
                    )
                    st.session_state.awaiting_permission = True
                    st.session_state.pending_question = question
                    return

                context = "\n\n".join(doc.page_content for doc in docs)
                response = llm.invoke(
                    f"Answer ONLY using this context:\n{context}\n\nQuestion:\n{question}"
                )

                st.markdown(response.content)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response.content}
                )


if __name__ == "__main__":
    main()
