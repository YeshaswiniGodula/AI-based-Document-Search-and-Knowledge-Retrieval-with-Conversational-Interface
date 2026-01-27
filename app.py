import os
import streamlit as st
from dotenv import load_dotenv
from backend import (
    extract_pdf_pypdf,
    extract_pdf_unstructured,
    extract_txt,
    split_text,
    get_vectorstore,
    get_chain,
    get_summary_llm
)

# ENV
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(
    page_title="AI-Based Document Retrieval Bot",
    layout="wide"
)

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: #f8fafc;
    color: #0f172a;
    font-family: 'Segoe UI', system-ui, sans-serif;
}
h1 {
    font-size: 2rem !important;
    font-weight: 700;
}
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}
section[data-testid="stFileUploader"] {
    background: #f1f5f9;
    border: 1px dashed #cbd5f5;
    border-radius: 10px;
    padding: 12px;
}
.stChatMessage {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 12px;
}
.stChatMessage[data-testid="chat-message-user"] {
    background: #eff6ff;
    border-left: 4px solid #2563eb;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MAIN APP ----------------
def main():
    st.title("📄 AI Document Retrieval Bot")

    if not HF_TOKEN:
        st.error("HF_TOKEN not found")
        st.stop()

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "summaries" not in st.session_state:
        st.session_state.summaries = {}

    # ---------------- SIDEBAR ----------------
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
                st.session_state.summaries.clear()

                summary_llm = get_summary_llm()

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

                        chunks = split_text(text)
                        all_chunks.extend(chunks)

                        # -------- SUMMARY PER DOCUMENT --------
                        summary_prompt = f"""
                        Summarize the following document in 5 clear bullet points.
                        Keep it short and professional.

                        Document:
                        {text[:3000]}
                        """

                        summary_response = summary_llm.invoke(summary_prompt)

                        summary_text = (
                            summary_response.content
                            if hasattr(summary_response, "content")
                            else str(summary_response)
                        )

                        st.session_state.summaries[file.name] = summary_text

                    st.session_state.vectorstore = get_vectorstore(all_chunks)
                    st.success("✅ Documents processed successfully")

    # ---------------- DOCUMENT SUMMARIES ----------------
    if st.session_state.summaries:
        st.subheader("📄 Document Summaries")
        for doc, summary in st.session_state.summaries.items():
            with st.expander(f"📘 {doc}"):
                st.markdown(summary)

    # ---------------- CHAT HISTORY ----------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---------------- CHAT INPUT ----------------
    if question := st.chat_input("Ask a question across all documents"):
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("assistant"):
            if not st.session_state.vectorstore:
                st.warning("Upload and process documents first")
            else:
                with st.spinner("Thinking..."):
                    chain = get_chain(st.session_state.vectorstore)
                    response = chain.invoke(question)

                    answer = (
                        response.content
                        if hasattr(response, "content")
                        else str(response)
                    )

                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

if __name__ == "__main__":
    main()

