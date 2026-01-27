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

# ---------------- ENV ----------------
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(
    page_title="AI Document Retrieval Bot",
    layout="wide"
)

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp { background: #f8fafc; }
.stChatMessage { border-radius: 12px; padding: 12px; }
</style>
""", unsafe_allow_html=True)

# ---------------- MAIN APP ----------------
def main():
    st.title("📄 AI-Based Document Retrieval Bot")

    if not HF_TOKEN:
        st.error("HF_TOKEN not found")
        st.stop()

    # -------- SESSION STATE --------
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_history_backup" not in st.session_state:
        st.session_state.chat_history_backup = []

    if "summaries" not in st.session_state:
        st.session_state.summaries = {}

    if "show_chat" not in st.session_state:
        st.session_state.show_chat = True

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.header("📤 Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload PDF / TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )

        st.subheader("⚙️ Advanced Options")

        role = st.selectbox(
            "Answer Style",
            ["Student", "Beginner", "Expert", "Manager"]
        )

        summary_level = st.radio(
            "Summary Level",
            ["Short", "Medium", "Detailed"]
        )

        enable_comparison = st.checkbox("📊 Document Comparison")
        show_sources = st.checkbox("📌 Show Sources", value=True)

        # -------- CHAT CONTROL BUTTONS --------
        st.subheader("💬 Chat Controls")

        if st.button("🧹 Clear Chat"):
            st.session_state.chat_history_backup = st.session_state.messages.copy()
            st.session_state.messages = []
            st.success("Chat cleared")

        if st.button("👀 View Previous Chat"):
            st.session_state.show_chat = not st.session_state.show_chat

        # -------- PROCESS DOCUMENTS --------
        if st.button("🚀 Process Documents"):
            if not uploaded_files:
                st.warning("Upload at least one document")
            else:
                all_chunks = []
                st.session_state.summaries.clear()

                summary_llm = get_summary_llm()

                level_map = {
                    "Short": "2 bullet points",
                    "Medium": "5 bullet points",
                    "Detailed": "10 bullet points"
                }

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

                        summary_prompt = f"""
                        Summarize the document in {level_map[summary_level]}.
                        Use simple English.

                        Document:
                        {text[:3000]}
                        """

                        summary = summary_llm.invoke(summary_prompt)
                        st.session_state.summaries[file.name] = summary.content

                    st.session_state.vectorstore = get_vectorstore(all_chunks)
                    st.success("✅ Documents processed")

    # ---------------- DOCUMENT SUMMARIES ----------------
    if st.session_state.summaries:
        st.subheader("📄 Document Summaries")
        for name, summary in st.session_state.summaries.items():
            with st.expander(name):
                st.markdown(summary)

    # ---------------- DOCUMENT COMPARISON ----------------
    if enable_comparison and len(st.session_state.summaries) >= 2:
        st.subheader("📊 Document Comparison")

        compare_text = ""
        for k, v in st.session_state.summaries.items():
            compare_text += f"\nDocument: {k}\n{v}\n"

        compare_llm = get_summary_llm()
        result = compare_llm.invoke(
            f"Compare these documents:\n{compare_text}"
        )

        st.markdown(result.content)

    # ---------------- PREVIOUS CHAT (OPTIONAL VIEW) ----------------
    if not st.session_state.show_chat and st.session_state.chat_history_backup:
        st.subheader("🕘 Previous Chat History")
        for msg in st.session_state.chat_history_backup:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ---------------- CURRENT CHAT ----------------
    if st.session_state.show_chat:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # ---------------- CHAT INPUT ----------------
    if question := st.chat_input("Ask a question from the documents"):
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("assistant"):
            if not st.session_state.vectorstore:
                st.warning("Upload and process documents first")
            else:
                with st.spinner("Thinking..."):
                    chain, memory = get_chain(
                        st.session_state.vectorstore,
                        role=role
                    )

                    response = chain.invoke(question)
                    answer = response.content

                    if show_sources:
                        answer += "\n\n📌 *Answer generated from uploaded documents*"

                    st.markdown(answer)

                    memory.save_context(
                        {"question": question},
                        {"answer": answer}
                    )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

    # ---------------- FUTURE FEATURES ----------------
    st.sidebar.subheader("🚀 Coming Soon")
    st.sidebar.button("🎤 Voice Input")
    st.sidebar.button("🌐 Multilingual Support")
    st.sidebar.button("📈 Confidence Score")
    st.sidebar.button("🔐 User Login")

if __name__ == "__main__":
    main()

