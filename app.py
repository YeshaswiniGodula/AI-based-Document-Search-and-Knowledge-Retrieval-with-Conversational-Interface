import os
import streamlit as st
from dotenv import load_dotenv
from backend import extract_pdf_pypdf, extract_pdf_unstructured, extract_txt, split_text, get_vectorstore, get_chain
# ENV

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")  # to Host on streamlit cloud which secrets Token


# STREAMLIT CONFIG

st.set_page_config(
    page_title="AI-Based Document Retrieval Bot",
    layout="wide"
)

# Styling 

st.markdown("""
<style>
h1 { font-size: 1.7rem !important; font-weight: 600; }
.stButton > button {
    background-color: #2563eb;
    color: white;
    border-radius: 6px;
    padding: 0.35rem 0.8rem;
    font-size: 0.85rem;
    transition: all 0.25s ease-in-out;
}
.stButton > button:hover {
    background-color: #1e40af;
    color : black;
    transform: translateY(-2px) scale(1.04);
    box-shadow: 0 6px 16px rgba(37,99,235,0.45);
}
.stChatMessage {
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    transition: all 0.25s ease-in-out;
}
.stChatMessage:hover {
     transform: translateY(2px) scale(1.01);
}

</style>
""", unsafe_allow_html=True)


def main():
    st.title("📄 AI-Based Document Retrieval Bot")

    if not HF_TOKEN:
        st.error("HF_TOKEN not found")
        st.stop()

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar 
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
                                st.info(f"OCR used for: {file.name}")
                                file.seek(0)
                                text = extract_pdf_unstructured(file)

                        elif file.name.endswith(".txt"):
                            text = extract_txt(file)

                        if len(text.strip()) < 50:
                            st.warning(f"Skipped (no text): {file.name}")
                            continue

                        chunks = split_text(text)
                        all_chunks.extend(chunks)

                    if not all_chunks:
                        st.error("No readable text found")
                        st.stop()

                    st.session_state.vectorstore = get_vectorstore(all_chunks)
                    st.success(f"✅ {len(uploaded_files)} documents uploaded Successfully ")


    # Chat History 

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


    # Chat Input 
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



