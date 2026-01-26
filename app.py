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

/* App background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: #e5e7eb;
    font-family: 'Segoe UI', sans-serif;
}

/* Main title */
h1 {
    font-size: 2rem !important;
    font-weight: 700;
    color: #f8fafc;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #020617);
    border-right: 1px solid #1e293b;
}

/* Sidebar headers */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #93c5fd;
}

/* File uploader */
section[data-testid="stFileUploader"] {
    background: #020617;
    border: 1px dashed #334155;
    border-radius: 10px;
    padding: 10px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    color: white;
    border-radius: 10px;
    padding: 0.45rem 1.2rem;
    font-size: 0.9rem;
    border: none;
    transition: all 0.25s ease-in-out;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e3a8a);
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(59,130,246,0.4);
}

/* Chat messages */
.stChatMessage {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 12px;
    margin-bottom: 12px;
}

/* User message */
.stChatMessage[data-testid="chat-message-user"] {
    background: linear-gradient(135deg, #1e3a8a, #1e40af);
    color: #f8fafc;
}

/* Assistant message */
.stChatMessage[data-testid="chat-message-assistant"] {
    background: #020617;
    color: #e5e7eb;
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: #020617;
    border: 1px solid #334155;
    color: #f8fafc;
    border-radius: 12px;
}

/* Success / info messages */
.stAlert {
    border-radius: 10px;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 10px;
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
