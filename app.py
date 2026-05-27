import streamlit as st
import os

from src.pdf_loader import load_multiple_pdfs
from src.text_chunker import split_text
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.rag_pipeline import get_conversation_chain

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="AI Document Q&A Assistant",
    page_icon="📄",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
        color: white;
    }

    .stTextInput input {
        border-radius: 10px;
        padding: 10px;
    }

    .stButton button {
        border-radius: 10px;
        height: 45px;
        width: 120px;
        font-size: 18px;
        font-weight: bold;
    }

    .answer-box {
        background-color: #1E3A2F;
        padding: 20px;
        border-radius: 12px;
        color: white;
        font-size: 18px;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    .chat-box {
        background-color: #1A1D24;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #2E3440;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------- TITLE ---------------- #

st.title("📄 AI Document Q&A Assistant")

st.markdown(
    """
    Upload PDFs and ask intelligent questions using AI-powered RAG.
    """
)

# ---------------- SESSION STATE ---------------- #

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None

if "latest_sources" not in st.session_state:
    st.session_state.latest_sources = []

# ---------------- SIDEBAR ---------------- #

with st.sidebar:

    st.header("📂 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} PDF(s) uploaded"
        )

        os.makedirs(
            "data/uploaded_pdfs",
            exist_ok=True
        )

        # Save PDFs
        for uploaded_file in uploaded_files:

            save_path = os.path.join(
                "data/uploaded_pdfs",
                uploaded_file.name
            )

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Process PDFs
        with st.spinner("Processing PDFs..."):

            raw_text = load_multiple_pdfs(
                "data/uploaded_pdfs"
            )

            chunks = split_text(raw_text)

            embeddings = get_embedding_model()

            vector_store = create_vector_store(
                chunks,
                embeddings
            )

            st.session_state.conversation_chain = (
                get_conversation_chain(
                    vector_store
                )
            )

        st.success(
            "✅ PDFs processed successfully!"
        )

    st.markdown("---")

    # CLEAR CHAT
    if st.button("🗑 Clear Chat"):

        st.session_state.chat_history = []
        st.session_state.latest_sources = []

        st.success("Chat Cleared")

# ---------------- QUESTION SECTION ---------------- #

st.header("💬 Ask Questions")

user_question = st.text_input(
    "Enter your question"
)

# ---------------- SEND BUTTON ---------------- #

if st.button("Send"):

    if not st.session_state.conversation_chain:

        st.warning(
            "Please upload PDFs first."
        )

    elif not user_question:

        st.warning(
            "Please enter a question."
        )

    else:

        try:

            with st.spinner(
                "Generating detailed answer..."
            ):

                response = (
                    st.session_state.conversation_chain.invoke(
                        {
                            "question": user_question,
                            "chat_history": [
                                (
                                    chat["question"],
                                    chat["answer"]
                                )
                                for chat in st.session_state.chat_history
                            ]
                        }
                    )
                )

                answer = response["answer"].strip()

                # SAVE SOURCES
                sources = []

                if "source_documents" in response:

                    for doc in response["source_documents"]:

                        source = doc.metadata.get(
                            "source",
                            "PDF"
                        )

                        if source not in sources:
                            sources.append(source)

                st.session_state.latest_sources = sources

                # SAVE CHAT HISTORY
                st.session_state.chat_history.append(
                    {
                        "question": user_question,
                        "answer": answer
                    }
                )

                st.rerun()

        except Exception as e:

            st.error(f"Error: {str(e)}")

# ---------------- CHAT HISTORY BELOW SEND ---------------- #

if st.session_state.chat_history:

    st.markdown("---")

    st.subheader("🧠 Chat History")

    # SHOW NEWEST ANSWER FIRST
    for chat in reversed(
        st.session_state.chat_history
    ):

        st.markdown(
            f"""
            <div class="chat-box">

            <h4>👤 Question</h4>
            <p>{chat['question']}</p>

            <h4>🤖 Answer</h4>
            <p>{chat['answer']}</p>

            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- SOURCES ---------------- #

if st.session_state.latest_sources:

    st.markdown("---")

    st.subheader("📚 Sources Used")

    for source in st.session_state.latest_sources:

        st.write(f"• {source}")

# ---------------- FOOTER ---------------- #

st.markdown("---")

st.markdown(
    """
    🚀 Built with Streamlit, LangChain, FAISS, HuggingFace, and RAG Architecture
    """
)