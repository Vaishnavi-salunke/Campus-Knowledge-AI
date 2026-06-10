import os
import streamlit as st
from src.rag.rag_engine import ask_ai

st.set_page_config(
    page_title="Campus Knowledge AI",
    page_icon="🎓",
    layout="wide"
)

# Sidebar
with st.sidebar:

    st.title("🎓 Campus Knowledge AI")
     # Document Selector
    document_names = []

    if os.path.exists("data"):

        document_names = [
            file
            for file in os.listdir("data")
            if file.endswith(".pdf") or file.endswith(".txt")
        ]

    selected_document = st.selectbox(
        "📄 Search In",
        ["All Documents"] + document_names
    )


    uploaded_files = st.file_uploader(
        "📄 Upload College Documents",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

if uploaded_files:

    os.makedirs("data", exist_ok=True)

    for file in uploaded_files:

        file_path = os.path.join(
            "data",
            file.name
        )

        with open(
            file_path,
            "wb"
        ) as f:

            f.write(
                file.getbuffer()
            )

    st.success(
        f"{len(uploaded_files)} document(s) uploaded successfully"
    )

    st.subheader("Uploaded Documents")

    for file in uploaded_files:

        st.write(f"📄 {file.name}")

    st.markdown("---")

    st.subheader("System Status")
    st.success("Online")

    st.markdown("---")

    st.subheader("AI Models")

    st.markdown("""
    - Groq (Llama 3.3 70B)
    - all-MiniLM-L6-v2
    - Cosine Similarity Search
    """)

    st.markdown("---")

    st.subheader("Knowledge Base")

    st.markdown("""
    - Files from data folder
    - Uploaded PDFs
    - Uploaded TXT files
    """)

    st.markdown("---")

    st.subheader("Example Questions")

    st.markdown("""
    - When is the library open?
    - What engineering branches are offered?
    - What facilities are available?
    """)

# Main Area

st.title("🎓 Campus Knowledge AI Assistant")

st.caption(
    "Retrieval-Augmented Generation (RAG) using Sentence Transformers and Groq Llama 3.3 70B"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask a question about the college..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Searching knowledge base..."):

        answer, context, score,source = ask_ai(question, selected_document)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.expander("🔍 Retrieval Details"):

        st.metric(
            "Similarity Score",
            f"{score:.3f}"
        )
        st.write(
           f"📄 Source Document: {source}"
        )

        st.text_area(
            "Retrieved Context",
            context,
            height=150
        )