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
    uploaded_files = st.file_uploader(
    "📄 Upload College Documents",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:

    st.success(
        f"{len(uploaded_files)} document(s) uploaded"
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
    - Gemini 2.5 Flash
    - all-MiniLM-L6-v2
    - Cosine Similarity Search
    """)

    st.markdown("---")

    st.subheader("Knowledge Base")

    st.markdown("""
    - college_info.txt
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
    "Retrieval-Augmented Generation (RAG) using Sentence Transformers and Gemini 2.5 Flash"
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

        answer, context, score = ask_ai(question)

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

        st.text_area(
            "Retrieved Context",
            context,
            height=150
        )