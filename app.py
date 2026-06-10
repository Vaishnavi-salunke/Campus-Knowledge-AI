import streamlit as st

from src.rag.rag_engine import ask_ai

st.set_page_config(
    page_title="Campus Knowledge AI Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Campus Knowledge AI Assistant")

st.caption(
    "Retrieval-Augmented Generation (RAG) using Sentence Transformers"
)

question = st.text_input(
    "Ask a question about the college..."
)

if question:

    try:

        with st.spinner(
            "Searching knowledge base..."
        ):

            answer, context, score = ask_ai(
                question
            )

        if score < 0.25:

            st.warning(
                "Low confidence result. The answer may not be present in the documents."
            )

        st.success(answer)

        with st.expander(
            "🔍 Retrieval Details",
            expanded=True
        ):

            st.write(
                f"Similarity Score: {score:.3f}"
            )

            st.text_area(
                "Retrieved Context",
                context,
                height=300
            )

    except Exception as e:

        st.error(
            f"Application Error: {e}"
        )