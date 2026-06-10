from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.document_loader import load_documents
from src.rag.text_chunker import chunk_text


def create_vector_store():

    documents = load_documents("data")

    print(f"Loaded {len(documents)} documents")

    chunks = []

    for document in documents:

        document_chunks = chunk_text(
            document["content"]
        )

        for chunk in document_chunks:

            chunks.append(
                {
                    "chunk": chunk,
                    "source": document["filename"]
                }
            )

    print(f"Total chunks created: {len(chunks)}")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    vector_store = []

    for item in chunks:

        chunk = item["chunk"]

        if not chunk.strip():
            continue

        embedding = model.encode(
            chunk,
            normalize_embeddings=True
        )

        vector_store.append(
            {
                "chunk": chunk,
                "source": item["source"],
                "embedding": embedding
            }
        )

    print(
        f"Vector store size: {len(vector_store)}"
    )

    return vector_store, model


def retrieve(
    query,
    vector_store,
    model,
    selected_document="All Documents"
):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    scores = []

    for item in vector_store:

        if (
            selected_document != "All Documents"
            and item["source"] != selected_document
        ):
            continue

        similarity = cosine_similarity(
            [query_embedding],
            [item["embedding"]]
        )[0][0]

        scores.append(
            (
                similarity,
                item["chunk"],
                item["source"]
            )
        )

    scores.sort(
        key=lambda x: x[0],
        reverse=True
    )

    if not scores:
        return "", 0.0, "No Source Found"

    top_chunks = scores[:3]

    print("\nTop Matches:")

    for rank, (score, _, source) in enumerate(
        top_chunks,
        start=1
    ):
        print(
            f"{rank}. Score={score:.4f} | Source={source}"
        )

    context = "\n\n".join(
        [chunk for _, chunk, _ in top_chunks]
    )

    best_score = top_chunks[0][0]

    source = top_chunks[0][2]

    return context, best_score, source


# Create vector store once
vector_store, model = create_vector_store()


def ask_ai(question, selected_document="All Documents"):

    context, score, source = retrieve(
        question,
        vector_store,
        model,
        selected_document
    )

    print(f"\nBest Similarity Score: {score:.4f}")

    if score < 0.25:

        return (
            "I could not find relevant information in the documents.",
            context,
            score,
            source
        )

    prompt = f"""
You are Campus Knowledge AI.

Use ONLY the information provided in the context.

Do not make up facts.
Do not use outside knowledge.

If the answer is not present in the context, reply exactly:

I could not find that information in the documents.

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.choices[0].message.content

    return (
        answer,
        context,
        score,
        source
    )