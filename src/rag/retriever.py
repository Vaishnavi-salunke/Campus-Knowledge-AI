from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.document_loader import load_documents
from src.rag.text_chunker import chunk_text


def create_vector_store():

    documents = load_documents("data")

    print(f"Loaded {len(documents)} documents")

    chunks = []

    for document in documents:

        chunks.extend(
            chunk_text(document["content"])
        )

    print(f"Total chunks created: {len(chunks)}")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    vector_store = []

    for chunk in chunks:

        if not chunk.strip():
            continue

        embedding = model.encode(
            chunk,
            normalize_embeddings=True
        )

        vector_store.append(
            {
                "chunk": chunk,
                "embedding": embedding
            }
        )

    print(
        f"Vector store size: {len(vector_store)}"
    )

    return vector_store, model


def retrieve(query, vector_store, model, top_k=3):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

    scores = []

    for item in vector_store:

        similarity = cosine_similarity(
            [query_embedding],
            [item["embedding"]]
        )[0][0]

        scores.append(
            (
                similarity,
                item["chunk"]
            )
        )

    scores.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return scores[:top_k]


# Build vector store
vector_store, model = create_vector_store()

# Test Query
query = "When is the library open?"

results = retrieve(
    query,
    vector_store,
    model
)

print("\nQuestion:")
print(query)

print("\nTop Matches:")

for rank, (score, chunk) in enumerate(results, start=1):

    print(f"\n--- Match {rank} ---")
    print(f"Similarity Score: {score:.4f}")
    print(chunk[:1000])