from sentence_transformers import SentenceTransformer

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

    print(f"Total chunks: {len(chunks)}")

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

    return vector_store


vector_store = create_vector_store()

print("\n===== VECTOR STORE SUMMARY =====")

if vector_store:

    print(
        f"Embedding Dimension: {len(vector_store[0]['embedding'])}"
    )

    print(
        f"Total Stored Chunks: {len(vector_store)}"
    )

    print("\nFirst 3 Chunks Preview:")

    for i, item in enumerate(vector_store[:3], start=1):

        print(f"\n--- Chunk {i} ---")

        print(item["chunk"][:500])

        print(
            f"Embedding Length: {len(item['embedding'])}"
        )

else:

    print("Vector store is empty.")