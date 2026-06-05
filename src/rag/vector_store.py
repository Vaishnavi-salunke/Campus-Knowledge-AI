from sentence_transformers import SentenceTransformer

from document_loader import load_documents
from text_chunker import chunk_text


def create_vector_store():

    documents = load_documents("data")

    chunks = []

    for document in documents:
        chunks.extend(chunk_text(document["content"]))

    model = SentenceTransformer("all-MiniLM-L6-v2")

    vector_store = []

    for chunk in chunks:

        embedding = model.encode(chunk)

        vector_store.append(
            {
                "chunk": chunk,
                "embedding": embedding
            }
        )

    return vector_store


vector_store = create_vector_store()

for item in vector_store:

    print("\nChunk:")
    print(item["chunk"])

    print("Embedding Length:", len(item["embedding"]))