from sentence_transformers import SentenceTransformer

from src.rag.document_loader import load_documents
from src.rag.text_chunker import chunk_text

documents = load_documents("data")

chunks = []

for document in documents:
    chunks.extend(chunk_text(document["content"]))

print(f"Total Chunks: {len(chunks)}")

model = SentenceTransformer("all-MiniLM-L6-v2")

for i, chunk in enumerate(chunks[:5], start=1):

    embedding = model.encode(
        chunk,
        normalize_embeddings=True
    )

    print(f"\nChunk {i}")
    print(chunk[:300])

    print(
        f"Embedding Length: {len(embedding)}"
    )