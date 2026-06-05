from sentence_transformers import SentenceTransformer

from document_loader import load_documents
from text_chunker import chunk_text

documents = load_documents("data")

chunks = []

for document in documents:
    chunks.extend(chunk_text(document["content"]))

model = SentenceTransformer("all-MiniLM-L6-v2")

for i, chunk in enumerate(chunks, start=1):
    embedding = model.encode(chunk)

    print(f"\nChunk {i}")
    print(chunk)
    print("Embedding Length:", len(embedding))