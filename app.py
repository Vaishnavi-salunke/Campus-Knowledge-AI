from src.rag.document_loader import load_documents
from src.rag.text_chunker import chunk_text

documents = load_documents()

for doc in documents:

    print(f"\nDocument: {doc['filename']}")

    chunks = chunk_text(doc["content"])

    print("\nChunks:")

    for idx, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {idx}:")
        print(chunk)