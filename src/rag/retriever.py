from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

    return vector_store, model


vector_store, model = create_vector_store()

query = "When is the library open?"

query_embedding = model.encode(query)

best_chunk = ""
best_score = -1

for item in vector_store:

    similarity = cosine_similarity(
        [query_embedding],
        [item["embedding"]]
    )[0][0]

    if similarity > best_score:
        best_score = similarity
        best_chunk = item["chunk"]

print("\nQuestion:")
print(query)

print("\nBest Match:")
print(best_chunk)

print("\nSimilarity Score:")
print(best_score)