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


def retrieve(query, vector_store, model):

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

        if best_score < 0.5:
           return "I could not find relevant information in the knowledge base."

        return best_chunk


vector_store, model = create_vector_store()

while True:

    question = input("\nAsk a question (type exit to quit): ")

    if question.lower() == "exit":
        break

    answer = retrieve(question, vector_store, model)

    print("\nAnswer:")
    print(answer)