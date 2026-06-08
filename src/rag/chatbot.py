from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from document_loader import load_documents
from text_chunker import chunk_text

from dotenv import load_dotenv
from google import genai
import os

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


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

    if best_score < 0.3:
        return "I could not find relevant information in the knowledge base."

    return best_chunk


vector_store, model = create_vector_store()

while True:

    question = input("\nAsk a question (type exit to quit): ")

    if question.lower() == "exit":
        break

    context = retrieve(question, vector_store, model)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
You are a helpful college information assistant.

Use ONLY the provided context to answer the question.

Give a natural and concise answer.
Do not copy the context word-for-word.

Context:
{context}

Question:
{question}

If the answer is not available in the context, reply exactly:
I could not find relevant information in the documents.
"""
    )

    print("\nAnswer:")
    print(response.text)