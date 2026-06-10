from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.rag.document_loader import load_documents
from src.rag.text_chunker import chunk_text

from dotenv import load_dotenv
from google import genai
import os

# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

print("API Key:", api_key[:10])

client = genai.Client(
    api_key=api_key
)


def create_vector_store():

    documents = load_documents("data")

    chunks = []

    for document in documents:
        chunks.extend(chunk_text(document["content"]))

    print(f"\nTotal Chunks Created: {len(chunks)}")

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

    print("Vector Store Created Successfully")

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

    print(f"\nBest Similarity Score: {best_score:.4f}")

    if best_score < 0.3:
        return "I could not find relevant information in the knowledge base."

    return best_chunk


# Create Vector Store Once
vector_store, model = create_vector_store()

while True:

    question = input("\nAsk a question (type exit to quit): ")

    if question.lower() == "exit":
        break

    context = retrieve(question, vector_store, model)

    print("\nRetrieved Context:")
    print(context[:1000])

    try:

        prompt = f"""
Answer ONLY using the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context, say:
'I could not find the answer in the knowledge base.'
"""

        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=prompt
        )

        print("\nAnswer:")
        print(response.text)

    except Exception as e:

        print(f"\nGemini Error: {e}")