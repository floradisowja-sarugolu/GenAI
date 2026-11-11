
import requests
import numpy as np
import faiss

from io import BytesIO
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# Ollama configuration
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.2:3b"


# Load embedding model
@__import__("functools").lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# Extract text from uploaded documents
def extract_text(files):

    documents = []

    for file in files:

        filename = file.name.lower()

        if filename.endswith(".pdf"):

            pdf = PdfReader(BytesIO(file.getvalue()))

            text = ""

            for page in pdf.pages:
                text += page.extract_text() or ""

        elif filename.endswith(".txt"):

            text = file.getvalue().decode(
                "utf-8", errors="ignore"
            )

        else:
            continue

        if text.strip():
            documents.append({
                "name": file.name,
                "text": text
            })

    return documents


# Split document text into chunks
def split_text(documents, chunk_size=800, overlap=150):

    chunks = []

    for document in documents:

        text = document["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append({
                    "text": chunk,
                    "source": document["name"]
                })

            start += chunk_size - overlap

    return chunks


# Create FAISS vector index
def create_vector_store(chunks):

    model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings, dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# Retrieve relevant chunks
def retrieve_documents(question, chunks, index, top_k=3):

    model = get_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding, dtype="float32"
    )

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx >= 0:
            results.append({
                "text": chunks[idx]["text"],
                "source": chunks[idx]["source"],
                "score": float(score)
            })

    return results


# Generate answer using Ollama
def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(
        [
            f"Source: {item['source']}\n{item['text']}"
            for item in retrieved_chunks
        ]
    )

    prompt = f"""
You are a helpful AI document assistant.

Answer the user's question using the supplied document context.

Instructions:
1. Give clear and understandable answers.
2. Use the document context as your primary source.
3. Do not invent information.
4. If the answer is not available in the documents,
   clearly say that the uploaded documents do not contain
   enough information.
5. Use headings and bullet points when appropriate.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    return response.json()["message"]["content"]
