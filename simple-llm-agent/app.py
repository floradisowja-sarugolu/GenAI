
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import os
import traceback

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

# Ollama models
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"

# Initialize local Ollama models
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0.2
)

# In-memory document index
vector_store = None
index_info = {}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/index", methods=["POST"])
def index_document():
    global vector_store, index_info

    if "file" not in request.files:
        return jsonify({
            "success": False,
            "error": "Please select a PDF document."
        }), 400

    uploaded_file = request.files["file"]

    if not uploaded_file.filename:
        return jsonify({
            "success": False,
            "error": "No file selected."
        }), 400

    if not uploaded_file.filename.lower().endswith(".pdf"):
        return jsonify({
            "success": False,
            "error": "Only PDF files are supported."
        }), 400

    temp_path = None

    try:
        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:
            uploaded_file.save(temp_file.name)
            temp_path = temp_file.name

        # Extract PDF content
        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        if not documents:
            raise ValueError("No pages were found in this PDF.")

        extracted_text = "\n".join(
            doc.page_content for doc in documents
        )

        if not extracted_text.strip():
            raise ValueError(
                "No readable text found. The PDF may contain scanned images."
            )

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise ValueError("Unable to create document chunks.")

        # Generate embeddings and build FAISS index
        new_vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        # Check embedding dimensions
        sample_embedding = embeddings.embed_query(
            "Embedding dimension test"
        )

        # Update the active index only after successful processing
        vector_store = new_vector_store

        index_info = {
            "filename": uploaded_file.filename,
            "pages": len(documents),
            "chunks": len(chunks),
            "characters": len(extracted_text),
            "embedding_dimension": len(sample_embedding),
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": LLM_MODEL,
            "vector_count": new_vector_store.index.ntotal,
            "chunk_size": 500,
            "chunk_overlap": 50
        }

        # Preview first five chunks
        chunk_previews = []

        for i, chunk in enumerate(chunks[:5]):
            chunk_previews.append({
                "number": i + 1,
                "page": chunk.metadata.get("page", 0) + 1,
                "text": chunk.page_content
            })

        return jsonify({
            "success": True,
            "message": "Document indexed successfully.",
            "info": index_info,
            "previews": chunk_previews
        })

    except Exception as error:
        print(traceback.format_exc())

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.route("/api/chat", methods=["POST"])
def chat():
    global vector_store

    if vector_store is None:
        return jsonify({
            "success": False,
            "error": "Please upload and index a PDF first."
        }), 400

    data = request.get_json(silent=True) or {}

    question = data.get("question", "").strip()
    history = data.get("history", [])

    if not question:
        return jsonify({
            "success": False,
            "error": "Please enter a question."
        }), 400

    try:
        # Retrieve the four most relevant document chunks
        retrieved_docs = vector_store.similarity_search(
            question,
            k=min(4, vector_store.index.ntotal)
        )

        if not retrieved_docs:
            return jsonify({
                "success": True,
                "answer": "Information not found in the document.",
                "sources": []
            })

        # Combine retrieved content
        context_parts = []
        sources = []

        for doc in retrieved_docs:
            page_number = doc.metadata.get("page", 0) + 1

            context_parts.append(
                f"[Page {page_number}]\n{doc.page_content}"
            )

            if page_number not in [s["page"] for s in sources]:
                sources.append({"page": page_number})

        context = "\n\n".join(context_parts)

        # Include recent conversation for follow-up questions
        conversation = ""

        if isinstance(history, list):
            for item in history[-6:]:
                if not isinstance(item, dict):
                    continue

                role = item.get("role", "")
                content = item.get("content", "")

                if role in ("user", "assistant") and isinstance(content, str):
                    conversation += (
                        f"{role.capitalize()}: {content[:1500]}\n"
                    )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a helpful PDF-based AI assistant.

Answer the user's question using ONLY the supplied document context.

Rules:
1. Do not use outside knowledge or make up information.
2. If the answer is not available in the context, respond exactly:
   Information not found in the document.
3. Use conversation history only to understand follow-up questions.
   All factual answers must still be supported by the document context.
4. Give clear, natural and useful answers.
5. Use bullet points or numbered steps when appropriate.
6. Do not mention information that is not supported by the document.

DOCUMENT CONTEXT:
{context}

RECENT CONVERSATION:
{conversation}"""
            ),
            (
                "human",
                "{question}"
            )
        ])

        # Generate answer using Ollama
        response = (prompt | llm).invoke({
            "context": context,
            "conversation": conversation or "No previous conversation.",
            "question": question
        })

        answer = response.content

        return jsonify({
            "success": True,
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(retrieved_docs)
        })

    except Exception as error:
        print(traceback.format_exc())

        return jsonify({
            "success": False,
            "error": str(error)
        }), 500


@app.route("/api/status")
def status():
    return jsonify({
        "indexed": vector_store is not None,
        "info": index_info if vector_store is not None else {}
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
