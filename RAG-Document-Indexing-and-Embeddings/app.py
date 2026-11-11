
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import tempfile
import os
import traceback

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

EMBEDDING_MODEL = "nomic-embed-text"

# Local Ollama embedding model
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

# In-memory vector store for this running application
vector_store = None

# Store indexing information
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
        # Save PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            uploaded_file.save(temp_file.name)
            temp_path = temp_file.name

        # Load PDF and extract text
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

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise ValueError("Unable to create document chunks.")

        # Generate embeddings and create FAISS vector store
        new_vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        # Save in memory only after successful indexing
        vector_store = new_vector_store

        # Calculate embedding dimension
        sample_embedding = embeddings.embed_query(
            "Embedding dimension test"
        )

        index_info = {
            "filename": uploaded_file.filename,
            "pages": len(documents),
            "chunks": len(chunks),
            "characters": len(extracted_text),
            "embedding_dimension": len(sample_embedding),
            "embedding_model": EMBEDDING_MODEL,
            "vector_count": new_vector_store.index.ntotal,
            "chunk_size": 500,
            "chunk_overlap": 50
        }

        # Preview first few chunks
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


@app.route("/api/search", methods=["POST"])
def inspect_search():
    global vector_store

    if vector_store is None:
        return jsonify({
            "success": False,
            "error": "Please index a PDF before testing semantic search."
        }), 400

    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({
            "success": False,
            "error": "Enter a search query."
        }), 400

    try:
        # Retrieve the three most relevant chunks
        results = vector_store.similarity_search_with_score(
            query,
            k=min(3, vector_store.index.ntotal)
        )

        matches = []

        for document, score in results:
            matches.append({
                "page": document.metadata.get("page", 0) + 1,
                "text": document.page_content,
                "distance": round(float(score), 4)
            })

        return jsonify({
            "success": True,
            "query": query,
            "matches": matches
        })

    except Exception as error:
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
