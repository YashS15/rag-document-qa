import os
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from rag.document_processor import process_pdf
from rag.embeddings import embed_query, embed_texts
from rag.llm import generate_answer
from rag.vector_store import VectorStore

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

Path("uploads").mkdir(exist_ok=True)

store = VectorStore()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/documents", methods=["GET"])
def list_documents():
    return jsonify(store.list_documents())


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        doc_data = process_pdf(filepath, filename)
        if not doc_data["chunks"]:
            return jsonify({"error": "Could not extract text from this PDF"}), 400

        texts = [c["text"] for c in doc_data["chunks"]]
        embeddings = embed_texts(texts)

        doc_meta = {k: v for k, v in doc_data.items() if k != "chunks"}
        store.add_document(doc_meta, doc_data["chunks"], embeddings)

        return jsonify({
            "doc_id": doc_data["doc_id"],
            "name": filename,
            "page_count": doc_data["page_count"],
            "chunk_count": doc_data["chunk_count"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or not data.get("question", "").strip():
        return jsonify({"error": "No question provided"}), 400

    question = data["question"].strip()
    model = data.get("model", "llama3.2")
    k = int(data.get("k", 5))

    if store.index.ntotal == 0:
        return jsonify({"error": "No documents indexed yet. Please upload a PDF first."}), 400

    query_emb = embed_query(question)
    relevant_chunks = store.search(query_emb, k=k)

    if not relevant_chunks:
        return jsonify({"error": "No relevant content found in the indexed documents."}), 404

    answer = generate_answer(question, relevant_chunks, model=model)

    return jsonify({
        "answer": answer,
        "sources": [
            {
                "doc_name": c["doc_name"],
                "page": c["page"],
                "text": c["text"][:300] + "…" if len(c["text"]) > 300 else c["text"],
                "score": round(c["score"], 4),
            }
            for c in relevant_chunks
        ],
    })


@app.route("/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    if store.delete_document(doc_id):
        return jsonify({"message": "Document removed"})
    return jsonify({"error": "Document not found"}), 404


@app.route("/models", methods=["GET"])
def list_models():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        return jsonify({"models": models or ["llama3.2", "mistral", "llama2"]})
    except Exception:
        return jsonify({"models": ["llama3.2", "mistral", "llama2"]})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
