import json
import os
from pathlib import Path

import requests
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from werkzeug.utils import secure_filename

from rag.document_processor import process_pdf
from rag.embeddings import embed_query, embed_texts
from rag.llm import stream_answer, summarize_document
from rag.vector_store import VectorStore

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

Path("uploads").mkdir(exist_ok=True)

store = VectorStore()

GROQ_MODEL_OPTIONS = [
    {"id": "llama3.1-8b", "name": "Llama 3.1 8B — Fast"},
    {"id": "llama3.3-70b", "name": "Llama 3.3 70B — Powerful"},
    {"id": "gemma2-9b", "name": "Gemma 2 9B — Google"},
]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


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

        summary = summarize_document(doc_data["chunks"])

        return jsonify({
            "doc_id": doc_data["doc_id"],
            "name": filename,
            "page_count": doc_data["page_count"],
            "chunk_count": doc_data["chunk_count"],
            "summary": summary,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/ask/stream", methods=["POST"])
def ask_stream():
    data = request.get_json()
    if not data or not data.get("question", "").strip():
        return jsonify({"error": "No question provided"}), 400

    question = data["question"].strip()
    model = data.get("model", "llama3.1-8b")
    k = int(data.get("k", 5))
    history = data.get("history", [])
    doc_id = data.get("doc_id")  # optional: restrict search to one document

    if not store.documents:
        return jsonify({"error": "No documents indexed yet. Please upload a PDF first."}), 400

    query_emb = embed_query(question)
    relevant_chunks = store.search(query_emb, k=k, doc_id=doc_id or None)

    if not relevant_chunks:
        return jsonify({"error": "No relevant content found in the indexed documents."}), 404

    sources = [
        {
            "doc_name": c["doc_name"],
            "page": c["page"],
            "text": c["text"][:300] + "…" if len(c["text"]) > 300 else c["text"],
            "score": round(c["score"], 4),
        }
        for c in relevant_chunks
    ]

    def generate():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        try:
            for token in stream_answer(question, relevant_chunks, model, history):
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.route("/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    if store.delete_document(doc_id):
        return jsonify({"message": "Document removed"})
    return jsonify({"error": "Document not found"}), 404


@app.route("/models", methods=["GET"])
def list_models():
    if os.environ.get("GROQ_API_KEY"):
        return jsonify({"models": GROQ_MODEL_OPTIONS, "provider": "groq"})
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        ollama_models = [
            {"id": m["name"], "name": m["name"]}
            for m in resp.json().get("models", [])
        ]
        return jsonify({"models": ollama_models or [{"id": "llama3.2", "name": "llama3.2"}], "provider": "ollama"})
    except Exception:
        return jsonify({"models": [{"id": "llama3.2", "name": "llama3.2"}], "provider": "ollama"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = not os.environ.get("GROQ_API_KEY")
    app.run(host="0.0.0.0", port=port, debug=debug)
