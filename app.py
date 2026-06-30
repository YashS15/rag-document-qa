import json
import logging
import os
from pathlib import Path

import requests

import uvicorn
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Request,
)
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from werkzeug.utils import secure_filename

from rag.document_processor import process_pdf
from rag.embeddings import embed_query, embed_texts, get_model
from rag.llm import stream_answer, summarize_document
from rag.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_FOLDER = "uploads"
MAX_CONTENT_LENGTH = 50 * 1024 * 1024

templates = Jinja2Templates(directory="templates")

Path("uploads").mkdir(exist_ok=True)

# Pre-load the embedding model at startup so the first upload doesn't trigger
# a slow Hugging Face download mid-request (which can cause a gunicorn timeout)
try:
    logger.info("Loading embedding model...")
    get_model()
    logger.info("Embedding model ready.")
except Exception as exc:
    logger.warning("Could not pre-load embedding model: %s", exc)

store = VectorStore()

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc):
    logger.exception(exc)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )

GROQ_MODEL_OPTIONS = [
    {"id": "llama3.1-8b", "name": "Llama 3.1 8B — Fast"},
    {"id": "llama3.3-70b", "name": "Llama 3.3 70B — Powerful"},
    {"id": "gemma2-9b", "name": "Gemma 2 9B — Google"},
]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/documents")
async def list_documents():
    return store.list_documents()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    with open(filepath, "wb") as f:
        f.write(await file.read())

    try:
        doc_data = process_pdf(filepath, filename)

        if not doc_data["chunks"]:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from this PDF",
            )

        texts = [c["text"] for c in doc_data["chunks"]]
        embeddings = embed_texts(texts)

        doc_meta = {
            k: v
            for k, v in doc_data.items()
            if k != "chunks"
        }

        store.add_document(
            doc_meta,
            doc_data["chunks"],
            embeddings,
        )

        summary = summarize_document(doc_data["chunks"])

        return {
            "doc_id": doc_data["doc_id"],
            "name": filename,
            "page_count": doc_data["page_count"],
            "chunk_count": doc_data["chunk_count"],
            "summary": summary,
        }

    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

class AskRequest(BaseModel):
    question: str
    model: str = "llama3.1-8b"
    k: int = 5
    history: list = []
    doc_id: str | None = None

@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    question = req.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="No question provided",
        )

    model = req.model
    k = req.k
    history = req.history
    doc_id = req.doc_id

    if not store.documents:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Please upload a PDF first.",
        )

    query_emb = embed_query(question)
    relevant_chunks = store.search(
        query_emb,
        k=k,
        doc_id=doc_id,
    )

    if not relevant_chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found in the indexed documents.",
        )

    sources = [
        {
            "doc_name": c["doc_name"],
            "page": c["page"],
            "text": (
                c["text"][:300] + "…"
                if len(c["text"]) > 300
                else c["text"]
            ),
            "score": round(c["score"], 4),
        }
        for c in relevant_chunks
    ]

    async def generate():
        yield (
            f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        )

        try:
            for token in stream_answer(
                question,
                relevant_chunks,
                model,
                history,
            ):
                yield (
                    f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
                )

        except Exception as e:
            yield (
                f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            )

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    if store.delete_document(doc_id):
        return {"message": "Document removed"}

    raise HTTPException(
        status_code=404,
        detail="Document not found",
    )

@app.get("/models")
async def list_models():
    if os.environ.get("GROQ_API_KEY"):
        return {
            "models": GROQ_MODEL_OPTIONS,
            "provider": "groq",
        }

    try:
        resp = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5,
        )
        resp.raise_for_status()

        ollama_models = [
            {
                "id": model["name"],
                "name": model["name"],
            }
            for model in resp.json().get("models", [])
        ]

        return {
            "models": ollama_models
            or [
                {
                    "id": "llama3.2",
                    "name": "llama3.2",
                }
            ],
            "provider": "ollama",
        }

    except Exception:
        return {
            "models": [
                {
                    "id": "llama3.2",
                    "name": "llama3.2",
                }
            ],
            "provider": "ollama",
        }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
