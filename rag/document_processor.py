import pdfplumber
import uuid
from datetime import datetime


def extract_text_from_pdf(filepath: str) -> list[dict]:
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": i + 1, "text": text.strip()})
    return pages


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def process_pdf(filepath: str, filename: str) -> dict:
    doc_id = str(uuid.uuid4())
    pages = extract_text_from_pdf(filepath)

    chunks = []
    for page_data in pages:
        for chunk in chunk_text(page_data["text"]):
            chunks.append({
                "id": str(uuid.uuid4()),
                "doc_id": doc_id,
                "doc_name": filename,
                "page": page_data["page"],
                "text": chunk,
            })

    return {
        "doc_id": doc_id,
        "name": filename,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "uploaded_at": datetime.now().isoformat(),
        "chunks": chunks,
    }
