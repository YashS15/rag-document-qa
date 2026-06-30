# RAG Document Q&A System — Full Technical Documentation

> A production-grade Retrieval-Augmented Generation (RAG) system that lets users upload PDF documents and ask natural-language questions about their content. Answers are grounded in the actual document text and streamed in real time.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Technology Stack](#3-technology-stack)
4. [RAG Pipeline — How It Works](#4-rag-pipeline--how-it-works)
5. [Component Deep Dives](#5-component-deep-dives)
   - 5.1 Document Processor
   - 5.2 Embedding Engine
   - 5.3 Vector Store (FAISS)
   - 5.4 LLM Integration (Groq / Ollama)
   - 5.5 Flask Backend
   - 5.6 Frontend (HTML / CSS / JS)
6. [API Reference](#6-api-reference)
7. [Streaming (SSE)](#7-streaming-sse)
8. [Multi-Turn Conversation Memory](#8-multi-turn-conversation-memory)
9. [Deployment](#9-deployment)
10. [Docker Setup](#10-docker-setup)
11. [Interview Questions & Answers](#11-interview-questions--answers)

---

## 1. Project Overview

### What It Does

Users upload one or more PDF files. The system:
1. Extracts and chunks the text
2. Converts chunks into dense vector embeddings
3. Stores embeddings in a FAISS vector index persisted to disk
4. On a user query, finds the most semantically similar chunks
5. Feeds those chunks as context to an LLM (Groq or Ollama)
6. Streams the answer back token-by-token via Server-Sent Events (SSE)

### Key Features

| Feature | Description |
|---|---|
| PDF ingestion | Drag-and-drop upload, text extraction with pdfplumber |
| Semantic search | FAISS cosine similarity over fastembed vectors |
| Streaming answers | SSE — tokens appear as they are generated |
| Multi-turn memory | Last 6 Q&A turns sent as chat history |
| Document filter | Restrict search to one specific document |
| Upload summary | LLM auto-summarises each PDF on upload |
| New conversation | One-click to clear chat and reset history |
| Export chat | Downloads full Q&A session as Markdown |
| Copy button | One-click clipboard copy per answer |
| Docker support | Containerised with Ollama bundled |
| Cloud deployment | Render.com with Groq API (free tier) |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          BROWSER (Client)                           │
│  ┌──────────────┐   ┌──────────────────────────────────────────┐    │
│  │  Sidebar     │   │  Chat Panel                              │    │
│  │  - Upload    │   │  - Streaming SSE consumer                │    │
│  │  - Doc list  │   │  - Conversation history (JS array)       │    │
│  │  - Model     │   │  - Copy / Export / New Chat              │    │
│  │  - Filter    │   │                                          │    │
│  └──────┬───────┘   └──────────────────┬───────────────────────┘    │
└─────────┼────────────────────────────── ┼ ──────────────────────────┘
          │ POST /upload                  │ POST /ask/stream (SSE)
          ▼                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FLASK BACKEND (app.py)                      │
│                                                                     │
│  /upload ──► document_processor ──► embeddings ──► vector_store    │
│                                                         │           │
│  /ask/stream ──► embed_query ──► vector_store.search    │           │
│                      │                 │                │           │
│                      │         relevant chunks          │           │
│                      │                 │                │           │
│                      └────────► llm.stream_answer ──► SSE stream   │
└─────────────────────────────────────────────────────────────────────┘
          │                               │
          ▼                               ▼
┌─────────────────┐             ┌──────────────────────┐
│  fastembed      │             │  Groq API (cloud)    │
│  ONNX Runtime   │             │  or Ollama (local)   │
│  all-MiniLM-L6  │             │  llama / gemma       │
│  384-dim vecs   │             │                      │
└─────────────────┘             └──────────────────────┘
          │
          ▼
┌─────────────────┐
│  FAISS Index    │
│  (disk-backed)  │
│  IndexFlatIP    │
│  vectorstore/   │
└─────────────────┘
```

### Request Flow for a Question

```
User types question
       │
       ▼
JS sends POST /ask/stream  {question, model, history, doc_id}
       │
       ▼
Flask: embed_query(question)  →  384-dim float32 vector
       │
       ▼
FAISS: IndexFlatIP.search(query_vec, k=15)  →  top-N candidate indices
       │
       ▼
Filter: skip deleted docs, skip non-matching doc_id
       →  top-5 relevant chunks
       │
       ▼
Build messages array:
  [ {system: context}, ...history, {user: question} ]
       │
       ▼
LLM (Groq or Ollama) streaming API
       │
       ▼
Flask yields SSE events:
  data: {"type":"sources", ...}
  data: {"type":"token", "token":"The"}
  data: {"type":"token", "token":" answer"}
  ...
  data: [DONE]
       │
       ▼
JS ReadableStream reader appends tokens to bubble in real time
```

---

## 3. Technology Stack

### Backend

| Library | Version | Purpose |
|---|---|---|
| **Flask** | 3.x | Web framework — routing, request/response, SSE |
| **gunicorn** | 21.x | Production WSGI server (replaces Flask dev server) |
| **pdfplumber** | 0.11.x | PDF text extraction per page (handles layout better than PyPDF2) |
| **fastembed** | 0.8.x | ONNX-based embedding model — no PyTorch, low RAM |
| **faiss-cpu** | 1.8.x | Facebook AI Similarity Search — cosine similarity at scale |
| **requests** | 2.x | HTTP client for Groq API and Ollama REST calls |
| **numpy** | 2.x | Float32 array operations for embedding vectors |
| **werkzeug** | 3.x | `secure_filename` for safe file upload handling |

### Frontend

| Technology | Purpose |
|---|---|
| Vanilla HTML5 | Semantic structure — no framework overhead |
| CSS3 (custom) | Flexbox layout, CSS variables for theming, animations |
| Vanilla JavaScript (ES2022) | Fetch API, ReadableStream, async/await, EventSource-style SSE |

### AI / ML

| Component | Tool | Why |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed | 384-dim, ONNX runtime (~100MB RAM vs ~500MB PyTorch) |
| Vector DB | FAISS `IndexFlatIP` | Exact cosine similarity, disk-persistent, no server needed |
| LLM (cloud) | Groq API — `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `gemma2-9b-it` | Free tier, fastest inference API available (~500 tok/s) |
| LLM (local) | Ollama — `llama3.2`, `mistral` | Fully offline, no API key needed |

### DevOps

| Tool | Purpose |
|---|---|
| Docker | Container packaging the Flask app |
| docker-compose | Orchestrates app + Ollama together locally |
| Render.com | Cloud PaaS — auto-deploys on push to main |
| GitHub | Source control + PR workflow |

---

## 4. RAG Pipeline — How It Works

RAG stands for **Retrieval-Augmented Generation**. The core idea is: instead of asking the LLM to answer from its training data (which may be stale or hallucinated), you first *retrieve* relevant text from your own documents, then *augment* the LLM's prompt with that text, and let it *generate* a grounded answer.

### Phase 1 — Indexing (happens on upload)

```
PDF File
   │
   ▼ pdfplumber
Raw text per page: [{page:1, text:"..."}, {page:2, text:"..."}, ...]
   │
   ▼ chunk_text(size=1000, overlap=200)
Overlapping chunks: ["first 1000 chars", "chars 800-1800", "chars 1600-2600", ...]
   │
   ▼ fastembed.TextEmbedding.embed()
384-dimensional float32 vectors, L2-normalised
   │
   ▼ faiss.IndexFlatIP.add()
Stored in FAISS index (in-memory + written to vectorstore/index.faiss)
   │
   ▼ vectorstore/metadata.json
Chunk metadata saved: {id, doc_id, doc_name, page, text, faiss_idx}
```

**Why chunking?**
- LLMs have context windows (e.g., 8192 tokens). A 100-page PDF can be 100,000 tokens — too large.
- Chunking splits text into manageable pieces so only *relevant* pieces are sent.
- Overlap (200 chars) prevents information loss at chunk boundaries.

**Why normalised vectors + IndexFlatIP?**
- `IndexFlatIP` computes the inner product (dot product) between vectors.
- When both vectors are L2-normalised (unit length), inner product = cosine similarity.
- Cosine similarity is direction-only — a short chunk and a long chunk about the same topic score equally.

### Phase 2 — Retrieval (happens on each query)

```
User question: "What are the side effects?"
   │
   ▼ fastembed.embed([question])
Query vector: [0.023, -0.147, 0.891, ...]  (384 floats)
   │
   ▼ FAISS.search(query_vec, k=15)
Top-15 candidates by cosine similarity + their FAISS indices
   │
   ▼ Filter
- Skip chunks whose doc_id is not in active documents dict
- Skip chunks not matching selected doc_id filter (if set)
→ Top-5 relevant chunks
```

### Phase 3 — Generation (happens after retrieval)

```
Build messages array:
[
  {role: "system", content: "Answer using this context:\n[chunk1]\n[chunk2]..."},
  {role: "user",   content: "previous question 1"},      ← history
  {role: "assistant", content: "previous answer 1"},     ← history
  {role: "user",   content: "What are the side effects?"}  ← current
]
   │
   ▼ Groq streaming API (or Ollama /api/chat)
Token stream: "The", " side", " effects", " include", ...
   │
   ▼ SSE events to browser
Tokens appended to bubble in real time
```

---

## 5. Component Deep Dives

### 5.1 Document Processor (`rag/document_processor.py`)

**pdfplumber** opens the PDF and iterates pages. It handles multi-column layouts, tables, and headers better than PyPDF2 because it uses PDFMiner under the hood and preserves spatial information.

```python
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap   # slide back by overlap amount
    return chunks
```

**Sliding window chunking**: each chunk starts `overlap` characters before where the previous chunk ended. This ensures a sentence split across a boundary still appears in full in at least one chunk.

Each chunk is stored with its metadata:
```python
{
  "id": "uuid4",
  "doc_id": "uuid4",        # parent document
  "doc_name": "paper.pdf",
  "page": 3,                # original page number
  "text": "...",
  "faiss_idx": 142          # position in FAISS index
}
```

### 5.2 Embedding Engine (`rag/embeddings.py`)

**fastembed** is a lightweight library that runs ONNX-exported embedding models. The model used is `BAAI/bge-small-en-v1.5`:

- **384 dimensions** — same as `all-MiniLM-L6-v2`
- **ONNX Runtime** — no PyTorch, no CUDA required
- **RAM** — ~100MB at runtime vs ~450MB for sentence-transformers + PyTorch
- **Speed** — ~3000 sentences/sec on CPU

```python
_model = None  # lazy singleton — loaded only on first call

def get_model():
    global _model
    if _model is None:
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _model

def embed_texts(texts):
    embeddings = list(get_model().embed(texts))   # returns a generator
    return np.array(embeddings, dtype=np.float32)  # shape: (N, 384)
```

The model is a **singleton** — loaded once at first use and reused for every subsequent call. Loading takes ~2-3 seconds (downloads ONNX model on first use).

### 5.3 Vector Store (`rag/vector_store.py`)

**FAISS** (Facebook AI Similarity Search) is a C++ library for efficient similarity search, wrapped in Python. It supports billions of vectors and multiple index types.

**Index type used: `IndexFlatIP`**
- `Flat` = brute-force exact search (no approximation)
- `IP` = Inner Product
- With normalised vectors = cosine similarity
- O(N) per query, but with 384-dim vectors, handles ~1M chunks in < 1 second

**Persistence design:**
```
vectorstore/
  index.faiss      ← binary FAISS index (vectors only)
  metadata.json    ← chunk metadata + document registry
```

FAISS stores only the raw float32 matrix. The metadata (page, filename, text) lives in `metadata.json`. The connection: `chunks[faiss_idx]` maps every FAISS position to its metadata.

**Soft delete design:**
FAISS does not support per-vector deletion. Instead:
1. `documents` dict acts as an allowlist of active doc_ids
2. When a document is deleted, its `doc_id` is removed from `documents`
3. During search, chunks whose `doc_id` is not in `documents` are silently skipped
4. FAISS index is never rebuilt — deleted vectors are simply invisible

**Trade-off**: over time the FAISS index accumulates "ghost" vectors. For a demo this is fine; production would compact periodically.

**Document filter (new feature):**
```python
def search(self, query_embedding, k=5, doc_id=None):
    ...
    for score, idx in zip(scores[0], indices[0]):
        chunk = self.chunks[idx]
        if chunk["doc_id"] not in self.documents:
            continue          # deleted
        if doc_id and chunk["doc_id"] != doc_id:
            continue          # filtered out
        results.append(...)
```

### 5.4 LLM Integration (`rag/llm.py`)

The LLM layer supports two backends transparently:

**Groq API** — used when `GROQ_API_KEY` env var is set (cloud deployment):
- OpenAI-compatible REST endpoint: `POST /openai/v1/chat/completions`
- Accepts `messages` array (system + user + assistant turns)
- `stream: true` returns SSE — `data: {"choices":[{"delta":{"content":"token"}}]}`
- Model map: old Ollama model names → valid Groq model IDs

**Ollama** — used when no API key (local dev):
- `POST /api/chat` with same `messages` format
- Returns NDJSON stream: `{"message":{"content":"token"},"done":false}`

Both backends use the same **chat-format messages** for multi-turn support:
```python
def _build_messages(question, context_chunks, history):
    context = format_context(context_chunks)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)}
    ]
    for turn in (history or []):
        messages.append({"role": "user",      "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": question})
    return messages
```

Using the `messages` format (instead of a single concatenated string) is more reliable because:
- The LLM's attention mechanism is tuned for the role-based format
- It properly separates context (system) from conversation (user/assistant)
- It works identically for both Groq and Ollama

**Groq Model Fix:**
`mixtral-8x7b-32768` was deprecated by Groq in early 2025. The updated mapping:

```python
GROQ_MODELS = {
    "llama3.1-8b":  "llama-3.1-8b-instant",       # ~500 tok/s, free
    "llama3.3-70b": "llama-3.3-70b-versatile",    # best quality, free
    "gemma2-9b":    "gemma2-9b-it",                # Google's model, free
}
```

### 5.5 Flask Backend (`app.py`)

Flask is a micro web framework. The app defines HTTP routes using decorators:

```python
app = Flask(__name__)

@app.route("/ask/stream", methods=["POST"])
def ask_stream():
    ...
```

**Key design decisions:**

1. **Module-level singleton**: `store = VectorStore()` is created once at startup. All requests share the same in-memory FAISS index. This is why `--workers 1` is used in gunicorn — multiple workers would each have their own index and see different state.

2. **File cleanup**: uploaded PDFs are deleted after indexing:
   ```python
   finally:
       if os.path.exists(filepath):
           os.remove(filepath)
   ```
   The raw PDF is never stored permanently; only extracted text chunks are kept.

3. **`stream_with_context`**: Flask's `stream_with_context` ensures the request context remains available while the generator runs (needed for streaming responses).

4. **SSE headers**: 
   ```python
   headers={
       "Cache-Control": "no-cache",
       "X-Accel-Buffering": "no",   # disables Nginx buffering
       "Connection": "keep-alive",
   }
   ```

### 5.6 Frontend (HTML / CSS / JS)

**Layout** — CSS Flexbox split:
```
.app (display: flex)
├── .sidebar (width: 320px, flex-shrink: 0)
│   ├── upload section
│   ├── model selector
│   ├── document filter
│   └── document list
└── .chat-area (flex: 1)
    ├── .chat-header
    ├── .messages (flex: 1, overflow-y: auto)
    └── .chat-input-area
```

**CSS Variables** for consistent theming:
```css
:root {
    --primary: #6366f1;      /* indigo */
    --sidebar-bg: #1e1e2e;   /* dark navy */
    --card: #ffffff;
    --border: #e2e8f0;
}
```

**Streaming in JavaScript:**

`EventSource` only supports GET requests. For a POST-based SSE stream, we use `fetch()` with a `ReadableStream` reader:

```javascript
const resp = await fetch('/ask/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, model, history, doc_id })
});

const reader = resp.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();   // keep incomplete line

    for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6);
        if (raw === '[DONE]') return;
        const parsed = JSON.parse(raw);
        if (parsed.type === 'token') {
            fullAnswer += parsed.token;
            bubbleEl.textContent = fullAnswer;   // live update
        }
    }
}
```

**Conversation history** stored in a JS array, capped at 6 turns:
```javascript
let conversationHistory = [];

// after answer completes:
conversationHistory.push({ question, answer: fullAnswer });
if (conversationHistory.length > 6) conversationHistory.shift();
```

---

## 6. API Reference

### `GET /`
Returns the main HTML page.

---

### `GET /health`
Health check for Render's load balancer.

**Response:**
```json
{ "status": "ok" }
```

---

### `GET /documents`
List all indexed documents.

**Response:**
```json
[
  {
    "doc_id": "uuid",
    "name": "report.pdf",
    "page_count": 12,
    "chunk_count": 47,
    "uploaded_at": "2025-01-15T10:30:00"
  }
]
```

---

### `POST /upload`
Upload and index a PDF.

**Request:** `multipart/form-data` with field `file`.

**Response:**
```json
{
  "doc_id": "uuid",
  "name": "report.pdf",
  "page_count": 12,
  "chunk_count": 47,
  "summary": "This report analyses Q3 sales data across 5 regions..."
}
```

**Errors:** 400 if no file, not a PDF, or no extractable text. 500 on internal error.

---

### `POST /ask/stream`
Ask a question. Returns an SSE stream.

**Request body:**
```json
{
  "question": "What are the main findings?",
  "model": "llama3.1-8b",
  "k": 5,
  "history": [
    { "question": "Who wrote this?", "answer": "Dr. Smith." }
  ],
  "doc_id": "uuid-or-null"
}
```

**Response:** `text/event-stream`
```
data: {"type":"sources","sources":[{"doc_name":"report.pdf","page":3,"text":"...","score":0.87}]}

data: {"type":"token","token":"The"}

data: {"type":"token","token":" main"}

data: {"type":"token","token":" findings"}

data: [DONE]
```

---

### `DELETE /documents/<doc_id>`
Remove a document from the index.

**Response:** `{"message": "Document removed"}`

---

### `GET /models`
List available LLM models.

**Response (Groq):**
```json
{
  "provider": "groq",
  "models": [
    { "id": "llama3.1-8b",  "name": "Llama 3.1 8B — Fast" },
    { "id": "llama3.3-70b", "name": "Llama 3.3 70B — Powerful" },
    { "id": "gemma2-9b",    "name": "Gemma 2 9B — Google" }
  ]
}
```

---

## 7. Streaming (SSE)

**Server-Sent Events (SSE)** is an HTTP protocol where the server keeps a connection open and pushes data in a standard format:

```
Content-Type: text/event-stream

data: first event\n\n
data: second event\n\n
data: [DONE]\n\n
```

Each event starts with `data: ` and ends with two newlines. The `[DONE]` sentinel signals the end of the stream.

**Why SSE instead of WebSockets?**
- SSE is unidirectional (server → client) which is exactly what we need
- Works over standard HTTP/1.1 — no protocol upgrade
- Automatically reconnects on network failure
- Simpler to implement and debug

**Why it makes answers feel faster:**
A 200-token answer at 100 tok/s takes 2 seconds. Without streaming, the user stares at a spinner for 2 seconds then sees the full answer. With streaming, they read the answer as it's written — the *perceived* latency drops to near zero.

**Flask generator pattern:**
```python
def generate():
    yield f"data: {json.dumps({'type': 'sources', ...})}\n\n"
    for token in stream_answer(...):
        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"
    yield "data: [DONE]\n\n"

return Response(stream_with_context(generate()), mimetype="text/event-stream")
```

---

## 8. Multi-Turn Conversation Memory

The system maintains **short-term memory** across a conversation session. The last 6 Q&A turns are stored in a JavaScript array and sent with every new question:

**Why this matters:**
Without memory, every question is independent. The user cannot ask "Can you elaborate on that?" or "What about the third point?" because the LLM has no context of the previous exchange.

**Implementation:**
```
Turn 1: Q:"What is the study about?"  A:"It studies..."
Turn 2: Q:"Who conducted it?"         A:"Dr. Smith..."
Turn 3: Q:"What were his findings?"   ← LLM knows "his" = Dr. Smith
```

The messages array sent to the LLM:
```python
[
  {"role": "system",    "content": "Context: [relevant chunks]"},
  {"role": "user",      "content": "What is the study about?"},
  {"role": "assistant", "content": "It studies..."},
  {"role": "user",      "content": "Who conducted it?"},
  {"role": "assistant", "content": "Dr. Smith..."},
  {"role": "user",      "content": "What were his findings?"},  ← current
]
```

**Why cap at 6 turns?**
LLMs have context windows (~8192 tokens for Llama 3.1 8B). Each Q&A turn can be 200-500 tokens. 6 turns ≈ 2000 tokens of history, leaving plenty of room for the context chunks and the answer. Going beyond risks hitting the context limit and causing errors.

---

## 9. Deployment

### Architecture on Render (Free Tier)

```
Internet
   │
   ▼
Render Web Service
  ├── Python 3.11 runtime
  ├── gunicorn (1 worker)
  ├── fastembed model downloaded on first request
  └── vectorstore/ (ephemeral — resets on restart)
       │
       ▼
  Groq API (external — paid separately, free tier)
```

**Constraints of Render free tier:**
- 512 MB RAM — enough with fastembed (no PyTorch), tight with sentence-transformers
- Service sleeps after 15 min inactivity — cold start takes ~30 seconds
- Ephemeral filesystem — vectorstore resets on every deploy/restart
- No persistent disk on free tier

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes (cloud) | Your Groq API key from console.groq.com |
| `PORT` | Auto-set by Render | Port to listen on |

### Build & Start Commands

```
Build:  pip install -r requirements.txt
Start:  gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
```

The `--timeout 120` gives the embedding model and LLM enough time to respond, especially on first request (model download).

---

## 10. Docker Setup

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Only system build tools needed (no CUDA, no GPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p uploads vectorstore

EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1"]
```

### docker-compose.yml

```yaml
version: '3.8'
services:
  app:
    build: .
    ports: ["5000:5000"]
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY:-}
    volumes:
      - vectorstore_data:/app/vectorstore   # persist index across restarts
    depends_on: [ollama]

  ollama:
    image: ollama/ollama
    ports: ["11434:11434"]
    volumes:
      - ollama_data:/root/.ollama           # persist downloaded models

volumes:
  vectorstore_data:
  ollama_data:
```

**Named volumes** ensure the FAISS index and Ollama model weights survive container restarts. Without them, every `docker compose up` would start fresh.

**Run:**
```bash
docker compose up --build
# Then pull a model inside ollama container:
docker compose exec ollama ollama pull llama3.2
```

---

## 11. Interview Questions & Answers

### Section A — RAG & AI Fundamentals

---

**Q1. What is RAG and why is it better than fine-tuning for document Q&A?**

**A:** RAG (Retrieval-Augmented Generation) combines a retrieval system with an LLM. Instead of asking the LLM to memorise all knowledge (which requires expensive fine-tuning and goes stale), RAG fetches relevant documents at query time and includes them in the prompt.

**Advantages over fine-tuning:**
- **No training required** — works immediately on new documents
- **Up-to-date** — add new PDFs anytime without retraining
- **Explainable** — you can show exactly which passages the answer came from
- **Cheaper** — fine-tuning large models costs thousands of dollars; RAG costs cents per query
- **Lower hallucination** — the LLM is grounded by real text, not its compressed memory

**When fine-tuning IS better:** when you want the model to learn a new *style* or *format* (e.g., always reply as a customer service agent) rather than new facts.

---

**Q2. Explain vector embeddings. What does a 384-dimensional vector represent?**

**A:** An embedding is a fixed-size numerical representation of text that captures its *semantic meaning*. Text with similar meaning has similar embeddings (close in vector space), regardless of exact wording.

A **384-dimensional vector** means each piece of text is represented as an array of 384 floating-point numbers. These 384 dimensions don't have human-interpretable meanings individually — they're learned latent features. Together they encode things like:
- Topic (medicine, finance, law)
- Sentiment (positive, negative)
- Entity types (people, places, dates)
- Syntactic structure

Example: "dog" and "puppy" will have very similar 384-dim vectors even though they share no characters. "dog" and "stock market" will be far apart.

**Why 384?** It's a balance — enough dimensions to capture rich semantics, small enough to be fast and memory-efficient. Models like `text-embedding-ada-002` use 1536 dims for higher accuracy but at 4× the memory and compute cost.

---

**Q3. What is cosine similarity and why is it preferred over Euclidean distance for embeddings?**

**A:** 

**Cosine similarity** measures the angle between two vectors:
```
cos(θ) = (A · B) / (|A| × |B|)
```
Range: -1 (opposite) to +1 (identical direction).

**Euclidean distance** measures the straight-line distance between two points:
```
d = sqrt(Σ(Ai - Bi)²)
```

**Why cosine for embeddings:**
1. **Magnitude invariance** — "dog" and "The dog is big and fluffy and runs fast" should be similar, but the longer text has a larger magnitude vector. Cosine ignores magnitude, Euclidean doesn't.
2. **Text length normalisation** — longer documents naturally have larger embedding magnitudes. Cosine prevents bias toward short texts.
3. **FAISS optimisation** — with L2-normalised vectors (unit length), inner product = cosine similarity, which FAISS computes very efficiently with `IndexFlatIP`.

---

**Q4. What is FAISS? How does IndexFlatIP work?**

**A:** FAISS (Facebook AI Similarity Search) is a C++ library (with Python bindings) optimised for searching among large collections of dense vectors.

`IndexFlatIP`:
- `Flat` = stores all vectors in a flat array, performs exact (brute-force) search
- `IP` = Inner Product — computes dot product between query and every stored vector
- Returns top-K vectors by highest score

**Complexity:** O(N × D) per query, where N = number of vectors, D = dimensions (384).

**Why exact search for this project?**
- Our dataset (a few PDFs = few thousand chunks) is small enough that brute-force is fast (<1ms)
- Approximate methods (FAISS HNSW, IVF) sacrifice accuracy for speed — only worth it at millions of vectors

**Alternatives:**
- `IndexHNSWFlat` — graph-based approximate search, O(log N), great for millions of vectors
- `IndexIVFFlat` — clusters vectors, searches only relevant clusters, good for billions
- Pinecone, Weaviate, Qdrant — hosted vector databases with additional features (metadata filtering, replication)

---

**Q5. What is chunking and how do you choose chunk size?**

**A:** Chunking splits a long document into smaller overlapping pieces that fit within the LLM's context window and can be retrieved individually.

**Choosing chunk size involves trade-offs:**

| Smaller chunks (200-500 chars) | Larger chunks (1000-2000 chars) |
|---|---|
| More precise retrieval | More context per chunk |
| Risk losing context across boundaries | Risk including irrelevant content |
| More chunks → slower indexing | Fewer chunks → faster indexing |
| Good for fact lookup | Good for reasoning over paragraphs |

**This project uses 1000 chars with 200 overlap** — a common middle ground. The overlap ensures a sentence straddling a boundary appears complete in at least one chunk.

**Advanced techniques:**
- **Semantic chunking** — split at sentence/paragraph boundaries, not fixed character counts
- **Hierarchical chunking** — store both small and large chunks; retrieve small, expand to large for context
- **Sliding window** — what we use; simple and effective

---

**Q6. How do you handle hallucinations in a RAG system?**

**A:** Hallucinations occur when the LLM generates confident-sounding text that isn't in the provided context.

**Mitigations used in this project:**
1. **System prompt instruction**: "Answer based solely on the context above. If the context lacks enough information, say so clearly."
2. **Source citations**: showing the user which chunks were used lets them verify the answer
3. **Temperature 0.3**: lower temperature = more deterministic, less creative (less likely to fabricate)
4. **Retrieval quality**: better embeddings + chunking = more relevant context = less need to guess

**Additional mitigations (production):**
- **Faithfulness scoring** — use a second LLM to check if the answer is supported by the retrieved chunks
- **Answer with quotes** — prompt the LLM to include direct quotes from the sources
- **Confidence threshold** — if cosine similarity of top chunk < 0.5, warn the user: "No highly relevant content found"
- **Reranking** — use a cross-encoder model to re-score retrieved chunks for relevance before sending to LLM

---

### Section B — System Design

---

**Q7. How does the FAISS vector store handle document deletion without rebuilding the index?**

**A:** FAISS `IndexFlatIP` has no built-in delete operation — once a vector is added at position N, it stays there. We use a **soft delete** pattern:

1. `documents` dict is the **allowlist** of active document IDs
2. `chunks` list mirrors FAISS positions and is **never shrunk**
3. On delete: remove `doc_id` from `documents`, leave chunks and FAISS untouched
4. On search: iterate FAISS results, check `chunk["doc_id"] not in self.documents`, skip if deleted

**Trade-off:** Over time, deleted vectors waste memory and slightly slow search (more candidates to check). Production systems compact the index periodically by rebuilding it without deleted vectors.

**Alternative:** FAISS `IDMap` + `remove_ids()` — but this requires tracking IDs separately and is complex. The soft-delete approach is simpler and sufficient for moderate scale.

---

**Q8. Why is gunicorn used instead of Flask's built-in server in production?**

**A:** Flask's built-in server (`flask run` or `app.run()`) is a development server:
- Single-threaded (one request at a time)
- Not designed for concurrent connections
- Has overhead from debug mode (auto-reloader, debugger)
- Not production-hardened

**Gunicorn** (Green Unicorn) is a battle-tested Python WSGI server:
- Manages multiple worker processes
- Handles signals properly (SIGTERM for graceful shutdown)
- Buffers slow clients
- Used in production by thousands of companies

**Why `--workers 1` for this app?**
The VectorStore is a module-level Python object — each gunicorn worker would have its own copy with its own in-memory FAISS index. Multiple workers = inconsistent state (one worker sees a newly uploaded doc, another doesn't). With 1 worker we avoid this. For true multi-worker scalability, move the vector store to a shared external service (Redis + Qdrant, etc.).

---

**Q9. Design the system to support 1000 concurrent users.**

**A:** The current architecture has several single-point bottlenecks:

**Current bottlenecks:**
1. Single gunicorn worker → one request at a time
2. In-memory FAISS index → not shared across processes
3. Groq API rate limits → free tier has request quotas

**Scaled architecture:**

```
Load Balancer (Nginx / Render)
       │
  ┌────┼────┐
  ▼    ▼    ▼
Worker Worker Worker   (gunicorn workers or k8s pods)
  │    │    │
  └────┼────┘
       │
  ┌────▼────────────────────┐
  │  Qdrant / Pinecone      │  ← shared vector DB (replaces FAISS)
  │  (shared, persistent)   │
  └─────────────────────────┘
       │
  ┌────▼────────────────────┐
  │  LLM API pool           │  ← multiple API keys, load-balanced
  └─────────────────────────┘
       │
  ┌────▼────────────────────┐
  │  Redis (session cache)  │  ← cache frequent query results
  └─────────────────────────┘
```

**Additionally:**
- Move document metadata to PostgreSQL (not a JSON file)
- Add a task queue (Celery + Redis) for PDF processing (async, non-blocking)
- CDN for static assets
- Rate limiting per user/IP

---

**Q10. How would you add user authentication to this system?**

**A:** The system currently has no auth — any user can upload/query/delete. Adding auth:

**Simple approach — JWT (JSON Web Tokens):**
1. Add `POST /auth/register` and `POST /auth/login` routes
2. On login, issue a signed JWT: `{"user_id": 123, "exp": ...}`
3. Client stores JWT in localStorage, sends it as `Authorization: Bearer <token>`
4. Flask middleware decodes and validates the token on protected routes

**Document isolation:**
```python
# each chunk needs a user_id field
{
  "doc_id": "uuid",
  "user_id": "user_123",   ← add this
  "text": "..."
}

# search filters by user_id
if chunk["user_id"] != current_user_id:
    continue
```

**More robust:** Use session cookies + database-backed sessions, or OAuth2 (Google login via Flask-Dance).

---

### Section C — Python & Flask

---

**Q11. What is `stream_with_context` in Flask and why is it needed?**

**A:** Flask uses a **request context** — a thread-local object that stores the current request's data (headers, cookies, form data). When you return a generator from a route, Flask tears down the request context before the generator finishes, causing errors if the generator tries to access `request`, `g`, or `current_app`.

`stream_with_context(generator)` wraps the generator to keep the request context alive for the entire duration of the stream.

```python
# Without: request context is gone when generator starts
return Response(generate(), mimetype="text/event-stream")  # ❌

# With: request context available throughout streaming
return Response(stream_with_context(generate()), mimetype="text/event-stream")  # ✓
```

---

**Q12. Explain Python generators and how they enable memory-efficient streaming.**

**A:** A **generator** is a function that uses `yield` instead of `return`. It produces values one at a time, pausing between each:

```python
def count():
    yield 1
    yield 2
    yield 3

for n in count():
    print(n)   # 1, 2, 3
```

**Memory advantage:** A generator never holds all its values in memory at once. For streaming 1000 tokens, a list would allocate all 1000 upfront. A generator produces one token, sends it to the client, then discards it.

**In this project:**
```python
def generate():                          # generator function
    yield f"data: {json.dumps(...)}\n\n"  # sources event
    for token in stream_answer(...):      # token by token
        yield f"data: {json.dumps({'type':'token','token':token})}\n\n"
    yield "data: [DONE]\n\n"
```

Flask's `Response` accepts a generator — it calls `next()` on it each time it's ready to send data, resulting in a true streaming HTTP response.

---

**Q13. What is the difference between `@app.route` and `app.add_url_rule` in Flask?**

**A:** They are equivalent — `@app.route` is syntactic sugar:

```python
# These are identical:
@app.route('/upload', methods=['POST'])
def upload(): ...

# Equivalent to:
def upload(): ...
app.add_url_rule('/upload', 'upload', upload, methods=['POST'])
```

`add_url_rule` is useful when you want to register routes dynamically (e.g., from a config file, or in a factory pattern).

---

**Q14. How does `secure_filename` protect against directory traversal attacks?**

**A:** Without sanitisation, an attacker could upload a file named `../../../../etc/passwd` or `../app.py`, overwriting sensitive files.

`werkzeug.utils.secure_filename` strips:
- Path separators (`/`, `\`)
- Leading dots (`../`)
- Special characters
- Null bytes

```python
secure_filename("../../etc/passwd")   # → "etc_passwd"
secure_filename("my file (1).pdf")    # → "my_file_1.pdf"
secure_filename("../app.py")          # → "app.py"
```

We also delete the file immediately after processing (`finally: os.remove(filepath)`) so even a sanitised but malicious file doesn't linger.

---

### Section D — JavaScript & Frontend

---

**Q15. Why use `fetch()` with `ReadableStream` instead of `EventSource` for SSE?**

**A:** The browser's `EventSource` API is built for SSE but only supports **GET** requests. Our `/ask/stream` endpoint needs to be a POST (to send a JSON body with the question, history, and model choice).

With `fetch()`:
```javascript
const resp = await fetch('/ask/stream', {
    method: 'POST',
    body: JSON.stringify({ question, history, model })
});
const reader = resp.body.getReader();   // ReadableStream
```

This gives us streaming POST support. The trade-off: `EventSource` handles reconnection automatically; with `fetch()` we handle errors manually.

---

**Q16. Explain the buffer accumulation pattern used in the SSE reader.**

**A:** Network data arrives in arbitrary-sized chunks. A single `reader.read()` call may return half a line, or multiple lines. We need to reassemble complete SSE lines before parsing:

```javascript
let buffer = '';

while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    
    const lines = buffer.split('\n');
    buffer = lines.pop();   // last element may be incomplete — keep in buffer
    
    for (const line of lines) {  // process only complete lines
        if (line.startsWith('data: ')) { ... }
    }
}
```

`lines.pop()` removes and saves the last element. If the chunk ended mid-line (e.g., `data: {"type":"to`), that incomplete string stays in `buffer` and gets prepended to the next chunk's data.

---

**Q17. What is XSS and how does the `esc()` function prevent it?**

**A:** **Cross-Site Scripting (XSS)** is when malicious JavaScript is injected into a page via user-supplied content. If a PDF contained `<script>alert('hacked')</script>` and we inserted it directly into innerHTML, the browser would execute it.

Our `esc()` function HTML-encodes dangerous characters:
```javascript
function esc(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')     // < → &lt; (not a tag anymore)
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
```

Result: `<script>` becomes `&lt;script&gt;` — displayed as text, never executed.

**We use `element.textContent = text` for streaming** (not innerHTML) — `textContent` never interprets HTML, so it's inherently XSS-safe.

---

### Section E — DevOps & Deployment

---

**Q18. What is Docker and what problem does it solve?**

**A:** Docker packages an application and all its dependencies (Python version, libraries, system packages) into a **container** — an isolated, reproducible unit that runs the same way everywhere.

**Problem it solves:** "It works on my machine." Different developers have different OS versions, Python versions, library versions. Docker standardises the environment.

**Key concepts:**
- **Image** — read-only blueprint (built from Dockerfile)
- **Container** — running instance of an image
- **Layer caching** — each `RUN` instruction in a Dockerfile creates a cached layer. Unchanged layers aren't rebuilt.

**Why `COPY requirements.txt .` before `COPY . .`?**
```dockerfile
COPY requirements.txt .          # layer 1 — changes rarely
RUN pip install -r requirements.txt  # layer 2 — cached as long as requirements.txt unchanged
COPY . .                         # layer 3 — changes on every code edit
```
If we copied all files first, any code change would invalidate the pip layer and reinstall all packages. This ordering means `pip install` is only re-run when `requirements.txt` changes.

---

**Q19. What is the difference between `CMD` and `ENTRYPOINT` in a Dockerfile?**

**A:**

| | `CMD` | `ENTRYPOINT` |
|---|---|---|
| Purpose | Default command | Fixed command |
| Override | `docker run image <new-cmd>` replaces it | `docker run image <args>` appends to it |
| Use case | Default that users might change | Core executable that's always run |

```dockerfile
# CMD — easily overridden:
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
# docker run myapp python test.py    ← replaces gunicorn with python test.py

# ENTRYPOINT — always runs gunicorn:
ENTRYPOINT ["gunicorn", "app:app"]
CMD ["--bind", "0.0.0.0:5000"]      ← default args (overridable)
# docker run myapp --workers 4       ← appends to ENTRYPOINT
```

For web apps, using CMD is common — it lets you override with a shell for debugging (`docker run -it myapp /bin/bash`).

---

**Q20. Explain named volumes in docker-compose and why they're used here.**

**A:** Docker containers have ephemeral filesystems — all changes are lost when the container is removed. **Named volumes** are directories managed by Docker that persist across container restarts and removals.

```yaml
volumes:
  vectorstore_data:/app/vectorstore   # FAISS index persists
  ollama_data:/root/.ollama           # LLM model weights persist
```

**Without volumes:** Every `docker compose down && docker compose up` would:
- Delete the FAISS index (re-upload all PDFs)
- Delete Ollama models (re-download llama3.2 = 2GB every time)

**With volumes:** Data lives in Docker-managed directories on the host. The container mounts them at the specified paths. Data survives restarts.

---

**Q21. What is a WSGI server and why can't Flask's development server handle production traffic?**

**A:** **WSGI** (Web Server Gateway Interface) is the Python standard for how web servers communicate with Python web applications.

**Flask's dev server:**
- Single-threaded by default
- Handles 1 request at a time
- Auto-reloads on code change (overhead)
- Has an interactive debugger (security risk in production)
- Not optimised for sustained load

**Gunicorn (production WSGI server):**
- **Pre-fork worker model** — master process forks N worker processes at startup
- Each worker handles requests independently
- Workers are restarted if they crash (resilience)
- Proper signal handling (SIGTERM → graceful shutdown)
- Used in production by Instagram, Pinterest, Reddit

```bash
gunicorn app:app --workers 4 --bind 0.0.0.0:5000
# 4 parallel workers, each handling requests independently
```

---

### Section F — General CS & Problem Solving

---

**Q22. What is the time and space complexity of FAISS IndexFlatIP search?**

**A:**
- **Time:** O(N × D) per query, where N = number of indexed vectors, D = dimensions (384)
  - For 10,000 chunks and 384 dims: 3.84M float multiplications per query
  - On modern CPU with SIMD: < 1ms
- **Space:** O(N × D × 4 bytes) for float32 storage
  - 10,000 × 384 × 4 = ~15 MB for the index

**Approximate search (HNSW) comparison:**
- Time: O(log N × D) — much faster at scale
- Space: O(N × D + graph_edges) — slightly more
- Accuracy: ~99% recall (not exact) — acceptable for most RAG use cases

---

**Q23. How would you evaluate the quality of a RAG system?**

**A:** RAG quality has two components: retrieval quality and generation quality.

**Retrieval metrics:**
- **Recall@K** — what fraction of relevant chunks are in the top-K results?
- **MRR (Mean Reciprocal Rank)** — how high is the first relevant chunk ranked?
- **Precision@K** — what fraction of top-K results are relevant?

**Generation metrics:**
- **Faithfulness** — is the answer supported by the retrieved context? (check with an LLM judge)
- **Answer relevance** — does the answer address the question?
- **Context precision** — are the retrieved chunks actually used in the answer?

**Frameworks:**
- **RAGAS** — Python library that automates all above metrics using an LLM judge
- **TruLens** — similar, open-source
- **Human evaluation** — gold standard; ask domain experts to rate answer quality

---

**Q24. What are the pros and cons of storing embeddings locally (FAISS) vs. a hosted vector DB (Pinecone)?**

**A:**

| | FAISS (local) | Pinecone / Qdrant (hosted) |
|---|---|---|
| **Cost** | Free | $70-100/month for production tier |
| **Setup** | pip install, 5 lines of code | API key, network calls |
| **Scale** | ~10M vectors on a single machine | Billions of vectors, multi-region |
| **Persistence** | Disk files (lost on ephemeral hosts) | Managed, always persistent |
| **Filtering** | Manual (Python loop) | Native metadata filtering |
| **Deletion** | No native delete | Native delete by ID |
| **Multi-tenancy** | Manual implementation | Built-in namespaces |
| **Best for** | Prototypes, single-server apps | Production, multi-user apps |

---

**Q25. How would you implement semantic caching to reduce LLM API costs?**

**A:** Many users ask similar questions. Instead of calling the LLM every time, we can cache answers for semantically similar queries.

```python
# Semantic cache: (query_embedding → cached_answer)

cache = []  # list of {embedding, answer}
SIMILARITY_THRESHOLD = 0.95

def get_cached_answer(query_embedding):
    for entry in cache:
        similarity = cosine_similarity(query_embedding, entry["embedding"])
        if similarity > SIMILARITY_THRESHOLD:
            return entry["answer"]
    return None

def ask_with_cache(question, context_chunks, model):
    query_emb = embed_query(question)
    cached = get_cached_answer(query_emb)
    if cached:
        return cached   # free!
    answer = generate_answer(question, context_chunks, model)
    cache.append({"embedding": query_emb, "answer": answer})
    return answer
```

**In production:** Use Redis to store the cache (with TTL expiry) so it persists across restarts and is shared across workers. Libraries like `GPTCache` implement this pattern with FAISS-based similarity lookup.

---

**Q26. Explain the difference between synchronous and asynchronous Python web servers. Should this app use async (FastAPI) instead of Flask?**

**A:**

**Synchronous (Flask + gunicorn):**
- Each worker handles one request at a time
- During an LLM API call (2-5 seconds), the worker blocks — it can't handle other requests
- Solved by having multiple workers (e.g., 4 workers = 4 concurrent requests)
- Simple to reason about, no async/await needed

**Asynchronous (FastAPI + uvicorn):**
- A single worker can handle thousands of concurrent connections
- During the LLM API call, the event loop handles other requests
- Requires `async def` routes and `await` for all I/O
- Better for I/O-bound, high-concurrency applications

**For this project:** The main I/O is the LLM API call (slow, blocking). An async server would genuinely help here at scale. However:
- Flask is simpler and more familiar
- With `--workers 4` gunicorn handles typical demo traffic fine
- Switching to FastAPI would require rewriting all routes with `async def`

**Recommendation:** Keep Flask for a portfolio project. Mention in interviews that you'd use FastAPI + async streaming for production.

---

**Q27. What is the system prompt and why does its design matter?**

**A:** The **system prompt** is an instruction given to the LLM that shapes its behavior for all subsequent messages. It's the `{"role": "system", ...}` message in the messages array.

**Our system prompt:**
```
You are a helpful assistant that answers questions using only the provided document context.

Context:
[retrieved chunks here]

Answer based solely on the context above. If the context lacks enough information, say so clearly.
```

**Why it matters:**
1. **Grounding instruction**: "using only the provided context" reduces hallucinations
2. **Fallback instruction**: "say so clearly" prevents the LLM from making up answers when context is insufficient
3. **Context injection**: we put the retrieved chunks *in the system message* (not the user message) because it's clearly LLM instructions, not user content
4. **Temperature 0.3**: lower = more factual, less creative

**Common mistakes:**
- Too long system prompt → eats context window, costs tokens
- No fallback instruction → LLM fabricates confidently
- Putting context in the user message → can confuse the model's role distinction

---

**Q28. How would you implement document versioning (allow re-uploading an updated PDF)?**

**A:** The current system treats each upload as a new document (new UUID). To handle versions:

**Approach:**
1. On upload, compute a hash of the PDF content (`hashlib.md5(pdf_bytes).hexdigest()`)
2. Check if a document with the same filename already exists
3. If yes, ask user: "Replace existing version?" (or auto-replace)
4. On replace: call `store.delete_document(old_doc_id)` then index the new version

**Schema addition:**
```python
documents[doc_id] = {
    "name": filename,
    "version": 2,                    # ← add
    "previous_doc_id": old_doc_id,   # ← add for audit trail
    "content_hash": "abc123",        # ← add for dedup
    ...
}
```

---

**Q29. How would you add support for DOCX and TXT files?**

**A:** The document processor is the only layer that's format-specific. The embedding, FAISS, and LLM layers work on plain text strings.

**For DOCX:**
```python
from docx import Document   # python-docx

def extract_text_from_docx(filepath):
    doc = Document(filepath)
    pages = []
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip():
            pages.append({"page": i + 1, "text": para.text})
    return pages
```

**For TXT:**
```python
def extract_text_from_txt(filepath):
    with open(filepath, encoding='utf-8') as f:
        text = f.read()
    return [{"page": 1, "text": text}]
```

**Dispatch:**
```python
def extract_text(filepath, filename):
    ext = filename.rsplit('.', 1)[-1].lower()
    if ext == 'pdf':   return extract_text_from_pdf(filepath)
    elif ext == 'docx': return extract_text_from_docx(filepath)
    elif ext == 'txt':  return extract_text_from_txt(filepath)
    else: raise ValueError(f"Unsupported format: {ext}")
```

---

**Q30. Walk me through how you would debug a situation where the RAG system gives correct answers for some questions but wrong/irrelevant answers for others.**

**A:** This is a retrieval quality problem. Debugging steps:

**Step 1 — Check retrieval quality:**
Add a debug endpoint that shows what chunks were actually retrieved:
```python
# Already visible via the "sources" in the response
# Check: are the retrieved chunks actually relevant to the question?
```

**Step 2 — Check embedding quality:**
Are semantically related phrases mapping to similar vectors?
```python
from rag.embeddings import embed_texts
import numpy as np

a = embed_texts(["What is the revenue?"])
b = embed_texts(["Total sales for Q3 were $5M"])
similarity = np.dot(a[0], b[0])
print(similarity)  # Should be > 0.7 for related phrases
```

**Step 3 — Check chunking:**
Is the answer in the source document but split across chunk boundaries? Reduce overlap or increase chunk size.

**Step 4 — Check the prompt:**
Print the full messages array being sent to the LLM:
```python
print(json.dumps(messages, indent=2))
# Are the retrieved chunks actually relevant?
# Is the context window being exceeded?
```

**Step 5 — Increase k:**
Try `k=10` instead of `k=5`. More candidates may include the relevant chunk.

**Step 6 — Add reranking:**
Use a cross-encoder (like `cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-score the top-20 chunks by relevance to the query, then take the top 5. Cross-encoders are slower but much more accurate than bi-encoder similarity.

---

*End of Documentation*
