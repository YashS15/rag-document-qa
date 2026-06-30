import json
import os

import requests

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

# Groq model IDs (mixtral-8x7b-32768 was deprecated — replaced with working models)
GROQ_MODELS = {
    "llama3.1-8b": "llama-3.1-8b-instant",
    "llama3.3-70b": "llama-3.3-70b-versatile",
    "gemma2-9b": "gemma2-9b-it",
}
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using only the provided document context.

Context from documents:
{context}

Answer based solely on the context above. If the context lacks enough information, say so clearly. Be concise and accurate."""


def _build_messages(question: str, context_chunks: list[dict], history: list[dict] | None) -> list[dict]:
    context = "\n\n---\n\n".join(
        f"[Source: {c['doc_name']}, Page {c['page']}]\n{c['text']}"
        for c in context_chunks
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    for turn in (history or []):
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": turn["answer"]})
    messages.append({"role": "user", "content": question})
    return messages


def stream_answer(question: str, context_chunks: list[dict], model: str, history: list[dict] | None = None):
    """Generator that yields text tokens one by one."""
    messages = _build_messages(question, context_chunks, history)
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        groq_model = GROQ_MODELS.get(model, DEFAULT_GROQ_MODEL)
        yield from _stream_groq(messages, groq_model, groq_key)
    else:
        yield from _stream_ollama(messages, model)


def _stream_groq(messages: list[dict], groq_model: str, api_key: str):
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": groq_model, "messages": messages, "stream": True, "temperature": 0.3},
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue
            payload = line[6:]
            if payload == b"[DONE]":
                break
            try:
                token = json.loads(payload)["choices"][0]["delta"].get("content", "")
                if token:
                    yield token
            except (KeyError, json.JSONDecodeError):
                pass
    except Exception as e:
        yield f"\n\n[Error: {e}]"


def _stream_ollama(messages: list[dict], model: str):
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={"model": model, "messages": messages, "stream": True},
            stream=True,
            timeout=180,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
                token = data.get("message", {}).get("content", "")
                if token:
                    yield token
                if data.get("done"):
                    break
            except json.JSONDecodeError:
                pass
    except requests.exceptions.ConnectionError:
        yield "Error: Cannot connect to Ollama. Run `ollama serve` and pull a model first."
    except Exception as e:
        yield f"Error: {e}"


def summarize_document(chunks: list[dict], model: str = "llama3.1-8b") -> str:
    """Return a 2-3 sentence summary of the document."""
    sample = "\n\n".join(c["text"] for c in chunks[:6])[:3000]
    messages = [
        {"role": "user", "content": f"Summarize this document in 2-3 sentences:\n\n{sample}"}
    ]
    groq_key = os.environ.get("GROQ_API_KEY")
    try:
        if groq_key:
            groq_model = GROQ_MODELS.get(model, DEFAULT_GROQ_MODEL)
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {groq_key}"},
                json={"model": groq_model, "messages": messages, "temperature": 0.3, "max_tokens": 200},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            resp = requests.post(
                OLLAMA_CHAT_URL,
                json={"model": model if model in ("llama3.2", "mistral", "llama2") else "llama3.2",
                      "messages": messages, "stream": False},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
    except Exception:
        return ""
