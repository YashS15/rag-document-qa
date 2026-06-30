import os

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

GROQ_MODELS = {
    "llama3.2": "llama-3.1-8b-instant",
    "llama2": "llama-3.1-8b-instant",
    "mistral": "mixtral-8x7b-32768",
    "llama3.2:70b": "llama-3.3-70b-versatile",
}
DEFAULT_GROQ_MODEL = "llama-3.1-8b-instant"

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions using only the provided document context.

Context:
{context}

Question: {question}

Answer based solely on the context above. If the context lacks enough information, say so clearly. Be concise and accurate."""


def generate_answer(question: str, context_chunks: list[dict], model: str = "llama3.2") -> str:
    context_parts = [
        f"[Source: {c['doc_name']}, Page {c['page']}]\n{c['text']}"
        for c in context_chunks
    ]
    context = "\n\n---\n\n".join(context_parts)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        return _generate_groq(prompt, model, groq_key)
    return _generate_ollama(prompt, model)


def _generate_groq(prompt: str, model: str, api_key: str) -> str:
    groq_model = GROQ_MODELS.get(model, DEFAULT_GROQ_MODEL)
    try:
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": groq_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1024,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error calling Groq API: {e}"


def _generate_ollama(prompt: str, model: str) -> str:
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        return (
            "Error: Cannot connect to Ollama. "
            "Make sure Ollama is running (`ollama serve`) and the model is pulled (`ollama pull llama3.2`)."
        )
    except Exception as e:
        return f"Error calling Ollama: {e}"
