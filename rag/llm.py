import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

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
        return f"Error generating answer: {e}"
