import numpy as np
from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    embeddings = get_model().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
