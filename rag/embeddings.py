import numpy as np
from fastembed import TextEmbedding

_model = None


def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        # BAAI/bge-small-en-v1.5: 384-dim, ONNX-based (~100MB RAM vs ~500MB for PyTorch)
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    embeddings = list(get_model().embed(texts))
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
