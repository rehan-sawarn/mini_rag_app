# src/embeddings.py
from sentence_transformers import SentenceTransformer
from src.util import new_id
from src.text_splitter import split_text

# LOAD LOCAL MODEL
model = SentenceTransformer("all-MiniLM-L12-v2")

def embed_text(texts: list[str]) -> list[list[float]]:
    """
    EMBED A LIST OF TEXTS USING SENTENCE-TRANSFORMERS.
    RETURNS A LIST OF FLOAT VECTORS.
    """
    embeddings = model.encode(texts, convert_to_numpy=True)
    return embeddings.tolist()  # Qdrant expects list of floats

def prepare_docs(text: str, source: str = "paste") -> list[dict]:
    """
    SPLIT, EMBED, AND PACKAGE DOCS WITH METADATA FOR QDRANT
    """
    chunks = split_text(text)
    vectors = embed_text(chunks)
    docs = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        docs.append({
            "id": new_id("chunk"),
            "vector": vector,
            "payload": {
                "content": chunk,
                "source": source,
                "position": i
            }
        })
    return docs
