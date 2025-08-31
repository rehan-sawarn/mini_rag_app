from src.settings import settings
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

def get_qdrant():
    return QdrantClient(url=settings.QDRANT_URL)

def ensure_collection():
    client = get_qdrant()
    if not client.HAS_COLLECTION(settings.QDRANT_COLLECTION):
        client.recreate_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vector_size=settings.EMBEDDING_DIM,
            distance="Cosine"
        )
    return settings.QDRANT_COLLECTION

def upsert_docs(docs):
    client = get_qdrant()
    points = [PointStruct(id=doc["id"], vector=doc["vector"], payload=doc["payload"]) for doc in docs]
    client.upsert(collection_name=settings.QDRANT_COLLECTION, points=points)

def query_qdrant(query, top_k=5):
    """Embeds the query using same embeddings model and returns top_k chunks."""
    from src.embeddings import embed_text
    client = get_qdrant()
    query_vector = embed_text([query])[0]  # single query
    hits = client.search(
        collection_name=settings.QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k
    )
    return hits
