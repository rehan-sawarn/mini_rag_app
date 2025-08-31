from src.vectorstore import get_qdrant
from sentence_transformers import SentenceTransformer

# Initialize embeddings model
model = SentenceTransformer("all-MiniLM-L12-v2")
qdrant = get_qdrant()
collection = "mini_rag"

def search(query: str, top_k: int = 3):
    vector = model.encode([query])[0].tolist()
    results = qdrant.search(
        collection_name=collection,
        query_vector=vector,
        limit=top_k
    )
    for r in results:
        print(r.payload["text"], "\n---")

# Example query
search("your search query here")
