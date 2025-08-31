# ingest.py
import os
from src.settings import Settings
from src.vectorstore import get_qdrant
from src.embeddings import embed_text 
from src.text_splitter import split_text

def load_documents(folder_path: str):
    docs = []
    for fname in os.listdir(folder_path):
        if fname.endswith(".txt"):
            with open(os.path.join(folder_path, fname), "r", encoding="utf-8") as f:
                docs.append(f.read())
    return docs

def ingest():
    settings = Settings()
    qdrant = get_qdrant()
    collection = settings.QDRANT_COLLECTION

    # Load .txt files
    docs = load_documents("data")
    
    all_chunks = []
    for doc in docs:
        chunks = split_text(doc, chunk_size=500, overlap=50)
        all_chunks.extend(chunks)

    # Generate embeddings via OpenAI
    embeddings = embed_text(all_chunks)

    # Prepare vectors for Qdrant
    vectors = []
    for idx, emb in enumerate(embeddings):
        vectors.append({
            "id": idx,
            "vector": emb,
            "payload": {"text": all_chunks[idx]},
        })

    # Upsert to Qdrant collection
    qdrant.upsert(
        collection_name=collection,
        points=vectors
    )

    print(f"[✔] Ingested {len(all_chunks)} chunks into Qdrant collection '{collection}'")

if __name__ == "__main__":
    ingest()
