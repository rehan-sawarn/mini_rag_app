import time
import streamlit as st
from src.settings import settings
from src.embeddings import prepare_docs
from src.vectorstore import ensure_collection, upsert_docs, query_qdrant
from src.llm import generate_answer

st.set_page_config(page_title="Mini RAG", page_icon="📚")
st.title("Mini RAG")
st.caption("A Retrieval-Augmented Generation demo using Qdrant + local embeddings + local LLM.")

# -----------------------------
# 1. Environment & DB Health Check
# -----------------------------
with st.expander("Current settings", expanded=False):
    st.write({
        "QDRANT_URL": bool(settings.QDRANT_URL),
        "QDRANT_COLLECTION": settings.QDRANT_COLLECTION,
    })

if st.button("Connect to Qdrant & ensure collection"):
    try:
        col = ensure_collection()
        st.success(f"✅ Connected. Collection '{col}' is ready.")
    except Exception as e:
        st.error(f"❌ Connection failed: {e}")

# -----------------------------
# 2. Text Ingestion
# -----------------------------
st.header("📥 Ingest Text")
user_text = st.text_area("Paste text here", height=200)

if st.button("Ingest pasted text"):
    if not user_text.strip():
        st.warning("Please paste some text first.")
    else:
        docs = prepare_docs(user_text, source="paste")  # chunk + embed
        upsert_docs(docs)
        st.success(f"✅ Ingested {len(docs)} chunks into Qdrant.")

# -----------------------------
st.header("💬 Ask a Question")
query = st.text_input("Enter your query:")

if st.button("Get Answer") and query.strip():
    import time
    start_time = time.time()
    
    # Step 1: Retrieve top-k chunks from Qdrant
    top_chunks = query_qdrant(query, top_k=5)
    
    # Step 2: Display retrieved sources
    st.subheader("Relevant Sources / Chunks")
    for i, chunk in enumerate(top_chunks):
        st.markdown(f"**Chunk {i+1}:** {chunk['payload']['content']}")
    
    # Step 3: Generate answer using Flan-T5-Small (local LLM)
    context = "\n".join([c['payload']['content'] for c in top_chunks])
    
    # Pass query first, context second (matches updated llm.py)
    answer = generate_answer(query=query, context=context)
    
    st.subheader("Answer")
    st.write(answer)
    
    # Step 4: Display timing and rough token count
    elapsed = time.time() - start_time
    st.info(f"Request took ~{elapsed:.2f} seconds, {len(context.split())} words retrieved.")