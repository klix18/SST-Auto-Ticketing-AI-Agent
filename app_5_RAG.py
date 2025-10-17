# ==========================================================
# Conducts Semantic and Keyword search, combines results into a
# ==========================================================
import os
import re
import sqlite3
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from concurrent.futures import ThreadPoolExecutor
from langchain.tools import tool

# Import variable from your variables file
from app_5_variables import request_description

# ==========================================================
# 🔧 CONFIG
# ==========================================================
load_dotenv()
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST    = "https://sst-master-db-xxzkqyr.svc.aped-4627-b74a.pinecone.io"

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 512

TOP_K_SEMANTIC = 2
TOP_K_KEYWORD  = 2

SQLITE_DB_PATH = "/Users/kevinli_home/Desktop/SST-Ticketing-Agent/DB_Setup/SQLite_DB/pinecone_db_2.sqlite"

# ==========================================================
# 🔌 Connect to Pinecone + Embeddings
# ==========================================================
def get_pinecone_and_embeddings():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_HOST)
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        dimensions=EMBED_DIM,
        api_key=OPENAI_API_KEY,
    )
    return index, embeddings

# ==========================================================
# 🔍 Semantic RAG Retrieval (Pinecone)
# ==========================================================
def rag_retrieve_semantic(query_text: str, top_k: int = TOP_K_SEMANTIC):
    index, embeddings = get_pinecone_and_embeddings()
    query_vector = embeddings.embed_query(query_text)
    res = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    semantic_chunks = []
    for match in res.get("matches", []) or []:
        meta = match.get("metadata", {}) or {}
        semantic_chunks.append({
            "chunk_id": match.get("id"),
            "chunk_title": meta.get("title", ""),
            "chunk_text": meta.get("text", ""),
            "chunk_score": match.get("score", 0.0),
            "source": "semantic"
        })
    return semantic_chunks

# ==========================================================
# 🧠 Keyword RAG Retrieval (SQLite)
# ==========================================================
def rag_retrieve_keyword(query_text: str, top_k: int = TOP_K_KEYWORD):
    """
    Perform true keyword search on local SQLite DB.
    Match against both title and text fields.
    """
    if not os.path.exists(SQLITE_DB_PATH):
        raise FileNotFoundError(f"SQLite database not found at {SQLITE_DB_PATH}")

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()

    # Clean and tokenize query text
    keywords = [w.lower() for w in re.findall(r"\w+", query_text) if len(w) > 2]

    # Build LIKE-based query for all keywords
    conditions = []
    params = []
    for k in keywords:
        conditions.append("(LOWER(title) LIKE ? OR LOWER(text) LIKE ?)")
        params.extend([f"%{k}%", f"%{k}%"])
    where_clause = " OR ".join(conditions)

    sql = f"""
        SELECT id, title, text
        FROM rag_chunks
        WHERE {where_clause}
        LIMIT {top_k * 3}
    """

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    # Simple scoring by keyword frequency
    scored = []
    for (cid, title, text) in rows:
        score = sum(text.lower().count(k) + title.lower().count(k) * 2 for k in keywords)
        scored.append({
            "chunk_id": cid,
            "chunk_title": title,
            "chunk_text": text,
            "chunk_score": float(score),
            "source": "keyword"
        })

    scored.sort(key=lambda x: x["chunk_score"], reverse=True)
    return scored[:top_k]

# ==========================================================
# 🔗 Combine Semantic + Keyword Results
# ==========================================================
def combine_results(semantic_chunks, keyword_chunks):
    """
    Merge semantic and keyword results, avoiding duplicates
    if titles match. Combine text from both.
    """
    combined_texts = []
    combined_titles = []
    combined_ids = []

    # Add top semantic results
    for s in semantic_chunks:
        combined_ids.append(s["chunk_id"])
        combined_titles.append(s["chunk_title"])
        combined_texts.append(f"### {s['chunk_title']}\n{s['chunk_text']}")

    # Add keyword results if titles differ
    for k in keyword_chunks:
        if k["chunk_title"] not in combined_titles:
            combined_ids.append(k["chunk_id"])
            combined_titles.append(k["chunk_title"])
            combined_texts.append(f"### {k['chunk_title']}\n{k['chunk_text']}")

    RAG_combined_id = " + ".join(combined_ids)
    RAG_combined_title = " + ".join(combined_titles)
    RAG_combined_text = "\n\n---\n\n".join(combined_texts)
    RAG_combined_total_chunk_number = len(combined_ids)  # number of unique chunks merged



    RAG_result = {
        "RAG_combined_id": RAG_combined_id,
        "RAG_combined_title": RAG_combined_title,
        "RAG_combined_text": RAG_combined_text,
        "RAG_combined_total_chunk_number": RAG_combined_total_chunk_number,
    }

    # Return it
    return RAG_result

# ==========================================================
# 🚀 Master Function
# ==========================================================
def master_rag(query: str) -> dict:
    """Perform hybrid RAG (semantic + SQLite keyword) concurrently."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_semantic = executor.submit(rag_retrieve_semantic, query)
        future_keyword = executor.submit(rag_retrieve_keyword, query)
        semantic_chunks = future_semantic.result()
        keyword_chunks = future_keyword.result()

    RAG_result = combine_results(semantic_chunks, keyword_chunks)
    return RAG_result

#Making this a tool that the parser llm can call
@tool("master_rag_tool", return_direct=True)
def master_rag_tool(query: str) -> str:
    """
    Hybrid retrieval function that performs both semantic and keyword search.
    Use this tool to find more information regarding Skyshowtime imagery production.
    Input: query string.
    Output: Combined top results from Pinecone and SQLite, formatted text.
    """
    RAG_result = master_rag(query)
    combined_text = RAG_result.get("RAG_combined_text", "")
    return combined_text

# ==========================================================
# 🧩 Manual Test
# ==========================================================
if __name__ == "__main__":
    #print("🧠 Performing Combined RAG Retrieval for:")
    print(f"Request Description: {request_description}\n")

    RAG_result = master_rag(request_description)

    #print("✅ Combined RAG Output:")
    #print(RAG_result["RAG_combined_id"])
    print(RAG_result["RAG_combined_title"])
    print(RAG_result["RAG_combined_total_chunk_number"]) # if 0, then that means no chunks were successfully retrieved
    #print(RAG_result["RAG_combined_text"])

