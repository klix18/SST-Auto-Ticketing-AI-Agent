# ==========================================================
# app_3_llm.py — Handles RAG + LLM
# ==========================================================
import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone

# --- Load ENV ---
load_dotenv()
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST    = "https://sst-master-db-xxzkqyr.svc.aped-4627-b74a.pinecone.io"
EMBED_MODEL      = "text-embedding-3-small"
EMBED_DIM        = 512

# --- Categories ---
CATEGORIES = [
    "Make New Package",
    "Publish Artwork to Platform",
    "Change Existing Image Assets",
    "Add Missing Image Assets",
]

# ==========================================================
# 🔧 RAG + LLM
# ==========================================================
def get_pc_clients():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_HOST)
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        dimensions=EMBED_DIM,
        api_key=OPENAI_API_KEY,
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    return index, embeddings, llm


def get_llm_json():
    """Strict JSON-mode LLM for classification."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def rag_retrieve_top3(query_text: str):
    """Embed query and fetch top 3 chunks."""
    index, embeddings, _ = get_pc_clients()
    qvec = embeddings.embed_query(query_text)
    res = index.query(vector=qvec, top_k=3, include_metadata=True)
    chunks = []
    for m in res.get("matches", []):
        md = m.get("metadata", {}) or {}
        chunks.append({
            "id": m.get("id"),
            "text": md.get("text", ""),
            "title": md.get("title", ""),
            "score": m.get("score", 0.0),
        })
    return chunks[:3]


def rag_classify(request_type: str, request_description: str):
    """RAG classification with confidence threshold logic."""
    _, _, _chat_llm = get_pc_clients()
    llm = get_llm_json()
    matches = rag_retrieve_top3(request_description)

    if not matches:
        return {
            "result": "Insufficient Context",
            "confidence": 0,
            "explanation": "No RAG chunks retrieved; cannot classify.",
            "context_used": "",
            "chunks_used": [],
            "chunk_used": None,
        }

    rag_context = "\n---\n".join([m["text"] for m in matches])
    chunk_ids = [m["id"] for m in matches]

    system_prompt = (
        "You are a classification assistant for artwork ticket types. "
        "Use ONLY the provided RAG context below to decide which of the 4 categories best matches "
        "the user's request description.\n\n"
        "⚠️ IMPORTANT: You must return your answer in strict JSON format, "
        "for example: {\"result\": \"Make New Package\", \"confidence\": 95, \"explanation\": \"Reason...\"}\n\n"
        "Valid categories:\n"
        "1 Make New Package\n"
        "2 Publish Artwork to Platform\n"
        "3 Change Existing Image Assets\n"
        "4 Add Missing Image Assets\n"
    )


    user_prompt = f"Request Description:\n{request_description}\n\nRAG CONTEXT:\n{rag_context}\nChunk IDs:\n{chunk_ids}\n"
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

    resp = llm.invoke(msgs)
    raw = (resp.content or "").strip()
    data = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    result = str(data.get("result", "")).strip()
    conf = int(data.get("confidence", 0)) if isinstance(data.get("confidence"), (int, float)) else 0


    result_norm = result
    if result_norm and result_norm[0].isdigit():
        parts = result_norm.split(" ", 1)
        if len(parts) > 1:
            result_norm = parts[1].strip()

    if result_norm not in CATEGORIES and result_norm != "Insufficient Context":
        for c in CATEGORIES:
            if c.lower() in result_norm.lower():
                result_norm = c
                break
        else:
            result_norm = "Insufficient Context"

    if conf < 60:
        result_norm = "Insufficient Context"
        explanation = "Confidence below 60%; insufficient context."
    else:
        explanation = data.get("explanation", f"Matched '{result_norm}' with {conf}% confidence.")

    return {
        "result": result_norm,
        "confidence": conf,
        "explanation": explanation,
        "context_used": rag_context,
        "chunks_used": chunk_ids,
        "chunk_used": data.get("chunk_used", chunk_ids[0] if chunk_ids else None),
    }


def summarize_type_with_rag(type_label: str, rag_context: str) -> str:
    """Summarize request type strictly from RAG context."""
    _, _, llm = get_pc_clients()
    if not rag_context.strip():
        return f"Insufficient context to summarize '{type_label}'."

    system_prompt = (
        "You are an expert production ops assistant. "
        "Use ONLY the provided RAG context. Do NOT use outside knowledge."
    )
    user_prompt = (
        f"Explain '{type_label}' in 1–2 sentences based ONLY on this context:\n{rag_context}\n"
    )
    resp = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    text = (resp.content or "").strip()
    return text or f"Insufficient context to summarize '{type_label}'."


def llm_chat_reply(user_msg: str, rag_context: str, request_type: str, request_description: str, suggested_type: str):
    """Strictly RAG-grounded chat reply."""
    _, _, llm = get_pc_clients()
    sys = (
        "You assist users about artwork ticket types using ONLY the provided RAG context."
        "If not in context, ask clarifying questions. Always end with: "
        f"Would you like me to change the Request Type to '{suggested_type}' or continue?"
    )
    primer = (
        f"User chose: '{request_type}'. RAG suggests '{suggested_type}' based on '{request_description}'.\n"
        f"RAG CONTEXT:\n{rag_context}\n"
    )
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": primer + user_msg}]
    resp = llm.invoke(msgs)
    return resp.content.strip()
