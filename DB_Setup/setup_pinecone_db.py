import os
import re
import uuid
from typing import List, Dict
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone
from pinecone.exceptions.exceptions import NotFoundException
from dotenv import load_dotenv

# ---------- CONFIG ----------
MD_PATH = "/Users/kevinli_home/Desktop/SST-Ticketing-Agent/pinecone_db_2.md"


# Pinecone (pods) host
load_dotenv()
PINECONE_HOST = os.getenv("PINECONE_HOST")
# Default namespace
PINECONE_NAMESPACE = None

# Embedding settings
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 512
EXPECTED_SECTIONS = 8

# ---------- HELPERS ----------
def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_md_by_h2(md_text: str) -> List[Dict]:
    """Split Markdown by H2 headings (`##`)."""
    pattern = re.compile(r"(?m)^(##)\s+(.+?)\s*$")
    matches = list(pattern.finditer(md_text))
    if not matches:
        raise RuntimeError("No H2 (##) headings found in Markdown file.")
    sections = []
    for i, m in enumerate(matches):
        start_body = m.end()
        end_body = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        title = m.group(2).strip()
        body = md_text[start_body:end_body].strip()
        body = re.sub(r"[ \t]+\n", "\n", body)
        body = re.sub(r"\n{3,}", "\n\n", body)
        sections.append({"title": title, "text": body})
    return sections

def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s or uuid.uuid4().hex[:8]

# ---------- MAIN ----------
def main():
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    if not OPENAI_API_KEY or not PINECONE_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY or PINECONE_API_KEY in environment.")

    # Init clients
    oa = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_HOST)

    # Step 1: Read + split
    md = read_text(MD_PATH)
    sections = split_md_by_h2(md)
    if len(sections) != EXPECTED_SECTIONS:
        titles = [s["title"] for s in sections]
        raise RuntimeError(f"Expected {EXPECTED_SECTIONS} H2 sections, found {len(sections)}. Found: {titles}")
    print(f"✅ Found {len(sections)} H2 sections (OK)")

    # Step 2: Clear default namespace
    print("Clearing default namespace…")
    try:
        index.delete(delete_all=True, namespace=None)
        print("Default namespace cleared.")
    except NotFoundException:
        print("Nothing to clear (namespace not found).")

    # Step 3: Embed + upsert
    texts = [f"Title: {sec['title']}\n\n{sec['text']}" for sec in sections]
    print("Embedding sections with OpenAI...")
    emb = oa.embeddings.create(model=EMBED_MODEL, input=texts, dimensions=EMBED_DIM)

    vectors = []
    for i, (sec, e) in enumerate(zip(sections, emb.data), start=1):
        vid = f"h2-{i:02d}-{slugify(sec['title'])}"
        vectors.append({
            "id": vid,
            "values": e.embedding,
            "metadata": {
                "title": sec["title"],
                "text": f"Title: {sec['title']}\n\n{sec['text']}",
                "order": i,
                "source": os.path.basename(MD_PATH),
                "section_type": "h2",
                "version": "v2"
            }
        })

    print(f"Upserting {len(vectors)} vectors to Pinecone…")
    index.upsert(vectors=vectors, namespace=None)
    print("✅ Upsert complete.")

    # Step 4: Sanity query
    sanity_query = "push the artwork to Atom"
    q_emb = oa.embeddings.create(model=EMBED_MODEL, input=sanity_query, dimensions=EMBED_DIM).data[0].embedding
    res = index.query(vector=q_emb, top_k=3, include_metadata=True, namespace=None)

    print("\nTop-3 sanity results:")
    for m in res.get("matches", []) or []:
        meta = m.get("metadata") or {}
        snippet = (meta.get("text") or "")[:160].replace("\n", " ")
        print(f"- score={m.get('score',0):.4f} | id={m.get('id')} | title={meta.get('title')}")
        print(f"  {snippet}…")

if __name__ == "__main__":
    main()
