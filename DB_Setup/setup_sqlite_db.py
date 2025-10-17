# ==========================================================
# create_sqlite_from_md.py — Split Markdown by H2 and save to SQLite
# ==========================================================
import os
import re
import uuid
import sqlite3
from typing import List, Dict

# ==========================================================
# 🔧 FILE PATHS
# ==========================================================
MD_PATH = "/Users/kevinli_home/Desktop/SST-Ticketing-Agent/pinecone_db_2.md"
DB_DIR  = "/Users/kevinli_home/Desktop/SST-Ticketing-Agent/DB_Setup/SQLite_DB"
DB_PATH = os.path.join(DB_DIR, "pinecone_db_2.sqlite")

# Ensure target directory exists
os.makedirs(DB_DIR, exist_ok=True)

# ==========================================================
# 📖 READ MARKDOWN
# ==========================================================
def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# ==========================================================
# ✂️ SPLIT MARKDOWN INTO H2 SECTIONS
# ==========================================================
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

# ==========================================================
# 🆔 GENERATE UNIQUE ID
# ==========================================================
def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return s or uuid.uuid4().hex[:8]

# ==========================================================
# 🧱 CREATE TABLE
# ==========================================================
def create_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            text TEXT NOT NULL
        )
    """)
    conn.commit()

# ==========================================================
# 💾 INSERT CHUNKS
# ==========================================================
def insert_chunks(conn, sections: List[Dict]):
    cursor = conn.cursor()
    for sec in sections:
        chunk_id = slugify(sec["title"])
        cursor.execute(
            "INSERT OR REPLACE INTO rag_chunks (id, title, text) VALUES (?, ?, ?)",
            (chunk_id, sec["title"], sec["text"])
        )
    conn.commit()

# ==========================================================
# 🚀 MAIN RUNNER
# ==========================================================
def main():
    print(f"📄 Reading Markdown from:\n   {MD_PATH}")
    md_text = read_text(MD_PATH)
    sections = split_md_by_h2(md_text)
    print(f"✅ Found {len(sections)} H2 sections.")

    if len(sections) != 8:
        print(f"⚠️ Expected 8 sections, but found {len(sections)} — continuing anyway.")

    print(f"💾 Creating SQLite DB at:\n   {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    create_table(conn)
    insert_chunks(conn, sections)
    conn.close()

    print(f"🎉 Done! Saved {len(sections)} chunks to '{DB_PATH}'")

# ==========================================================
# 🏁 RUN DIRECTLY
# ==========================================================
if __name__ == "__main__":
    main()
