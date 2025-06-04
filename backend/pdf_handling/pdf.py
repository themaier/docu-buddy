import io
import os
import re
from typing import List

from pypdf import PdfReader
import tiktoken
import openai
from supabase import create_client

# ---------------  configuration  --------------- #
openai.api_key = os.getenv("OPENAI_API_KEY")          # must be set in env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
EMBED_MODEL = "text-embedding-3-small"                # 1536-dim
TOKENS_PER_CHUNK = 800                                # safe vs. 8192 limit
TOKEN_OVERLAP = 100                                   # for cross-chunk recall
ENCODER = tiktoken.encoding_for_model(EMBED_MODEL)
# ----------------------------------------------- #


def _extract_text(file_like: io.BufferedIOBase) -> str:
    """Extract raw text from every page of the PDF."""
    reader = PdfReader(file_like)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)


def _preprocess(text: str) -> str:
    """Light cleanup: collapse ≥2 blank lines → exactly one, trim."""
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def _chunk_text(text: str,
                max_tokens: int = TOKENS_PER_CHUNK,
                overlap: int = TOKEN_OVERLAP) -> List[str]:
    """
    Paragraph-aware, token-bounded chunker.

    • Split on double newline → natural paragraphs / bullet blocks  
    • Accumulate until `max_tokens` would be exceeded  
    • Add `overlap` tokens from the end of the previous chunk to the start
    """
    paragraphs = text.split("\n\n")
    chunks, cur, cur_tokens = [], [], 0

    for para in paragraphs:
        tokens = len(ENCODER.encode(para))
        if cur_tokens + tokens > max_tokens:
            if cur:                                           # flush current
                chunk = "\n\n".join(cur)
                chunks.append(chunk)
                if overlap:
                    # prepend last `overlap` tokens of the flushed chunk
                    tail = ENCODER.decode(
                        ENCODER.encode(chunk)[-overlap:]
                    )
                    cur, cur_tokens = [tail], len(ENCODER.encode(tail))
                else:
                    cur, cur_tokens = [], 0
        cur.append(para)
        cur_tokens += tokens

    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def _embed(chunks: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    """Call the OpenAI embeddings endpoint once (batch mode)."""
    resp = openai.embeddings.create(input=chunks, model=model)
    # The API returns results in the original order
    return [d.embedding for d in resp.data]


def _insert_supabase(chunks: List[str],
                     embeddings: List[List[float]],
                     table: str) -> int:
    """Write (content, embedding) rows to Supabase."""
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    sb.table(table).delete().neq("id", 0).execute()
    rows = [{"content": c, "embedding": e}
            for c, e in zip(chunks, embeddings)]
    sb.table(table).insert(rows).execute()
    return len(rows)


def process_pdf_and_upload(file_like: io.BufferedIOBase, table: str) -> dict:
    """
    High-level helper: extract → preprocess → chunk → embed → insert.
    Returns a simple stats dict for the API response.
    """
    raw = _extract_text(file_like)
    clean = _preprocess(raw)
    chunks = _chunk_text(clean)
    embeds = _embed(chunks)
    inserted = _insert_supabase(chunks, embeds, table)
    return {
        "pages_processed": raw.count("\f") + 1,   # crude page count
        "chunks_created": len(chunks),
        "rows_inserted": inserted,
        "model": EMBED_MODEL
    }
