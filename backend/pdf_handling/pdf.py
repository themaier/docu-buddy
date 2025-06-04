import io
import os
import re
from typing import List

import requests                    # still imported in case you need it later
from pypdf import PdfReader
import tiktoken
import openai
from supabase import create_client

# ---------------  configuration  --------------- #
openai.api_key = os.getenv("OPENAI_API_KEY")          # must be set in env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
EMBED_MODEL = "text-embedding-3-small"                # 1536-dim
TOKENS_PER_CHUNK = 800                                # safe vs. 8 192 limit
TOKEN_OVERLAP = 100                                   # for cross-chunk recall
ENCODER = tiktoken.encoding_for_model(EMBED_MODEL)
# ----------------------------------------------- #

def _debug(msg: str) -> None:
    """Consistent, flush-immediate debug helper."""
    print(f"[DEBUG] {msg}", flush=True)

# ---------- pipeline helpers ---------- #

def _extract_text(file_like: io.BufferedIOBase) -> str:
    _debug("Starting text extraction from PDF")
    reader = PdfReader(file_like)

    pages = []
    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        _debug(f"Page {idx + 1}/{len(reader.pages)} → {len(page_text)} chars")
        pages.append(page_text)

    full_text = "\n".join(pages)
    _debug(f"Total extracted length: {len(full_text)} chars")
    return full_text


def _preprocess(text: str) -> str:
    _debug("Pre-processing raw text")
    cleaned = re.sub(r"\n{2,}", "\n\n", text).strip()
    _debug(f"Pre-processed length: {len(cleaned)} chars")
    return cleaned


def _chunk_text(text: str,
                max_tokens: int = TOKENS_PER_CHUNK,
                overlap: int = TOKEN_OVERLAP) -> List[str]:
    _debug("Chunking text")
    paragraphs = text.split("\n\n")
    chunks, cur_buf, cur_tokens = [], [], 0

    for para_idx, para in enumerate(paragraphs):
        tokens = len(ENCODER.encode(para))

        if cur_tokens + tokens > max_tokens and cur_buf:
            chunk = "\n\n".join(cur_buf)
            chunks.append(chunk)
            _debug(f"Chunk {len(chunks)} → {cur_tokens} tokens")

            if overlap:
                tail_tokens = ENCODER.encode(chunk)[-overlap:]
                tail = ENCODER.decode(tail_tokens)
                cur_buf, cur_tokens = [tail], len(tail_tokens)
            else:
                cur_buf, cur_tokens = [], 0

        cur_buf.append(para)
        cur_tokens += tokens

    if cur_buf:
        chunks.append("\n\n".join(cur_buf))
        _debug(f"Chunk {len(chunks)} (final) → {cur_tokens} tokens")

    _debug(f"Total chunks produced: {len(chunks)}")
    return chunks


def _embed(chunks: List[str], model: str = EMBED_MODEL) -> List[List[float]]:
    _debug(f"Embedding {len(chunks)} chunks with model “{model}”")
    resp = openai.embeddings.create(input=chunks, model=model)
    vectors = [d.embedding for d in resp.data]
    _debug("Embedding request completed")
    return vectors


def _insert_supabase(chunks: List[str],
                     embeddings: List[List[float]],
                     table: str) -> int:
    _debug(f"Uploading to Supabase table “{table}”")
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Safety wipe of existing rows
    _debug("Deleting existing rows …")
    try:
        sb.table(table).delete().not_.is_("id", "null").execute()
        _debug("Existing rows deleted")
    except Exception as e:
        _debug(f"Delete failed: {e}")

    rows = [{"content": c, "embedding": e}
            for c, e in zip(chunks, embeddings)]

    _debug(f"Inserting {len(rows)} new rows")
    try:
        sb.table(table).insert(rows).execute()
        _debug("Insert completed")
    except Exception as e:
        _debug(f"Insert failed: {e}")
        raise                       # bubble up so the caller sees the error too

    return len(rows)

# ---------- public API ---------- #

def process_pdf_and_upload(file_like: io.BufferedIOBase, table: str) -> dict:
    _debug("===== Pipeline start =====")
    raw   = _extract_text(file_like)
    clean = _preprocess(raw)
    chunks = _chunk_text(clean)
    embeds = _embed(chunks)
    inserted = _insert_supabase(chunks, embeds, table)

    stats = {
        "pages_processed": raw.count("\f") + 1,  # crude page count
        "chunks_created": len(chunks),
        "rows_inserted": inserted,
        "model": EMBED_MODEL,
    }
    _debug(f"Pipeline finished → {stats}")
    return stats
