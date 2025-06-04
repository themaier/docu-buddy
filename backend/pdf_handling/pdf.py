import uuid
import fitz  # PyMuPDF
from openai import OpenAI
import requests

embedding_model = "text-embedding-ada-002"

def extract_text_from_pdf(file) -> list[str]:
    """Extract text from a PDF file-like object using PyMuPDF."""
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return paragraphs

def chunk_text(paragraphs: list[str], max_chars: int = 1000) -> list[str]:
    """Chunks text into parts of ~max_chars length."""
    chunks = []
    chunk = ""
    for para in paragraphs:
        if len(chunk) + len(para) > max_chars:
            chunks.append(chunk.strip())
            chunk = para
        else:
            chunk += " " + para
    if chunk:
        chunks.append(chunk.strip())
    return chunks

def get_embedding(text: str, openai_client: OpenAI) -> list[float]:
    """Get embedding using passed OpenAI client."""
    response = openai_client.embeddings.create(
        input=text,
        model=embedding_model
    )
    return response.data[0].embedding

def insert_to_supabase(supabase_url: str, supabase_key: str, table: str, content: str, embedding: list[float]) -> requests.Response:
    """Insert text and embedding to Supabase."""
    url = f"{supabase_url}/rest/v1/{table}"
    headers = {
        "apikey": supabase_key,
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "id": str(uuid.uuid4()),
        "content": content,
        "embedding": embedding
    }
    return requests.post(url, headers=headers, json=payload)

def process_pdf_and_upload(file, table: str, openai_key: str, supabase_url: str, supabase_key: str) -> dict:
    """Orchestrates full pipeline from PDF to Supabase."""
    paragraphs = extract_text_from_pdf(file)
    chunks = chunk_text(paragraphs)
    uploaded = 0

    print("---")
    print(openai_key)
    print("---")
    client = OpenAI(api_key=openai_key)

    for chunk in chunks:
        embedding = get_embedding(chunk, client)
        res = insert_to_supabase(supabase_url, supabase_key, table, chunk, embedding)
        if not res.ok:
            return {"error": res.text}
        uploaded += 1

    return {
        "message": f"Successfully uploaded {uploaded} chunks to '{table}'",
        "chunks_uploaded": uploaded
    }