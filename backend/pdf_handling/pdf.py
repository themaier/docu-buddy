import uuid
import fitz  # PyMuPDF
from openai import OpenAI
from supabase import create_client, Client
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(supabase_url, supabase_key)
openai_client = OpenAI(api_key=openai_key)

embedding_model = "text-embedding-ada-002"

def extract_text_from_pdf(file) -> list[str]:
    """Extract text from a PDF file-like object using PyMuPDF."""
    file.seek(0)
    doc = fitz.open(stream=file.read(), filetype="pdf")
    paragraphs = []
    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4].strip()
            if text:
                paragraphs.append(text)
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

def get_embedding(text: str) -> list[float]:
    """Get embedding using OpenAI client."""
    response = openai_client.embeddings.create(
        input=text,
        model=embedding_model
    )
    return response.data[0].embedding

def insert_to_supabase(table: str, content: str, embedding: list[float]):
    """Insert text and embedding to Supabase using supabase.Client."""
    data = {
        "id": str(uuid.uuid4()),
        "content": content,
        "embedding": embedding
    }
    response = supabase.table(table).insert(data).execute()
    return response

def process_pdf_and_upload(file, table: str) -> dict:
    """Main orchestration: extract text, embed, and upload to Supabase."""
    paragraphs = extract_text_from_pdf(file)
    chunks = chunk_text(paragraphs)
    uploaded = 0

    for chunk in chunks:
        embedding = get_embedding(chunk)
        res = insert_to_supabase(table, chunk, embedding)

        if res.status_code >= 400:
            return {"error": res.data}
        uploaded += 1

    return {
        "message": f"Successfully uploaded {uploaded} chunks to '{table}'",
        "chunks_uploaded": uploaded
    }
