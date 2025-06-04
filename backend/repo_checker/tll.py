import os
import re
from typing import Set

import json
from pprint import pprint
import openai
from supabase import create_client
import tiktoken

# ── env / config ──────────────────────────────────────────────────────────── #
openai.api_key = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

EMBED_MODEL = "text-embedding-3-small"
TOKENIZER = tiktoken.encoding_for_model(EMBED_MODEL)
SB = create_client(SUPABASE_URL, SUPABASE_KEY)

TABLE = "tll"          # vector table holding the PDF
TOP_K = 3              # chunks to retrieve per query
SIM_THRESHOLD = 0.28   # ANN match distance heuristic
# ──────────────────────────────────────────────────────────────────────────── #

# ── 1. 100 % LLM-based extension → language mapping ──────────────────────── #
_FEW_SHOT_EXAMPLES = [
    (".py",  "Python"),
    (".tsx", "React"),
    (".vue", "Vue.js"),
    (".rs",  "Rust"),
    (".go",  "Go"),
]

def _infer_lang_from_ext(ext: str) -> str:
    """
    Query GPT-4o-mini for the MOST LIKELY primary language / framework that
    owns files with `ext`.  Returns a short canonical name (≤2 words).
    """
    examples_txt = "\n".join(f"{e[0]} → {e[1]}" for e in _FEW_SHOT_EXAMPLES)

    prompt = (
        "You are a senior build-tool consultant. "
        "Given ONLY a software project file-extension (including its leading dot), "
        "output the single most probable programming language OR software framework "
        "associated with that extension. "
        "Respond with **just that name** (max. two words). "
        "Do NOT add punctuation, explanations, or quotes.\n\n"
        "Examples (DO NOT repeat):\n"
        f"{examples_txt}\n\n"
        f"Extension: {ext}"
    )

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    guess = resp.choices[0].message.content.strip()
    # Keep at most two words, Title-case for consistency
    return " ".join(guess.split()[:2]).title() or "Unknown"

# ── 2. walk the repo ──────────────────────────────────────────────────────── #
def _scan_extensions(repo_path: str) -> Set[str]:
    """Collect every unique extension (with leading dot) in the repo."""
    exts = set()
    for root, _, files in os.walk(repo_path):
        for fname in files:
            _, ext = os.path.splitext(fname)
            if ext:      # skip extension-less files
                exts.add(ext.lower())
    return exts

# ── 3. ask the PDF if that language/tool is allowed ──────────────────────── #
SIM_THRESHOLD = 0.35         # cosine distance (tune once)

def _compile_loose_regex(term: str) -> re.Pattern:
    """
    Same idea as before, but:
      • collapses spaces / hyphens
      • keeps a leading \b
      • adds a trailing \b **only if** the term ends with [A-Za-z0-9_]
    """
    norm = re.sub(r"\s+", " ", term.strip())
    parts = [
        re.escape(tok) if tok != " " else r"[\s\-]*"
        for tok in norm.split(" ")
    ]
    body = "".join(parts)

    # Leading boundary is always safe
    pattern = r"\b" + body

    # Add closing \b only when the last char is "word"-ish
    if re.search(r"\w$", norm):
        pattern += r"\b"
    else:
        pattern += r"(?!\w)"      # “not followed by a word char”

    return re.compile(pattern, flags=re.I)


def _allowed_or_not(term: str) -> bool:
    vec = openai.embeddings.create(
        input=[term], model=EMBED_MODEL
    ).data[0].embedding

    res = SB.rpc(
        "match_documents",
        {
            "query_embedding": vec,
            "match_threshold": 1 - SIM_THRESHOLD,  # cosine → distance
            "match_count":     TOP_K
        },
    ).execute()

    if not res.data:
        return False

    chunk  = res.data[0]["content"]
    dist   = res.data[0]["distance"]

    # case-, space- & hyphen-insensitive literal check
    if _compile_loose_regex(term).search(chunk):
        return True

    return dist < SIM_THRESHOLD

# ── 4. public API ─────────────────────────────────────────────────────────── #
def check_repo(repo_path: str = "./repo") -> dict:
    """
    Scans `repo_path` and returns:
        {
            "used_and_allowed":      [ "Python", "React", ... ],
            "used_but_not_allowed":  [ "Vue.js", ... ]
        }
    """
    extensions = _scan_extensions(repo_path)
    languages = {_infer_lang_from_ext(ext) for ext in extensions}
    print(languages)

    allowed, forbidden = [], []
    for lang in sorted(languages):
        (allowed if _allowed_or_not(lang) else forbidden).append(lang)

    return {
        "used_and_allowed": allowed,
        "used_but_not_allowed": forbidden,
    }
    
IGNORE_TERMS = {
    "json", "csv", "xml", "txt",           # plain data / meta files
    "yaml", "yml", "markdown", "md",
    "svg", "image", "archive",
    "sha256", "checksum", "temporary file",
    "json schema", "interpreter", "text file"
}
    
def categorise(repo_path: str = "./repo") -> dict:
    """
    Return three buckets:
      • ignored              – trivial data / meta file types
      • used_and_allowed     – present in repo AND allowed by PDF
      • used_but_not_allowed – present in repo but NOT allowed by PDF
    """
    raw = check_repo(repo_path)            # reuse existing logic

    ignored = []
    allowed = []
    forbidden = []

    for term in raw["used_and_allowed"]:
        (ignored if term.lower() in IGNORE_TERMS else allowed).append(term)

    for term in raw["used_but_not_allowed"]:
        (ignored if term.lower() in IGNORE_TERMS else forbidden).append(term)

    return {
        "ignored": ignored,
        "used_and_allowed": allowed,
        "used_but_not_allowed": forbidden,
    }

# ── 5. minimal runner ─────────────────────────────────────────────────────── #
if __name__ == "__main__":
    # <<< EDIT THESE TWO LINES ONLY >>> ------------------------------------- #
    REPO_PATH   = "./repo"   # folder you want to scan
    OUTPUT_JSON = False      # True → raw JSON, False → pretty-print
    # ----------------------------------------------------------------------- #
    result = categorise(REPO_PATH)
    if OUTPUT_JSON:
        print(json.dumps(result, ensure_ascii=False))
    else:
        pprint(result, width=80, sort_dicts=False)