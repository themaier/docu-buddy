import os
import re
import json
from pprint import pprint
from typing import Dict, List, Tuple

import openai
from supabase import create_client
import tiktoken
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple


# ── env / config ──────────────────────────────────────────────────────────── #
openai.api_key  = os.getenv("OPENAI_API_KEY")
SUPABASE_URL    = os.getenv("SUPABASE_URL")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY")

EMBED_MODEL = "text-embedding-3-small"
TOKENIZER   = tiktoken.encoding_for_model(EMBED_MODEL)
SB          = create_client(SUPABASE_URL, SUPABASE_KEY)

TABLE          = "tll"         # vector table holding the PDF
TOP_K          = 3             # chunks to retrieve per query
SIM_THRESHOLD  = 0.85          # ANN match distance heuristic
# ──────────────────────────────────────────────────────────────────────────── #


# ── 1. 100 % LLM-based extension → language mapping ───────────────────────── #
_FEW_SHOT_EXAMPLES = [
    (".py",  "Python"),
    (".tsx", "React"),
    (".vue", "Vue.js"),
    (".rs",  "Rust"),
    (".go",  "Go"),
    (".bat", "Batch"),
    (".txt", "Text"),
]

def _infer_lang_from_ext(ext: str) -> str:
    """Return the most probable language / framework for `ext` (≤ 2 words)."""
    examples_txt = "\n".join(f"{e[0]} → {e[1]}" for e in _FEW_SHOT_EXAMPLES)

    prompt = (
        "You are a senior build-tool consultant. "
        "Given ONLY a software project file-extension (including its leading dot), "
        "output the single most probable programming language OR software framework "
        "associated with that extension. "
        "Respond with **just that name** (max. two words). "
        "Do NOT add punctuation, explanations, or quotes.\n\n"
        "Do NOT add file, or script or other phrases as a second word to the output."
        "Examples (DO NOT repeat):\n"
        f"{examples_txt}\n\n"
        f"Extension: {ext}"
    )

    resp  = openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    guess = resp.choices[0].message.content.strip()
    return " ".join(guess.split()[:2]).title() or "Unknown"


# ── 2. walk the repo ──────────────────────────────────────────────────────── #
def _scan_extensions(repo_path: str) -> List[str]:
    """Collect every unique extension (with leading dot) in the repo."""
    exts = set()
    for root, _, files in os.walk(repo_path):
        for fname in files:
            _, ext = os.path.splitext(fname)
            if ext:                             # skip extension-less files
                exts.add(ext.lower())
    return sorted(exts)


# ── 3. ask the PDF if that language/tool is allowed ──────────────────────── #
def _compile_loose_regex(term: str) -> re.Pattern:
    """
    Collapses whitespace/hyphens and adds word-boundaries
    so that “Csharp” matches “C-Sharp”, etc.
    """
    norm  = re.sub(r"\s+", " ", term.strip())
    parts = [
        re.escape(tok) if tok != " " else r"[\s\-]*"
        for tok in norm.split(" ")
    ]
    pattern = r"\b" + "".join(parts)
    pattern += r"\b" if re.search(r"\w$", norm) else r"(?!\w)"
    return re.compile(pattern, flags=re.I)

def _simplify(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()

def _allowed_or_not(term: str) -> bool:
    """Returns True if `term` is allowed according to the PDF."""
    # ① embed the term
    vec = openai.embeddings.create(
        input=[term], model=EMBED_MODEL
    ).data[0].embedding

    # ② query Supabase
    res = SB.rpc(
        "match_documents",
        {
            "query_embedding": vec,
            "match_threshold": 1 - SIM_THRESHOLD,
            "match_count": TOP_K,
        },
    ).execute()

    term_simple = _simplify(term)

    # ③ cheap contains-check
    for row in res.data:
        if term_simple in _simplify(row["content"]):
            return True

    # ④ fallback regex
    pattern = _compile_loose_regex(term)
    for row in res.data:
        if pattern.search(row["content"]):
            return True

    # ⑤ ANN distance check
    if any(row.get("distance", 1.0) < SIM_THRESHOLD for row in res.data):
        return True

    return False


# ── 4. public API ─────────────────────────────────────────────────────────── #
IGNORE_TERMS = {
    "json", "csv", "xml", "txt",           # plain data / meta files
    "yaml", "yml", "markdown", "md",
    "svg", "image", "archive",
    "sha256", "checksum", "temporary",
    "json schema", "interpreter", "text",
    "shell", "batch", "2",
}

def _is_ignored(ext: str, lang: str) -> bool:
    """
    Ignore if either the bare extension or the language name
    is in the IGNORE_TERMS set (case-insensitive).
    """
    bare_ext = ext.lstrip(".").lower()
    return bare_ext in IGNORE_TERMS or lang.lower() in IGNORE_TERMS

def categorise(repo_path: str = "./repo") -> Dict[str, List[Tuple[str, str]]]:
    """
    Returns three buckets of **(extension, language)** tuples:
      • ignored
      • used_and_allowed
      • used_but_not_allowed
    """
    # ① find extensions
    exts = _scan_extensions(repo_path)

    # ② map ext → language
    ext_lang: Dict[str, str] = {ext: _infer_lang_from_ext(ext) for ext in exts}

    used_and_allowed     : List[Tuple[str, str]] = []
    used_but_not_allowed : List[Tuple[str, str]] = []
    ignored              : List[Tuple[str, str]] = []

    for ext, lang in ext_lang.items():
        if _is_ignored(ext, lang):
            ignored.append((ext, lang))
            continue

        if _allowed_or_not(lang):
            used_and_allowed.append((ext, lang))
        else:
            used_but_not_allowed.append((ext, lang))

    # sort each bucket for stable output
    for bucket in (ignored, used_and_allowed, used_but_not_allowed):
        bucket.sort()

    return {
        "ignored":              ignored,
        "used_and_allowed":     used_and_allowed,
        "used_but_not_allowed": used_but_not_allowed,
    }
    

def _collect_files_with_exts(repo_path: str, exts: Set[str]) -> List[Path]:
    """Return every file under repo_root whose extension is in `exts`."""
    matches: List[Path] = []
    for root, _, files in os.walk(repo_path):
        for fn in files:
            if Path(fn).suffix.lower() in exts:
                matches.append(Path(root, fn).resolve())
    return matches


def list_excluded_files_with_links(
    repo_path: str = "./repo",
    include_ignored: bool = False,
) -> Dict[str, List[str]]:
    """
    1. Runs `categorise(repo_path)` to identify excluded extensions.  
       • Always includes 'used_but_not_allowed'.  
       • Optionally also includes 'ignored' when `include_ignored=True`.
    2. Finds every file with those extensions and converts the path to a
       GitHub URL of the form:
           https://github.com/USER/REPO/blob/BRANCH/rel/path/to/file

    Returns a mapping:
        {
            ".vue": [
                "https://github.com/USER/REPO/blob/main/src/App.vue",
                ...
            ],
            ".csv": [
                ...
            ]
        }
    """
    report          = categorise(repo_path)
    excluded_buckets = ["used_but_not_allowed"]
    if include_ignored:
        excluded_buckets.append("ignored")

    # gather the set of excluded extensions (lower-case, leading dot)
    excluded_exts: Set[str] = {
        ext.lower()
        for bucket in excluded_buckets
        for ext, _ in report.get(bucket, [])
    }

    if not excluded_exts:
        return {}

    # collect files & build URLs
    remote_base, branch = "https://github.com/openrewrite/rewrite", "main"
    repo_root           = Path(repo_path).resolve()
    url_map: Dict[str, List[str]] = {ext: [] for ext in excluded_exts}

    for path in _collect_files_with_exts(repo_path, excluded_exts):
        rel_path = path.relative_to(repo_root).as_posix()
        url      = f"{remote_base}/blob/{branch}/{rel_path}"
        url_map[path.suffix.lower()].append(url)

    # keep deterministic ordering
    for urls in url_map.values():
        urls.sort()

    return url_map




def _insert_supabase(
    link_map: Dict[str, List[str]],          # {".vue": [url1, url2], ...}
    *,
    table: str = "disallowed_files",
) -> int:
    """
    Flatten `link_map` and push rows {file_url, extension, name} to Supabase.

    • `name` is looked up once per extension via `_infer_lang_from_ext`.  
    • No dedup / upsert – runs the simplest possible `.insert()` call.
    """
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    sb.table(table).delete().neq("id", 0).execute()
    ext_lang = {ext: _infer_lang_from_ext(ext) for ext in link_map}
    rows = [
        {"file_url": url, "extension": ext, "name": ext_lang[ext]}
        for ext, urls in link_map.items()
        for url in urls
    ]

    # 4️⃣  push the fresh data
    sb.table(table).insert(rows).execute()
    return len(rows)

if __name__ == "__main__":
    REPO_PATH         = "./repo"
    OUTPUT_EXCLUDED   = True          # <- toggle to print the link list
    OUTPUT_JSON_LINKS = False         # <- toggle JSON vs pretty print

    if OUTPUT_EXCLUDED:
        link_map = list_excluded_files_with_links(REPO_PATH, include_ignored=False)
        inserted = _insert_supabase(link_map)
        if OUTPUT_JSON_LINKS:
            print(json.dumps(link_map, indent=2, ensure_ascii=False))
        else:
            pprint(link_map, width=120, sort_dicts=False)