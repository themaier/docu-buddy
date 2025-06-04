"""license_audit.py – Audit third‑party dependencies against an allowed‑license matrix.

Extends the original script by **also crawling each dependency’s repository** when the public
registries (npm, PyPI, Maven Central…) don’t expose an SPDX license.  Currently a best‑effort
Git‑first strategy is implemented (GitHub/GitLab/Bitbucket).  If a license is still unresolved,
``UNKNOWN`` is reported.

The script prints two tables:
1. A full report of every discovered dependency (✔︎ allowed / ✘ not allowed)
2. A compact *violations* table listing only the dependencies whose license did **not** match
   any embedding in the Supabase `matrix` table – this is the list you asked for.

Required environment variables (⚠ unchanged):
    OPENAI_API_KEY   – for embeddings
    SUPABASE_URL     – Supabase instance URL
    SUPABASE_KEY     – service key
    GITHUB_TOKEN     – optional, to raise GitHub API rate‑limit (recommended)

Add
    pip install supabase openai tiktoken requests lxml python‑slugify
if these libraries aren’t present.
"""

from __future__ import annotations

import json
import os
import re
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from supabase import create_client
import openai
import tiktoken

# ── configuration ─────────────────────────────────────────────────────────────
EMBED_MODEL     = "text-embedding-3-small"
TOKENIZER       = tiktoken.encoding_for_model(EMBED_MODEL)

SUPABASE_URL    = os.getenv("SUPABASE_URL")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
GITHUB_TOKEN    = os.getenv("GITHUB_TOKEN")  # optional, bumps GitHub rate-limit to 5k/h

if not all((SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY)):
    raise RuntimeError("Missing one of SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY env vars")

SB              = create_client(SUPABASE_URL, SUPABASE_KEY)

TABLE           = "matrix"
TOP_K           = 3
SIM_THRESHOLD   = 0.85
REQUEST_TIMEOUT = 10

# ── helpers ───────────────────────────────────────────────────────────────────

HEADERS_JSON = {"Accept": "application/json"}
if GITHUB_TOKEN:
    HEADERS_JSON["Authorization"] = f"Bearer {GITHUB_TOKEN}"

REPO_CACHE:    Dict[str, Optional[str]] = {}
LICENSE_NAME_CACHE: Dict[str, Optional[str]] = {}


def _safe_request(url: str, headers: dict | None = None) -> Optional[dict]:
    """Return parsed JSON for *url* or ``None`` on network / HTTP / JSON errors."""
    try:
        hdrs = {**HEADERS_JSON, **(headers or {})}
        resp = requests.get(url, headers=hdrs, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ████████ 1. Parse manifest files → dependency identifiers ███████████████████

_MANIFEST_PATTERNS = {
    "package.json":         lambda p: _deps_from_package_json(p),
    "requirements":         lambda p: _deps_from_requirements(p),
    "pom.xml":              lambda p: _deps_from_pom(p),
    "build.gradle":         lambda p: _deps_from_gradle(p),
    "build.gradle.kts":     lambda p: _deps_from_gradle(p),
    "ivy.xml":              lambda p: _deps_from_ivy(p),
    "build.sbt":            lambda p: _deps_from_sbt(p),
    "build.sc":             lambda p: _deps_from_sbt(p),
}

# ── JavaScript ────────────────────────────────────────────────────────────────

def _deps_from_package_json(path: Path) -> Set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    deps: Dict[str, str] = {}
    for field in ("dependencies", "devDependencies", "peerDependencies"):
        deps.update(data.get(field, {}))
    return set(deps.keys())

# ── Python ────────────────────────────────────────────────────────────────────

dep_line = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")

def _deps_from_requirements(path: Path) -> Set[str]:
    reqs: Set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.split("#", 1)[0].strip()
        if match := dep_line.match(line):
            reqs.add(match.group(1))
    return reqs

# ── Maven ─────────────────────────────────────────────────────────────────────

def _deps_from_pom(path: Path) -> Set[str]:
    ns = {"m": "http://maven.apache.org/POM/4.0.0"}
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except ET.ParseError:
        return set()
    deps: Set[str] = set()
    for dep in root.findall(".//m:dependency", ns):
        group    = dep.findtext("m:groupId", default="", namespaces=ns)
        artifact = dep.findtext("m:artifactId", default="", namespaces=ns)
        if artifact:
            deps.add(f"{group}:{artifact}" if group else artifact)
    return deps

# ── Gradle (Groovy/Kotlin DSL) ────────────────────────────────────────────────

_gradle_dep_re = re.compile(r"""[\"']([\w.\-]+):([\w.\-]+):([\w.\-]+)[\"']""")

def _deps_from_gradle(path: Path) -> Set[str]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return {f"{g}:{a}" for g, a, _ in _gradle_dep_re.findall(txt)}

# ── Ivy ───────────────────────────────────────────────────────────────────────

def _deps_from_ivy(path: Path) -> Set[str]:
    try:
        root = ET.fromstring(path.read_text(encoding="utf-8"))
    except ET.ParseError:
        return set()
    deps: Set[str] = set()
    for dep in root.findall("./dependencies/dependency"):
        org  = dep.get("org", "")
        name = dep.get("name")
        if name:
            deps.add(f"{org}:{name}" if org else name)
    return deps

# ── sbt / mill ────────────────────────────────────────────────────────────────

_sbt_dep_re = re.compile(r'"([\w.\-]+)"\s*%%?\s*"([\w.\-]+)"\s*%\s*"([\w.\-]+)"')

def _deps_from_sbt(path: Path) -> Set[str]:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return {f"{g}:{a}" for g, a, _ in _sbt_dep_re.findall(txt)}


# ████████ 2. Discover repository URL for a dependency ███████████████████████

def _repo_url_for_dep(dep: str) -> Optional[str]:
    if dep in REPO_CACHE:
        return REPO_CACHE[dep]

    repo_url: Optional[str] = None

    if "/" in dep or dep.islower():  # npm
        data = _safe_request(f"https://registry.npmjs.org/{dep}")
        if data:
            info = data.get("repository") or data.get("homepage")
            repo_url = info.get("url") if isinstance(info, dict) else info

    if repo_url is None and dep.replace("-", "_").islower():  # PyPI
        data = _safe_request(f"https://pypi.org/pypi/{dep}/json")
        if data:
            urls = data["info"].get("project_urls", {})
            repo_url = urls.get("Source") or urls.get("Homepage") or data["info"].get("home_page")

    if repo_url is None and ":" in dep:  # Maven Central
        group, artifact = dep.split(":", 1)
        q = f"g:\"{group}\" AND a:\"{artifact}\""
        solr = _safe_request(f"https://search.maven.org/solrsearch/select?q={q}&rows=1&wt=json")
        if solr and solr["response"]["numFound"]:
            doc = solr["response"]["docs"][0]
            repo_url = doc.get("scm") or doc.get("repositoryId") or doc.get("homepage")

    if repo_url:
        for prefix in ("git+", "scm:"):
            if repo_url.startswith(prefix):
                repo_url = repo_url[len(prefix):]
        if repo_url.endswith(".git"):
            repo_url = repo_url[:-4]

    REPO_CACHE[dep] = repo_url
    return repo_url


# ████████ 3. Embedding helper ██████████████████████████████████████████████

openai.api_key = OPENAI_API_KEY


def _embed(text: str) -> List[float]:
    """Request an embedding from OpenAI and print diagnostics so you see when
    the LLM is contacted."""
    snippet = textwrap.shorten(text, width=60, placeholder="…")
    tokens  = len(TOKENIZER.encode(text))
    print(f"[LLM] embedding request → model={EMBED_MODEL}, tokens={tokens}, text=\"{snippet}\"")
    try:
        emb = openai.embeddings.create(input=[text], model=EMBED_MODEL).data[0].embedding
    except Exception as exc:
        print(f"[LLM] ❌ embedding failed: {exc}")
        raise
    print(f"[LLM] embedding received ← dim={len(emb)}")
    return emb


# ████████ 4. Resolve dependency → SPDX license █████████████████████████████

def _heuristic_license_name(block: str) -> Optional[str]:
    s = block.lower()
    if "mit license" in s:
        return "MIT"
    if "apache license" in s:
        return "Apache-2.0"
    if "gnu general public license" in s or "gpl" in s:
        return "GPL"
    if "bsd license" in s:
        return "BSD"
    if "mozilla public license" in s:
        return "MPL"
    return None


def _license_from_github(repo_url: str) -> Optional[str]:
    m = re.search(r"github.com[:/](?P<owner>[\w.-]+)/(?P<repo>[\w.-]+)", repo_url)
    if not m:
        return None
    owner, repo = m.group("owner"), m.group("repo")
    api = f"https://api.github.com/repos/{owner}/{repo}/license"
    data = _safe_request(api)
    if data and (spdx := data.get("license", {}).get("spdx_id")) and spdx != "NOASSERTION":
        return spdx
    for branch in ("main", "master"):
        for name in ("LICENSE", "LICENSE.txt", "COPYING", "COPYING.txt"):
            raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{name}"
            try:
                txt = requests.get(raw, timeout=REQUEST_TIMEOUT).text[:1024]
            except Exception:
                continue
            guessed = _heuristic_license_name(txt)
            if guessed:
                return guessed
    return None


def _license_from_generic(repo_url: str) -> Optional[str]:
    for path in ("/LICENSE", "/LICENSE.txt"):
        url = repo_url.rstrip("/") + path
        try:
            txt = requests.get(url, timeout=REQUEST_TIMEOUT).text[:1024]
        except Exception:
            continue
        if txt.strip():
            guess = _heuristic_license_name(txt)
            if guess:
                return guess
    return None


def get_license(dep: str) -> Optional[str]:
    if dep in LICENSE_NAME_CACHE:
        return LICENSE_NAME_CACHE[dep]

    lic: Optional[str] = None

    # npm
    if "/" in dep or dep.islower():
        data = _safe_request(f"https://registry.npmjs.org/{dep}")
        if data:
            latest = data.get("dist-tags", {}).get("latest")
            meta   = data.get("versions", {}).get(latest, {})
            raw    = meta.get("license") or data.get("license")
            lic    = raw.strip() if isinstance(raw, str) else None

    # PyPI
    if lic is None and dep.replace("-", "_").islower():
        data
