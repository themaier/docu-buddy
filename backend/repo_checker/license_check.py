"""liccheck.py – complete pipeline (Phases 0‑8)
====================================================

A self‑contained script that

1. initialises OpenAI + Supabase clients (env: ``OPENAI_API_KEY``,
   ``SUPABASE_URL``, ``SUPABASE_KEY``)
2. lists recognised dependency‑manifest files under ``./repo``
3. uses GPT to extract up to **five** dependencies total
4. fetches each dependency’s licence (npm registry, PyPI, or Maven →
   fall‑back to GPT)
5. classifies to a single SPDX identifier
6. checks whether that identifier is already present in the embedded
   *Licence Matrix* (ANN search via Supabase RPC)
7. prints **JSON** – a list of ``[license, "pkg@version"]`` tuples – for
   all licences **not** found in the matrix

No files are written; the JSON appears on stdout.  There is no CLI –
just run ``python liccheck.py``.
"""
from __future__ import annotations

import json
import os
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import openai  # type: ignore
import requests  # type: ignore
from supabase import create_client  # type: ignore

# ────────────────────────────────────────────
# Environment & constants
# ────────────────────────────────────────────
SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_KEY: str = os.environ["SUPABASE_KEY"]
openai.api_key = os.environ["OPENAI_API_KEY"]

EMBED_MODEL: str = "text-embedding-3-small"
SIM_THRESHOLD: float = 0.85
TOP_K: int = 3

SB = create_client(SUPABASE_URL, SUPABASE_KEY)

# Manifest files we recognise
_MANIFEST_NAMES = {
    "package.json",
    "requirements.txt",
    "requirements.in",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "ivy.xml",
    "build.sbt",
    "build.sc",
}

# SPDX regex hints (very small – expanded on demand)
_SPDX_HINTS = {
    "MIT": re.compile(r"\bMIT\b", re.I),
    "Apache-2.0": re.compile(r"Apache\s*License.*2\.0", re.I),
    "GPL-3.0": re.compile(r"GNU\s+General\s+Public\s+License.*3", re.I),
}

# ────────────────────────────────────────────
# Data model
# ────────────────────────────────────────────
@dataclass(frozen=True)
class Dependency:
    name: str
    version: Optional[str]
    ecosystem: str  # "npm" | "pypi" | "maven" | "unknown"

    @property
    def display(self) -> str:
        return f"{self.name}:{self.version or 'latest'}" if self.version else self.name

# ────────────────────────────────────────────
# Licence‑matrix helpers
# ────────────────────────────────────────────

def _simplify(txt: str) -> str:
    return "".join(c.lower() for c in txt if c.isalnum())


def _compile_loose_regex(term: str) -> re.Pattern[str]:
    # allow flexible whitespace / punctuation between words
    escaped = re.escape(term)
    pattern = re.sub(r"\\\s+", r"\\s+", escaped)
    return re.compile(pattern, re.I)


def _allowed_or_not(term: str) -> bool:
    """True if *term* appears in the licence matrix embeddings."""
    vec = (
        openai.embeddings.create(input=[term], model=EMBED_MODEL)  # type: ignore
        .data[0]
        .embedding
    )

    res = SB.rpc(
        "match_documents",
        {
            "query_embedding": vec,
            "match_threshold": 1 - SIM_THRESHOLD,
            "match_count": TOP_K,
        },
    ).execute()

    if res.error:
        raise RuntimeError(f"Supabase RPC error: {res.error}")

    term_simple = _simplify(term)

    # 1) direct substring
    for row in res.data:
        if term_simple in _simplify(row["content"]):
            return True

    # 2) loose regex
    pattern = _compile_loose_regex(term)
    for row in res.data:
        if pattern.search(row["content"]):
            return True

    # 3) ANN distance check (if the RPC returns distance)
    if any(row.get("distance", 1.0) < SIM_THRESHOLD for row in res.data):
        return True

    return False

# ────────────────────────────────────────────
# Phase 2 – repo scan
# ────────────────────────────────────────────

def find_manifest_files(repo_root: Path) -> Iterable[Path]:
    for path in repo_root.rglob("*"):
        if path.is_file() and path.name in _MANIFEST_NAMES:
            yield path

# ────────────────────────────────────────────
# Phase 3 – GPT dependency extraction (max 5)
# ────────────────────────────────────────────

_DEP_SYSTEM_PROMPT = """You are a developer tool that extracts dependencies from build files.\n\nGiven the content of a build/manifest file, return a JSON list.\nEach element must be an object with keys: \"name\", \"version\" (may be null).\nReturn *only* the JSON list, with no extra keys or prose."""


def _gpt_extract_dependencies(content: str) -> List[Dependency]:
    messages = [
        {"role": "system", "content": _DEP_SYSTEM_PROMPT},
        {"role": "user", "content": content[:32_000]},  # safety truncation
    ]
    try:
        resp = openai.chat.completions.create(  # type: ignore
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as exc:
        raise RuntimeError(f"GPT extraction failed: {exc}") from exc

    deps: List[Dependency] = []
    for item in data:
        name: str = item.get("name")
        version: Optional[str] = item.get("version")
        if not name:
            continue
        ecosystem = _guess_ecosystem(name)
        deps.append(Dependency(name=name, version=version, ecosystem=ecosystem))
    return deps


def _guess_ecosystem(pkg_name: str) -> str:
    if "/" in pkg_name or pkg_name.startswith("@"):
        return "npm"
    if re.match(r"[A-Z]", pkg_name):
        # crude Maven heuristic (camel‑cased Java packages)
        return "maven"
    return "pypi"

# ────────────────────────────────────────────
# Phase 4 – fetch licence text / identifier
# ────────────────────────────────────────────

def _npm_license(dep: Dependency) -> Optional[str]:
    url = f"https://registry.npmjs.org/{dep.name}"
    try:
        j = requests.get(url, timeout=10).json()
    except Exception:
        return None
    version = dep.version or j.get("dist-tags", {}).get("latest")
    if not version or version not in j.get("versions", {}):
        return None
    info = j["versions"][version]
    lic = info.get("license")
    if isinstance(lic, dict):
        lic = lic.get("type")
    return lic


def _pypi_license(dep: Dependency) -> Optional[str]:
    url = f"https://pypi.org/pypi/{dep.name}/json"
    try:
        j = requests.get(url, timeout=10).json()
    except Exception:
        return None
    info = j.get("info", {})
    lic = info.get("license") or None
    if not lic:
        # try classifiers
        for c in info.get("classifiers", []):
            if c.startswith("License ::"):
                lic = c.split("::")[-1].strip()
                break
    return lic


def _maven_license(dep: Dependency) -> Optional[str]:
    # We usually need groupId:artifactId. Use search API.
    url = "https://search.maven.org/solrsearch/select"
    params = {"q": dep.name, "rows": 1, "wt": "json"}
    try:
        res = requests.get(url, params=params, timeout=10).json()
        docs = res.get("response", {}).get("docs", [])
    except Exception:
        docs = []

    if not docs:
        return None
    lic = docs[0].get("licenses")
    if isinstance(lic, list):
        lic = lic[0] if lic else None
    return lic


def _fallback_gpt_license(dep: Dependency) -> Optional[str]:
    prompt = textwrap.dedent(
        f"""You are a software‑licence assistant.\n\nGive me *only* the SPDX licence identifier (e.g. MIT, Apache‑2.0) for the open‑source package:\n{dep.display}\nIf unknown, reply \"UNKNOWN\"."""
    )
    messages = [
        {"role": "system", "content": "You answer with a single SPDX identifier or UNKNOWN"},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = openai.chat.completions.create(  # type: ignore
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
        )
        lic = resp.choices[0].message.content.strip()
        return lic if lic and lic != "UNKNOWN" else None
    except Exception:
        return None


def identify_license(dep: Dependency) -> Optional[str]:
    """Return SPDX identifier, or None if it can’t be found."""
    lic: Optional[str]
    if dep.ecosystem == "npm":
        lic = _npm_license(dep)
    elif dep.ecosystem == "pypi":
        lic = _pypi_license(dep)
    elif dep.ecosystem == "maven":
        lic = _maven_license(dep)
    else:
        lic = None

    if lic is None:
        lic = _fallback_gpt_license(dep)

    if lic is None:
        return None

    return _normalise_spdx(lic)


def _normalise_spdx(raw: str) -> str:
    raw = raw.strip()
    for spdx, rx in _SPDX_HINTS.items():
        if rx.search(raw):
            return spdx
    # fallback: remove whitespace
    return raw.replace(" ", "")

# ────────────────────────────────────────────
# Pipeline orchestrator
# ────────────────────────────────────────────

def run_scan() -> None:
    repo_root = Path("./repo")
    if not repo_root.exists():
        raise SystemExit("./repo not found – nothing to scan")

    # Phase 2 – collect manifests
    manifests = list(find_manifest_files(repo_root))

    # Phase 3 – extract deps (max 5)
    dependencies: List[Dependency] = []

    for m in manifests:
        if len(dependencies) >= 5:
            break
        content = m.read_text(encoding="utf-8", errors="ignore")
        try:
            deps = _gpt_extract_dependencies(content)
        except Exception as e:
            print(f"[warn] failed to parse {m}: {e}")
            continue
        for d in deps:
            if len(dependencies) >= 5:
                break
            if d not in dependencies:
                dependencies.append(d)

    # Phase 4‑6 – licence fetch → compare with matrix
    missing: List[Tuple[str, str]] = []  # (licence, dep disp)

    for dep in dependencies:
        licence = identify_license(dep)
        if not licence:
            continue  # couldn’t determine – ignore for now
        if not _allowed_or_not(licence):
            missing.append((licence, dep.display))

    # Phase 8 – print JSON
    print(json.dumps(missing, indent=2))

# ────────────────────────────────────────────
# Hard‑wired entry‑point
# ────────────────────────────────────────────
if __name__ == "__main__":
    run_scan()
