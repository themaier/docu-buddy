#!/usr/bin/env python3
# fmt: off
"""
Phase 1: Structural Code Complexity Pre-Analysis
Goal: Analyze codebase and rank functions/code parts by complexity
Output: Top 100 most complex code sections for further LLM analysis
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ComplexityMetrics:
    """Container for all complexity metrics"""

    cyclomatic_complexity: int = 0
    nesting_depth: int = 0
    function_length: int = 0
    parameter_count: int = 0
    cognitive_complexity: int = 0
    documentation_score: int = 0
    total_score: float = 0.0


class CodeComplexityAnalyzer:
    def __init__(self):
        # Language-specific patterns for different file types
        self.language_patterns = {
            "python": {
                "extensions": [".py"],
                "function_pattern": r"^\s*def\s+(\w+)\s*\(",
                "class_pattern": r"^\s*class\s+(\w+)",
                "branching_keywords": ["if", "elif", "for", "while", "try", "except", "with"],
                "comment_patterns": [r"#.*", r'"""[\s\S]*?"""', r"'''[\s\S]*?'''"],
            },
            "java": {
                "extensions": [".java"],
                "function_pattern": r"^\s*(?:public|protected|private)?\s*(?:static\s+)?(?:[\w<>\[\]]+\s+)+(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+(?:\s*,\s*\w+)*)?\s*\{",
                "class_pattern": r"\b(?:public|private)?\s*class\s+(\w+)",
                "branching_keywords": ["if", "else", "for", "while", "switch", "case", "try", "catch"],
                "comment_patterns": [r"//.*", r"/\*[\s\S]*?\*/"],
            },
            "go": {
                "extensions": [".go"],
                "function_pattern": r"\bfunc\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(",
                "class_pattern": r"\btype\s+(\w+)\s+struct",
                "branching_keywords": ["if", "for", "switch", "case", "select"],
                "comment_patterns": [r"//.*", r"/\*[\s\S]*?\*/"],
            },
            "csharp": {
                "extensions": [".cs"],
                "function_pattern": r"\b(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\(",
                "class_pattern": r"\b(?:public|private)?\s*class\s+(\w+)",
                "branching_keywords": ["if", "else", "for", "while", "switch", "case", "try", "catch"],
                "comment_patterns": [r"//.*", r"/\*[\s\S]*?\*/"],
            },
            "cpp": {
                "extensions": [".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"],
                "function_pattern": r"\b\w+\s+(\w+)\s*\([^)]*\)\s*{",
                "class_pattern": r"\bclass\s+(\w+)",
                "branching_keywords": ["if", "else", "for", "while", "switch", "case", "try", "catch"],
                "comment_patterns": [r"//.*", r"/\*[\s\S]*?\*/"],
            },
        }

        # Complexity scoring weights
        self.complexity_weights = {
            "cyclomatic_complexity": 3.0,
            "nesting_depth": 2.5,
            "function_length": 1.5,
            "parameter_count": 1.0,
            "cognitive_complexity": 2.0,
            "documentation_penalty": 2.0,  # Penalty for poor documentation
        }

    def should_skip_directory(self, dirpath: str) -> bool:
        """Check if directory should be skipped (infrastructure/non-code directories)"""
        skip_dirs = {".git", ".svn", ".hg", ".bzr", "node_modules", "bower_components", "vendor", 
                     "packages", ".gradle", ".maven", "target", "build", "bin", "obj", "out", ".vscode",
                     ".idea", ".eclipse", "__pycache__", ".pytest_cache", ".mypy_cache", "venv", "env", 
                     ".env", "virtualenv", "dist", "coverage", ".nyc_output", "logs", "log", "tmp", 
                     "temp", ".docker", "docker-compose", ".terraform", ".aws", "migrations", "assets", 
                     "static", "public", "resources", "docs", "documentation", "wiki", "test", "tests",
                     "spec", "specs", ".settings", ".metadata"}

        dir_name = os.path.basename(dirpath.rstrip(os.sep))
        return dir_name in skip_dirs or dir_name.startswith(".")

    def should_skip_file(self, filepath: str) -> bool:
        """Check if file should be skipped (infrastructure/config files)"""
        filename = os.path.basename(filepath).lower()

        # Skip common infrastructure and config files
        skip_files = {"package.json", "package-lock.json", "yarn.lock", "pom.xml", "build.gradle", 
                      "settings.gradle", "gradle.properties", "build.xml", "ivy.xml", "makefile", 
                      "cmake", "cmakecache.txt", "requirements.txt", "pipfile", "pipfile.lock", 
                      "poetry.lock", "composer.json", "composer.lock", "gemfile", "gemfile.lock", 
                      ".gitignore", ".gitattributes", ".gitmodules", ".dockerignore", "dockerfile", 
                      "readme.md", "readme.txt", "readme.rst", "license", "license.txt", "license.md", 
                      "changelog.md", "changelog.txt", "contributing.md", "code_of_conduct.md", 
                      ".editorconfig", ".eslintrc", ".prettierrc", "tsconfig.json", "jsconfig.json", 
                      ".babelrc", "webpack.config.js", ".travis.yml", ".circleci", "appveyor.yml", 
                      "jenkinsfile", ".github", "schema.sql", "seeds.sql", "todo.txt", "notes.txt", 
                      "manifest.mf", "meta-inf"}

        # Check exact filename matches
        if filename in skip_files:
            return True

        # Check file extensions for non-code files
        skip_extensions = {".md", ".txt", ".rst", ".pdf", ".doc", ".docx", ".json", ".xml", ".yaml", 
                           ".yml", ".ini", ".cfg", ".conf", ".properties", ".env", ".local", ".png", 
                           ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".bmp", ".mp3", ".mp4", ".avi", 
                           ".mov", ".wav", ".zip", ".tar", ".gz", ".7z", ".rar", ".exe", ".dll", ".so", 
                           ".dylib", ".jar", ".war", ".ear", ".db", ".sqlite", ".sqlite3", ".mdb", 
                           ".log", ".tmp", ".temp", ".cache", ".pem", ".key", ".crt", ".cert", ".g4", 
                           ".sh", ".bash", ".zsh", ".fish", ".bat", ".cmd", ".ps1", ".psm1", ".lock"}

        ext = Path(filepath).suffix.lower()
        return ext in skip_extensions

    def detect_language(self, filepath: str) -> str:
        """Detect programming language based on file extension"""
        # First check if we should skip this file
        if self.should_skip_file(filepath):
            return "skip"

        ext = Path(filepath).suffix.lower()

        for lang, config in self.language_patterns.items():
            if ext in config["extensions"]:
                return lang
        return "unknown"
    
    def strip_string_literals(self, line: str) -> str:
        """Remove all string literals (single and double quoted) from the line."""
        return re.sub(r'(["\'])(?:\\.|[^\\])*?\1', '', line)

    def extract_functions(self, content: str, language: str) -> List[Dict[str, Any]]:
        if language == "unknown":
            return []

        config = self.language_patterns[language]
        functions = []
        lines = content.splitlines()
        total_lines = len(lines)
        i = 0

        while i < total_lines:
            line = lines[i]
            func_match = re.search(config["function_pattern"], line)

            if func_match:
                func_name = func_match.group(1)
                start_line = i + 1
                function_lines = [line]

                # Capture full multi-line signature
                while "{" not in line and i + 1 < total_lines:
                    i += 1
                    line = lines[i]
                    function_lines.append(line)
                    if "{" in line:
                        break

                if "{" not in line:
                    i += 1
                    continue  # skip malformed/abstract

                # Begin brace counting (ignore braces inside strings)
                brace_count = self.strip_string_literals(line).count("{") - self.strip_string_literals(line).count("}")
                i += 1

                while i < total_lines and brace_count > 0:
                    line = lines[i]
                    code_only = self.strip_string_literals(line)
                    brace_count += code_only.count("{") - code_only.count("}")
                    function_lines.append(line)
                    i += 1

                end_line = start_line + len(function_lines) - 1
                functions.append({
                    "name": func_name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": function_lines,
                    "language": language,
                })
            else:
                i += 1

        return functions


    def calculate_cyclomatic_complexity(self, content: str, language: str) -> int:
        """Calculate cyclomatic complexity (number of decision points + 1)"""
        if language == "unknown":
            return 1

        config = self.language_patterns[language]
        complexity = 1  # Base complexity

        for keyword in config["branching_keywords"]:
            # Count branching keywords
            pattern = r"\b" + keyword + r"\b"
            complexity += len(re.findall(pattern, content, re.IGNORECASE))

        # Add complexity for boolean operators
        complexity += len(re.findall(r"\b(and|or|&&|\|\|)\b", content, re.IGNORECASE))

        return complexity

    def calculate_nesting_depth(self, content: str, language: str) -> int:
        """Calculate maximum nesting depth"""
        lines = content.split("\n")
        max_depth = 0
        current_depth = 0

        if language == "python":
            # Use indentation for Python
            for line in lines:
                if line.strip():
                    indent = len(line) - len(line.lstrip())
                    depth = indent // 4  # Assuming 4-space indentation
                    max_depth = max(max_depth, depth)
        else:
            # Use braces for other languages
            for line in lines:
                current_depth += line.count("{") - line.count("}")
                max_depth = max(max_depth, current_depth)

        return max_depth

    def calculate_function_length(self, content: str) -> int:
        """Calculate function length in lines of code (excluding empty lines and comments)"""
        lines = content.split("\n")
        code_lines = 0

        for line in lines:
            stripped = line.strip()
            if (
                stripped
                and not stripped.startswith("#")
                and not stripped.startswith("//")
            ):
                code_lines += 1

        return code_lines

    def count_parameters(self, function_content: str, language: str) -> int:
        """Count function parameters"""
        if language == "unknown":
            return 0

        # Extract function signature (first line typically)
        first_line = function_content.split("\n")[0]

        # Find parameter list in parentheses
        param_match = re.search(r"\(([^)]*)\)", first_line)
        if not param_match:
            return 0

        params = param_match.group(1).strip()
        if not params:
            return 0

        # Count commas + 1, but handle edge cases
        param_count = params.count(",") + 1

        # Adjust for empty parameters or self/this parameters
        if language == "python" and "self" in params:
            param_count -= 1

        return max(0, param_count)

    def calculate_cognitive_complexity(self, content: str, language: str) -> int:
        """Calculate cognitive complexity (more nuanced than cyclomatic)"""
        if language == "unknown":
            return 0

        config = self.language_patterns[language]
        cognitive_score = 0
        nesting_level = 0

        lines = content.split("\n")

        for line in lines:
            stripped = line.strip()

            # Increase nesting level
            if language == "python":
                if any(
                    keyword in stripped
                    for keyword in ["if", "for", "while", "try", "with"]
                ):
                    nesting_level += 1
                    cognitive_score += nesting_level
            else:
                if "{" in line:
                    nesting_level += 1
                if "}" in line:
                    nesting_level = max(0, nesting_level - 1)

                # Add complexity for control structures
                for keyword in config["branching_keywords"]:
                    if re.search(r"\b" + keyword + r"\b", stripped, re.IGNORECASE):
                        cognitive_score += nesting_level + 1

        return cognitive_score

    def calculate_documentation_score(self, content: str, language: str) -> int:
        """Calculate documentation quality score (0-10)"""
        if language == "unknown":
            return 5

        config = self.language_patterns[language]
        lines = content.split("\n")
        total_lines = len([l for l in lines if l.strip()])

        if total_lines == 0:
            return 5

        comment_lines = 0

        # Count comment lines
        for pattern in config["comment_patterns"]:
            comment_lines += len(re.findall(pattern, content, re.MULTILINE))

        # Calculate documentation ratio
        doc_ratio = comment_lines / total_lines if total_lines > 0 else 0

        # Score based on ratio (0-10 scale)
        if doc_ratio >= 0.3:
            return 10
        elif doc_ratio >= 0.2:
            return 8
        elif doc_ratio >= 0.15:
            return 6
        elif doc_ratio >= 0.1:
            return 4
        elif doc_ratio >= 0.05:
            return 2
        else:
            return 0

    def analyze_function(self, function_data: Dict[str, Any]) -> ComplexityMetrics:
        """Analyze a single function and return complexity metrics"""
        content = "\n".join(function_data["content"])
        language = function_data["language"]

        metrics = ComplexityMetrics()

        # Calculate individual metrics
        metrics.cyclomatic_complexity = self.calculate_cyclomatic_complexity(
            content, language
        )
        metrics.nesting_depth = self.calculate_nesting_depth(content, language)
        metrics.function_length = self.calculate_function_length(content)
        metrics.parameter_count = self.count_parameters(content, language)
        metrics.cognitive_complexity = self.calculate_cognitive_complexity(
            content, language
        )
        metrics.documentation_score = self.calculate_documentation_score(
            content, language
        )

        # Calculate total weighted score
        total_score = (
            metrics.cyclomatic_complexity
            * self.complexity_weights["cyclomatic_complexity"]
            + metrics.nesting_depth * self.complexity_weights["nesting_depth"]
            + (metrics.function_length / 10)
            * self.complexity_weights["function_length"]
            + metrics.parameter_count * self.complexity_weights["parameter_count"]
            + metrics.cognitive_complexity
            * self.complexity_weights["cognitive_complexity"]
        )

        # Apply documentation penalty
        doc_penalty = (
            (10 - metrics.documentation_score)
            / 10
            * self.complexity_weights["documentation_penalty"]
        )
        total_score += doc_penalty

        metrics.total_score = total_score

        return metrics

    def analyze_codebase(self, root_path: str) -> List[Dict[str, Any]]:
        """Analyze entire codebase and return ranked complexity results"""
        results = []
        skipped_dirs = set()

        # Walk through all files
        for root, dirs, files in os.walk(root_path):
            # Skip infrastructure directories
            if self.should_skip_directory(root):
                if root not in skipped_dirs:
                    print(f"Skipping directory: {root}")
                    skipped_dirs.add(root)
                continue

            # Remove directories that should be skipped from dirs list
            # This prevents os.walk from entering them
            dirs[:] = [
                d for d in dirs if not self.should_skip_directory(os.path.join(root, d))
            ]

            for file in files:
                filepath = os.path.join(root, file)
                language = self.detect_language(filepath)

                if language in ["unknown", "skip"]:
                    continue

                try:
                    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # Extract functions from file
                    functions = self.extract_functions(content, language)

                    for func in functions:
                        metrics = self.analyze_function(func)
                        rel_path = os.path.relpath(filepath, root_path).replace(
                            "\\", "/"
                        )
                        file_url = f"file:///{filepath}"
                        # github_repo_url should be passed in or set globally
                        github_url = f"{self.github_repo_url}{rel_path}#L{func['start_line']}-L{func['end_line']}"

                        result = {
                            "function_name": func["name"],
                            "file_url": file_url,
                            "github_url": github_url,
                            "start_line": func["start_line"],
                            "end_line": func["end_line"],
                            "language": language,
                            "rule_analysis": {
                                "cyclomatic_complexity": metrics.cyclomatic_complexity,
                                "nesting_depth": metrics.nesting_depth,
                                "function_length": metrics.function_length,
                                "parameter_count": metrics.parameter_count,
                                "cognitive_complexity": metrics.cognitive_complexity,
                                "documentation_score": metrics.documentation_score,
                                "rule_score": metrics.total_score,
                            },
                        }
                        results.append(result)

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    continue

        # Sort by complexity score (descending) and return top 100
        results.sort(key=lambda x: x["rule_analysis"]["rule_score"], reverse=True)
        return results[:100]


def main(repo_url: str):
    """Analyze a codebase for function complexity and output the results."""

    analyzer = CodeComplexityAnalyzer()
    codebase_path = r"./repo"
    analyzer.github_repo_url = repo_url
    print(f"\n🔍 Analyzing codebase at: {codebase_path}...\n")
    top_complex_functions = analyzer.analyze_codebase(codebase_path)
    total_files_analyzed = len({func["file_url"] for func in top_complex_functions})
    languages_found = sorted({func["language"] for func in top_complex_functions})
    summary = (
        "\n📌 Analysis Summary:\n"
        f"   📂 Files analyzed: {total_files_analyzed}\n"
        f"   🧬 Languages found: {', '.join(languages_found)}\n"
        f"   🔍 Functions analyzed: {len(top_complex_functions)}\n"
        "\n✅ Results saved to complex_functions.json\n"
    )
    print(summary)
    with open("./complex_functions.json", "w", encoding="utf-8") as f:
        json.dump(top_complex_functions, f, indent=2)


if __name__ == "__main__":
    main("https://github.com/openrewrite/rewrite/blob/main/")
