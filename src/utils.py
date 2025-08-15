# utils.py

import os
import json
import hashlib
from functools import lru_cache
from typing import List, Tuple
from typing import Any
import yaml


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_filename_from_path(path: str) -> str:
    # Preserve extension; sanitize directories
    base = path.replace(os.sep, "__")
    return base


def add_line_numbers_preserve(original_code: str, start_line: int = 1) -> str:
    numbered = []
    for i, line in enumerate(original_code.splitlines(), start=start_line):
        # L123: actual code
        numbered.append(f"L{i:04d}: {line}")
    return "\n".join(numbered)


def chunk_code_by_lines(original_code: str, max_lines: int = 400) -> List[Tuple[int, str]]:
    """
    Returns list of (start_line_number, code_chunk_str) preserving original line numbers.
    """
    lines = original_code.splitlines()
    chunks = []
    for start in range(0, len(lines), max_lines):
        end = min(start + max_lines, len(lines))
        chunk_text = "\n".join(lines[start:end])
        chunks.append((start + 1, chunk_text))
    return chunks


def detect_language_from_extension(ext: str) -> str:
    ext = ext.lower()
    if ext in [".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx", ".h", ".c++", ".h++", ".C", ".CPP", ".tpp", ".ipp"]:
        return "cpp"
    if ext in [".py"]:
        return "python"
    if ext in [".js", ".ts", ".jsx", ".tsx"]:
        return "javascript"
    if ext in [".java"]:
        return "java"
    return "generic"


def save_json(path: str, data: dict):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_text(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def content_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def load_models_config(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def list_source_files(root: str, extensions: List[str]) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if any(fn.endswith(ext) for ext in extensions):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def extract_json_from_text(text: str) -> str:
    """
    Try to extract the first {...} JSON object from a text blob.
    Handles cases where models wrap JSON in prose or code fences.
    """
    import re
    # Prefer fenced blocks first
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    # Fallback: first { ... } balanced-ish
    braces = re.search(r"\{.*\}", text, re.DOTALL)
    if braces:
        return braces.group(0)
    return text  # best effort

# ---------- Token utilities ----------


@lru_cache(maxsize=64)
def _get_encoding_for_model(model: str):
    try:
        import tiktoken
    except Exception:
        raise RuntimeError(
            "tiktoken is required for token estimation. Please install it.")
    try:
        # Best effort: ask tiktoken for the model; fall back sensibly
        return tiktoken.encoding_for_model(model)
    except Exception:
        # Prefer o200k_base (4o-family); then cl100k_base
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")


def count_tokens_text(model: str, text: str) -> int:
    """
    Approximate token count for a single text message to a chat model.
    Does not include minor chat-message overhead; close enough for planning.
    """
    enc = _get_encoding_for_model(model)
    return len(enc.encode(text))


def get_model_name(llm: Any, default: str = "gpt-4o") -> str:
    """
    Best-effort retrieval of the model id from a LangChain ChatOpenAI (or similar) instance.
    Falls back to `default` if not found. Keeps our preflight robust across lib versions.
    """
    for attr in ("model", "model_name", "model_id"):
        val = getattr(llm, attr, None)
        if isinstance(val, str) and val:
            return val
    # Some wrappers keep config in a dict-like .kwargs/.model_kwargs; try those lightly
    for attr in ("kwargs", "model_kwargs"):
        d = getattr(llm, attr, None)
        if isinstance(d, dict) and isinstance(d.get("model"), str):
            return d["model"]
    return default
