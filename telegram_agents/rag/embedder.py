# Chunks source texts and loads embeddings into ChromaDB

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logging.getLogger("pypdf").setLevel(logging.ERROR)

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader

CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma")
TEXTS_PATH = os.getenv("TEXTS_PATH", "./texts")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400        # tokens (approximate, use word count / 0.75)
CHUNK_OVERLAP = 50
MIN_PAGE_WORDS = 40

_AGENTS = ("nihilist", "existentialist", "absurdist")

# Folder names on disk may use the tradition (existentialism) rather than the handle.
_AGENT_DIRS = {
    "nihilist": ("nihilist", "nihilism"),
    "existentialist": ("existentialist", "existentialism"),
    "absurdist": ("absurdist", "absurdism"),
}

_SOFT_HYPHEN = re.compile(r"[\u00ad]")
_LINE_BREAK_HYPHEN = re.compile(r"(\w)[\-‐‑]\s+(\w)")
_WHITESPACE = re.compile(r"\s+")


def get_chroma_client(chroma_path: str = CHROMA_PATH) -> chromadb.Client:
    return chromadb.PersistentClient(path=chroma_path)


def get_embedding_function(model_name: str = EMBEDDING_MODEL):
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = _SOFT_HYPHEN.sub("", text)
    text = _LINE_BREAK_HYPHEN.sub(r"\1\2", text)
    text = text.replace("\r", "\n")
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    words = clean_text(text).split()
    if not words:
        return []
    overlap = min(overlap, max(chunk_size - 1, 0))
    step = max(chunk_size - overlap, 1)
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def _extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        text = clean_text(raw)
        if len(text.split()) < MIN_PAGE_WORDS:
            continue
        pages.append((index, text))
    return pages


def _read_source(path: Path) -> tuple[str, list[int]]:
    """Return full text and a page number per word (1-based; 0 if unknown)."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        pages = _extract_pdf_pages(path)
        words: list[str] = []
        page_of_word: list[int] = []
        for page_num, text in pages:
            page_words = text.split()
            words.extend(page_words)
            page_of_word.extend([page_num] * len(page_words))
        return " ".join(words), page_of_word
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    words = text.split()
    return text, [0] * len(words)


def _chunk_words(
    words: list[str],
    page_of_word: list[int],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[str, int, int]]:
    if not words:
        return []
    overlap = min(overlap, max(chunk_size - 1, 0))
    step = max(chunk_size - overlap, 1)
    chunks: list[tuple[str, int, int]] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        page_slice = page_of_word[start:end]
        page_start = next((p for p in page_slice if p), 0)
        page_end = next((p for p in reversed(page_slice) if p), 0)
        chunks.append((" ".join(words[start:end]), page_start, page_end))
        if end >= len(words):
            break
        start += step
    return chunks


def _source_files(agent: str, texts_path: str) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for folder in _AGENT_DIRS.get(agent, (agent,)):
        agent_dir = Path(texts_path) / folder
        if not agent_dir.is_dir():
            continue
        for pattern in ("*.txt", "*.md", "*.pdf"):
            for path in sorted(agent_dir.glob(pattern)):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                files.append(path)
    return files


def load_texts_for_agent(
    agent: str,
    texts_path: str = TEXTS_PATH,
    chroma_path: str = CHROMA_PATH,
) -> int:
    files = _source_files(agent, texts_path)
    if not files:
        print(f"[RAG] No source files found for {agent} under {texts_path}")
        return 0

    client = get_chroma_client(chroma_path)
    ef = get_embedding_function()
    collection = client.get_or_create_collection(
        name=f"{agent}_texts",
        embedding_function=ef,
    )

    existing_ids = set(collection.get(include=[]).get("ids") or [])
    total_loaded = 0

    for path in files:
        filename = path.name
        stem = path.stem
        already = [chunk_id for chunk_id in existing_ids if chunk_id.startswith(f"{agent}_{stem}_")]
        if already:
            print(f"[RAG] {filename} already loaded ({len(already)} chunks)")
            continue

        print(f"[RAG] Reading {path}...")
        text, page_of_word = _read_source(path)
        words = text.split()
        chunk_rows = _chunk_words(words, page_of_word)
        if not chunk_rows:
            print(f"[RAG] Skipping {path.name}: no extractable text")
            continue

        ids = [f"{agent}_{stem}_{i}" for i in range(len(chunk_rows))]

        documents = [row[0] for row in chunk_rows]
        metadatas = []
        for i, (_, page_start, page_end) in enumerate(chunk_rows):
            meta: dict = {"source": filename, "agent": agent, "chunk_index": i}
            if page_start:
                meta["page_start"] = page_start
                meta["page_end"] = page_end
            metadatas.append(meta)

        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        existing_ids.update(ids)
        total_loaded += len(chunk_rows)
        print(f"[RAG] Loaded {len(chunk_rows)} chunks from {filename}")

    return total_loaded


def load_all_texts(
    texts_path: str = TEXTS_PATH,
    chroma_path: str = CHROMA_PATH,
) -> dict[str, int]:
    return {
        agent: load_texts_for_agent(agent, texts_path=texts_path, chroma_path=chroma_path)
        for agent in _AGENTS
    }
