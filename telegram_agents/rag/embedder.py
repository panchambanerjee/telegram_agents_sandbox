# Chunks source texts and loads embeddings into ChromaDB

from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_PATH = "./data/chroma"
TEXTS_PATH = "./texts"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400        # tokens (approximate, use word count / 0.75)
CHUNK_OVERLAP = 50

_AGENTS = ("nihilist", "existentialist", "absurdist")


def get_chroma_client(chroma_path: str = CHROMA_PATH) -> chromadb.Client:
    return chromadb.PersistentClient(path=chroma_path)


def get_embedding_function(model_name: str = EMBEDDING_MODEL):
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    words = text.split()
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


def load_texts_for_agent(
    agent: str,
    texts_path: str = TEXTS_PATH,
    chroma_path: str = CHROMA_PATH,
) -> int:
    agent_dir = Path(texts_path) / agent
    if not agent_dir.is_dir():
        return 0

    client = get_chroma_client(chroma_path)
    ef = get_embedding_function()
    collection = client.get_or_create_collection(
        name=f"{agent}_texts",
        embedding_function=ef,
    )

    existing_ids = set(collection.get(include=[]).get("ids") or [])
    total_loaded = 0

    for txt_path in sorted(agent_dir.glob("*.txt")):
        text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(text)
        if not chunks:
            continue

        filename = txt_path.name
        stem = txt_path.stem
        ids = [f"{agent}_{stem}_{i}" for i in range(len(chunks))]
        if all(chunk_id in existing_ids for chunk_id in ids):
            continue

        collection.upsert(
            ids=ids,
            documents=chunks,
            metadatas=[
                {"source": filename, "agent": agent, "chunk_index": i}
                for i in range(len(chunks))
            ],
        )
        existing_ids.update(ids)
        total_loaded += len(chunks)

    return total_loaded


def load_all_texts(
    texts_path: str = TEXTS_PATH,
    chroma_path: str = CHROMA_PATH,
) -> dict[str, int]:
    return {
        agent: load_texts_for_agent(agent, texts_path=texts_path, chroma_path=chroma_path)
        for agent in _AGENTS
    }
