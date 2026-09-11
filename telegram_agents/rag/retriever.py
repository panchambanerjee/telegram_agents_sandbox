# Retrieves relevant passages from ChromaDB given a query

from __future__ import annotations

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_PATH = "./data/chroma"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 3


def _embedding_function(model_name: str = EMBEDDING_MODEL):
    return SentenceTransformerEmbeddingFunction(model_name=model_name)


def get_retriever(agent: str, chroma_path: str = CHROMA_PATH):
    client = chromadb.PersistentClient(path=chroma_path)
    ef = _embedding_function()
    name = f"{agent}_texts"
    try:
        collection = client.get_collection(name=name, embedding_function=ef)
    except Exception as exc:
        raise ValueError(f"Collection {name!r} does not exist") from exc

    if collection.count() == 0:
        raise ValueError(f"Collection {name!r} is empty")

    return {"collection": collection, "ef": ef}


def retrieve(
    agent: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    conviction_score: float = 0.5,
    chroma_path: str = CHROMA_PATH,
) -> list[dict]:
    retriever = get_retriever(agent, chroma_path=chroma_path)
    collection = retriever["collection"]

    if conviction_score < 0.35:
        query = f"{query} uncertainty doubt questioning"
    elif conviction_score > 0.65:
        query = f"{query} conviction certainty core argument"

    n_results = min(top_k, collection.count())
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    passages = []
    for text, metadata, distance in zip(documents, metadatas, distances):
        metadata = metadata or {}
        passages.append(
            {
                "text": text,
                "source": metadata.get("source"),
                "chunk_index": metadata.get("chunk_index"),
                "distance": distance,
            }
        )

    passages.sort(key=lambda item: item["distance"])
    return passages


def format_retrieved_passages(passages: list[dict], max_chars: int = 1200) -> str:
    blocks = []
    for passage in passages:
        source = passage.get("source", "")
        text = passage.get("text", "")
        blocks.append(f"[Source: {source}]\n{text}")
    formatted = "\n\n".join(blocks)
    return formatted[:max_chars]
