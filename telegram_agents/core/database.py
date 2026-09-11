# Async SQLite layer: message log, agent cursors, conviction scores

from __future__ import annotations

import os
from typing import Any

import aiosqlite

DB_PATH = "./data/conversations.db"

_AGENTS = ("nihilist", "existentialist", "absurdist")


async def _connect(db_path: str) -> aiosqlite.Connection:
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    return db


def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
    return dict(row)


async def init_db(db_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
    async with await _connect(db_path) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS message_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                sender TEXT NOT NULL,
                sender_name TEXT NOT NULL,
                content TEXT NOT NULL,
                telegram_msg_id INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_cursor (
                agent TEXT PRIMARY KEY,
                last_seen_id INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS conviction_scores (
                agent TEXT PRIMARY KEY,
                score REAL NOT NULL DEFAULT 0.5,
                updated_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        await db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_message_log_chat_created
            ON message_log(chat_id, created_at)
            """
        )
        await db.commit()


async def log_message(
    chat_id: int,
    sender: str,
    sender_name: str,
    content: str,
    telegram_msg_id: int | None = None,
    db_path: str = DB_PATH,
) -> int:
    async with await _connect(db_path) as db:
        cursor = await db.execute(
            """
            INSERT INTO message_log (chat_id, sender, sender_name, content, telegram_msg_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (chat_id, sender, sender_name, content, telegram_msg_id),
        )
        await db.commit()
        return cursor.lastrowid


async def get_recent_messages(
    chat_id: int,
    limit: int = 20,
    db_path: str = DB_PATH,
) -> list[dict]:
    async with await _connect(db_path) as db:
        cursor = await db.execute(
            """
            SELECT * FROM (
                SELECT * FROM message_log
                WHERE chat_id = ?
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
            """,
            (chat_id, limit),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]


async def get_unprocessed_messages(
    agent: str,
    chat_id: int,
    db_path: str = DB_PATH,
) -> list[dict]:
    async with await _connect(db_path) as db:
        cursor = await db.execute(
            "SELECT last_seen_id FROM agent_cursor WHERE agent = ?",
            (agent,),
        )
        row = await cursor.fetchone()
        last_seen_id = row["last_seen_id"] if row else 0
        cursor = await db.execute(
            """
            SELECT * FROM message_log
            WHERE chat_id = ?
              AND id > ?
              AND sender != ?
            ORDER BY id ASC
            """,
            (chat_id, last_seen_id, agent),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]


async def advance_cursor(agent: str, last_id: int, db_path: str = DB_PATH) -> None:
    async with await _connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO agent_cursor (agent, last_seen_id)
            VALUES (?, ?)
            ON CONFLICT(agent) DO UPDATE SET last_seen_id = excluded.last_seen_id
            """,
            (agent, last_id),
        )
        await db.commit()


async def message_already_logged(
    telegram_msg_id: int,
    db_path: str = DB_PATH,
) -> bool:
    async with await _connect(db_path) as db:
        cursor = await db.execute(
            """
            SELECT 1 FROM message_log
            WHERE telegram_msg_id = ?
            LIMIT 1
            """,
            (telegram_msg_id,),
        )
        return await cursor.fetchone() is not None


async def clear_chat(chat_id: int, db_path: str = DB_PATH) -> None:
    async with await _connect(db_path) as db:
        await db.execute("DELETE FROM message_log WHERE chat_id = ?", (chat_id,))
        await db.execute("UPDATE agent_cursor SET last_seen_id = 0")
        await db.commit()


async def get_conviction(agent: str, db_path: str = DB_PATH) -> float:
    async with await _connect(db_path) as db:
        cursor = await db.execute(
            "SELECT score FROM conviction_scores WHERE agent = ?",
            (agent,),
        )
        row = await cursor.fetchone()
        return float(row["score"]) if row else 0.5


async def update_conviction(agent: str, delta: float, db_path: str = DB_PATH) -> float:
    current = await get_conviction(agent, db_path=db_path)
    new_score = max(0.0, min(1.0, current + delta))
    async with await _connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO conviction_scores (agent, score, updated_at)
            VALUES (?, ?, datetime('now'))
            ON CONFLICT(agent) DO UPDATE SET
                score = excluded.score,
                updated_at = datetime('now')
            """,
            (agent, new_score),
        )
        await db.commit()
    return new_score


async def get_all_convictions(db_path: str = DB_PATH) -> dict[str, float]:
    scores = {agent: 0.5 for agent in _AGENTS}
    async with await _connect(db_path) as db:
        cursor = await db.execute("SELECT agent, score FROM conviction_scores")
        rows = await cursor.fetchall()
        for row in rows:
            scores[row["agent"]] = float(row["score"])
    return scores
