"""SQLite persistence for chat feedback (golden dataset)."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("FEEDBACK_DB_PATH", "/data/feedback.db"))


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                context_id TEXT NOT NULL,
                message_id TEXT,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                citations TEXT NOT NULL DEFAULT '[]',
                rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                comment TEXT,
                source TEXT NOT NULL DEFAULT 'kubeflow-docs-bot'
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating)"
        )
        conn.commit()


@contextmanager
def get_connection():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_feedback(record: dict[str, Any]) -> dict[str, Any]:
    feedback_id = record.get("id") or str(uuid.uuid4())
    created_at = record.get("created_at") or datetime.now(timezone.utc).isoformat()
    citations = record.get("citations") or []
    if not isinstance(citations, list):
        citations = []

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO feedback (
                id, created_at, context_id, message_id, query, response,
                citations, rating, comment, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                feedback_id,
                created_at,
                record["context_id"],
                record.get("message_id"),
                record["query"],
                record["response"],
                json.dumps(citations),
                int(record["rating"]),
                record.get("comment"),
                record.get("source", "kubeflow-docs-bot"),
            ),
        )

    return {
        "id": feedback_id,
        "created_at": created_at,
        "context_id": record["context_id"],
        "message_id": record.get("message_id"),
        "rating": int(record["rating"]),
    }


def list_feedback(
    *,
    min_rating: int | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[Any] = []

    if min_rating is not None:
        clauses.append("rating >= ?")
        params.append(min_rating)

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.extend([limit, offset])

    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT id, created_at, context_id, message_id, query, response,
                   citations, rating, comment, source
            FROM feedback
            {where_sql}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()

    results: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["citations"] = json.loads(item.get("citations") or "[]")
        except json.JSONDecodeError:
            item["citations"] = []
        results.append(item)
    return results
