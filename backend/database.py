"""
SQLite persistence layer for the gallery.

Connection helpers, schema migrations (PRAGMA user_version based), and CRUD
operations for images and analyses. FTS5 index kept in sync via triggers.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from backend.config import settings

logger = logging.getLogger(__name__)


# ── Connection ────────────────────────────────────────────────────────────

def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """Short-lived connection for a single request."""
    conn = _connect(settings.database_file)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Migrations ────────────────────────────────────────────────────────────

def _migration_v1(conn: sqlite3.Connection) -> None:
    """Initial schema: images, analyses, FTS5 index + sync triggers."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
            name TEXT,
            keywords TEXT,
            description TEXT,
            search_query TEXT,
            prompt_used TEXT,
            model_used TEXT,
            duration_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_analyses_image_id ON analyses(image_id);
        CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at DESC);

        CREATE VIRTUAL TABLE IF NOT EXISTS analyses_fts USING fts5(
            name, keywords, description, search_query,
            content='analyses', content_rowid='id'
        );

        CREATE TRIGGER IF NOT EXISTS analyses_ai AFTER INSERT ON analyses BEGIN
            INSERT INTO analyses_fts(rowid, name, keywords, description, search_query)
            VALUES (
                new.id,
                new.name,
                REPLACE(REPLACE(COALESCE(new.keywords, ''), '[', ''), ']', ''),
                new.description,
                new.search_query
            );
        END;

        CREATE TRIGGER IF NOT EXISTS analyses_ad AFTER DELETE ON analyses BEGIN
            INSERT INTO analyses_fts(analyses_fts, rowid, name, keywords, description, search_query)
            VALUES (
                'delete',
                old.id,
                old.name,
                REPLACE(REPLACE(COALESCE(old.keywords, ''), '[', ''), ']', ''),
                old.description,
                old.search_query
            );
        END;

        CREATE TRIGGER IF NOT EXISTS analyses_au AFTER UPDATE ON analyses BEGIN
            INSERT INTO analyses_fts(analyses_fts, rowid, name, keywords, description, search_query)
            VALUES (
                'delete',
                old.id,
                old.name,
                REPLACE(REPLACE(COALESCE(old.keywords, ''), '[', ''), ']', ''),
                old.description,
                old.search_query
            );
            INSERT INTO analyses_fts(rowid, name, keywords, description, search_query)
            VALUES (
                new.id,
                new.name,
                REPLACE(REPLACE(COALESCE(new.keywords, ''), '[', ''), ']', ''),
                new.description,
                new.search_query
            );
        END;
    """)


MIGRATIONS = [_migration_v1]


def init_db() -> None:
    """Apply any pending migrations. Called at app startup."""
    db_file = settings.database_file
    logger.info(f"Initialisiere SQLite-Datenbank: {db_file}")
    with _connect(db_file) as conn:
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        target = len(MIGRATIONS)
        if current >= target:
            logger.info(f"DB-Schema aktuell (v{current}).")
            return
        for idx in range(current, target):
            version = idx + 1
            logger.info(f"Migration v{version} wird angewendet…")
            MIGRATIONS[idx](conn)
            conn.execute(f"PRAGMA user_version = {version}")
            conn.commit()
        logger.info(f"DB-Schema auf v{target} aktualisiert.")


# ── CRUD: images ──────────────────────────────────────────────────────────

def insert_image(
    conn: sqlite3.Connection,
    filename: str,
    original_name: str,
    file_path: str,
    file_size: int,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO images (filename, original_name, file_path, file_size)
        VALUES (?, ?, ?, ?)
        """,
        (filename, original_name, file_path, file_size),
    )
    return cur.lastrowid


def get_image(conn: sqlite3.Connection, image_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM images WHERE id = ?", (image_id,)
    ).fetchone()


def delete_image(conn: sqlite3.Connection, image_id: int) -> sqlite3.Row | None:
    """Delete image row. Returns the row before deletion (for file cleanup) or None."""
    row = get_image(conn, image_id)
    if row is None:
        return None
    conn.execute("DELETE FROM images WHERE id = ?", (image_id,))
    return row


def list_images(
    conn: sqlite3.Connection,
    page: int = 1,
    per_page: int = 20,
    sort: str = "created_at",
    order: str = "desc",
) -> tuple[list[dict[str, Any]], int]:
    """List images with their most recent analysis. Returns (rows, total_count)."""
    allowed_sort = {"created_at", "id", "original_name"}
    allowed_order = {"asc", "desc"}
    if sort not in allowed_sort:
        sort = "created_at"
    if order.lower() not in allowed_order:
        order = "desc"

    offset = max(0, (page - 1) * per_page)
    total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    rows = conn.execute(
        f"""
        SELECT
            i.*,
            a.id           AS analysis_id,
            a.name         AS analysis_name,
            a.keywords     AS analysis_keywords,
            a.description  AS analysis_description,
            a.search_query AS analysis_search_query,
            a.model_used   AS analysis_model_used,
            a.created_at   AS analysis_created_at
        FROM images i
        LEFT JOIN (
            SELECT a1.*
            FROM analyses a1
            JOIN (
                SELECT image_id, MAX(id) AS max_id
                FROM analyses
                GROUP BY image_id
            ) latest ON latest.max_id = a1.id
        ) a ON a.image_id = i.id
        ORDER BY i.{sort} {order.upper()}
        LIMIT ? OFFSET ?
        """,
        (per_page, offset),
    ).fetchall()

    return [_row_to_gallery_item(r) for r in rows], total


def _row_to_gallery_item(row: sqlite3.Row) -> dict[str, Any]:
    keywords_raw = row["analysis_keywords"]
    try:
        keywords = json.loads(keywords_raw) if keywords_raw else []
    except json.JSONDecodeError:
        keywords = []
    return {
        "id": row["id"],
        "filename": row["filename"],
        "original_name": row["original_name"],
        "file_path": row["file_path"],
        "file_size": row["file_size"],
        "created_at": row["created_at"],
        "latest_analysis": (
            {
                "id": row["analysis_id"],
                "name": row["analysis_name"],
                "keywords": keywords,
                "description": row["analysis_description"],
                "search_query": row["analysis_search_query"],
                "model_used": row["analysis_model_used"],
                "created_at": row["analysis_created_at"],
            }
            if row["analysis_id"] is not None
            else None
        ),
    }


# ── CRUD: analyses ────────────────────────────────────────────────────────

def insert_analysis(
    conn: sqlite3.Connection,
    image_id: int,
    name: str | None,
    keywords: list[str] | None,
    description: str | None,
    search_query: str | None,
    prompt_used: str | None,
    model_used: str | None,
    duration_ms: int | None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO analyses
            (image_id, name, keywords, description, search_query,
             prompt_used, model_used, duration_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_id,
            name,
            json.dumps(keywords or [], ensure_ascii=False),
            description,
            search_query,
            prompt_used,
            model_used,
            duration_ms,
        ),
    )
    return cur.lastrowid


def get_analyses_for_image(
    conn: sqlite3.Connection, image_id: int
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT * FROM analyses
        WHERE image_id = ?
        ORDER BY created_at DESC, id DESC
        """,
        (image_id,),
    ).fetchall()
    return [_analysis_row_to_dict(r) for r in rows]


def get_latest_analysis(
    conn: sqlite3.Connection, image_id: int
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT * FROM analyses
        WHERE image_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (image_id,),
    ).fetchone()
    return _analysis_row_to_dict(row) if row else None


def _analysis_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    try:
        keywords = json.loads(row["keywords"]) if row["keywords"] else []
    except json.JSONDecodeError:
        keywords = []
    return {
        "id": row["id"],
        "image_id": row["image_id"],
        "name": row["name"],
        "keywords": keywords,
        "description": row["description"],
        "search_query": row["search_query"],
        "prompt_used": row["prompt_used"],
        "model_used": row["model_used"],
        "duration_ms": row["duration_ms"],
        "created_at": row["created_at"],
    }


# ── Search (FTS5) ─────────────────────────────────────────────────────────

def search_analyses(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Full-text search; returns gallery items (image + matching analysis)."""
    if not query.strip():
        return []
    sanitized = _sanitize_fts_query(query)
    rows = conn.execute(
        """
        SELECT
            i.*,
            a.id           AS analysis_id,
            a.name         AS analysis_name,
            a.keywords     AS analysis_keywords,
            a.description  AS analysis_description,
            a.search_query AS analysis_search_query,
            a.model_used   AS analysis_model_used,
            a.created_at   AS analysis_created_at
        FROM analyses_fts f
        JOIN analyses a ON a.id = f.rowid
        JOIN images   i ON i.id = a.image_id
        WHERE analyses_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (sanitized, limit),
    ).fetchall()
    return [_row_to_gallery_item(r) for r in rows]


def _sanitize_fts_query(query: str) -> str:
    """Tolerant FTS5 query: wrap each term as a prefix match, drop operators."""
    cleaned = "".join(c if c.isalnum() or c in " -_" else " " for c in query)
    terms = [t for t in cleaned.split() if t]
    if not terms:
        return '""'
    return " ".join(f'"{t}"*' for t in terms)
