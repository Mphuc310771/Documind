import json
import time
import sqlite3
import logging
import threading

logger = logging.getLogger(__name__)


class SQLStore:
    """
    Relational persistence layer (SQLite, stdlib only).

    Source of truth for Notebooks, Documents and ChatHistory so they survive
    browser cache clears and are shared across devices. ChromaDB is kept solely
    for semantic vector search.
    """

    def __init__(self, db_path: str = "./app_data.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock, self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS notebooks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    notebook_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    doc_type TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    UNIQUE(notebook_id, filename)
                );
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    notebook_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT,
                    citations TEXT,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_nb ON documents(notebook_id);
                CREATE INDEX IF NOT EXISTS idx_chat_nb ON chat_messages(notebook_id);
                """
            )
            # Migration: add owner column to notebooks if missing (multi-tenant support)
            cols = [r[1] for r in c.execute("PRAGMA table_info(notebooks)").fetchall()]
            if "owner" not in cols:
                c.execute("ALTER TABLE notebooks ADD COLUMN owner TEXT")

            # Update default notebook name from "Sổ tay mặc định" to "DocuMind Workspace" if it exists
            c.execute(
                "UPDATE notebooks SET name = 'DocuMind Workspace' WHERE id = 'default' AND name = 'Sổ tay mặc định'"
            )

            cur = c.execute("SELECT COUNT(*) FROM notebooks WHERE id = 'default'")
            if cur.fetchone()[0] == 0:
                c.execute(
                    "INSERT INTO notebooks (id, name, created_at) VALUES (?, ?, ?)",
                    ("default", "DocuMind Workspace", time.time()),
                )

    # ----- Notebooks -----
    def list_notebooks(self, owner: str | None = None) -> list[dict]:
        with self._lock, self._conn() as c:
            if owner is None:
                rows = c.execute(
                    "SELECT id, name, created_at FROM notebooks ORDER BY created_at ASC"
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT id, name, created_at FROM notebooks WHERE owner = ? ORDER BY created_at ASC",
                    (owner,),
                ).fetchall()
            return [dict(r) for r in rows]

    def create_notebook(self, notebook_id: str, name: str, owner: str | None = None) -> dict:
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO notebooks (id, name, created_at, owner) VALUES "
                "(?, ?, COALESCE((SELECT created_at FROM notebooks WHERE id = ?), ?), ?)",
                (notebook_id, name, notebook_id, time.time(), owner),
            )
        return {"id": notebook_id, "name": name}

    def notebook_owner(self, notebook_id: str) -> str | None:
        with self._lock, self._conn() as c:
            row = c.execute("SELECT owner FROM notebooks WHERE id = ?", (notebook_id,)).fetchone()
            return row["owner"] if row else None

    def delete_notebook(self, notebook_id: str):
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM notebooks WHERE id = ?", (notebook_id,))
            c.execute("DELETE FROM documents WHERE notebook_id = ?", (notebook_id,))
            c.execute("DELETE FROM chat_messages WHERE notebook_id = ?", (notebook_id,))

    # ----- Documents -----
    def add_document(self, notebook_id: str, filename: str, doc_type: str = "file", chunk_count: int = 0):
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO documents (notebook_id, filename, doc_type, chunk_count, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(notebook_id, filename) DO UPDATE SET "
                "doc_type = excluded.doc_type, chunk_count = excluded.chunk_count",
                (notebook_id, filename, doc_type, chunk_count, time.time()),
            )

    def remove_document(self, notebook_id: str, filename: str):
        with self._lock, self._conn() as c:
            c.execute(
                "DELETE FROM documents WHERE notebook_id = ? AND filename = ?",
                (notebook_id, filename),
            )

    def list_documents(self, notebook_id: str) -> list[dict]:
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT filename, doc_type, chunk_count, created_at FROM documents "
                "WHERE notebook_id = ? ORDER BY created_at ASC",
                (notebook_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ----- Chat history -----
    def add_chat_message(self, notebook_id: str, role: str, content: str, citations=None):
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO chat_messages (notebook_id, role, content, citations, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (notebook_id, role, content, json.dumps(citations, ensure_ascii=False) if citations else None, time.time()),
            )

    def get_chat_messages(self, notebook_id: str) -> list[dict]:
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT role, content, citations, created_at FROM chat_messages "
                "WHERE notebook_id = ? ORDER BY id ASC",
                (notebook_id,),
            ).fetchall()
            out = []
            for r in rows:
                item = dict(r)
                if item.get("citations"):
                    try:
                        item["citations"] = json.loads(item["citations"])
                    except Exception:
                        item["citations"] = None
                out.append(item)
            return out

    def clear_chat(self, notebook_id: str):
        with self._lock, self._conn() as c:
            c.execute("DELETE FROM chat_messages WHERE notebook_id = ?", (notebook_id,))

    # ----- Users (optional auth) -----
    def create_user(self, user_id: str, username: str, password_hash: str) -> dict:
        with self._lock, self._conn() as c:
            c.execute(
                "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (user_id, username, password_hash, time.time()),
            )
        return {"id": user_id, "username": username}

    def get_user_by_username(self, username: str) -> dict | None:
        with self._lock, self._conn() as c:
            row = c.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            return dict(row) if row else None

    def get_user_by_id(self, user_id: str) -> dict | None:
        with self._lock, self._conn() as c:
            row = c.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row) if row else None
