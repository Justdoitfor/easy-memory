import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SQLiteManager:
    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._migrate_history_table()
        self._create_history_table()
        self._create_messages_table()

    def _migrate_history_table(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                cur = self.connection.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='history'")
                if cur.fetchone() is None:
                    self.connection.execute("COMMIT")
                    return
                cur.execute("PRAGMA table_info(history)")
                old_cols = {row[1] for row in cur.fetchall()}
                expected_cols = {
                    "id", "memory_id", "old_memory", "new_memory", "event",
                    "created_at", "updated_at", "is_deleted", "actor_id", "role",
                }
                if old_cols == expected_cols:
                    self.connection.execute("COMMIT")
                    return
                logger.info("Migrating history table to new schema.")
                cur.execute("DROP TABLE IF EXISTS history_old")
                cur.execute("ALTER TABLE history RENAME TO history_old")
                cur.execute("""
                    CREATE TABLE history (
                        id TEXT PRIMARY KEY, memory_id TEXT, old_memory TEXT,
                        new_memory TEXT, event TEXT, created_at DATETIME,
                        updated_at DATETIME, is_deleted INTEGER, actor_id TEXT, role TEXT
                    )
                """)
                intersecting = list(expected_cols & old_cols)
                if intersecting:
                    cols_csv = ", ".join(intersecting)
                    cur.execute(f"INSERT INTO history ({cols_csv}) SELECT {cols_csv} FROM history_old")
                cur.execute("DROP TABLE history_old")
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                logger.error(f"History table migration failed: {e}")
                raise

    def _create_history_table(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS history (
                        id TEXT PRIMARY KEY, memory_id TEXT, old_memory TEXT,
                        new_memory TEXT, event TEXT, created_at DATETIME,
                        updated_at DATETIME, is_deleted INTEGER, actor_id TEXT, role TEXT
                    )
                """)
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                raise

    def _create_messages_table(self) -> None:
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id TEXT PRIMARY KEY, session_scope TEXT, role TEXT,
                        content TEXT, name TEXT, created_at DATETIME
                    )
                """)
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                raise

    def add_history(self, memory_id, old_memory, new_memory, event, *,
                    created_at=None, updated_at=None, is_deleted=0, actor_id=None, role=None):
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute(
                    "INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, updated_at, is_deleted, actor_id, role) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (str(uuid.uuid4()), memory_id, old_memory, new_memory, event, created_at, updated_at, is_deleted, actor_id, role),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                raise

    def batch_add_history(self, records):
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.executemany(
                    "INSERT INTO history (id, memory_id, old_memory, new_memory, event, created_at, updated_at, is_deleted, actor_id, role) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(str(uuid.uuid4()), r.get("memory_id"), r.get("old_memory"), r.get("new_memory"), r.get("event"), r.get("created_at"), r.get("updated_at"), r.get("is_deleted", 0), r.get("actor_id"), r.get("role")) for r in records],
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                raise

    def get_history(self, memory_id):
        with self._lock:
            cur = self.connection.execute(
                "SELECT id, memory_id, old_memory, new_memory, event, created_at, updated_at, is_deleted, actor_id, role FROM history WHERE memory_id = ? ORDER BY created_at ASC, DATETIME(updated_at) ASC",
                (memory_id,),
            )
            rows = cur.fetchall()
        return [
            {"id": r[0], "memory_id": r[1], "old_memory": r[2], "new_memory": r[3], "event": r[4],
             "created_at": r[5], "updated_at": r[6], "is_deleted": bool(r[7]), "actor_id": r[8], "role": r[9]}
            for r in rows
        ]

    def save_messages(self, messages, session_scope):
        if not messages:
            return
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                now = datetime.now(timezone.utc).isoformat()
                for msg in messages:
                    self.connection.execute(
                        "INSERT INTO messages (id, session_scope, role, content, name, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                        (str(uuid.uuid4()), session_scope, msg.get("role"), msg.get("content"), msg.get("name"), now),
                    )
                self.connection.execute(
                    "DELETE FROM messages WHERE session_scope = ? AND id NOT IN (SELECT id FROM (SELECT id FROM messages WHERE session_scope = ? ORDER BY created_at DESC LIMIT 10))",
                    (session_scope, session_scope),
                )
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                raise

    def get_last_messages(self, session_scope, limit=10):
        with self._lock:
            cur = self.connection.execute(
                "SELECT role, content, name, created_at FROM (SELECT role, content, name, created_at FROM messages WHERE session_scope = ? ORDER BY created_at DESC LIMIT ?) ORDER BY created_at ASC",
                (session_scope, limit),
            )
            rows = cur.fetchall()
        return [{"role": r[0], "content": r[1], "name": r[2], "created_at": r[3]} for r in rows]

    def reset(self):
        with self._lock:
            try:
                self.connection.execute("BEGIN")
                self.connection.execute("DROP TABLE IF EXISTS history")
                self.connection.execute("DROP TABLE IF EXISTS messages")
                self.connection.execute("COMMIT")
            except Exception as e:
                self.connection.execute("ROLLBACK")
                raise
        self._create_history_table()
        self._create_messages_table()

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None

    def __del__(self):
        self.close()