"""
Long-Term Analytical Memory Store
==================================
SQLite-backed persistent memory for the Agentic RAG system.
Stores analytical findings, insights, and recommendations
so the agent can recall past investigations across sessions.
"""

import sqlite3
import json
import os
from datetime import datetime

MEMORY_DB = "data/agent_memory.db"


class MemoryStore:
    """Persistent memory for analytical findings and conversation context."""

    def __init__(self, db_path: str = MEMORY_DB):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                topic TEXT NOT NULL,
                finding TEXT NOT NULL,
                source TEXT DEFAULT 'agent',
                importance TEXT DEFAULT 'normal'
            );
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                key_topics TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_findings_topic ON findings(topic);
            CREATE INDEX IF NOT EXISTS idx_conv_thread ON conversation_summaries(thread_id);
        """)
        conn.close()

    def store_finding(self, topic: str, finding: str, source: str = "agent", importance: str = "normal"):
        """Store an analytical finding for future recall."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO findings (timestamp, topic, finding, source, importance) VALUES (?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), topic, finding, source, importance),
            )
            conn.commit()
        finally:
            conn.close()

    def recall_findings(self, topic: str = None, limit: int = 5) -> str:
        """Retrieve past findings, optionally filtered by topic."""
        conn = sqlite3.connect(self.db_path)
        try:
            if topic:
                rows = conn.execute(
                    "SELECT timestamp, topic, finding, importance FROM findings "
                    "WHERE topic LIKE ? ORDER BY timestamp DESC LIMIT ?",
                    (f"%{topic}%", limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT timestamp, topic, finding, importance FROM findings "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()

            if not rows:
                return "No previous findings stored."

            lines = ["[Retrieved Past Findings]\n"]
            for ts, tp, finding, imp in rows:
                date_str = ts[:10]
                lines.append(f"  [{date_str}] ({tp}) {finding}")
            return "\n".join(lines)
        finally:
            conn.close()

    def store_conversation_summary(self, thread_id: str, summary: str, key_topics: list):
        """Store a conversation summary for long-term recall."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO conversation_summaries (timestamp, thread_id, summary, key_topics) VALUES (?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), thread_id, summary, json.dumps(key_topics)),
            )
            conn.commit()
        finally:
            conn.close()

    def recall_conversation_context(self, thread_id: str, limit: int = 3) -> str:
        """Get recent conversation summaries for a thread."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT timestamp, summary FROM conversation_summaries "
                "WHERE thread_id = ? ORDER BY timestamp DESC LIMIT ?",
                (thread_id, limit),
            ).fetchall()
            if not rows:
                return ""
            lines = ["[Previous Session Context]\n"]
            for ts, summary in rows:
                lines.append(f"  [{ts[:10]}] {summary}")
            return "\n".join(lines)
        finally:
            conn.close()

    def get_all_topics(self) -> list:
        """Return all distinct topics from findings."""
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute("SELECT DISTINCT topic FROM findings ORDER BY topic").fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()
