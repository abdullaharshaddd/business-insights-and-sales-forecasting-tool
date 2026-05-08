"""
Long-Term Analytical Memory Store
==================================
PostgreSQL-backed persistent memory for the Agentic RAG system.
Stores analytical findings, insights, and recommendations
so the agent can recall past investigations across sessions.
"""

from sqlalchemy import create_engine, text, Column, Integer, String, Text, DateTime
import json
import yaml
from datetime import datetime

class MemoryStore:
    """Persistent memory for analytical findings and conversation context."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.db_url = cfg["database"]["url"]
        self.engine = create_engine(self.db_url)
        # Tables are handled by Prisma migration, but we ensure they exist here as a fallback
        self._init_db()

    def _init_db(self):
        # We assume tables are created via Prisma, but we can verify/create here if needed.
        # For simplicity in this unification, we rely on the global Prisma migration.
        pass

    def store_finding(self, topic: str, finding: str, source: str = "agent", importance: str = "normal"):
        """Store an analytical finding for future recall."""
        sql = "INSERT INTO findings (timestamp, topic, finding, source, importance) VALUES (NOW(), :topic, :finding, :source, :importance)"
        with self.engine.connect() as conn:
            conn.execute(text(sql), {"topic": topic, "finding": finding, "source": source, "importance": importance})
            conn.commit()

    def recall_findings(self, topic: str = None, limit: int = 5) -> str:
        """Retrieve past findings, optionally filtered by topic."""
        if topic:
            sql = "SELECT timestamp, topic, finding, importance FROM findings WHERE topic ILIKE :topic ORDER BY timestamp DESC LIMIT :limit"
            params = {"topic": f"%{topic}%", "limit": limit}
        else:
            sql = "SELECT timestamp, topic, finding, importance FROM findings ORDER BY timestamp DESC LIMIT :limit"
            params = {"limit": limit}

        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params).fetchall()
            
            if not result:
                return "No previous findings stored."

            lines = ["[Retrieved Past Findings]\n"]
            for row in result:
                ts, tp, finding, imp = row
                date_str = str(ts)[:10]
                lines.append(f"  [{date_str}] ({tp}) {finding}")
            return "\n".join(lines)

    def store_conversation_summary(self, thread_id: str, summary: str, key_topics: list):
        """Store a conversation summary for long-term recall."""
        sql = "INSERT INTO conversation_summaries (timestamp, thread_id, summary, key_topics) VALUES (NOW(), :thread_id, :summary, :key_topics)"
        with self.engine.connect() as conn:
            conn.execute(text(sql), {
                "thread_id": thread_id, 
                "summary": summary, 
                "key_topics": json.dumps(key_topics)
            })
            conn.commit()

    def recall_conversation_context(self, thread_id: str, limit: int = 3) -> str:
        """Get recent conversation summaries for a thread."""
        sql = "SELECT timestamp, summary FROM conversation_summaries WHERE thread_id = :thread_id ORDER BY timestamp DESC LIMIT :limit"
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), {"thread_id": thread_id, "limit": limit}).fetchall()
            if not result:
                return ""
            lines = ["[Previous Session Context]\n"]
            for row in result:
                ts, summary = row
                lines.append(f"  [{str(ts)[:10]}] {summary}")
            return "\n".join(lines)

    def get_all_topics(self) -> list:
        """Return all distinct topics from findings."""
        sql = "SELECT DISTINCT topic FROM findings ORDER BY topic"
        with self.engine.connect() as conn:
            result = conn.execute(text(sql)).fetchall()
            return [r[0] for r in result]
