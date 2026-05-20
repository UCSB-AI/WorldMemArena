
from dataclasses import dataclass
from datetime import datetime
import os
import sqlite3
from typing import Optional

@dataclass
class RawMessage:
    """Raw conversation message (Layer 1)"""
    msg_id: int
    timestamp: datetime
    role: str  # "user" or "assistant" or speaker name
    text: str
    image_path: Optional[str] = None  # Path to image file if exists
    

class RawMessageStore:
    """
    Layer 1: Sequential raw messages
    - No vectorization, only temporal and ID indexing
    - Immutable: messages are never modified/deleted
    """
    
    def __init__(self, db_path: str = "raw_messages.db", reuse=False):
        if not reuse and os.path.exists(db_path):
            print(f"Remove existing raw db: {db_path}")
            os.remove(db_path)

        if os.path.dirname(db_path) and not os.path.exists(os.path.dirname(db_path)):
            os.makedirs(os.path.dirname(db_path)) 
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                msg_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                image_path TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON messages(timestamp)")
        conn.commit()
        conn.close()
    
    def append(self, msg: RawMessage) -> int:
        """Append a new message, returns assigned ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO messages (timestamp, role, text, image_path)
            VALUES (?, ?, ?, ?)
        """, (msg.timestamp.isoformat(), msg.role, msg.text, msg.image_path))
        msg_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return msg_id
    
    def fetch_by_ids(self, id_ranges: list[list[int]]) -> list[RawMessage]:
        """
        Fetch messages by ID ranges
        Args:
            id_ranges: [[start1, end1], [start2, end2], ...]
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        results = []
        for start, end in id_ranges:
            cursor.execute("""
                SELECT msg_id, timestamp, role, text, image_path
                FROM messages
                WHERE msg_id BETWEEN ? AND ?
                ORDER BY msg_id
            """, (start, end))
            
            for row in cursor.fetchall():
                results.append(RawMessage(
                    msg_id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    role=row[2],
                    text=row[3],
                    image_path=row[4]
                ))
        
        conn.close()
        return results
    
    def fetch_by_timerange(self, start: datetime, end: datetime) -> list[RawMessage]:
        """Fetch messages within a time range"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT msg_id, timestamp, role, text, image_path
            FROM messages
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY msg_id
        """, (start.isoformat(), end.isoformat()))
        
        results = []
        for row in cursor.fetchall():
            results.append(RawMessage(
                msg_id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                role=row[2],
                text=row[3],
                image_path=row[4]
            ))
        
        conn.close()
        return results
    
    def get_latest_n(self, n: int = 10) -> list[RawMessage]:
        """Get the latest N messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT msg_id, timestamp, role, text, image_path
            FROM messages
            ORDER BY msg_id DESC
            LIMIT ?
        """, (n,))
        
        results = []
        for row in cursor.fetchall():
            results.append(RawMessage(
                msg_id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                role=row[2],
                text=row[3],
                image_path=row[4]
            ))
        
        conn.close()
        return list(reversed(results))  # Return in chronological order

    def dump_db(self, path):
        """Dump raw messages to JSON file"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        res = cursor.execute("""
                SELECT msg_id, timestamp, role, text, image_path
                FROM messages
                ORDER BY msg_id
            """, ).fetchall()

        res = [
            {
                "msg_id": row[0],
                "timestamp": row[1],
                "role": row[2],
                "text": row[3],
                "image_path": row[4]
            }
            for row in res
        ]

        conn.close()

        import json
        with open(path, 'w') as f:
            json.dump(res, f, indent=2)
