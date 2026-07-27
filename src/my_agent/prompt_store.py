import sqlite3
import random
from datetime import datetime

class PromptStore:
    def __init__(self, db_path='prompts.db'):
        self.conn = sqlite3.connect(db_path)
        self.create_table()
    
    def create_table(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    gray_release REAL DEFAULT 100.0,
                    created_at TEXT NOT NULL,
                    is_active BOOLEAN DEFAULT 1
                )
            ''')
    
    def add_version(self, content, gray_release=100.0):
        version = self.get_latest_version() + 1
        created_at = datetime.now().isoformat()
        with self.conn:
            self.conn.execute('''
                INSERT INTO prompts (version, content, gray_release, created_at, is_active)
                VALUES (?, ?, ?, ?, 1)
            ''', (version, content, gray_release, created_at))
    
    def get_latest_version(self):
        cursor = self.conn.execute('SELECT MAX(version) FROM prompts')
        result = cursor.fetchone()[0]
        return result if result is not None else 0
    
    def get_current_prompt(self):
        cursor = self.conn.execute('SELECT version, content, gray_release FROM prompts WHERE is_active=1')
        versions = cursor.fetchall()
        if not versions:
            return None
        
        total = sum(v[2] for v in versions)
        rand = random.uniform(0, total)
        current = 0
        for version, content, gray in versions:
            current += gray
            if rand <= current:
                return content
        return versions[-1][1]  # Fallback
    
    def rollback_to(self, version_id):
        with self.conn:
            self.conn.execute('UPDATE prompts SET is_active=0')
            self.conn.execute('UPDATE prompts SET is_active=1 WHERE id=?', (version_id,))