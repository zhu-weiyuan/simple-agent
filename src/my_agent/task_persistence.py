import sqlite3
import json
from datetime import datetime

class TaskPersistence:
    def __init__(self, db_path='tasks.db'):
        self.conn = sqlite3.connect(db_path)
        self.create_table()
    
    def create_table(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    last_turn INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
    
    def save_checkpoint(self, task_id, state, current_turn, checkpoint_interval=5):
        if current_turn % checkpoint_interval != 0:
            return
        created_at = datetime.now().isoformat()
        updated_at = created_at
        with self.conn:
            self.conn.execute('''
                INSERT OR REPLACE INTO tasks (task_id, state, last_turn, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (task_id, json.dumps(state), current_turn, created_at, updated_at))
    
    def load_checkpoint(self, task_id):
        cursor = self.conn.execute('SELECT state FROM tasks WHERE task_id=?', (task_id,))
        result = cursor.fetchone()
        if result:
            return json.loads(result[0])
        return None
    
    def clear_checkpoint(self, task_id):
        with self.conn:
            self.conn.execute('DELETE FROM tasks WHERE task_id=?', (task_id,))