import sqlite3
import json
from datetime import datetime

class EvalPersistence:
    def __init__(self, db_path='evals.db'):
        self.conn = sqlite3.connect(db_path)
        self.create_table()
    
    def create_table(self):
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS eval_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metrics TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
    
    def log_result(self, metrics):
        created_at = datetime.now().isoformat()
        with self.conn:
            self.conn.execute('''
                INSERT INTO eval_results (metrics, created_at)
                VALUES (?, ?)
            ''', (json.dumps(metrics), created_at))
    
    def get_trends(self, metric_name):
        cursor = self.conn.execute('''
            SELECT metrics FROM eval_results ORDER BY created_at
        ''')
        results = []
        for row in cursor:
            metrics = json.loads(row[0])
            if metric_name in metrics:
                results.append(metrics[metric_name])
        return results
    
    def export_json(self, file_path):
        cursor = self.conn.execute('SELECT * FROM eval_results')
        rows = cursor.fetchall()
        data = []
        for row in rows:
            data.append({
                'id': row[0],
                'metrics': json.loads(row[1]),
                'created_at': row[2]
            })
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)