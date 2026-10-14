"""SQLite helper - stores every prediction."""
import sqlite3
import datetime

from config import DB_PATH


def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_label TEXT,
                    prediction TEXT,
                    confidence REAL,
                    created_at TEXT)""")
    return conn


def log_prediction(patient_label, prediction, confidence):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO history (patient_label, prediction, confidence, created_at) "
            "VALUES (?, ?, ?, ?)",
            (patient_label, prediction, confidence,
             datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))


def get_history(limit=50):
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def clear_history():
    with _conn() as conn:
        conn.execute("DELETE FROM history")
