import json
import re
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

from config import DB_PATH, STORAGE_DIR, UPLOAD_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    company TEXT NOT NULL,
    salt BLOB NOT NULL,
    pw_hash BLOB NOT NULL,
    failed INTEGER NOT NULL DEFAULT 0,
    locked_until TEXT,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS machines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    owner_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    code TEXT NOT NULL,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    location TEXT,
    notes TEXT,
    created TEXT NOT NULL,
    UNIQUE(owner_id, code)
);
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id INTEGER NOT NULL REFERENCES machines(id) ON DELETE CASCADE,
    recorded_at TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_path TEXT,
    status TEXT NOT NULL,
    level TEXT NOT NULL,
    score REAL NOT NULL,
    threshold REAL NOT NULL,
    health REAL NOT NULL,
    flag_ratio REAL NOT NULL,
    peak_score REAL NOT NULL,
    novelty REAL NOT NULL,
    reliable INTEGER NOT NULL,
    distance REAL NOT NULL,
    profile TEXT NOT NULL,
    segment_scores TEXT NOT NULL,
    model_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS baselines (
    machine_id INTEGER PRIMARY KEY REFERENCES machines(id) ON DELETE CASCADE,
    created TEXT NOT NULL,
    health REAL NOT NULL,
    profile TEXT NOT NULL,
    clips INTEGER NOT NULL,
    model_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS machine_ideals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id INTEGER NOT NULL UNIQUE REFERENCES machines(id) ON DELETE CASCADE,
    recorded_at TEXT NOT NULL,
    file_name TEXT NOT NULL,
    health REAL NOT NULL,
    score REAL NOT NULL,
    profile TEXT NOT NULL,
    model_id TEXT NOT NULL
);

"""


@contextmanager
def session():
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    with session() as conn:
        conn.executescript(SCHEMA)


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_user(username, company, salt, pw_hash):
    try:
        with session() as conn:
            conn.execute(
                "INSERT INTO users (username, company, salt, pw_hash, created) VALUES (?, ?, ?, ?, ?)",
                (username, company, salt, pw_hash, now()),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def get_user(username):
    with session() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return dict(row) if row else None


def set_attempts(user_id, failed, locked_until):
    with session() as conn:
        conn.execute("UPDATE users SET failed = ?, locked_until = ? WHERE id = ?",
                     (failed, locked_until, user_id))


def add_machine(owner_id, code, name, mtype, location, notes):
    try:
        with session() as conn:
            conn.execute(
                "INSERT INTO machines (owner_id, code, name, type, location, notes, created) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (owner_id, code, name, mtype, location, notes, now()),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def list_machines(owner_id):
    with session() as conn:
        return pd.read_sql_query(
            "SELECT id, code, name, type, location, notes, created FROM machines "
            "WHERE owner_id = ? ORDER BY code", conn, params=(owner_id,))


def delete_machine(owner_id, machine_id):
    with session() as conn:
        conn.execute("DELETE FROM machines WHERE id = ? AND owner_id = ?", (machine_id, owner_id))
    shutil.rmtree(UPLOAD_DIR / str(owner_id) / str(machine_id), ignore_errors=True)


def owns(owner_id, machine_id):
    with session() as conn:
        row = conn.execute("SELECT 1 FROM machines WHERE id = ? AND owner_id = ?",
                           (machine_id, owner_id)).fetchone()
    return row is not None


def save_upload(owner_id, machine_id, name, data):
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)[:80]
    folder = UPLOAD_DIR / str(owner_id) / str(machine_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe}"
    path.write_bytes(data)
    return str(path)


def add_reading(owner_id, machine_id, file_name, file_path, result, model_id):
    if not owns(owner_id, machine_id):
        raise PermissionError("Machine does not belong to this account")
    with session() as conn:
        conn.execute(
            "INSERT INTO readings (machine_id, recorded_at, file_name, file_path, status, level, score, "
            "threshold, health, flag_ratio, peak_score, novelty, reliable, distance, profile, "
            "segment_scores, model_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (machine_id, now(), file_name, file_path, result["status"], result["level"], result["score"],
             result["threshold"], result["health"], result["flag_ratio"], result["peak_score"],
             result["novelty"], int(result["reliable"]), result["distance"],
             json.dumps(result["profile"]), json.dumps([round(float(v), 4) for v in result["segment_scores"]]),
             model_id),
        )

def set_ideal(owner_id, machine_id, file_name, result, model_id):
    if not owns(owner_id, machine_id):
        raise PermissionError("Machine does not belong to this account")
    with session() as conn:
        conn.execute("DELETE FROM machine_ideals WHERE machine_id = ?", (machine_id,))
        conn.execute(
            "INSERT INTO machine_ideals (machine_id, recorded_at, file_name, health, score, "
            "profile, model_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (machine_id, now(), file_name, result["health"], result["score"],
             json.dumps(result["profile"]), model_id),
        )


def get_ideal(owner_id, machine_id):
    with session() as conn:
        row = conn.execute(
            "SELECT i.* FROM machine_ideals i JOIN machines m ON m.id = i.machine_id "
            "WHERE m.owner_id = ? AND i.machine_id = ?",
            (owner_id, machine_id),
        ).fetchone()
    if row is None:
        return None
    ideal = dict(row)
    ideal["profile"] = json.loads(ideal["profile"])
    return ideal



def readings(owner_id, machine_id, model_id=None):
    query = (
        "SELECT r.* FROM readings r JOIN machines m ON m.id = r.machine_id "
        "WHERE m.owner_id = ? AND r.machine_id = ?"
    )
    params = [owner_id, machine_id]
    if model_id is not None:
        query += " AND r.model_id = ?"
        params.append(model_id)
    query += " ORDER BY r.recorded_at, r.id"
    with session() as conn:
        frame = pd.read_sql_query(query, conn, params=params)
    frame["recorded_at"] = pd.to_datetime(frame["recorded_at"])
    frame["profile"] = frame["profile"].map(json.loads)
    frame["segment_scores"] = frame["segment_scores"].map(json.loads)
    return frame


def delete_reading(owner_id, reading_id):
    with session() as conn:
        conn.execute(
            "DELETE FROM readings WHERE id = ? AND machine_id IN (SELECT id FROM machines WHERE owner_id = ?)",
            (reading_id, owner_id),
        )


def fleet(owner_id):
    machines = list_machines(owner_id)
    rows = []
    for m in machines.itertuples():
        history = readings(owner_id, m.id)
        last = history.iloc[-1] if len(history) else None
        rows.append({
            "Code": m.code, "Name": m.name, "Type": m.type, "Location": m.location,
            "Readings": len(history),
            "Last reading": last["recorded_at"].strftime("%Y-%m-%d %H:%M") if last is not None else "",
            "Last status": last["status"] if last is not None else "",
            "Last level": last["level"] if last is not None else "",
            "Last health": round(float(last["health"])) if last is not None else None,
        })
    return pd.DataFrame(rows)

def set_baseline(owner_id, machine_id, health, profile, clips, model_id):
    if not owns(owner_id, machine_id):
        raise PermissionError("Machine does not belong to this account")
    with session() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO baselines (machine_id, created, health, profile, clips, model_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (machine_id, now(), float(health), json.dumps([float(v) for v in profile]), clips, model_id),
        )


def get_baseline(owner_id, machine_id, model_id):
    with session() as conn:
        row = conn.execute(
            "SELECT b.* FROM baselines b JOIN machines m ON m.id = b.machine_id "
            "WHERE m.owner_id = ? AND b.machine_id = ? AND b.model_id = ?",
            (owner_id, machine_id, model_id),
        ).fetchone()
    if row is None:
        return None
    base = dict(row)
    base["profile"] = json.loads(base["profile"])
    return base