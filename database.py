from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

DB_PATH = Path("visiontar.db")
DEFECT_CLASSES = ("cotto", "poco", "blob", "rotto")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    return connection


def init_db(db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS production_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                production_date TEXT NOT NULL,
                product_description TEXT NOT NULL,
                lot_code TEXT NOT NULL,
                profile_key TEXT NOT NULL,
                source_type TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS defect_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                track_id INTEGER,
                defect_class TEXT NOT NULL,
                confidence REAL,
                x_px INTEGER,
                y_px INTEGER,
                x_norm REAL,
                y_norm REAL,
                frame_finale INTEGER,
                timestamp_s REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES production_sessions(id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_defect_session ON defect_events(session_id)"
        )


def create_session(
    production_date: date,
    product_description: str,
    lot_code: str,
    profile_key: str,
    source_type: str,
    db_path: Path = DB_PATH,
) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO production_sessions (
                production_date,
                product_description,
                lot_code,
                profile_key,
                source_type,
                started_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                production_date.isoformat(),
                product_description.strip(),
                lot_code.strip(),
                profile_key,
                source_type,
                now,
            ),
        )
        return int(cursor.lastrowid)


def close_session(session_id: int, db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE production_sessions
            SET ended_at = ?
            WHERE id = ?
            """,
            (datetime.now().isoformat(timespec="seconds"), session_id),
        )


def add_defect_event(
    session_id: int,
    defect_class: str,
    *,
    track_id: int | None = None,
    confidence: float | None = None,
    x_px: int | None = None,
    y_px: int | None = None,
    x_norm: float | None = None,
    y_norm: float | None = None,
    frame_finale: int | None = None,
    timestamp_s: float | None = None,
    db_path: Path = DB_PATH,
) -> None:
    if defect_class not in DEFECT_CLASSES:
        raise ValueError(f"Classe difetto non valida: {defect_class}")

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO defect_events (
                session_id,
                track_id,
                defect_class,
                confidence,
                x_px,
                y_px,
                x_norm,
                y_norm,
                frame_finale,
                timestamp_s,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                track_id,
                defect_class,
                confidence,
                x_px,
                y_px,
                x_norm,
                y_norm,
                frame_finale,
                timestamp_s,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )


def get_session(session_id: int, db_path: Path = DB_PATH) -> dict[str, Any] | None:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM production_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None


def get_session_summary(session_id: int, db_path: Path = DB_PATH) -> dict[str, int]:
    summary = {name: 0 for name in DEFECT_CLASSES}

    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT defect_class, COUNT(*) AS total
            FROM defect_events
            WHERE session_id = ?
            GROUP BY defect_class
            """,
            (session_id,),
        ).fetchall()

    for row in rows:
        summary[str(row["defect_class"])] = int(row["total"])

    summary["totale_scarti"] = sum(summary[name] for name in DEFECT_CLASSES)
    return summary
