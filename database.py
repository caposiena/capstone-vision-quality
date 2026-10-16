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
        session_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(production_sessions)").fetchall()
        }
        if "total_analyzed" not in session_columns:
            conn.execute(
                "ALTER TABLE production_sessions ADD COLUMN total_analyzed INTEGER NOT NULL DEFAULT 0"
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
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_defect_session_track
            ON defect_events(session_id, track_id)
            WHERE track_id IS NOT NULL
            """
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


def close_session(
    session_id: int,
    total_analyzed: int | None = None,
    db_path: Path = DB_PATH,
) -> None:
    with get_connection(db_path) as conn:
        if total_analyzed is None:
            conn.execute(
                """
                UPDATE production_sessions
                SET ended_at = ?
                WHERE id = ?
                """,
                (datetime.now().isoformat(timespec="seconds"), session_id),
            )
        else:
            conn.execute(
                """
                UPDATE production_sessions
                SET ended_at = ?, total_analyzed = ?
                WHERE id = ?
                """,
                (
                    datetime.now().isoformat(timespec="seconds"),
                    max(0, int(total_analyzed)),
                    session_id,
                ),
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
) -> bool:
    if defect_class not in DEFECT_CLASSES:
        raise ValueError(f"Classe difetto non valida: {defect_class}")

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO defect_events (
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
        return cursor.rowcount > 0


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
    session = get_session(session_id, db_path)
    summary["totale_analizzati"] = int(
        (session or {}).get("total_analyzed") or 0
    )
    summary["buoni"] = max(
        0,
        summary["totale_analizzati"] - summary["totale_scarti"],
    )
    return summary


def build_lot_report_rows(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Una riga per lotto, aggregando correttamente tutte le rilevazioni."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            WITH session_stats AS (
                SELECT
                    s.id,
                    s.production_date,
                    s.product_description,
                    s.lot_code,
                    COALESCE(s.total_analyzed, 0) AS totale_analizzati,
                    COUNT(d.id) AS totale_scarti,
                    COALESCE(SUM(CASE WHEN d.defect_class = 'cotto' THEN 1 ELSE 0 END), 0) AS cotto,
                    COALESCE(SUM(CASE WHEN d.defect_class = 'poco' THEN 1 ELSE 0 END), 0) AS poco,
                    COALESCE(SUM(CASE WHEN d.defect_class = 'blob' THEN 1 ELSE 0 END), 0) AS blob,
                    COALESCE(SUM(CASE WHEN d.defect_class = 'rotto' THEN 1 ELSE 0 END), 0) AS rotto
                FROM production_sessions s
                LEFT JOIN defect_events d ON d.session_id = s.id
                WHERE s.ended_at IS NOT NULL
                  AND COALESCE(s.total_analyzed, 0) > 0
                GROUP BY
                    s.id,
                    s.production_date,
                    s.product_description,
                    s.lot_code,
                    s.total_analyzed
            )
            SELECT
                production_date,
                product_description,
                lot_code,
                COUNT(*) AS sessions,
                SUM(totale_analizzati) AS totale_analizzati,
                SUM(totale_scarti) AS totale_scarti,
                SUM(cotto) AS cotto,
                SUM(poco) AS poco,
                SUM(blob) AS blob,
                SUM(rotto) AS rotto
            FROM session_stats
            GROUP BY production_date, product_description, lot_code
            ORDER BY production_date, product_description, lot_code
            """
        ).fetchall()

    result = []
    for row in rows:
        item = dict(row)
        analyzed = int(item.get("totale_analizzati") or 0)
        for key in ("totale_scarti", "cotto", "poco", "blob", "rotto"):
            item[key] = int(item.get(key) or 0)
        item["scarto_pct"] = (
            round(item["totale_scarti"] * 100.0 / analyzed, 2)
            if analyzed > 0
            else 0.0
        )
        for key in DEFECT_CLASSES:
            item[f"{key}_pct"] = (
                round(item[key] * 100.0 / analyzed, 2)
                if analyzed > 0
                else 0.0
            )
        result.append(item)
    return result


def build_session_report_rows(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """Restituisce una riga per ogni rilevazione/sessione, senza aggregare lotti ripetuti."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                s.id AS rilevazione_id,
                s.production_date,
                s.product_description,
                s.lot_code,
                s.started_at,
                s.ended_at,
                COALESCE(s.total_analyzed, 0) AS totale_analizzati,
                COUNT(d.id) AS totale_scarti,
                COALESCE(SUM(CASE WHEN d.defect_class = 'cotto' THEN 1 ELSE 0 END), 0) AS cotto,
                COALESCE(SUM(CASE WHEN d.defect_class = 'poco' THEN 1 ELSE 0 END), 0) AS poco,
                COALESCE(SUM(CASE WHEN d.defect_class = 'blob' THEN 1 ELSE 0 END), 0) AS blob,
                COALESCE(SUM(CASE WHEN d.defect_class = 'rotto' THEN 1 ELSE 0 END), 0) AS rotto
            FROM production_sessions s
            LEFT JOIN defect_events d ON d.session_id = s.id
            WHERE s.ended_at IS NOT NULL
            GROUP BY
                s.id,
                s.production_date,
                s.product_description,
                s.lot_code,
                s.started_at,
                s.ended_at,
                s.total_analyzed
            ORDER BY s.id
            """
        ).fetchall()

    result = []
    for row in rows:
        item = dict(row)
        analyzed = int(item.get("totale_analizzati") or 0)
        for key in ("totale_scarti", "cotto", "poco", "blob", "rotto"):
            item[key] = int(item.get(key) or 0)
        item["scarto_pct"] = (
            round(item["totale_scarti"] * 100.0 / analyzed, 2)
            if analyzed > 0
            else 0.0
        )
        for key in DEFECT_CLASSES:
            item[f"{key}_pct"] = (
                round(item[key] * 100.0 / analyzed, 2)
                if analyzed > 0
                else 0.0
            )
        result.append(item)
    return result
