"""
database.py — Async SQLite audit trail for every bot decision.

Tables:
  - analysis_log : every M15 cycle — prompt, AI response, latency, decision
  - trades       : order lifecycle — entry, exit, PnL
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Optional

import aiosqlite

from config import DB_PATH
from logger import get_logger

log = get_logger("database")

# ────────────────────────────────────────────────────────────────
# Schema
# ────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    timeframe       TEXT    NOT NULL,
    session         TEXT,
    market_report   TEXT,
    ai_raw_response TEXT,
    ai_signal       TEXT,
    ai_confidence   REAL,
    ai_risk         TEXT,
    ai_reasoning    TEXT,
    strategy_signal TEXT,
    lot_size        REAL,
    was_traded      INTEGER DEFAULT 0,
    latency_ms      REAL,
    dry_run         INTEGER DEFAULT 0,
    ensemble_meta   TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket          INTEGER UNIQUE,
    symbol          TEXT    NOT NULL,
    signal          TEXT,
    lot             REAL,
    entry_price     REAL,
    stop_loss       REAL,
    take_profit     REAL,
    open_ts         TEXT,
    close_ts        TEXT,
    close_price     REAL,
    pnl             REAL,
    status          TEXT    DEFAULT 'open',
    analysis_log_id INTEGER,
    FOREIGN KEY (analysis_log_id) REFERENCES analysis_log(id)
);

CREATE INDEX IF NOT EXISTS idx_analysis_ts     ON analysis_log(ts);
CREATE INDEX IF NOT EXISTS idx_trades_symbol   ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status   ON trades(status);
"""


# ────────────────────────────────────────────────────────────────
# AuditDB
# ────────────────────────────────────────────────────────────────
class AuditDB:
    """Async SQLite wrapper for the bot's full audit trail."""

    def __init__(self) -> None:
        self._path = str(DB_PATH)

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        async with aiosqlite.connect(self._path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()
        log.info("Database initialized", path=self._path)

    async def record_cycle(
        self,
        symbol: str,
        timeframe: str,
        market_report: str,
        ai_response: dict,
        strategy_signal: str,
        lot_size: float,
        was_traded: bool,
        latency_ms: float,
        dry_run: bool = False,
    ) -> int:
        """
        Insert one analysis cycle record.
        Returns the row ID (used to link a trade entry).
        """
        ts = datetime.now(tz=timezone.utc).isoformat()
        entry_params = ai_response.get("entry_params", {})
        async with aiosqlite.connect(self._path) as db:
            cursor = await db.execute(
                """
                INSERT INTO analysis_log
                  (ts, symbol, timeframe, market_report,
                   ai_raw_response, ai_signal, ai_confidence, ai_risk,
                   ai_reasoning, strategy_signal, lot_size, was_traded,
                   latency_ms, dry_run, ensemble_meta)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    ts, symbol, timeframe, market_report,
                    json.dumps(ai_response),
                    ai_response.get("signal"),
                    ai_response.get("confidence_score"),
                    ai_response.get("risk_assessment"),
                    ai_response.get("reasoning"),
                    strategy_signal,
                    lot_size,
                    int(was_traded),
                    latency_ms,
                    int(dry_run),
                    json.dumps(ai_response.get("ensemble_meta", []))
                ),
            )
            await db.commit()
            row_id = cursor.lastrowid
        log.debug("Cycle recorded", row_id=row_id, signal=ai_response.get("signal"))
        return row_id

    async def record_trade_open(
        self,
        ticket: int,
        symbol: str,
        signal: str,
        lot: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        analysis_log_id: Optional[int] = None,
    ) -> None:
        ts = datetime.now(tz=timezone.utc).isoformat()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO trades
                  (ticket, symbol, signal, lot, entry_price, stop_loss,
                   take_profit, open_ts, status, analysis_log_id)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (ticket, symbol, signal, lot, entry_price, stop_loss,
                 take_profit, ts, "open", analysis_log_id),
            )
            await db.commit()
        log.info("Trade opened in DB", ticket=ticket, symbol=symbol)

    async def record_trade_close(
        self,
        ticket: int,
        close_price: float,
        pnl: float,
    ) -> None:
        ts = datetime.now(tz=timezone.utc).isoformat()
        async with aiosqlite.connect(self._path) as db:
            await db.execute(
                """
                UPDATE trades
                SET close_ts=?, close_price=?, pnl=?, status='closed'
                WHERE ticket=?
                """,
                (ts, close_price, pnl, ticket),
            )
            await db.commit()
        log.info("Trade closed in DB", ticket=ticket, pnl=pnl)

    async def get_recent_cycles(self, limit: int = 10) -> list[dict]:
        async with aiosqlite.connect(self._path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM analysis_log ORDER BY id DESC LIMIT ?", (limit,)
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]
