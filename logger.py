"""
logger.py — Structured, rotating file + console logger for the trading bot.

Each module gets a BoundLogger via get_logger(name).
BoundLogger supports structlog-style keyword extras:
    log.info("msg", symbol="EURUSD", price=1.085)
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
import json
from datetime import datetime, timezone

from config import LOG_FILE, LOG_LEVEL


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts":     datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        extras = getattr(record, "_extras", {})
        payload.update(extras)
        return json.dumps(payload, default=str)


class _HumanFormatter(logging.Formatter):
    LEVEL_COLORS = {
        "DEBUG":    "\033[36m",
        "INFO":     "\033[32m",
        "WARNING":  "\033[33m",
        "ERROR":    "\033[31m",
        "CRITICAL": "\033[35m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelname, "")
        ts    = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%H:%M:%S")
        prefix = f"{color}[{record.levelname:<8}]{self.RESET}"
        extras = getattr(record, "_extras", {})
        extra_str = (" | " + " ".join(f"{k}={v}" for k, v in extras.items())) if extras else ""
        return f"{ts} {prefix} [{record.name}] {record.getMessage()}{extra_str}"


# ── BoundLogger wrapper ───────────────────────────────────────────
class BoundLogger:
    """
    Wraps a stdlib Logger and lets callers pass structured extras as kwargs.
    Usage: log.info("message", symbol="EURUSD", bid=1.085)
    """

    def __init__(self, inner: logging.Logger) -> None:
        self._inner = inner

    def _emit(self, level: int, msg: str, exc_info=False, **kwargs) -> None:
        if self._inner.isEnabledFor(level):
            record = self._inner.makeRecord(
                self._inner.name, level,
                "(unknown)", 0, msg, (), None,
            )
            record._extras = kwargs  # type: ignore[attr-defined]
            if exc_info:
                import sys as _sys
                record.exc_info = _sys.exc_info()
            self._inner.handle(record)

    def debug   (self, msg: str, **kw) -> None: self._emit(logging.DEBUG,    msg, **kw)
    def info    (self, msg: str, **kw) -> None: self._emit(logging.INFO,     msg, **kw)
    def warning (self, msg: str, **kw) -> None: self._emit(logging.WARNING,  msg, **kw)
    def error   (self, msg: str, **kw) -> None: self._emit(logging.ERROR,    msg, **kw)
    def critical(self, msg: str, **kw) -> None: self._emit(logging.CRITICAL, msg, **kw)

    def isEnabledFor(self, level: int) -> bool:
        return self._inner.isEnabledFor(level)


def _build_root_logger() -> logging.Logger:
    root = logging.getLogger("tradingbot")
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    root.handlers.clear()

    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(_JsonFormatter())
    fh.setLevel(logging.DEBUG)
    root.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_HumanFormatter())
    ch.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    root.addHandler(ch)

    return root


_root: logging.Logger = _build_root_logger()


def get_logger(name: str) -> BoundLogger:
    """Return a structured BoundLogger for the given component name."""
    return BoundLogger(_root.getChild(name))
