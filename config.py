"""
config.py — Central configuration for the AI Trading Bot.
All env vars are loaded here; individual modules import from this file.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ──────────────────────────────────────────────
# MT5 Connection
# ──────────────────────────────────────────────
MT5_LOGIN: int = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD: str = os.getenv("MT5_PASSWORD", "")
MT5_SERVER: str = os.getenv("MT5_SERVER", "")
MT5_PATH: str = os.getenv("MT5_PATH", "")  # Optional: path to terminal64.exe

# ──────────────────────────────────────────────
# Trading Parameters
# ──────────────────────────────────────────────
SYMBOLS: list[str] = [
    s.strip() for s in os.getenv("SYMBOLS", os.getenv("SYMBOL", "EURUSD-T")).replace(";", ",").split(",") 
    if s.strip()
]
PRIORITY_SYMBOLS: list[str] = [
    "EURUSD-T", "GBPUSD-T", "USDJPY-T", "GOLD-T", "US100-T", "BRENT-T", "AUDUSD-T", "USDCAD-T", "GBPJPY-T",
    "[USA30]-T", "[GER40]-T"
]
TIMEFRAMES: list[str] = [t.strip() for t in os.getenv("TIMEFRAMES", "M5,M15,M30,H1").split(",")]
BARS_TO_FETCH: int = int(os.getenv("BARS_TO_FETCH", "500"))
SCALPING_MODE: bool = os.getenv("SCALPING_MODE", "false").lower() in ("true", "1", "yes")


# ──────────────────────────────────────────────
# Risk Parameters
# ──────────────────────────────────────────────
MAGIC_NUMBER: int = 20260224
TRADE_COOLDOWN_MINUTES: int = 5
MAX_SPREAD_RATIO: float = float(os.getenv("MAX_SPREAD_RATIO", "2.0")) # Veto if spread > 2x average
MAX_DRAWDOWN_PCT: float = float(os.getenv("MAX_DRAWDOWN_PCT", "5.0"))   # % of initial bal
MAX_OPEN_TRADES: int = int(os.getenv("MAX_OPEN_TRADES", "3"))
DAILY_DRAWDOWN_LIMIT_PCT: float = float(os.getenv("DAILY_DRAWDOWN_LIMIT_PCT", "2.0")) # Prop-firm daily limit
MAX_LOT_CAP: float = float(os.getenv("MAX_LOT_CAP", "2.0"))   # Maximum lots allowed per trade
RISK_HIGH_CONFIDENCE: float = float(os.getenv("RISK_HIGH_CONF", "1.0"))  # % of free margin
RISK_MED_CONFIDENCE: float = float(os.getenv("RISK_MED_CONF", "0.5"))
RISK_LOW_CONFIDENCE: float = float(os.getenv("RISK_LOW_CONF", "0.2"))
CONFIDENCE_HIGH_THRESHOLD: float = 0.60
CONFIDENCE_MED_THRESHOLD: float = 0.40
MIN_CONFIDENCE_TO_TRADE: float = 0.01  # Force test trade

# Tactical Profit Protection
PROFIT_LOCK_BE_USD: float = 5.0
PROFIT_LOCK_TRAIL_USD: float = 15.0
TRAILING_STOP_INTERVAL_SECONDS: int = 10

# ──────────────────────────────────────────────
# LLM / AI Provider
# ──────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini").lower()  # openai|anthropic|gemini
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODELS: list[str] = [m.strip() for m in os.getenv("OPENAI_MODEL", "gpt-4o").split(",")]
OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://apis.iflow.cn/v1")
AI_CONSENSUS_THRESHOLD: int = int(os.getenv("AI_CONSENSUS_THRESHOLD", "2"))
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
MAX_CONCURRENT_AI_CALLS: int = int(os.getenv("MAX_CONCURRENT_AI_CALLS", "3"))
AI_THROTTLE_SECONDS: float = 2.0

# ──────────────────────────────────────────────
# Telegram Notifier
# ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")

# ──────────────────────────────────────────────
# Retry / Resilience
# ──────────────────────────────────────────────
LLM_MAX_RETRIES: int = 3
LLM_BACKOFF_BASE: float = 1.0   # seconds; doubles each attempt
MT5_MAX_RETRIES: int = 5
MT5_HEARTBEAT_INTERVAL: int = 10  # seconds

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).parent
LOG_DIR: Path = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE_NAME: str = os.getenv("LOG_FILE_NAME", "bot.log")
LOG_FILE: Path = LOG_DIR / LOG_FILE_NAME
DB_PATH: Path = PROJECT_ROOT / "tradingbot_audit.db"

# ──────────────────────────────────────────────
# Operational Mode
# ──────────────────────────────────────────────
DRY_RUN: bool = os.getenv("DRY_RUN", "true").lower() in ("true", "1", "yes")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ──────────────────────────────────────────────
# MT5 Timeframe Map  (string → mt5 constant)
# ──────────────────────────────────────────────
import MetaTrader5 as mt5  # noqa: E402 — imported lazily for optional install

MT5_TIMEFRAME_MAP: dict[str, int] = {
    # Minutes
    "M1":  mt5.TIMEFRAME_M1,
    "M2":  mt5.TIMEFRAME_M2,
    "M3":  mt5.TIMEFRAME_M3,
    "M4":  mt5.TIMEFRAME_M4,
    "M5":  mt5.TIMEFRAME_M5,
    "M6":  mt5.TIMEFRAME_M6,
    "M10": mt5.TIMEFRAME_M10,
    "M12": mt5.TIMEFRAME_M12,
    "M15": mt5.TIMEFRAME_M15,
    "M20": mt5.TIMEFRAME_M20,
    "M30": mt5.TIMEFRAME_M30,
    # Hours
    "H1":  mt5.TIMEFRAME_H1,
    "H2":  mt5.TIMEFRAME_H2,
    "H3":  mt5.TIMEFRAME_H3,
    "H4":  mt5.TIMEFRAME_H4,
    "H6":  mt5.TIMEFRAME_H6,
    "H8":  mt5.TIMEFRAME_H8,
    "H12": mt5.TIMEFRAME_H12,
    # Days, Weeks, Months
    "D1":  mt5.TIMEFRAME_D1,
    "W1":  mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

# ──────────────────────────────────────────────
# Trading Sessions (UTC hours)
# ──────────────────────────────────────────────
SESSIONS: dict[str, tuple[int, int]] = {
    "Sydney":   (21, 6),
    "Tokyo":    (0, 9),
    "London":   (7, 16),
    "New York": (12, 21),
}

# Correlated pair groups — used by RiskManager
CORRELATION_GROUPS: list[list[str]] = [
    ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"],   # USD-negative basket
    ["USDCHF", "USDJPY", "USDCAD"],              # USD-positive basket
    ["XAUUSD", "XAGUSD"],                        # Metals
]
