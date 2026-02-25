# 🤖 AI Trading Bot

A production-grade, fully asynchronous trading bot that bridges **MetaTrader 5** execution with **LLM-powered** (GPT-4o / Claude / Gemini) decision-making on a configurable timeframe (default: M15).

---

## Architecture

```
tradingbot/
├── config.py          # All env vars & constants
├── logger.py          # JSON file + coloured console logger
├── mt5_client.py      # MT5 bridge: heartbeat, auto-reconnect, latency
├── data_manager.py    # OHLC + RSI/EMA/ATR/MACD + Market Situation Report
├── ai_brain.py        # LLM interface: multi-provider, backoff, JSON parsing
├── risk_manager.py    # Drawdown halt, lot sizing, trade cap, correlation
├── execution_engine.py# Order lifecycle, spread-aware pricing, trailing stops
├── database.py        # SQLite audit trail (every cycle + trades)
├── strategy.py        # EMA-200 trend + RSI mean-reversion
├── main.py            # asyncio orchestrator
└── tests/
    └── test_components.py
```

### Flow (every M15 candle close)

```
MT5 OHLC → Market Situation Report → LLM Analysis → Strategy Filter → Risk Check → Execute / HOLD → Audit DB
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your MT5 credentials and LLM API key
```

Minimum required settings in `.env`:

```env
MT5_LOGIN=12345678
MT5_PASSWORD=your_password
MT5_SERVER=ICMarkets-Demo
LLM_PROVIDER=openai          # openai | anthropic | gemini
OPENAI_API_KEY=sk-...
DRY_RUN=true                 # Start with dry run!
```

### 3. Run in dry-run mode first

```bash
python main.py
```

Watch `logs/bot.log` and confirm the bot:

- Connects to MT5 ✅
- Sends heartbeats every 10s ✅
- Generates a Market Situation Report at each M15 close ✅
- Receives and logs an AI JSON response ✅
- Prints `DRY RUN — order skipped` instead of placing real orders ✅

### 4. Enable live trading

Set `DRY_RUN=false` in `.env` when you're confident in the setup.

---

## Strategy

**Hybrid EMA-200 Trend + RSI Mean Reversion**

All 3 conditions must align for a trade signal:

| Rule | Condition |
|------|-----------|
| **Trend Filter** | Price must be on correct side of EMA-200 |
| **Entry Zone** | RSI rising from <40 (BUY) or falling from >60 (SELL) |
| **AI Confirmation** | LLM signal agrees + confidence ≥ 0.5 |

---

## Risk Controls

| Control | Default | Configurable |
|---------|---------|--------------|
| Max drawdown halt | 5% | `MAX_DRAWDOWN_PCT` |
| Max simultaneous trades | 3 | `MAX_OPEN_TRADES` |
| High confidence risk | 2% of free margin | `RISK_HIGH_CONF` |
| Medium confidence risk | 1% | `RISK_MED_CONF` |
| Low confidence risk | 0.5% | `RISK_LOW_CONF` |

---

## AI Response Contract

The LLM must respond with **only** this JSON:

```json
{
  "signal": "BUY" | "SELL" | "HOLD",
  "reasoning": "1-2 sentence justification",
  "entry_params": {
    "suggested_price": 1.08542,
    "stop_loss": 1.08300,
    "take_profit": 1.09100
  },
  "confidence_score": 0.82,
  "risk_assessment": "High" | "Medium" | "Low"
}
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests run without MT5 or live LLM — all external calls are mocked.

---

## Audit Trail

Every analysis cycle is stored in `tradingbot_audit.db`:

- `analysis_log` — market report, AI response, signal, lot size, latency
- `trades` — ticket, entry/exit prices, PnL

Open with any SQLite viewer (e.g. DB Browser for SQLite).

---

---

## 🚀 Hyper-Aggressive Testing Mode (Experimental)

The bot now supports a "Hyper-Aggressive" mode designed for high-frequency testing on large capital accounts ($10M+).

### Key Features

- **80% Capital Allocation**: Distributes 80% of available equity across the `MAX_OPEN_TRADES` (default 50).
- **Multi-Model Voting**: Uses an ensemble of LLMs to vote on signals. `AI_VOTING_ENABLED=True`.
- **Deep technical post-mortems**: Generates 25,600 token technical reports in Arabic for every closed trade.
- **Auto-Message Splitting**: Telegram reports are automatically split if they exceed the 4,096 character limit.

### Configuration

```python
# config.py
MAX_OPEN_TRADES = 50
CAPITAL_ALLOCATION_PCT = 80.0
AI_VOTING_ENABLED = True
POST_MORTEM_MAX_TOKENS = 25600
AI_THROTTLE_SECONDS = 0.5 # High-speed scanning
```

---

## 🛠️ Monitoring & Utilities

### 1. Progress Tracker

Check the status of the 50-trade milestone:

```bash
python check_progress.py
```

### 2. SL/TP Monitor

Watch real-time trailing stop updates:

```bash
python monitor_sl.py
```

---

## 🩹 Institutional-Grade Fixes

- **Trade Execution Resilience**: Added fallbacks for execution price fetching and partial deal data handling to ensure Telegram notifications are sent even when MT5 history synchronization is delayed.
- **Execution Price Fallback**: Automatically fetches current market prices if the MT5 execution result returns zero, ensuring accurate Telegram notifications.
- **Partial Data Survival**: The bot now sends closure notifications even if server-side deal history is delayed, using estimated prices as a fallback to prevent "silent closures".
- **Timezone-Resilient Tracking**: Uses raw UNIX timestamps for history queries to bypass MT5/Local system clock mismatches.

---

> ⚠️ **Disclaimer**: This software is for educational and research purposes. Trading financial instruments carries significant risk. Always test thoroughly in demo mode before deploying with real funds.
