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

## Supported LLM Providers

| Provider | Model (`config.py`) | Key Env Var |
|----------|---------------------|-------------|
| OpenAI | `gpt-4o` | `OPENAI_API_KEY` |
| Anthropic | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| Google | `gemini-1.5-pro` | `GEMINI_API_KEY` |

---

## Graceful Shutdown

Press `Ctrl+C` — the bot will:
1. Cancel all async tasks cleanly
2. Disconnect from MT5
3. Close the aiohttp session

---

> ⚠️ **Disclaimer**: This software is for educational and research purposes. Trading financial instruments carries significant risk. Always test thoroughly in demo mode before deploying with real funds.
