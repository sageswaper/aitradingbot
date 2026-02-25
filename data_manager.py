"""
data_manager.py — Data pipeline: OHLC fetching, indicator calculation,
                   and the Market Situation Report synthesizer.

All indicator maths use pandas/numpy vectorized operations for performance.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from config import (
    BARS_TO_FETCH, MT5_TIMEFRAME_MAP, SESSIONS, SYMBOLS, TIMEFRAMES,
)
from mt5_client import MT5Client
from logger import get_logger

log = get_logger("data_manager")


# ────────────────────────────────────────────────────────────────
# DataManager
# ────────────────────────────────────────────────────────────────
class DataManager:
    """
    Fetches OHLCV data from MT5, computes indicators, and serialises
    the current market state into a token-efficient narrative report.
    """

    def __init__(self, client: MT5Client) -> None:
        self._client = client

    # ── OHLC Fetching ─────────────────────────────────────────────

    async def get_ohlc(
        self,
        symbol: str,
        timeframe: str,
        bars: int = BARS_TO_FETCH,
    ) -> pd.DataFrame:
        """
        Fetch historical bars from MT5 and return a clean DataFrame.
        Runs the blocking MT5 call in a thread-pool executor.
        """
        tf_const = MT5_TIMEFRAME_MAP.get(timeframe)
        if tf_const is None:
            raise ValueError(f"Unknown timeframe '{timeframe}'")

        df = await asyncio.get_event_loop().run_in_executor(
            None, self._fetch_rates_sync, symbol, tf_const, bars
        )
        df = self._clean_ohlc(df)
        log.debug("OHLC fetched", symbol=symbol, timeframe=timeframe, bars=len(df))
        return df

    def _fetch_rates_sync(self, symbol: str, tf_const: int, bars: int) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, bars)
        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"copy_rates_from_pos failed for {symbol}: {mt5.last_error()}"
            )
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(
            columns={
                "open": "open", "high": "high", "low": "low",
                "close": "close", "tick_volume": "volume",
            },
            inplace=True,
        )
        return df[["open", "high", "low", "close", "volume"]]

    @staticmethod
    def _clean_ohlc(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop NaN rows and remove duplicate indices. 
        Avoids using asfreq() which can delete bars during holidays.
        """
        if df.empty:
            return df
        # Remove duplicates
        df = df[~df.index.duplicated(keep="last")]
        # Remove rows with NaN in core price columns
        df = df.dropna(subset=["open", "high", "low", "close"])
        # Ensure we keep the original bar count as much as possible
        return df

    # ── Technical Indicators ──────────────────────────────────────

    @staticmethod
    def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Wilder-smoothed RSI. Returns 100 when all losses are zero."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(
            alpha=1 / period, min_periods=period, adjust=False
        ).mean()
        avg_loss = loss.ewm(
            alpha=1 / period, min_periods=period, adjust=False
        ).mean()
        # Where avg_loss == 0: RSI = 100 (no downward pressure)
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        # Fill NaN that arose from 0/0 → RSI 100 (all gains, no losses)
        rsi = rsi.fillna(100)
        return rsi

    @staticmethod
    def calc_ema(close: pd.Series, period: int) -> pd.Series:
        return close.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_prev = (df["high"] - df["close"].shift()).abs()
        low_prev = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def calc_macd(
        close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Returns (macd_line, signal_line, histogram)."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calc_bb(close: pd.Series, period: int = 20, std_dev: int = 2) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Returns (upper_band, middle_band, lower_band)."""
        ma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = ma + (std_dev * std)
        lower = ma - (std_dev * std)
        return upper, ma, lower

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators and add as columns; returns augmented df."""
        df = df.copy()
        df["rsi"] = self.calc_rsi(df["close"])
        df["ema5"] = self.calc_ema(df["close"], 5)
        df["ema13"] = self.calc_ema(df["close"], 13)
        df["ema20"] = self.calc_ema(df["close"], 20)
        df["ema50"] = self.calc_ema(df["close"], 50)
        df["ema200"] = self.calc_ema(df["close"], 200)
        df["atr"] = self.calc_atr(df)
        df["macd"], df["macd_signal"], df["macd_hist"] = self.calc_macd(df["close"])
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = self.calc_bb(df["close"])
        return df

    # ── Session Detection ─────────────────────────────────────────

    @staticmethod
    def detect_session(server_ts: int) -> str:
        """Map broker server timestamp to named trading session."""
        utc_hour = datetime.fromtimestamp(server_ts, tz=timezone.utc).hour
        active = []
        for name, (start, end) in SESSIONS.items():
            if start <= end:
                if start <= utc_hour < end:
                    active.append(name)
            else:  # wraps midnight
                if utc_hour >= start or utc_hour < end:
                    active.append(name)
        if not active:
            return "Off-Session"
        if "London" in active and "New York" in active:
            return "London/New York Overlap"
        return "/".join(active)

    # ── Price Action Narrative ────────────────────────────────────

    @staticmethod
    def _candle_character(row: pd.Series) -> str:
        body = row["close"] - row["open"]
        body_abs = abs(body)
        range_ = row["high"] - row["low"]
        pct = (body_abs / range_ * 100) if range_ > 0 else 0
        direction = "Bullish" if body > 0 else "Bearish" if body < 0 else "Doji"
        size = "Strong" if pct > 60 else "Moderate" if pct > 30 else "Small/Doji"
        wick_top = row["high"] - max(row["open"], row["close"])
        wick_bot = min(row["open"], row["close"]) - row["low"]
        wicks = ""
        if wick_top > body_abs * 0.5:
            wicks += " upper-wick"
        if wick_bot > body_abs * 0.5:
            wicks += " lower-wick"
        return f"{size} {direction}{wicks}"

    def _build_price_action_narrative(self, df: pd.DataFrame, n: int = 10) -> str:
        last_n = df.tail(n)
        descriptions = []
        for ts, row in last_n.iterrows():
            char = self._candle_character(row)
            descriptions.append(char)

        # Pattern detection (simple)
        closes = last_n["close"].values
        bearish_streak = sum(1 for i in range(1, len(closes)) if closes[i] < closes[i-1])
        bullish_streak = sum(1 for i in range(1, len(closes)) if closes[i] > closes[i-1])
        streak_note = ""
        if bearish_streak >= 3:
            streak_note = f"Bearish pressure dominant ({bearish_streak}/{n-1} bars closing lower)."
        elif bullish_streak >= 3:
            streak_note = f"Bullish pressure dominant ({bullish_streak}/{n-1} bars closing higher)."

        candle_summary = " | ".join(descriptions[-5:])
        return f"Last 5 candles: {candle_summary}. {streak_note}"

    # ── Market Situation Report ───────────────────────────────────

    async def synthesize_market_report(
        self,
        symbol: str,
        timeframe: str,
    ) -> str:
        """
        Build a compact, high-signal narrative context string for the LLM.
        Token-efficient: replaces raw arrays with English sentences.
        """
        df = await self.get_ohlc(symbol, timeframe, BARS_TO_FETCH)
        df = self.add_indicators(df)
        if len(df) < 10:
            return f"Error: Insufficient data for {symbol} ({len(df)} bars)"

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        server_ts = await self._client.server_time_from_symbol(symbol)
        session = self.detect_session(server_ts)
        server_dt = datetime.fromtimestamp(server_ts, tz=timezone.utc)

        # Current price
        bid, ask = await self._client.get_current_price(symbol)
        spread_pts = round((ask - bid) / latest.get("point", 0.00001) if "point" in latest else (ask - bid) * 100000, 1)

        # 24h high / low
        h24 = df["high"].tail(96).max()   # ~24h for M15
        l24 = df["low"].tail(96).min()
        pct_from_h24 = round((ask - h24) / h24 * 100, 4)
        pct_from_l24 = round((ask - l24) / l24 * 100, 4)

        # Distance from EMA-200
        ema200 = latest["ema200"]
        pct_from_ema200 = round((ask - ema200) / ema200 * 100, 4)
        ema200_side = "above" if ask > ema200 else "below"

        # RSI trajectory
        rsi_now = round(latest["rsi"], 2)
        rsi_prev = round(prev["rsi"], 2)
        rsi_direction = "rising" if rsi_now > rsi_prev else "falling"
        rsi_zone = (
            "overbought" if rsi_now > 70 else
            "oversold" if rsi_now < 30 else
            "neutral"
        )

        # MACD histogram state
        macd_hist_now = latest["macd_hist"]
        macd_hist_prev = prev["macd_hist"]
        macd_state = (
            "Bullish and expanding" if macd_hist_now > 0 and macd_hist_now > macd_hist_prev else
            "Bullish but contracting" if macd_hist_now > 0 else
            "Bearish and expanding" if macd_hist_now < 0 and macd_hist_now < macd_hist_prev else
            "Bearish but contracting"
        )

        # ATR
        atr = round(latest["atr"], 5)

        # Average spread (last 20 bars proxy via tick volume)
        avg_vol = round(df["volume"].tail(20).mean(), 1)
        cur_vol = int(latest["volume"])

        # Price action narrative
        pa_narrative = self._build_price_action_narrative(df)

        report = f"""
=== MARKET SITUATION REPORT ===
[IDENTITY]
Symbol: {symbol} | Timeframe: {timeframe} | Session: {session}
Server Time (UTC): {server_dt.strftime('%Y-%m-%d %H:%M')}

[PRICE DYNAMICS]
Bid: {bid:.5f} | Ask: {ask:.5f} | Spread: {spread_pts} pts
24h High: {h24:.5f} ({'+' if pct_from_h24 >= 0 else ''}{pct_from_h24}% from Ask)
24h Low:  {l24:.5f} ({'+' if pct_from_l24 >= 0 else ''}{pct_from_l24}% from Ask)
EMA-200:  {ema200:.5f} — Price is {ema200_side} by {abs(pct_from_ema200):.4f}%
EMA-50:   {latest['ema50']:.5f} | EMA-20: {latest['ema20']:.5f}

[MOMENTUM INDICATORS]
RSI(14): {rsi_now} ({rsi_direction} from {rsi_prev}) — Zone: {rsi_zone}
MACD Histogram: {macd_state} (hist={round(macd_hist_now, 6)})
ATR(14): {atr} (baseline volatility per bar)

[PRICE ACTION — LAST 10 CANDLES]
{pa_narrative}

[LIQUIDITY CONTEXT]
Current Bar Volume: {cur_vol} ticks | 20-bar Avg: {avg_vol} ticks
Volume Ratio: {round(cur_vol / avg_vol, 2) if avg_vol > 0 else 'N/A'}x average
""".strip()

        log.debug("Market report synthesized", symbol=symbol, length=len(report))
        return report

    async def get_symbol_info_for_risk(self, symbol: str) -> dict:
        """Return point value and pip value for position sizing."""
        si = await self._client.symbol_info(symbol)
        return {
            "point": si.point,
            "digits": si.digits,
            "volume_min": si.volume_min,
            "volume_max": si.volume_max,
            "volume_step": si.volume_step,
            "tick_size": si.trade_tick_size,
            "contract_size": si.trade_contract_size,
        }
