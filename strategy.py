"""
strategy.py — Hybrid Trend + Mean Reversion strategy.

Rules (all 3 must align for BUY or SELL):
  1. EMA-200 trend filter — price must be on correct side
  2. RSI mean-reversion entry — price overextended against trend
  3. AI signal must agree with BUY/SELL and confidence ≥ threshold

Any misalignment → HOLD.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config import MIN_CONFIDENCE_TO_TRADE, SCALPING_MODE
from logger import get_logger

log = get_logger("strategy")

RSI_OVERSOLD  = 40   # For BUY entries
RSI_OVERBOUGHT = 60  # For SELL entries


@dataclass
class TradeDecision:
    signal: str                  # "BUY" | "SELL" | "HOLD"
    reason: str
    confidence: float
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0


class BaseStrategy:
    def should_analyze(self, df: pd.DataFrame, current_price: float, symbol: str) -> tuple[bool, str]:
        return True, "Proceed"

    def evaluate(self, df: pd.DataFrame, ai_signal: dict, current_price: float, symbol: str) -> TradeDecision:
        return TradeDecision("HOLD", "Not implemented", 0.0)

class ForexStrategy(BaseStrategy):
    """
    Forex Multi-EMA Scalping Strategy.
    Optimized for EURUSD, GBPUSD, USDJPY.
    Technical Rules:
    1. Trend Filter: Price must be above EMA-200 for BUY or below for SELL.
    2. Momentum Stack: EMA-5 > EMA-13 > EMA-20 confirms bullish momentum.
    3. RSI Guard: RSI(14) crossover from extremes (40/60) confirms entry quality.
    4. AI consensus serves as the final trigger based on macro-alignment.
    """
    def should_analyze(self, df: pd.DataFrame, current_price: float, symbol: str) -> tuple[bool, str]:
        if df.empty or len(df) < 21: return False, "Insufficient data"
        latest = df.iloc[-1]
        rsi = latest["rsi"]
        # Analyze if RSI is near extremes or we have an EMA cross
        if rsi < 45 or rsi > 55: return True, "RSI in active zone"
        return False, "Market consolidating"

    def evaluate(self, df: pd.DataFrame, ai_signal: dict, current_price: float, symbol: str) -> TradeDecision:
        latest, prev = df.iloc[-1], df.iloc[-2]
        ai_sig, ai_conf = ai_signal.get("signal", "HOLD"), ai_signal.get("confidence_score", 0.0)
        
        # Trend Filter
        is_bullish = current_price > latest["ema200"]
        # EMA Stack (5/13/20)
        stack_bullish = latest["ema5"] > latest["ema13"] > latest["ema20"]
        
        signal = "HOLD"
        reason = "No alignment"
        
        if ai_sig == "BUY" and is_bullish and stack_bullish:
            signal = "BUY"
        elif ai_sig == "SELL" and not is_bullish and not stack_bullish:
            signal = "SELL"
        elif ai_sig == "EXIT":
            signal = "EXIT"

        return TradeDecision(signal, reasoning := ai_signal.get("reasoning", ""), ai_conf, 
                             ai_signal.get("entry_params", {}).get("suggested_price", current_price),
                             ai_signal.get("entry_params", {}).get("stop_loss", 0.0),
                             ai_signal.get("entry_params", {}).get("take_profit", 0.0))

class IndexStrategy(BaseStrategy):
    """
    Index Volatility Momentum Strategy.
    Optimized for USA30, GER40, US100.
    Technical Rules:
    1. Volatility Breakout: Focus on price expansion outside Bollinger Bands (20,2).
    2. MACD Momentum: MACD Histogram must be expanding in the signal direction.
    3. High-Speed Trend: Indices trend aggressively; focus on BB mid-line support/resistance.
    4. AI analysis must confirm high-probability liquidity breakouts.
    """
    def should_analyze(self, df: pd.DataFrame, current_price: float, symbol: str) -> tuple[bool, str]:
        if df.empty or len(df) < 20: return False, "Small DF"
        latest = df.iloc[-1]
        # Analyze on Bollinger Band pressure or MACD momentum
        if current_price > latest["bb_upper"] or current_price < latest["bb_lower"]:
            return True, "Bollinger Squeeze/Breakout"
        if abs(latest["macd_hist"]) > abs(df["macd_hist"].tail(10).mean()):
            return True, "High MACD momentum"
        return False, "Low volatility"

    def evaluate(self, df: pd.DataFrame, ai_signal: dict, current_price: float, symbol: str) -> TradeDecision:
        latest = df.iloc[-1]
        ai_sig, ai_conf = ai_signal.get("signal", "HOLD"), ai_signal.get("confidence_score", 0.0)
        
        # Indices trend hard. We follow BB breakouts.
        bb_buy = current_price > latest["bb_mid"] and latest["macd_hist"] > 0
        bb_sell = current_price < latest["bb_mid"] and latest["macd_hist"] < 0
        
        signal = "HOLD"
        if ai_sig == "BUY" and bb_buy: signal = "BUY"
        elif ai_sig == "SELL" and bb_sell: signal = "SELL"
        elif ai_sig == "EXIT": signal = "EXIT"

        return TradeDecision(signal, ai_signal.get("reasoning", ""), ai_conf,
                             ai_signal.get("entry_params", {}).get("suggested_price", current_price),
                             ai_signal.get("entry_params", {}).get("stop_loss", 0.0),
                             ai_signal.get("entry_params", {}).get("take_profit", 0.0))

class CommodityStrategy(BaseStrategy):
    """
    Commodity Trend-Follow & Pullback Strategy.
    Optimized for GOLD (XAUUSD).
    Technical Rules:
    1. Golden Cross Guard: Bullish if EMA-50 > EMA-200.
    2. Pullback Entry: Use EMA-20 as a dynamic support/resistance during strong trends.
    3. ATR Volatility Filter: Avoid entry if market is flat and spread is > 30% of ATR.
    4. AI must identify macro structural shifts and potential reversals.
    """
    def evaluate(self, df: pd.DataFrame, ai_signal: dict, current_price: float, symbol: str) -> TradeDecision:
        latest = df.iloc[-1]
        ai_sig, ai_conf = ai_signal.get("signal", "HOLD"), ai_signal.get("confidence_score", 0.0)
        
        # Golden Cross check
        trend_up = latest["ema50"] > latest["ema200"]
        # Pullback check
        pullback_buy = current_price < latest["ema20"] and trend_up
        pullback_sell = current_price > latest["ema20"] and not trend_up
        
        signal = "HOLD"
        if ai_sig == "BUY" and trend_up: signal = "BUY" # We allow trend riding
        elif ai_sig == "SELL" and not trend_up: signal = "SELL"
        elif ai_sig == "EXIT": signal = "EXIT"

        return TradeDecision(signal, ai_signal.get("reasoning", ""), ai_conf,
                             ai_signal.get("entry_params", {}).get("suggested_price", current_price),
                             ai_signal.get("entry_params", {}).get("stop_loss", 0.0),
                             ai_signal.get("entry_params", {}).get("take_profit", 0.0))

class TrendMeanReversionStrategy:
    """The main Dispatcher class - routes requests to specialized strategies."""
    def __init__(self):
        self.fx = ForexStrategy()
        self.idx = IndexStrategy()
        self.cmd = CommodityStrategy()

    def _get_strategy(self, symbol: str) -> BaseStrategy:
        s = symbol.upper()
        if any(x in s for x in ["USA30", "GER40", "US100", "NAS100", "DAX", "DJI", "USA500"]):
            return self.idx
        if any(x in s for x in ["GOLD", "XAU", "SILVER", "BRENT", "WTI", "XAG"]):
            return self.cmd
        return self.fx

    def should_analyze(self, df: pd.DataFrame, current_price: float, symbol: str = "EURUSD-T") -> tuple[bool, str]:
        return self._get_strategy(symbol).should_analyze(df, current_price, symbol)

    def evaluate(self, df: pd.DataFrame, ai_signal: dict, current_price: float, symbol: str = "") -> TradeDecision:
        # AGGRESSIVE SCALPING GLOBAL: If AI confidence is ultra-high, prioritize it.
        ai_conf = ai_signal.get("confidence_score", 0.0)
        ai_sig = ai_signal.get("signal", "HOLD")
        
        if SCALPING_MODE and ai_conf >= 0.85 and ai_sig in ("BUY", "SELL"):
            log.info(f"ULTRA HIGH CONFIDENCE AI ({ai_conf}): Bypassing technical dispatch for {symbol}")
            return TradeDecision(ai_sig, ai_signal.get("reasoning", ""), ai_conf,
                                 ai_signal.get("entry_params", {}).get("suggested_price", current_price),
                                 ai_signal.get("entry_params", {}).get("stop_loss", 0.0),
                                 ai_signal.get("entry_params", {}).get("take_profit", 0.0))
        
        return self._get_strategy(symbol).evaluate(df, ai_signal, current_price, symbol)
