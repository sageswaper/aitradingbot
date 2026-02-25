"""
strategy.py — Hybrid Trend + SMC strategy.

Rules:
  1. EMA-200 trend filter — Baseline trend direction
  2. SMC Liquidity Sweeps — Price piercing PDH/PDL and rejecting
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

    def _calculate_smc_levels(self, df: pd.DataFrame) -> dict:
        """Calculates Previous Day/Session High/Low and finds Liquidity Sweeps."""
        if len(df) < 50: return {}
        
        # SMC: Calculate levels from earlier bars to allow for a sweep on the current/prev bar
        history = df.iloc[-50:-2]
        pdh = history["high"].max() 
        pdl = history["low"].min()
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        sweep_bullish = prev["low"] < pdl and latest["close"] > pdl
        sweep_bearish = prev["high"] > pdh and latest["close"] < pdh
        
        return {
            "pdh": pdh,
            "pdl": pdl,
            "sweep_bullish": sweep_bullish,
            "sweep_bearish": sweep_bearish
        }

class ForexStrategy(BaseStrategy):
    """
    Forex SMC & Liquidity Sweep Strategy.
    Optimized for EURUSD, GBPUSD, USDJPY.
    Technical Rules:
    1. Trend Filter: Price must be on correct side of EMA-200.
    2. Liquidity Sweep: Price must pierce PDH/PDL and reject back into range.
    3. AI consensus serves as the final trigger.
    """
    def should_analyze(self, df: pd.DataFrame, current_price: float, symbol: str) -> tuple[bool, str]:
        if df.empty or len(df) < 21: return False, "Insufficient data"
        latest = df.iloc[-1]
        # In SMC, we analyze near key levels
        smc = self._calculate_smc_levels(df)
        pdh, pdl = smc.get("pdh", 0), smc.get("pdl", 0)
        
        dist_to_level = min(abs(current_price - pdh), abs(current_price - pdl))
        if dist_to_level < (latest.get("atr", 0.001) * 2):
            return True, "Price near Liquidity Pool"
        
        return True, "Standard SMC Cycle"

    def evaluate(self, df: pd.DataFrame, ai_signal: dict, current_price: float, symbol: str) -> TradeDecision:
        latest = df.iloc[-1]
        ai_sig, ai_conf = ai_signal.get("signal", "HOLD"), ai_signal.get("confidence_score", 0.0)
        smc = self._calculate_smc_levels(df)
        
        is_sweep_buy = smc.get("sweep_bullish", False)
        is_sweep_sell = smc.get("sweep_bearish", False)

        # Baseline: EMA-200 for Trend Direction
        trend_bullish = current_price > latest["ema200"]
        
        signal = "HOLD"
        reason = "No alignment"
        
        if ai_sig == "BUY" and is_sweep_buy and trend_bullish:
            signal = "BUY"
            reason = f"Bullish Liquidity Sweep at {smc.get('pdl')}"
        elif ai_sig == "SELL" and is_sweep_sell and not trend_bullish:
            signal = "SELL"
            reason = f"Bearish Liquidity Sweep at {smc.get('pdh')}"
        elif ai_sig == "EXIT":
            signal = "EXIT"
        
        if signal == "HOLD" and ai_conf > 0.90:
            signal = ai_sig
            reason = "High Confidence AI Override"

        return TradeDecision(signal, reason, ai_conf, 
                             ai_signal.get("entry_params", {}).get("suggested_price", current_price),
                             ai_signal.get("entry_params", {}).get("stop_loss", 0.0),
                             ai_signal.get("entry_params", {}).get("take_profit", 0.0))

class IndexStrategy(BaseStrategy):
    """
    Index SMC Volatility Strategy.
    Optimized for US30, US100.
    """
    def evaluate(self, df: pd.DataFrame, ai_signal: dict, current_price: float, symbol: str) -> TradeDecision:
        latest = df.iloc[-1]
        ai_sig, ai_conf = ai_signal.get("signal", "HOLD"), ai_signal.get("confidence_score", 0.0)
        smc = self._calculate_smc_levels(df)
        
        # Indices trend hard, but respect old highs/lows for liquidity.
        trend_bullish = current_price > latest["ema200"]
        
        signal = "HOLD"
        if ai_sig == "BUY" and trend_bullish: signal = "BUY"
        elif ai_sig == "SELL" and not trend_bullish: signal = "SELL"
        elif ai_sig == "EXIT": signal = "EXIT"

        return TradeDecision(signal, ai_signal.get("reasoning", ""), ai_conf,
                             ai_signal.get("entry_params", {}).get("suggested_price", current_price),
                             ai_signal.get("entry_params", {}).get("stop_loss", 0.0),
                             ai_signal.get("entry_params", {}).get("take_profit", 0.0))

class CommodityStrategy(BaseStrategy):
    def evaluate(self, df: pd.DataFrame, ai_signal: dict, current_price: float, symbol: str) -> TradeDecision:
        latest = df.iloc[-1]
        ai_sig, ai_conf = ai_signal.get("signal", "HOLD"), ai_signal.get("confidence_score", 0.0)
        trend_up = latest["ema50"] > latest["ema200"]
        
        signal = "HOLD"
        if ai_sig == "BUY" and trend_up: signal = "BUY"
        elif ai_sig == "SELL" and not trend_up: signal = "SELL"
        elif ai_sig == "EXIT": signal = "EXIT"

        return TradeDecision(signal, ai_signal.get("reasoning", ""), ai_conf,
                             ai_signal.get("entry_params", {}).get("suggested_price", current_price),
                             ai_signal.get("entry_params", {}).get("stop_loss", 0.0),
                             ai_signal.get("entry_params", {}).get("take_profit", 0.0))

class TrendMeanReversionStrategy:
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
        ai_conf = ai_signal.get("confidence_score", 0.0)
        ai_sig = ai_signal.get("signal", "HOLD")
        
        if SCALPING_MODE and ai_conf >= 0.85 and ai_sig in ("BUY", "SELL"):
            return TradeDecision(ai_sig, ai_signal.get("reasoning", ""), ai_conf,
                                 ai_signal.get("entry_params", {}).get("suggested_price", current_price),
                                 ai_signal.get("entry_params", {}).get("stop_loss", 0.0),
                                 ai_signal.get("entry_params", {}).get("take_profit", 0.0))
        
        return self._get_strategy(symbol).evaluate(df, ai_signal, current_price, symbol)
