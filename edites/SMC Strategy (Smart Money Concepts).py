"""
strategy.py — Smart Money Concepts (SMC) & Liquidity Sweeps.
Aggressively patched: Removed naive EMA crosses. Added real institutional logic.
"""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from logger import get_logger

log = get_logger("strategy")

@dataclass
class TradeDecision:
    signal: str
    reason: str
    confidence: float
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0

class BaseStrategy:
    def should_analyze(self, df: pd.DataFrame, current_price: float, symbol: str) -> tuple[bool, str]:
        return True, "Proceed"

class SmartMoneyStrategy(BaseStrategy):
    """
    Institutional Logic:
    1. Liquidity Sweeps: Does price sweep a recent high/low and immediately reject?
    2. Volatility Breakout: Is ATR surging while we enter?
    3. AI MUST confirm the narrative. No dumb EMA stacking.
    """
    def should_analyze(self, df: pd.DataFrame, current_price: float, symbol: str) -> tuple[bool, str]:
        if df.empty or len(df) < 20: return False, "Insufficient data"
        latest = df.iloc[-1]
        
        # Volatility check - Is the market even moving?
        if latest["atr"] < (df["atr"].mean() * 0.5):
            return False, "Dead Market - Low ATR"
            
        return True, "Market Active"

    def evaluate(self, df: pd.DataFrame, ai_signal: dict, current_price: float, symbol: str) -> TradeDecision:
        latest = df.iloc[-1]
        ai_sig = ai_signal.get("signal", "HOLD")
        ai_conf = ai_signal.get("confidence_score", 0.0)
        
        # 1. Identify short-term liquidity pools (Last 10 bars max/min)
        recent_high = df['high'].tail(10).max()
        recent_low = df['low'].tail(10).min()
        
        # 2. SMC Sweep Logic
        # BUY condition: Price swept the low but closed above it, and AI confirms BUY
        liquidity_sweep_buy = current_price > recent_low and df['low'].iloc[-2] <= recent_low
        
        # SELL condition: Price swept the high but closed below it, and AI confirms SELL
        liquidity_sweep_sell = current_price < recent_high and df['high'].iloc[-2] >= recent_high

        # 3. Decision Matrix
        signal = "HOLD"
        reason = "Awaiting Liquidity Sweep"
        
        if ai_sig == "EXIT":
            signal = "EXIT"
            reason = "AI Panic Exit"
        elif ai_sig == "BUY" and liquidity_sweep_buy:
            signal = "BUY"
            reason = "SMC Buy: Liquidity Low Swept"
        elif ai_sig == "SELL" and liquidity_sweep_sell:
            signal = "SELL"
            reason = "SMC Sell: Liquidity High Swept"
        elif ai_sig in ("BUY", "SELL") and ai_conf > 0.85:
            # Override for extreme AI confidence despite no sweep
            signal = ai_sig
            reason = "AI Hyper-Confidence Breakout"

        return TradeDecision(signal, reason, ai_conf, 
                             ai_signal.get("entry_params", {}).get("suggested_price", current_price),
                             ai_signal.get("entry_params", {}).get("stop_loss", 0.0),
                             ai_signal.get("entry_params", {}).get("take_profit", 0.0))

class TrendMeanReversionStrategy:
    def __init__(self):
        # We replace all naive strategies with the Smart Money Strategy
        self.smc = SmartMoneyStrategy()

    def _get_strategy(self, symbol: str) -> BaseStrategy:
        return self.smc

    def should_analyze(self, df: pd.DataFrame, current_price: float, symbol: str = "EURUSD-T") -> tuple[bool, str]:
        return self.smc.should_analyze(df, current_price, symbol)

    def evaluate(self, df: pd.DataFrame, ai_signal: dict, current_price: float, symbol: str = "") -> TradeDecision:
        return self.smc.evaluate(df, ai_signal, current_price, symbol)