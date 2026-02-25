"""
risk_manager.py — The system's safety valve and position sizing engine.
"""
from __future__ import annotations

import math
from typing import Optional

import MetaTrader5 as mt5

from config import (
    MAX_DRAWDOWN_PCT, MAX_OPEN_TRADES,
    RISK_HIGH_CONFIDENCE, RISK_MED_CONFIDENCE, RISK_LOW_CONFIDENCE,
    CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MED_THRESHOLD,
    CORRELATION_GROUPS, SCALPING_MODE
)
from mt5_client import MT5Client
from logger import get_logger

log = get_logger("risk_manager")

class HaltTradingError(RuntimeError):
    """Raised when a critical risk rule is breached — stop all trading."""

class RiskVetoError(RuntimeError):
    """Raised when a specific trade is vetoed (not a full halt)."""

class RiskManager:
    def __init__(self, client: MT5Client) -> None:
        self._client = client
        self._peak_equity: Optional[float] = None

    async def check_all(self, symbol: str, signal: str, confidence: float) -> None:
        if signal == "HOLD":
            return

        await self.check_global_drawdown()
        await self.check_simultaneous_trades(symbol)
        await self.check_opposite_position(symbol, signal)
        await self.check_correlation_exposure(symbol, signal)
        log.info("Risk checks passed", symbol=symbol, signal=signal)

    async def check_global_drawdown(self) -> None:
        info = await self._client.account_info()
        equity = info.equity
        balance = info.balance

        if self._peak_equity is None:
            self._peak_equity = self._client.initial_balance or balance
        self._peak_equity = max(self._peak_equity, equity)

        drawdown_pct = (self._peak_equity - equity) / self._peak_equity * 100

        if drawdown_pct >= MAX_DRAWDOWN_PCT:
            msg = f"[CRITICAL] DRAWDOWN BREACH: {drawdown_pct:.2f}% >= {MAX_DRAWDOWN_PCT}% - ACTIVATING NUKE"
            log.critical(msg)
            # If we're inside the orchestrator context, we can attempt a direct nuke
            await self._nuke_all_positions()
            raise HaltTradingError(msg)

    async def _nuke_all_positions(self) -> None:
        """[NUCLEAR KILL SWITCH]: GTFO of every trade NOW. """
        log.critical("[GTFO] CLOSING ALL POSITIONS IMMEDIATELY")
        positions = await self._client.get_open_positions()
        from config import MAGIC_NUMBER
        for p in positions:
            if p.magic != MAGIC_NUMBER: continue
            is_buy = p.type == mt5.ORDER_TYPE_BUY
            price = (await self._client.get_current_price(p.symbol))[0 if is_buy else 1]
            request = {
                "action": int(mt5.TRADE_ACTION_DEAL),
                "symbol": p.symbol,
                "volume": p.volume,
                "type": int(mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY),
                "position": p.ticket,
                "price": float(price),
                "deviation": 50, # Extreme tolerance
                "magic": MAGIC_NUMBER,
                "comment": "NUCLEAR_PROTECTION",
                "type_time": int(mt5.ORDER_TIME_GTC),
                "type_filling": int(mt5.ORDER_FILLING_FOK)
            }
            await asyncio.get_event_loop().run_in_executor(None, mt5.order_send, request)
            log.info(f"Nuked position {p.ticket} for {p.symbol}")

    async def check_simultaneous_trades(self, symbol: str) -> None:
        positions = await self._client.get_open_positions()
        if len(positions) >= MAX_OPEN_TRADES:
            raise RiskVetoError(f"Max trades limit reached ({len(positions)})")

    async def check_spread_health(self, symbol: str, current_spread: float, avg_spread: float) -> None:
        from config import MAX_SPREAD_RATIO
        if avg_spread <= 0: return
        ratio = current_spread / avg_spread
        if ratio > MAX_SPREAD_RATIO:
            raise RiskVetoError(f"Spread too wide ({ratio:.2f}x avg)")
        if ratio < 0.1:
            raise RiskVetoError(f"Spread too thin ({ratio:.2f}x avg)")

    async def check_opposite_position(self, symbol: str, signal: str) -> None:
        positions = await self._client.get_open_positions(symbol)
        opp_type = mt5.ORDER_TYPE_SELL if signal == "BUY" else mt5.ORDER_TYPE_BUY
        for pos in positions:
            if pos.type == opp_type:
                raise RiskVetoError(f"Opposite position exists for {symbol}")

    async def check_correlation_exposure(self, symbol: str, signal: str) -> None:
        group = self._find_correlation_group(symbol)
        if not group: return
        positions = await self._client.get_open_positions()
        for pos in positions:
            if pos.symbol != symbol and pos.symbol in group:
                pos_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
                if pos_type == signal:
                    log.warning(f"Correlated exposure: {pos.symbol} and {symbol}")

    async def calculate_lot_size(self, symbol: str, sl_price: float, entry_price: float, confidence: float, symbol_info: dict, atr: float = 0.0) -> float:
        acct = await self._client.account_info()
        free_margin = acct.free_margin

        if confidence >= CONFIDENCE_HIGH_THRESHOLD: risk_pct = RISK_HIGH_CONFIDENCE
        elif confidence >= CONFIDENCE_MED_THRESHOLD: risk_pct = RISK_MED_CONFIDENCE
        else: risk_pct = RISK_LOW_CONFIDENCE

        risk_amount = free_margin * (risk_pct / 100)
        price_dist = abs(entry_price - sl_price)
        
        if atr > 0:
            max_sl_dist = atr * 3.5
            if price_dist > max_sl_dist:
                price_dist = max_sl_dist

        if price_dist == 0: return 0.0

        contract_size = symbol_info.get("contract_size", 100000)
        tick_size = symbol_info.get("tick_size", 0.00001)
        pip_value = tick_size * contract_size
        lot_raw = risk_amount / ((price_dist / tick_size) * pip_value)

        from config import MAX_LOT_CAP
        # Apply user's absolute limit of 2.0 OR config's cap
        lot_raw = min(lot_raw, MAX_LOT_CAP, 2.0)
        
        vol_step = symbol_info.get("volume_step", 0.01)
        lot_stepped = round(math.floor(lot_raw / vol_step) * vol_step, 2)
        return max(symbol_info.get("volume_min", 0.01), min(symbol_info.get("volume_max", 100.0), lot_stepped))

    def _find_correlation_group(self, symbol: str) -> Optional[list[str]]:
        for group in CORRELATION_GROUPS:
            if symbol in group: return group
        return None
