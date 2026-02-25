"""
risk_manager.py — The system's safety valve and position sizing engine.

Operates independently of AI confidence to enforce non-negotiable rules:
  - Max account drawdown halt (5%)
  - ATR-adjusted, confidence-scaled lot sizing
  - Maximum simultaneous open trades cap
  - Correlated-pair exposure detection
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


# ────────────────────────────────────────────────────────────────
# Custom exceptions
# ────────────────────────────────────────────────────────────────
class HaltTradingError(RuntimeError):
    """Raised when a critical risk rule is breached — stop all trading."""


class RiskVetoError(RuntimeError):
    """Raised when a specific trade is vetoed (not a full halt)."""


# ────────────────────────────────────────────────────────────────
# RiskManager
# ────────────────────────────────────────────────────────────────
class RiskManager:
    """
    All risk checks live here. Call check_all() before any order placement.
    """

    def __init__(self, client: MT5Client) -> None:
        self._client = client
        self._peak_equity: Optional[float] = None

    # ── Master Check ──────────────────────────────────────────────

    async def check_all(self, symbol: str, signal: str, confidence: float) -> None:
        """
        Run all risk checks in order.
        Raises HaltTradingError or RiskVetoError on any violation.
        Does nothing if signal == 'HOLD'.
        """
        if signal == "HOLD":
            return

        await self.check_global_drawdown()
        await self.check_simultaneous_trades(symbol)
        await self.check_opposite_position(symbol, signal)
        await self.check_correlation_exposure(symbol, signal)
        log.info(
            "Risk checks passed",
            symbol=symbol,
            signal=signal,
            confidence=confidence,
        )

    # ── Drawdown Guard ────────────────────────────────────────────

    async def check_global_drawdown(self) -> None:
        """
        Compare current equity to peak equity.
        If drawdown ≥ MAX_DRAWDOWN_PCT, raise HaltTradingError.
        """
        info = await self._client.account_info()
        equity = info.equity
        balance = info.balance

        # Track peak equity dynamically
        if self._peak_equity is None:
            self._peak_equity = self._client.initial_balance or balance
        self._peak_equity = max(self._peak_equity, equity)

        drawdown_pct = (self._peak_equity - equity) / self._peak_equity * 100

        log.debug(
            "Drawdown check",
            equity=equity,
            peak=self._peak_equity,
            drawdown_pct=round(drawdown_pct, 4),
            limit_pct=MAX_DRAWDOWN_PCT,
        )

        if drawdown_pct >= MAX_DRAWDOWN_PCT:
            msg = (
                f"DRAWDOWN LIMIT BREACHED: {drawdown_pct:.2f}% >= {MAX_DRAWDOWN_PCT}% — "
                f"HALTING ALL TRADING. Equity: {equity}, Peak: {self._peak_equity}"
            )
            log.critical(msg)
            raise HaltTradingError(msg)

    # ── Trade Count Cap ───────────────────────────────────────────

    async def check_simultaneous_trades(self, symbol: str) -> None:
        """Veto if total open positions >= MAX_OPEN_TRADES."""
        positions = await self._client.get_open_positions()
        count = len(positions)
        
        # FIXED: Removed the * 10 multiplier that crashed margin limits
        limit = MAX_OPEN_TRADES
        
        if count >= limit:
            msg = (
                f"Max simultaneous trades reached ({count}/{limit}). "
                f"Vetoing new {symbol} trade."
            )
            log.warning(msg)
            raise RiskVetoError(msg)

    # ── Hedging Prevention ──────────────────────────────────────

    async def check_opposite_position(self, symbol: str, signal: str) -> None:
        """Veto if an open position already exists in the opposite direction for the same symbol."""
        positions = await self._client.get_open_positions(symbol)
        opp_type = mt5.ORDER_TYPE_SELL if signal == "BUY" else mt5.ORDER_TYPE_BUY
        
        for pos in positions:
            if pos.type == opp_type:
                msg = f"Opposite position already exists for {symbol}. Vetoing {signal} to prevent hedging."
                log.warning(msg)
                raise RiskVetoError(msg)

    # ── Correlation Exposure ──────────────────────────────────────

    async def check_correlation_exposure(self, symbol: str, signal: str) -> None:
        """
        Detect if existing positions in the same correlation group
        would create conflicting exposure.
        e.g., if short USDCHF while trying to BUY EURUSD — both USD-negative → fine.
        But if long EURUSD while also trying to short GBPUSD (same group) → conflict.
        """
        group = self._find_correlation_group(symbol)
        if group is None:
            return  # Symbol not in any tracked group

        positions = await self._client.get_open_positions()
        for pos in positions:
            pos_symbol = pos.symbol
            if pos_symbol == symbol or pos_symbol not in group:
                continue
            pos_type = "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL"
            # Same group, same direction = concentrated exposure warning (not veto)
            if pos_type == signal:
                log.warning(
                    "Correlated exposure detected",
                    existing_symbol=pos_symbol,
                    existing_direction=pos_type,
                    new_symbol=symbol,
                    new_direction=signal,
                )
                # Log only — user can tighten to RiskVetoError if desired

    # ── Lot Sizing ────────────────────────────────────────────────

    async def calculate_lot_size(
        self,
        symbol: str,
        sl_price: float,
        entry_price: float,
        confidence: float,
        symbol_info: dict,
        atr: float = 0.0,
    ) -> float:
        """
        Risk-based position sizing:
          lot = (free_margin * risk_pct) / (sl_distance_in_price * contract_size)

        Clamped to broker volume limits and confidence-scaled.
        Returns 0.0 if parameters are invalid.
        """
        acct = await self._client.account_info()
        free_margin = acct.free_margin

        # Select risk % by confidence tier
        if confidence >= CONFIDENCE_HIGH_THRESHOLD:
            risk_pct = RISK_HIGH_CONFIDENCE
        elif confidence >= CONFIDENCE_MED_THRESHOLD:
            risk_pct = RISK_MED_CONFIDENCE
        else:
            risk_pct = RISK_LOW_CONFIDENCE

        risk_amount = free_margin * (risk_pct / 100)

        # RESCUE PLAN: ATR-based SL Cap
        # If SL is too wide ( > 3.5 * ATR), cap it to reduce risk and avoid "ghost" stops
        price_dist = abs(entry_price - sl_price)
        if atr > 0:
            max_sl_dist = atr * 3.5
            if price_dist > max_sl_dist:
                log.info(f"SL Cap triggered for {symbol}", original=round(price_dist, 5), capped=round(max_sl_dist, 5))
                price_dist = max_sl_dist
                # Adjust final SL price for lot calculation (keep direction)
                if sl_price < entry_price: # BUY
                    sl_price = entry_price - price_dist
                else: # SELL
                    sl_price = entry_price + price_dist

        if price_dist == 0:
            log.warning("SL distance is zero — cannot size position")
            return 0.0

        contract_size = symbol_info.get("contract_size", 100000)
        tick_size = symbol_info.get("tick_size", 0.00001)

        # dollar value of 1 pip move for 1 lot
        pip_value_per_lot = tick_size * contract_size

        sl_distance_ticks = price_dist / tick_size
        lot_raw = risk_amount / (sl_distance_ticks * pip_value_per_lot)

        # Clamp to broker limits
        vol_min = symbol_info.get("volume_min", 0.01)
        vol_max = symbol_info.get("volume_max", 100.0)
        vol_step = symbol_info.get("volume_step", 0.01)

        # RESCUE PLAN: Absolute Lot Cap
        # Prevent oversized positions on highly volatile assets
        lot_raw = min(lot_raw, 5.0) # Absolute cap for demo/safety

        # Round to nearest step
        lot_stepped = round(math.floor(lot_raw / vol_step) * vol_step, 2)
        lot_clamped = max(vol_min, min(vol_max, lot_stepped))

        log.info(
            "Lot size calculated",
            symbol=symbol,
            confidence=confidence,
            risk_pct=risk_pct,
            risk_amount_usd=round(risk_amount, 2),
            sl_distance=round(price_dist, 5),
            lot_raw=round(lot_raw, 4),
            lot_final=lot_clamped,
        )
        return lot_clamped

    # ── Helpers ───────────────────────────────────────────────────

    def _find_correlation_group(self, symbol: str) -> Optional[list[str]]:
        for group in CORRELATION_GROUPS:
            if symbol in group:
                return group
        return None
