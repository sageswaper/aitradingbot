"""
risk_manager.py — The system's ultra-hardened safety valve.
Aggressively patched: Concurrent Nuke, Daily Drawdown, Point-Aware Lot Sizing.
"""
from __future__ import annotations

import math
import asyncio
from typing import Optional
from datetime import datetime

import MetaTrader5 as mt5

from config import (
    MAX_DRAWDOWN_PCT, MAX_OPEN_TRADES,
    RISK_HIGH_CONFIDENCE, RISK_MED_CONFIDENCE, RISK_LOW_CONFIDENCE,
    CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MED_THRESHOLD,
    CORRELATION_GROUPS, DAILY_DRAWDOWN_LIMIT_PCT, MAX_LOT_CAP,
    MAX_SPREAD_RATIO
)
from mt5_client import MT5Client
from logger import get_logger
from news_manager import NewsManager

log = get_logger("risk_manager")

class HaltTradingError(RuntimeError):
    """Raised when a critical risk rule is breached — stop all trading."""

class RiskVetoError(RuntimeError):
    """Raised when a specific trade is vetoed (not a full halt)."""

class RiskManager:
    def __init__(self, client: MT5Client, news_manager: Optional[NewsManager] = None) -> None:
        self._client = client
        self._news = news_manager
        self._peak_equity: Optional[float] = None
        self._daily_start_equity: Optional[float] = None
        self._last_day_checked: Optional[str] = None
        self._nuked: bool = False

    async def check_all(self, symbol: str, signal: str, confidence: float) -> None:
        await self._sync_daily_balance()
        
        if signal == "HOLD" or self._nuked:
            return

        # 🚨 PROACTIVE NEWS VETO (The Red-Folder Radar) 🚨
        if self._news:
            is_news, reason = self._news.is_blackout(symbol)
            if is_news:
                raise RiskVetoError(f"Proactive Veto: {reason}")

        await self.check_global_drawdown()
        await self.check_daily_drawdown()
        await self.check_simultaneous_trades(symbol)
        await self.check_opposite_position(symbol, signal)
        await self.check_correlation_exposure(symbol, signal)
        log.info("Risk checks passed", symbol=symbol, signal=signal)

    async def _sync_daily_balance(self) -> None:
        """
        [PROP-FIRM SYNC]: Tracks equity at 00:00 UTC.
        Automatically resets the 'nuke' state for a fresh trading day.
        """
        from datetime import timezone
        now_utc = datetime.now(timezone.utc)
        today_utc = now_utc.strftime("%Y-%m-%d")
        
        if self._last_day_checked != today_utc:
            info = await self._client.account_info()
            self._daily_start_equity = info.equity # Prop-firm use equity snapshots
            self._last_day_checked = today_utc
            
            if self._nuked:
                log.info(f"New trading day {today_utc} detected. Resetting safety locks.")
                self._nuked = False
            
            log.info(f"Daily Snapshot Captured (UTC): {self._daily_start_equity}")

    async def check_global_drawdown(self) -> None:
        """Protects against total account drawdown from the peak equity."""
        info = await self._client.account_info()
        equity, balance = info.equity, info.balance

        if self._peak_equity is None:
            self._peak_equity = self._client.initial_balance or balance
        self._peak_equity = max(self._peak_equity, equity)

        drawdown_pct = (self._peak_equity - equity) / self._peak_equity * 100

        if drawdown_pct >= MAX_DRAWDOWN_PCT:
            msg = f"[CRITICAL] GLOBAL DRAWDOWN BREACH: {drawdown_pct:.2f}% >= {MAX_DRAWDOWN_PCT}%"
            log.critical(msg)
            await self._nuke_all_positions()
            raise HaltTradingError(msg)

    async def check_daily_drawdown(self) -> None:
        """Shields against daily bleeding (Max 2% loss from day start)."""
        info = await self._client.account_info()
        equity = info.equity
        
        if self._daily_start_equity is None: return
        
        daily_loss_pct = (self._daily_start_equity - equity) / self._daily_start_equity * 100
        
        if daily_loss_pct >= DAILY_DRAWDOWN_LIMIT_PCT:
            msg = f"[CRITICAL] DAILY DRAWDOWN BREACH: {daily_loss_pct:.2f}% >= {DAILY_DRAWDOWN_LIMIT_PCT}%"
            log.critical(msg)
            await self._nuke_all_positions()
            raise HaltTradingError(msg)

    async def _nuke_all_positions(self) -> None:
        """[NUCLEAR KILL SWITCH]: Paralellized GTFO logic."""
        if self._nuked: return
        self._nuked = True
        
        log.critical("[NUKE] ACTIVATING CONCURRENT KILL SWITCH")
        positions = await self._client.get_open_positions()
        from config import MAGIC_NUMBER
        
        tasks = []
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
                "deviation": 50, # Extreme tolerance for news volatility
                "magic": MAGIC_NUMBER,
                "comment": "NUCLEAR_PROTECTION",
                "type_time": int(mt5.ORDER_TIME_GTC),
                "type_filling": int(mt5.ORDER_FILLING_FOK)
            }
            # Launch all requests into the executor concurrently
            tasks.append(asyncio.get_event_loop().run_in_executor(None, mt5.order_send, request))
            
        if tasks:
            log.info(f"Bombarding broker with {len(tasks)} close orders...")
            await asyncio.gather(*tasks)
            log.critical("NUCLEAR STRIKE COMPLETE: ALL POSITIONS COMMANDED TO CLOSE")

    async def check_simultaneous_trades(self, symbol: str) -> None:
        positions = await self._client.get_open_positions()
        if len(positions) >= MAX_OPEN_TRADES:
            raise RiskVetoError(f"Max trades limit reached ({len(positions)})")

    async def check_spread_health(self, symbol: str, current_spread: float, avg_spread: float) -> None:
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
        
        if confidence >= CONFIDENCE_HIGH_THRESHOLD: risk_pct = RISK_HIGH_CONFIDENCE
        elif confidence >= CONFIDENCE_MED_THRESHOLD: risk_pct = RISK_MED_CONFIDENCE
        else: risk_pct = RISK_LOW_CONFIDENCE

        risk_amount = acct.free_margin * (risk_pct / 100)
        
        # [THE POINT TRAP FIX]: Use symbol's actual point value for safety
        point = symbol_info.get("point", 0.00001)
        min_dist_points = 50 * point # 5 pips minimum distance to avoid lot explosion
        
        price_dist = abs(entry_price - sl_price)
        if price_dist < min_dist_points:
            price_dist = min_dist_points # Cap at minimum distance
            log.warning(f"Stop distance too tight for {symbol}. Capped to {min_dist_points} units.")

        contract_size = symbol_info.get("contract_size", 100000)
        tick_size = symbol_info.get("tick_size", 0.00001)
        pip_value = tick_size * contract_size
        
        lot_raw = risk_amount / ((price_dist / tick_size) * pip_value)

        # [ASSET SPECIFIC LOT CAPS]: Limit Indices and Gold more strictly
        is_index_or_gold = any(x in symbol.upper() for x in ["US30", "US100", "NAS100", "XAU", "GOLD"])
        
        asset_cap = 0.5 if is_index_or_gold else 2.0
        final_cap = min(MAX_LOT_CAP, asset_cap)
        
        lot_raw = min(lot_raw, final_cap)
        
        vol_step = symbol_info.get("volume_step", 0.01)
        lot_stepped = round(math.floor(lot_raw / vol_step) * vol_step, 2)
        
        return max(symbol_info.get("volume_min", 0.01), min(symbol_info.get("volume_max", 100.0), lot_stepped))

    def _find_correlation_group(self, symbol: str) -> Optional[list[str]]:
        for group in CORRELATION_GROUPS:
            if symbol in group: return group
        return None
