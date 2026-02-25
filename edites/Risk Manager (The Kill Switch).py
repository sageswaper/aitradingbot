"""
risk_manager.py — The system's safety valve.
Aggressively patched: REAL Equity Protector (Nuke Function).
"""
from __future__ import annotations

import math
from typing import Optional
import MetaTrader5 as mt5

from config import (
    MAX_DRAWDOWN_PCT, MAX_OPEN_TRADES,
    RISK_HIGH_CONFIDENCE, RISK_MED_CONFIDENCE, RISK_LOW_CONFIDENCE,
    CONFIDENCE_HIGH_THRESHOLD, CONFIDENCE_MED_THRESHOLD
)
from mt5_client import MT5Client
from logger import get_logger

log = get_logger("risk_manager")

class HaltTradingError(RuntimeError): pass
class RiskVetoError(RuntimeError): pass

class RiskManager:
    def __init__(self, client: MT5Client) -> None:
        self._client = client
        self._peak_equity: Optional[float] = None
        self._nuked: bool = False

    async def check_all(self, symbol: str, signal: str, confidence: float) -> None:
        if signal == "HOLD" or self._nuked: return
        await self.check_global_drawdown()
        await self.check_simultaneous_trades(symbol)

    async def check_global_drawdown(self) -> None:
        info = await self._client.account_info()
        equity, balance = info.equity, info.balance

        if self._peak_equity is None: self._peak_equity = balance
        self._peak_equity = max(self._peak_equity, equity)

        drawdown_pct = (self._peak_equity - equity) / self._peak_equity * 100

        if drawdown_pct >= MAX_DRAWDOWN_PCT:
            msg = f"💥 DRAWDOWN BREACH: {drawdown_pct:.2f}% >= {MAX_DRAWDOWN_PCT}% 💥"
            log.critical(msg)
            await self._nuke_all_positions() # THE REAL SHIELD
            raise HaltTradingError(msg)

    async def _nuke_all_positions(self) -> None:
        """
        THE EMERGENCY KILL SWITCH. 
        Closes EVERY open position at market price IMMEDIATELY.
        """
        if self._nuked: return
        self._nuked = True
        log.critical("☢️ ACTIVATING NUCLEAR KILL SWITCH - CLOSING ALL TRADES ☢️")
        
        positions = await self._client.get_open_positions()
        from config import MAGIC_NUMBER
        
        for pos in positions:
            if pos.magic != MAGIC_NUMBER: continue
            
            is_buy = pos.type == mt5.ORDER_TYPE_BUY
            close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
            bid, ask = await self._client.get_current_price(pos.symbol)
            price = bid if is_buy else ask
            
            request = {
                "action": int(mt5.TRADE_ACTION_DEAL),
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": int(close_type),
                "position": pos.ticket,
                "price": float(price),
                "deviation": 50, # Accept high slippage to GTFO
                "magic": MAGIC_NUMBER,
                "comment": "EMERGENCY_NUKE",
                "type_time": int(mt5.ORDER_TIME_GTC),
                "type_filling": int(mt5.ORDER_FILLING_FOK)
            }
            
            import asyncio
            res = await asyncio.get_event_loop().run_in_executor(None, mt5.order_send, request)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                log.critical(f"Nuked position {pos.ticket} for {pos.symbol} successfully.")
            else:
                log.error(f"Failed to nuke {pos.ticket}. RC: {res.retcode if res else 'None'}")

    async def check_simultaneous_trades(self, symbol: str) -> None:
        positions = await self._client.get_open_positions()
        if len(positions) >= MAX_OPEN_TRADES:
            raise RiskVetoError(f"Max trades ({MAX_OPEN_TRADES}) reached. Vetoing {symbol}.")

    async def calculate_lot_size(self, symbol: str, sl_price: float, entry_price: float, confidence: float, symbol_info: dict, atr: float = 0.0) -> float:
        acct = await self._client.account_info()
        risk_pct = RISK_HIGH_CONFIDENCE if confidence >= CONFIDENCE_HIGH_THRESHOLD else RISK_MED_CONFIDENCE
        risk_amount = acct.free_margin * (risk_pct / 100)

        price_dist = abs(entry_price - sl_price)
        if price_dist == 0: 
            # HARD FALLBACK: If AI is dumb and gives 0 SL, we FORCE an ATR-based SL.
            price_dist = max(atr * 1.5, 0.001) # Minimum distance
            log.warning(f"AI gave 0 SL for {symbol}. Forcing ATR-based risk distance: {price_dist}")

        contract_size = symbol_info.get("contract_size", 100000)
        tick_size = symbol_info.get("tick_size", 0.00001)
        pip_value_per_lot = tick_size * contract_size
        
        sl_distance_ticks = price_dist / tick_size
        lot_raw = risk_amount / (sl_distance_ticks * pip_value_per_lot)

        vol_min = symbol_info.get("volume_min", 0.01)
        vol_max = symbol_info.get("volume_max", 100.0)
        vol_step = symbol_info.get("volume_step", 0.01)

        lot_clamped = max(vol_min, min(vol_max, round(math.floor(lot_raw / vol_step) * vol_step, 2)))
        # ABSOLUTE HARD CAP
        return min(lot_clamped, 2.0)