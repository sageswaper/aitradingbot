"""
execution_engine.py — Full trade lifecycle management.
Aggressively patched: Prop-Firm Safe Trailing (Anti-Spam), Dynamic Deviation, & Spread Guard.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from ai_brain import AIBrain

import MetaTrader5 as mt5

from config import DRY_RUN
from mt5_client import MT5Client
from logger import get_logger
from telegram_notifier import TelegramNotifier

log = get_logger("execution_engine")

_RETCODE_MAP: dict[int, tuple[str, bool]] = {
    mt5.TRADE_RETCODE_DONE:             ("Order executed successfully",         False),
    mt5.TRADE_RETCODE_REQUOTE:          ("Requote — price changed",              True),
    mt5.TRADE_RETCODE_REJECT:           ("Request rejected",                    False),
    mt5.TRADE_RETCODE_CANCEL:           ("Request cancelled",                   False),
    mt5.TRADE_RETCODE_PLACED:           ("Order placed (pending)",              False),
    mt5.TRADE_RETCODE_DONE_PARTIAL:     ("Partial fill",                        False),
    mt5.TRADE_RETCODE_ERROR:            ("Generic error",                       False),
    mt5.TRADE_RETCODE_TIMEOUT:          ("Timeout — retry",                      True),
    mt5.TRADE_RETCODE_INVALID:          ("Invalid request",                     False),
    mt5.TRADE_RETCODE_INVALID_VOLUME:   ("Invalid volume — recalculate",         True),
    mt5.TRADE_RETCODE_INVALID_PRICE:    ("Invalid price — refresh",              True),
    mt5.TRADE_RETCODE_INVALID_STOPS:    ("Invalid SL/TP levels",                False),
    mt5.TRADE_RETCODE_TRADE_DISABLED:   ("Trade disabled for symbol",           False),
    mt5.TRADE_RETCODE_MARKET_CLOSED:    ("Market closed",                       False),
    mt5.TRADE_RETCODE_NO_MONEY:         ("Insufficient funds",                  False),
    mt5.TRADE_RETCODE_PRICE_OFF:        ("Off quotes — retry",                   True),
    mt5.TRADE_RETCODE_CONNECTION:       ("No connection to broker",              True),
    mt5.TRADE_RETCODE_PRICE_CHANGED:    ("Price changed — retry",                True),
    0:                                  ("Order executed successfully (RC 0)",  False),
}
MAX_ORDER_RETRIES = 3


class ExecutionEngine:
    def __init__(self, client: MT5Client, brain: Optional[AIBrain] = None) -> None:
        self._client = client
        self._brain = brain

    async def place_order(self, symbol: str, signal: str, entry_params: dict, lot: float, comment: str = "AIBot") -> dict:
        if DRY_RUN: return {"success": True, "ticket": 0, "detail": "DRY RUN", "dry_run": True}
        if signal not in ("BUY", "SELL"): return {"success": False, "ticket": 0, "detail": f"Invalid signal: {signal}"}

        # 1. 🚨 MAXIMUM SPREAD GUARD (Prop-Firm Safety) 🚨
        bid, ask = await self._client.get_current_price(symbol)
        si = await self._client.symbol_info(symbol)
        spread_points = si.spread if si else (ask - bid) / (getattr(si, 'point', 0.00001))
        
        # Determine max allowed spread dynamically (e.g., 30 points for FX, 300 for Indices)
        is_volatile = any(x in symbol.upper() for x in ["US30", "USA30", "GER40", "US100", "BTC", "GOLD", "XAU", "JPY"])
        max_allowed_spread = 350 if is_volatile else 40
        
        if spread_points > max_allowed_spread:
            msg = f"Spread too high for {symbol}: {spread_points} pts > {max_allowed_spread}. Trade aborted to prevent instant loss."
            log.warning(msg)
            return {"success": False, "ticket": 0, "detail": msg}

        # AUTO-CLOSE OPPOSITE POSITION
        from config import MAGIC_NUMBER
        existing = await self._client.get_open_positions(symbol)
        opp_type = mt5.ORDER_TYPE_SELL if signal == "BUY" else mt5.ORDER_TYPE_BUY
        for pos in existing:
            if pos.magic == MAGIC_NUMBER and pos.type == opp_type:
                log.info(f"Auto-closing opposite position {pos.ticket} for {symbol} before opening {signal}")
                await self.close_position(pos.ticket, symbol, pos.volume)

        for attempt in range(1, MAX_ORDER_RETRIES + 1):
            result = await self._send_market_order(symbol, signal, entry_params, lot, comment)
            rc = result.get("retcode", -1)
            desc, should_retry = _RETCODE_MAP.get(rc, (f"Unknown retcode {rc}", False))

            if rc in (mt5.TRADE_RETCODE_DONE, 0, 10009):
                ticket = result.get("order", 0)
                if ticket == 0: await asyncio.sleep(1.0) # wait for history sync
                
                execution_price = result.get("price", entry_params.get("suggested_price", 0.0))
                
                await TelegramNotifier.notify_trade_open(
                    symbol=symbol, signal=signal, lot=lot, price=float(execution_price),
                    sl=float(entry_params.get("stop_loss", 0.0)), tp=float(entry_params.get("take_profit", 0.0)),
                    timeframe=comment.replace("AIBot_", "") if "AIBot_" in comment else "M15",
                    reasoning=entry_params.get("reasoning", "AI Breakdown"),
                    votes=entry_params.get("votes"), ticket=ticket
                )
                return {"success": True, "ticket": ticket, "detail": desc, "retcode": rc}

            last_error_msg = result.get("comment", desc)
            log.error(f"Order failed (Attempt {attempt}): {last_error_msg} RC: {rc}")
            
            if not should_retry or attempt == MAX_ORDER_RETRIES:
                await TelegramNotifier.notify_error(f"🚨 TRADE FAILED: {signal} {symbol}\nReason: {last_error_msg} (RC: {rc})")
                return {"success": False, "ticket": 0, "detail": last_error_msg, "retcode": rc}

            await asyncio.sleep(0.5)

        return {"success": False, "ticket": 0, "detail": "All retries exhausted"}

    async def _send_market_order(self, symbol: str, signal: str, entry_params: dict, lot: float, comment: str) -> dict:
        bid, ask = await self._client.get_current_price(symbol)
        si = await self._client.symbol_info(symbol)
        if not si: return {"retcode": -1, "comment": "Symbol info unavailable"}
            
        digits = si.digits
        order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
        price = ask if signal == "BUY" else bid
        
        # 2. 🚨 DYNAMIC SLIPPAGE (Deviation) GUARD 🚨
        # Instead of fixed 20 points, allow spread-based deviation to prevent requotes on fast markets
        dynamic_deviation = int(max(20, si.spread * 1.5)) 
        
        sl = float(entry_params.get("stop_loss", 0.0))
        tp = float(entry_params.get("take_profit", 0.0))
        
        # Stops level guard
        stops_level_pts = getattr(si, "trade_stops_level", 0)
        point = getattr(si, "point", 0.00001)
        spread = ask - bid
        min_dist = max((stops_level_pts + 10) * point, spread + (10 * point))

        if signal == "BUY":
            if sl > 0 and (bid - sl) < min_dist: sl = bid - min_dist
            if tp > 0 and (tp - ask) < min_dist: tp = ask + min_dist
        else:
            if sl > 0 and (sl - ask) < min_dist: sl = ask + min_dist
            if tp > 0 and (bid - tp) < min_dist: tp = bid - min_dist

        from config import MAGIC_NUMBER
        request = {
            "action":       int(mt5.TRADE_ACTION_DEAL),
            "symbol":       str(symbol),
            "volume":       float(lot),
            "type":         int(order_type),
            "price":        float(round(price, digits)),
            "sl":           float(round(sl, digits)),
            "tp":           float(round(tp, digits)),
            "deviation":    dynamic_deviation, # Applied dynamic deviation
            "magic":        int(MAGIC_NUMBER),
            "comment":      str(comment),
            "type_time":    int(mt5.ORDER_TIME_GTC),
            "type_filling": int(getattr(mt5, "ORDER_FILLING_FOK", 1)) # Simplified filling mode fallback
        }

        # Safe Filling Mode Detection
        SYM_FOK = getattr(mt5, "SYMBOL_FILLING_FOK", 1)
        SYM_IOC = getattr(mt5, "SYMBOL_FILLING_IOC", 2)
        if si.type_filling & SYM_FOK: request["type_filling"] = mt5.ORDER_FILLING_FOK
        elif si.type_filling & SYM_IOC: request["type_filling"] = mt5.ORDER_FILLING_IOC
        else: request["type_filling"] = mt5.ORDER_FILLING_RETURN

        check_result = await asyncio.get_event_loop().run_in_executor(None, lambda: mt5.order_check(request))
        if check_result and check_result.retcode not in (mt5.TRADE_RETCODE_DONE, 0, 10009):
            log.error(f"Order check failed for {symbol}: {check_result.comment} (RC: {check_result.retcode})")
            return check_result._asdict()

        result = await asyncio.get_event_loop().run_in_executor(None, lambda: mt5.order_send(request))
        return result._asdict() if result else {"retcode": -1, "comment": str(mt5.last_error())}

    async def handle_external_closure(self, ticket: int, symbol: str, volume: float) -> float:
        import MetaTrader5 as mt5
        from datetime import datetime, timedelta

        end = datetime.now()
        start = end - timedelta(minutes=5)
        deals = await asyncio.get_event_loop().run_in_executor(None, lambda: mt5.history_deals_get(start, end))
        
        profit, entry_price, close_price, trade_type = 0.0, 0.0, 0.0, "UNKNOWN"
        
        if deals:
            for d in deals:
                if d.position_id == ticket:
                    if d.entry == 1: 
                        profit = d.profit + d.commission + d.swap
                        close_price = d.price
                        trade_type = "BUY" if d.type == mt5.DEAL_TYPE_SELL else "SELL"
                    elif d.entry == 0:
                        entry_price = d.price
        
        if entry_price == 0: return time.time()

        trade_data = {"symbol": symbol, "profit": profit, "entry_price": entry_price, "close_price": close_price, "type": trade_type, "lot": volume}
        reasoning = await self._brain.analyze_trade_outcome(trade_data) if self._brain else "Auto-closure"
        
        await TelegramNotifier.notify_trade_close(
            symbol=symbol, ticket=ticket, profit=trade_data['profit'], detail="External closure",
            lot=trade_data['lot'], entry_price=trade_data['entry_price'], close_price=trade_data['close_price'], ai_explanation=reasoning
        )
        return time.time()

    async def close_position(self, ticket: int, symbol: str, lots: float, reason: str = "AIBot close") -> dict:
        if DRY_RUN: return {"success": True, "detail": "DRY RUN"}

        from config import MAGIC_NUMBER
        pos = await asyncio.get_event_loop().run_in_executor(None, mt5.positions_get, ticket)
        if not pos: return {"retcode": -1, "comment": "Position not found"}
        
        p = pos[0]
        si = await self._client.symbol_info(symbol)
        is_buy = p.type == mt5.ORDER_TYPE_BUY
        bid, ask = await self._client.get_current_price(symbol)
        
        request = {
            "action":     int(mt5.TRADE_ACTION_DEAL),
            "symbol":     str(symbol),
            "volume":     float(lots),
            "type":       int(mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY),
            "position":   int(ticket),
            "price":      float(bid if is_buy else ask),
            "deviation":  int(max(20, si.spread * 1.5)), # Dynamic
            "magic":      int(MAGIC_NUMBER),
            "comment":    str(reason),
            "type_time":  int(mt5.ORDER_TIME_GTC),
            "type_filling": int(mt5.ORDER_FILLING_FOK)
        }
        
        result = await asyncio.get_event_loop().run_in_executor(None, mt5.order_send, request)
        if result and result.retcode in (mt5.TRADE_RETCODE_DONE, 10009, 0):
            # Try to get actual profit
            await asyncio.sleep(0.5)
            history = await asyncio.get_event_loop().run_in_executor(None, lambda: mt5.history_deals_get(position=ticket))
            profit = sum([d.profit + d.commission + d.swap for d in history if d.entry == mt5.DEAL_ENTRY_OUT]) if history else 0.0
            
            await TelegramNotifier.notify_trade_close(
                symbol=symbol, ticket=ticket, profit=profit, detail=reason,
                lot=lots, entry_price=p.price_open, close_price=request["price"], ai_explanation="Manual/Bot Closure"
            )
            return {"success": True, "retcode": result.retcode, "profit": profit}
            
        return {"success": False, "retcode": result.retcode if result else -1, "comment": str(mt5.last_error())}

    async def manage_trailing_stop(self, symbol: str, atr: float, atr_multiplier: float = 1.5) -> None:
        if DRY_RUN: return

        positions = await self._client.get_open_positions(symbol)
        from config import MAGIC_NUMBER
        for pos in positions:
            if pos.magic != MAGIC_NUMBER: continue
            
            bid, ask = await self._client.get_current_price(symbol)
            si = await self._client.symbol_info(symbol)
            digits = si.digits if si else 5
            
            trail_dist = atr * atr_multiplier
            is_buy = pos.type == mt5.ORDER_TYPE_BUY

            new_sl = round(bid - trail_dist, digits) if is_buy else round(ask + trail_dist, digits)
            
            # 3. 🚨 PROP-FIRM SAFE: STEP TRAILING (Anti-Spam) 🚨
            # Only update SL if the move is significant (> 15% of ATR) to prevent server flooding.
            min_step = atr * 0.15 
            
            if is_buy:
                if new_sl <= pos.sl or (new_sl - pos.sl) < min_step: continue
            else:
                if pos.sl > 0 and (new_sl >= pos.sl or (pos.sl - new_sl) < min_step): continue

            await self._modify_stop_loss(pos.ticket, new_sl, pos.tp, symbol)

    async def manage_hard_stops(self) -> None:
        from config import MAGIC_NUMBER
        positions = await self._client.get_open_positions()
        for p in positions:
            if p.magic != MAGIC_NUMBER: continue
            bid, ask = await self._client.get_current_price(p.symbol)
            price = bid if p.type == mt5.ORDER_TYPE_BUY else ask
            is_buy = p.type == mt5.ORDER_TYPE_BUY
            
            sl_triggered = (p.sl > 0) and ((is_buy and price <= p.sl) or (not is_buy and price >= p.sl))
            tp_triggered = (p.tp > 0) and ((is_buy and price >= p.tp) or (not is_buy and price <= p.tp))
                
            if sl_triggered or tp_triggered:
                reason = "Failsafe: SL Triggered Locally" if sl_triggered else "Failsafe: TP Triggered Locally"
                log.warning(f"LOCAL CLOSE FALLBACK for {p.symbol}", ticket=p.ticket, pnl=p.profit)
                await self.close_position(p.ticket, p.symbol, float(p.volume), reason)

    async def manage_tactical_profit_locks(self, symbol: str) -> None:
        from config import MAGIC_NUMBER, PROFIT_LOCK_BE_USD
        positions = await self._client.get_open_positions(symbol)
        for p in positions:
            if p.magic != MAGIC_NUMBER: continue
            if p.profit >= PROFIT_LOCK_BE_USD:
                is_buy = p.type == mt5.ORDER_TYPE_BUY
                entry_price = p.price_open
                current_sl = p.sl
                si = await self._client.symbol_info(p.symbol)
                point, digits = getattr(si, "point", 0.00001), getattr(si, "digits", 5)
                
                be_price = round(entry_price + (point * 2 if is_buy else -point * 2), digits) # BE + 2 points cover commission
                
                should_move = (is_buy and current_sl < entry_price) or (not is_buy and (current_sl == 0 or current_sl > entry_price))
                if should_move:
                    await self._modify_stop_loss(p.ticket, be_price, p.tp, p.symbol)

    async def close_opposite_if_signal_reverses(self, symbol: str, ai_signal: str) -> None:
        if ai_signal == "HOLD": return
        positions = await asyncio.get_event_loop().run_in_executor(None, lambda: mt5.positions_get(symbol=symbol))
        if not positions: return

        from config import MAGIC_NUMBER
        for p in positions:
            if p.magic != MAGIC_NUMBER: continue
            is_buy = p.type == mt5.ORDER_TYPE_BUY
            if (ai_signal == "SELL" and is_buy) or (ai_signal == "BUY" and not is_buy):
                await self.close_position(p.ticket, symbol, float(p.volume), reason="Signal Reversed")

    async def _modify_stop_loss(self, ticket: int, new_sl: float, tp: float, symbol: str = "") -> None:
        request = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": new_sl, "tp": tp}
        result = await asyncio.get_event_loop().run_in_executor(None, mt5.order_send, request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info("Trailing stop updated", ticket=ticket, new_sl=new_sl)
        else:
            rc = result.retcode if result else -1
            if rc != 10004: log.warning(f"Trailing stop update failed for {symbol} (#{ticket}). RC: {rc}")