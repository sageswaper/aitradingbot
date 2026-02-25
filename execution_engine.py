"""
execution_engine.py — Full trade lifecycle management.

Responsibilities:
  - Market & pending order placement
  - Spread-aware entry price selection
  - MT5 error code translation & retry logic
  - Trailing stop management on profitable positions
  - Comprehensive result logging
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

# ────────────────────────────────────────────────────────────────
# MT5 Retcode → human description
# ────────────────────────────────────────────────────────────────
_RETCODE_MAP: dict[int, tuple[str, bool]] = {
    # (description, should_retry)
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


# ────────────────────────────────────────────────────────────────
# ExecutionEngine
# ────────────────────────────────────────────────────────────────
class ExecutionEngine:
    """
    Handles the complete lifecycle of trade orders.
    Uses spread-aware pricing and retries retcodes that warrant it.
    """

    def __init__(self, client: MT5Client, brain: Optional[AIBrain] = None) -> None:
        self._client = client
        self._brain = brain

    # ── Public API ────────────────────────────────────────────────

    async def place_order(
        self,
        symbol: str,
        signal: str,           # "BUY" | "SELL"
        entry_params: dict,    # {"suggested_price", "stop_loss", "take_profit"}
        lot: float,
        comment: str = "AIBot",
    ) -> dict:
        """
        Place a market order. Returns result dict with 'success', 'ticket', 'detail'.
        In DRY_RUN mode logs the intention without touching the broker.
        """
        if DRY_RUN:
            log.info(
                "DRY RUN — order skipped",
                symbol=symbol,
                signal=signal,
                lot=lot,
                entry_params=entry_params,
            )
            return {
                "success": True,
                "ticket": 0,
                "detail": "DRY RUN — no order placed",
                "dry_run": True,
            }

        if signal not in ("BUY", "SELL"):
            return {"success": False, "ticket": 0, "detail": f"Invalid signal: {signal}"}

        # AUTO-CLOSE OPPOSITE POSITION
        from config import MAGIC_NUMBER
        existing = await self._client.get_open_positions(symbol)
        opp_type = mt5.ORDER_TYPE_SELL if signal == "BUY" else mt5.ORDER_TYPE_BUY
        for pos in existing:
            # Only manage positions with our MAGIC_NUMBER
            if pos.magic == MAGIC_NUMBER and pos.type == opp_type:
                log.info(f"Auto-closing opposite position {pos.ticket} for {symbol} before opening {signal}")
                await self.close_position(pos.ticket, symbol, pos.volume)

        for attempt in range(1, MAX_ORDER_RETRIES + 1):
            result = await self._send_market_order(
                symbol, signal, entry_params, lot, comment,
            )
            
            rc = result.get("retcode", -1)
            desc, should_retry = _RETCODE_MAP.get(rc, (f"Unknown retcode {rc}", False))

            if rc in (mt5.TRADE_RETCODE_DONE, 0, 10009):
                # TICKET GUARD: Ensure we have a valid ticket
                ticket = result.get("order", 0)
                
                # If ticket is 0, wait slightly and check again (Secondary confirmation)
                if ticket == 0:
                    log.warning("Order reported DONE but ticket is 0. searching history...", symbol=symbol)
                    await asyncio.sleep(1.5)
                    # Check history more aggressively
                    from datetime import datetime, timedelta
                    end_time = datetime.now()
                    start_time = end_time - timedelta(minutes=1)
                    
                    history_orders = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: mt5.history_orders_get(start_time, end_time)
                    )
                    
                    if history_orders:
                        for o in history_orders:
                            if o.symbol == symbol and o.magic == MAGIC_NUMBER:
                                ticket = o.ticket
                                log.info("Ticket found in history", ticket=ticket, symbol=symbol)
                                break
                    
                    if ticket == 0:
                        # Final check of open positions
                        positions = await self._client.get_open_positions(symbol)
                        for p in positions:
                            if p.magic == MAGIC_NUMBER:
                                ticket = p.ticket
                                log.info("Ticket found in positions after delay", ticket=ticket, symbol=symbol)
                                break
                    
                    if ticket == 0:
                        log.error("Ticket still 0 after history and position check. Aborting notification to prevent ghost trade.", symbol=symbol)
                        return {
                            "success": False,
                            "ticket": 0,
                            "detail": "Failed to verify ticket after success retcode",
                            "retcode": rc,
                        }


                # FIX THE 0.0 PRICE BUG: Extract actual fill price from result
                execution_price = result.get("price", 0.0)
                
                # If price is still not in result, try to get it from position if possible
                if execution_price <= 0:
                    pos = await self._client.get_open_positions(symbol)
                    for p in pos:
                        if p.ticket == ticket:
                            execution_price = p.price_open
                            break
                
                if execution_price <= 0:
                    execution_price = entry_params.get("suggested_price", 0.0)
                
                # Send notification with full details
                await TelegramNotifier.notify_trade_open(
                    symbol=symbol,
                    signal=signal,
                    lot=lot,
                    price=float(execution_price),
                    sl=float(entry_params.get("stop_loss", 0.0)),
                    tp=float(entry_params.get("take_profit", 0.0)),
                    timeframe=comment.replace("AIBot_", "") if "AIBot_" in comment else "M15",
                    reasoning=entry_params.get("reasoning", "AI Breakdown"),
                    votes=entry_params.get("votes"),
                    ticket=ticket
                )
                
                log.info("Order executed successfully", symbol=symbol, ticket=ticket, price=execution_price)
                return {
                    "success": True,
                    "ticket": ticket,
                    "detail": desc,
                    "retcode": rc,
                }

            # If NOT Done, log error and potentially notify
            last_error_msg = result.get("comment", desc)
            log.error(
                "Order attempt failed",
                attempt=attempt, symbol=symbol, rc=rc, error=last_error_msg
            )
            
            if not should_retry or attempt == MAX_ORDER_RETRIES:
                # ERROR BROADCASTING to Telegram
                await TelegramNotifier.notify_error(
                    f"🚨 TRADE FAILED: {signal} {symbol}\nReason: {last_error_msg} (RC: {rc})"
                )
                return {
                    "success": False,
                    "ticket": 0,
                    "detail": last_error_msg,
                    "retcode": rc,
                }

            # Refresh price for retriable errors
            await asyncio.sleep(0.5)
            bid, ask = await self._client.get_current_price(symbol)
            entry_params = dict(entry_params)
            entry_params["suggested_price"] = ask if signal == "BUY" else bid

        return {"success": False, "ticket": 0, "detail": "All retries exhausted"}

    async def handle_external_closure(self, ticket: int, symbol: str, volume: float) -> None:
        """Processes a closure detected by the sync loop (SL, TP, etc)."""
        import MetaTrader5 as mt5
        from datetime import datetime, timedelta

        # 1. Fetch trade results from history
        end = datetime.now()
        start = end - timedelta(minutes=5)
        deals = await asyncio.get_event_loop().run_in_executor(
            None, lambda: mt5.history_deals_get(start, end)
        )
        
        profit = 0.0
        entry_price = 0.0
        close_price = 0.0
        trade_type = "UNKNOWN"
        
        if deals:
            # Sort deals to find the one matching this ticket
            for d in deals:
                if d.position_id == ticket:
                    # Deal type 1 is OUT (closure)
                    if d.entry == 1: 
                        profit = d.profit + d.commission + d.swap
                        close_price = d.price
                        trade_type = "BUY" if d.type == mt5.DEAL_TYPE_SELL else "SELL"
                    # Deal type 0 is IN (entry)
                    elif d.entry == 0:
                        entry_price = d.price
        
        # 2. If we couldn't find details, use placeholders
        if entry_price == 0:
            log.warning(f"Could not find history for ticket {ticket}")
            return # Don't report incomplete data

        trade_data = {
            "symbol": symbol,
            "profit": profit,
            "entry_price": entry_price,
            "close_price": close_price,
            "type": trade_type,
            "lot": volume
        }

        # 3. AI Analysis & Notification
        reasoning = "Auto-closure detected (TP/SL/Manual)"
        if self._brain:
            reasoning = await self._brain.analyze_trade_outcome(trade_data)
        
        # Corrected notification call to match method signature
        await self._notifier.notify_trade_close(
            symbol=symbol,
            ticket=ticket,
            profit=trade_data['profit'],
            detail="External closure (SL/TP/Manual)",
            lot=trade_data['lot'],
            entry_price=trade_data['entry_price'],
            close_price=trade_data['close_price'],
            ai_explanation=reasoning
        )
        log.info(f"AI Post-Mortem sent for external closure {ticket}", profit=profit)
        
        # Signal to multi_main that a trade just closed for cooldown purposes
        # Note: In a multi-component system, we often use event emitters or shared state
        # For now, we'll return the closure time if needed, OR multi_main handles it.
        return time.time()

    async def close_position(self, ticket: int, symbol: str, lots: float) -> dict:
        """Close an open position by ticket."""
        if DRY_RUN:
            log.info("DRY RUN — close position skipped", ticket=ticket)
            return {"success": True, "detail": "DRY RUN"}

        result = await self._send_close_order(ticket, symbol, lots)
        rc = result.get("retcode", -1)
        success = rc in (mt5.TRADE_RETCODE_DONE, 10009, 0)
        
        if success:
            # Calculate profit/loss from deal info if available
            profit = result.get("profit", 0.0) 

            # --- AI Post-Mortem Analysis ---
            ai_explanation = "تحليل فني سريع: تم إغلاق الصفقة وفقاً لمعطيات السوق."
            entry_price = 0.0
            close_price = result.get("price", 0.0)
            
            if self._brain:
                # Fetch more details for analysis
                pos_history = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: mt5.history_deals_get(position=ticket)
                )
                trade_type = "UNKNOWN"
                
                if pos_history:
                    for d in pos_history:
                        if d.entry == mt5.DEAL_ENTRY_IN:
                            entry_price = d.price
                            trade_type = "BUY" if d.type == mt5.DEAL_TYPE_BUY else "SELL"
                            break
                
                trade_data = {
                    "symbol": symbol,
                    "profit": profit,
                    "entry_price": entry_price,
                    "close_price": close_price,
                    "type": trade_type
                }
                ai_explanation = await self._brain.analyze_trade_outcome(trade_data)

            await TelegramNotifier.notify_trade_close(
                symbol=symbol, ticket=ticket, profit=profit, 
                detail=result.get("comment", "إغلاق يدوي أو آلي"),
                lot=lots, entry_price=entry_price, close_price=close_price,
                ai_explanation=ai_explanation
            )

        log.info("Position close result", ticket=ticket, retcode=rc, success=success, profit=result.get("profit", 0.0))
        return {"success": success, "retcode": rc}

    async def manage_trailing_stop(
        self,
        symbol: str,
        atr: float,
        atr_multiplier: float = 1.5,
    ) -> None:
        """
        Adjust SL on profitable positions to lock in gains.
        Trail distance = atr * atr_multiplier.
        """
        if DRY_RUN:
            return

        positions = await self._client.get_open_positions(symbol)
        from config import MAGIC_NUMBER
        for pos in positions:
            if pos.magic != MAGIC_NUMBER: continue
            
            bid, ask = await self._client.get_current_price(symbol)
            si = await self._client.symbol_info(symbol)
            digits = si.digits if si else 5
            
            trail_dist = atr * atr_multiplier
            is_buy = pos.type == mt5.ORDER_TYPE_BUY

            new_sl: Optional[float]
            if is_buy:
                new_sl = round(bid - trail_dist, digits)
                if new_sl <= pos.sl:
                    continue  # Only tighten
            else:
                new_sl = round(ask + trail_dist, digits)
                if pos.sl > 0 and new_sl >= pos.sl:
                    continue

            await self._modify_stop_loss(pos.ticket, new_sl, pos.tp, symbol)

    async def manage_hard_stops(self) -> None:
        """
        FALLBACK: If current price passes the defined SL/TP on an open position,
        explicitly close it locally. This protects against broker-level SL bypass.
        """
        from config import MAGIC_NUMBER
        positions = await self._client.get_open_positions()
        for p in positions:
            if p.magic != MAGIC_NUMBER: continue
            
            bid, ask = await self._client.get_current_price(p.symbol)
            price = bid if p.type == mt5.ORDER_TYPE_BUY else ask
            
            is_buy = p.type == mt5.ORDER_TYPE_BUY
            
            # SL Bypass Check
            sl_triggered = False
            if p.sl > 0:
                if is_buy and price <= p.sl: sl_triggered = True
                elif not is_buy and price >= p.sl: sl_triggered = True
                
            # TP Bypass Check
            tp_triggered = False
            if p.tp > 0:
                if is_buy and price >= p.tp: tp_triggered = True
                elif not is_buy and price <= p.tp: tp_triggered = True
                
            if sl_triggered or tp_triggered:
                reason = "Failsafe: SL Triggered Locally" if sl_triggered else "Failsafe: TP Triggered Locally"
                log.warning(f"LOCAL CLOSE FALLBACK for {p.symbol}", ticket=p.ticket, pnl=p.profit, reason=reason)
                await self.close_position(p.ticket, p.symbol, float(p.volume), reason)

    async def manage_tactical_profit_locks(self, symbol: str) -> None:
        """
        Secures gains by moving SL to breakeven or applying aggressive trailing stop
        based on dollar profit thresholds in config.
        """
        from config import MAGIC_NUMBER, PROFIT_LOCK_BE_USD, PROFIT_LOCK_TRAIL_USD
        positions = await self._client.get_open_positions(symbol)
        
        for p in positions:
            if p.magic != MAGIC_NUMBER: continue
            
            # 1. Breakeven Lock at $5 profit
            if p.profit >= PROFIT_LOCK_BE_USD:
                is_buy = p.type == mt5.ORDER_TYPE_BUY
                entry_price = p.price_open
                current_sl = p.sl
                
                # If SL is still at original (worse than entry), move to BE+1 point
                si = await self._client.symbol_info(p.symbol)
                point = getattr(si, "point", 0.00001)
                digits = getattr(si, "digits", 5)
                
                be_price = round(entry_price + (point if is_buy else -point), digits)
                
                should_move_to_be = False
                if is_buy and (current_sl < entry_price): should_move_to_be = True
                elif not is_buy and (current_sl == 0 or current_sl > entry_price): should_move_to_be = True
                
                if should_move_to_be:
                    log.info(f"Tactical BE Lock triggered for {p.symbol}", ticket=p.ticket, profit=p.profit)
                    await self._modify_stop_loss(p.ticket, be_price, p.tp, p.symbol)
                    
            # 2. Aggressive Trail at $15 profit
            if p.profit >= PROFIT_LOCK_TRAIL_USD:
                # We use a tighter multiplier for ATR when in deep profit
                # This is handled by the regular manage_trailing_stop but we can force it here
                # or just let the faster interval in multi_main handle it.
                log.debug(f"Deep profit reached for {p.symbol} - Tight trailing active", ticket=p.ticket, profit=p.profit)


    async def close_opposite_if_signal_reverses(self, symbol: str, ai_signal: str) -> None:
        """
        Check if we have an open position for 'symbol' that is opposite to the latest AI signal.
        If AI signal is BUY and we are SHORT, or vice versa, close the position.
        """
        if ai_signal == "HOLD":
            return

        # Get current positions for this symbol
        positions = await asyncio.get_event_loop().run_in_executor(
            None, lambda: mt5.positions_get(symbol=symbol)
        )
        if not positions:
            return

        from config import MAGIC_NUMBER
        for p in positions:
            if p.magic != MAGIC_NUMBER: continue
            
            is_buy = p.type == mt5.ORDER_TYPE_BUY
            # If AI says SELL but we are BUY, or AI says BUY but we are SELL
            if (ai_signal == "SELL" and is_buy) or (ai_signal == "BUY" and not is_buy):
                log.info(
                    "Signal reversal detected",
                    symbol=symbol,
                    ticket=p.ticket,
                    signal=ai_signal,
                    position_type="BUY" if is_buy else "SELL"
                )
                # MT5 close requires volume
                await self.close_position(p.ticket, symbol, float(p.volume))

    # ── Private Sync Methods (run in executor) ────────────────────

    async def _send_market_order(
        self,
        symbol: str,
        signal: str,
        entry_params: dict,
        lot: float,
        comment: str,
    ) -> dict:
        bid, ask = await self._client.get_current_price(symbol)
        si = await self._client.symbol_info(symbol)
        if not si:
            return {"retcode": -1, "comment": "Symbol info unavailable"}
            
        digits = si.digits
        order_type = mt5.ORDER_TYPE_BUY if signal == "BUY" else mt5.ORDER_TYPE_SELL
        price = ask if signal == "BUY" else bid
        deviation = 20  # max slippage in points

        sl = float(entry_params.get("stop_loss", 0.0))
        tp = float(entry_params.get("take_profit", 0.0))
        entry_price = float(price)

        # STOPS LEVEL GUARD: Ensure sl/tp are far enough from price
        # Rules: 
        # BUY: SL < Bid, TP > Bid. Min dist from Bid.
        # SELL: SL > Ask, TP < Ask. Min dist from Ask.
        stops_level_pts = getattr(si, "trade_stops_level", 0)
        point = getattr(si, "point", 0.00001)
        spread = ask - bid
        # Safety buffer: stops_level + spread + 30 points
        min_dist_fixed = (stops_level_pts + 30) * point
        min_dist = max(min_dist_fixed, spread + (10 * point))

        if signal == "BUY":
            # Check against Bid for SL, Ask for TP
            if sl > 0 and (bid - sl) < min_dist:
                sl = bid - min_dist
                log.info(f"Adjusting BUY SL for {symbol}", original=round(float(entry_params.get('stop_loss')), digits), new=round(sl, digits))
            if tp > 0 and (tp - ask) < min_dist:
                tp = ask + min_dist
                log.info(f"Adjusting BUY TP for {symbol}", original=round(float(entry_params.get('take_profit')), digits), new=round(tp, digits))
        else: # SELL
            # Check against Ask for SL, Bid for TP
            if sl > 0 and (sl - ask) < min_dist:
                sl = ask + min_dist
                log.info(f"Adjusting SELL SL for {symbol}", original=round(float(entry_params.get('stop_loss')), digits), new=round(sl, digits))
            if tp > 0 and (bid - tp) < min_dist:
                tp = bid - min_dist
                log.info(f"Adjusting SELL TP for {symbol}", original=round(float(entry_params.get('take_profit')), digits), new=round(tp, digits))


        sl = round(sl, digits)
        tp = round(tp, digits)
        price = round(entry_price, digits)

        # Detect filling mode dynamically from symbol info
        # Robust check for missing constants in some MT5 versions
        SYM_FOK = getattr(mt5, "SYMBOL_FILLING_FOK", 1)
        SYM_IOC = getattr(mt5, "SYMBOL_FILLING_IOC", 2)
        
        filling_mode = mt5.ORDER_FILLING_FOK # Default
        if si.type_filling & SYM_FOK:
            filling_mode = mt5.ORDER_FILLING_FOK
        elif si.type_filling & SYM_IOC:
            filling_mode = mt5.ORDER_FILLING_IOC
        else:
            filling_mode = mt5.ORDER_FILLING_RETURN

        from config import MAGIC_NUMBER
        request = {
            "action":       int(mt5.TRADE_ACTION_DEAL),
            "symbol":       str(symbol),
            "volume":       float(lot),
            "type":         int(order_type),
            "price":        float(price),
            "sl":           float(sl),
            "tp":           float(tp),
            "deviation":    int(deviation),
            "magic":        int(MAGIC_NUMBER),
            "comment":      str(comment),
            "type_time":    int(mt5.ORDER_TIME_GTC),
            "type_filling": int(filling_mode),
        }

        # PRE-FLIGHT CHECK
        check_result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: mt5.order_check(request)
        )
        if check_result is None:
            return {"retcode": -1, "comment": "Order check failed (None)"}
            
        if check_result.retcode not in (mt5.TRADE_RETCODE_DONE, 0, 10009):
            log.error(f"Order check failed for {symbol}: {check_result.comment} (RC: {check_result.retcode})")
            return check_result._asdict()

        # SEND ACTUAL ORDER
        result = await asyncio.get_event_loop().run_in_executor(
            None, lambda: mt5.order_send(request)
        )
        
        if result is None:
            return {"retcode": -1, "comment": str(mt5.last_error())}
            
        return result._asdict()

    async def _send_close_order(self, ticket: int, symbol: str, lots: float) -> dict:
        from config import MAGIC_NUMBER
        pos = await asyncio.get_event_loop().run_in_executor(None, mt5.positions_get, ticket)
        if not pos:
            return {"retcode": -1, "comment": "Position not found"}
        p = pos[0]
        si = await self._client.symbol_info(symbol)
        is_buy = p.type == mt5.ORDER_TYPE_BUY
        close_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY
        bid, ask = await self._client.get_current_price(symbol)
        price = bid if is_buy else ask

        filling_mode = mt5.ORDER_FILLING_FOK
        if si.type_filling & mt5.SYMBOL_FILLING_FOK:
            filling_mode = mt5.ORDER_FILLING_FOK
        elif si.type_filling & mt5.SYMBOL_FILLING_IOC:
            filling_mode = mt5.ORDER_FILLING_IOC
        else:
            filling_mode = mt5.ORDER_FILLING_RETURN

        request = {
            "action":     int(mt5.TRADE_ACTION_DEAL),
            "symbol":     str(symbol),
            "volume":     float(lots),
            "type":       int(close_type),
            "position":   int(ticket),
            "price":      float(price),
            "deviation":  int(20),
            "magic":      int(MAGIC_NUMBER),
            "comment":    str("AIBot close"),
            "type_time":  int(mt5.ORDER_TIME_GTC),
            "type_filling": int(filling_mode),
        }
        result = await asyncio.get_event_loop().run_in_executor(
            None, mt5.order_send, request
        )
        if result is None:
            return {"retcode": -1, "comment": str(mt5.last_error())}
            
        res_dict = result._asdict()
        
        # Try to get real profit from history deal
        if res_dict.get("retcode") == mt5.TRADE_RETCODE_DONE:
            await asyncio.sleep(0.5) # Wait for history to update
            history = await asyncio.get_event_loop().run_in_executor(
                None, lambda: mt5.history_deals_get(position=ticket)
            )
            if history:
                # Find the deal that closed this position
                for deal in history:
                    if deal.entry == mt5.DEAL_ENTRY_OUT:
                        res_dict["profit"] = deal.profit + deal.commission + deal.swap
                        break
        
        return res_dict

    async def _modify_stop_loss(self, ticket: int, new_sl: float, tp: float, symbol: str = "") -> None:
        request = {
            "action":   mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl":       new_sl,
            "tp":       tp,
        }
        result = await asyncio.get_event_loop().run_in_executor(
            None, mt5.order_send, request
        )
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info("Trailing stop updated", ticket=ticket, new_sl=new_sl)
        else:
            rc = result.retcode if result else -1
            msg = f"Trailing stop update failed for {symbol} (#{ticket}). RC: {rc}"
            log.warning(msg)
            if rc != 10004: # Ignore "Requote" to avoid spam
                await TelegramNotifier.notify_error(msg)

