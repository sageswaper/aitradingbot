"""
multi_main.py — Multi-timeframe orchestrator for the AI Trading Bot.

Runs concurrent loops for M1, M2, and M15 using a single shared MT5 connection.
This avoids "Call failed" errors caused by multiple processes competing for the MT5 terminal.
"""
from __future__ import annotations

import asyncio
import os
import signal as os_signal
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import List

from config import SYMBOLS, TIMEFRAMES, DRY_RUN
from logger import get_logger
from mt5_client import MT5Client, MT5ConnectionError
from data_manager import DataManager
from ai_brain import AIBrain
from risk_manager import RiskManager, HaltTradingError, RiskVetoError
from execution_engine import ExecutionEngine
from strategy import TrendMeanReversionStrategy
from database import AuditDB

log = get_logger("multi_main")

# Defaults (can still be overridden by env vars if needed)
# But config.py already handles SYMBOLS and TIMEFRAMES

_TIMEFRAME_SECONDS: dict[str, int] = {
    "M1": 60, "M2": 120, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}

def _seconds_to_next_candle(timeframe: str) -> float:
    tf_secs = _TIMEFRAME_SECONDS.get(timeframe, 900)
    now_ts = time.time()
    elapsed = now_ts % tf_secs
    wait = tf_secs - elapsed + 2
    return wait

class MultiTimeframeBot:
    def __init__(self, symbols: List[str], timeframes: List[str]) -> None:
        self.symbols    = symbols
        self.timeframes = timeframes
        self.client     = MT5Client()
        self.data       = DataManager(self.client)
        self.ai         = AIBrain()
        self.risk       = RiskManager(self.client)
        self.executor   = ExecutionEngine(self.client, brain=self.ai)
        self.strategy   = TrendMeanReversionStrategy()
        self.db         = AuditDB()
        self._running   = False
        self._halt      = False
        self.tasks: List[asyncio.Task] = []
        # RACE CONDITION GUARD: One analysis/execution per symbol at a time
        self._symbol_locks = {s: asyncio.Lock() for s in symbols}
        self._active_tickets: set[int] = set()


    async def start(self) -> None:
        log.info("Multi-bot starting", symbols=self.symbols, timeframes=self.timeframes, dry_run=DRY_RUN)
        
        # API Rate Limit Survival Strategy: Watchlist Trimming Directive
        total_cycles = len(self.symbols) * len(self.timeframes)
        if total_cycles > 60:
            print("\n" + "!" * 60)
            print(f"CRITICAL WARNING: Total Cycles ({total_cycles}) > 60.")
            print(f"With a 5s AI throttle, one sweep takes ~{round((total_cycles*5)/60, 1)} minutes.")
            print("To stay within the M5 window, please reduce SYMBOLS to < 15.")
            print("!" * 60 + "\n")
            log.warning("System will lag behind M5 candles due to high cycle count.")

        await self.db.initialize()
        await self.client.connect()
        
        self._last_closed_trades: dict[str, float] = {} # symbol -> timestamp

        # Ensure all symbols are in Market Watch
        for symbol in self.symbols:
            try:
                await self.client.ensure_symbol_visible(symbol)
            except Exception as e:
                log.error(f"Failed to enable symbol {symbol}: {e}")

        self.client.start_heartbeat()
        self._running = True
        # Initial ticket population
        init_pos = await self.client.get_open_positions()
        self._active_tickets = {p.ticket for p in init_pos}
        log.info("Initialized ticket tracker", count=len(self._active_tickets))

        # Start standard background loops
        self.tasks.append(asyncio.create_task(self._hard_stop_monitor(), name="hard_stop_monitor"))
        self.tasks.append(asyncio.create_task(self._trailing_stop_loop(), name="trailing_stop_loop"))
        self.tasks.append(asyncio.create_task(self._position_sync_loop(), name="position_sync_loop"))

        if DRY_RUN:
            log.warning("DRY RUN MODE ACTIVE")


        # Create a main loop for each (Symbol, Timeframe) pair
        # Add a small jitter to avoid thundering herd on MT5
        loop_tasks = []
        for symbol in self.symbols:
            for tf in self.timeframes:
                loop_tasks.append(self._main_loop(symbol, tf))
                await asyncio.sleep(0.1)
        
        try:
            await asyncio.gather(
                *loop_tasks,
                *self.tasks # Gather all background tasks
            )
        except asyncio.CancelledError:
            log.info("Tasks cancelled — shutting down")
        finally:
            await self._shutdown()
            log.info("Process loop ended.")

    async def _shutdown(self) -> None:
        self._running = False
        await self.client.disconnect()
        await self.ai.close()
        log.info("Shutdown complete")

    async def _hard_stop_monitor(self) -> None:
        """Background task to check for SL/TP bypass every 5 seconds."""
        log.info("Hard stop monitor started")
        while self._running:
            try:
                await self.executor.manage_hard_stops()
                await asyncio.sleep(5)
            except Exception as e:
                log.error("Hard stop monitor error", error=str(e))
                await asyncio.sleep(10)

    async def _main_loop(self, symbol: str, tf: str) -> None:
        log.info(f"Loop started for {symbol} @ {tf}")
        while self._running:
            if self._halt:
                await asyncio.sleep(60)
                continue

            wait_secs = _seconds_to_next_candle(tf)
            candle_time = datetime.now(tz=timezone.utc) + timedelta(seconds=wait_secs)
            log.info(
                f"[{symbol}][{tf}] Next candle close",
                wait_s=round(wait_secs, 1),
                scheduled_at=candle_time.strftime("%H:%M:%S UTC"),
            )
            await asyncio.sleep(wait_secs)

            if not self._running: break
            await self._run_analysis_cycle(symbol, tf)

    async def _run_analysis_cycle(self, symbol: str, tf: str) -> None:
        # Prevent concurrent analysis of the same symbol by different timeframes
        # This stops the M1 and M5 loops from both opening a trade at the same time
        async with self._symbol_locks.get(symbol, asyncio.Lock()):
            cycle_start = time.perf_counter()
            log.info(f"[{symbol}][{tf}] Analysis cycle starting")


        try:
            # [State Check Move]: Position check is now performed AFTER evaluation to allow reversal/exit signals.
            from config import TRADE_COOLDOWN_MINUTES

            # TIERED SCANNING: Fast tracks for M1/M5
            from config import PRIORITY_SYMBOLS
            is_fast_tf = tf in ["M1", "M5", "M15"]
            if is_fast_tf and symbol not in PRIORITY_SYMBOLS:
                log.debug(f"[Tiered Scan: Skipping non-priority {symbol} for low timeframe {tf}]")
                return
            
            # Further optimization: Non-priority symbols only on H1 and above
            if tf not in ["H1", "H4", "D1"] and symbol not in PRIORITY_SYMBOLS:
                log.debug(f"[Tiered Scan: Skipping {symbol} for {tf} - Only allowed on H1+]")
                return

            # ANTI-CHURNING: Post-Trade Cooldown
            # This logic is now handled by the position sync loop and external closure handler
            # last_close = self._last_closed_trades.get(symbol, 0)
            # cooldown_secs = TRADE_COOLDOWN_MINUTES * 60
            # if time.time() - last_close < cooldown_secs:
            #     remaining = round((cooldown_secs - (time.time() - last_close)) / 60, 1)
            #     log.info(f"[Cooldown: {symbol} in penalty for {remaining}m - Skipping]")
            #     return

            # 2. Indicators (Fetch early for filter)
            try:
                # Ensure connection is fresh before each cycle
                await self.client.ensure_connected()
                
                df = await self.data.get_ohlc(symbol, tf)
                df = self.data.add_indicators(df)
                bid, ask = await self.client.get_current_price(symbol)
                current_price = (bid + ask) / 2
                existing_pos = await self.client.get_open_positions(symbol)
            except Exception as e:
                log.error(f"[{symbol}][{tf}] Pre-cycle data fetch failed: {str(e)}")
                return # Skip cycle

            # 3. Pre-filter check
            should, reason = self.strategy.should_analyze(df, current_price)
            report = ""
            if not should:
                log.info(f"[{symbol}][{tf}] Local Filter: HOLD | {reason}")
                # Mock an AI "HOLD" response for auditing
                ai_response = {
                    "signal": "HOLD", "reasoning": f"Local technical filter: {reason}",
                    "confidence_score": 0.0, "entry_params": {}
                }
            else:
                # 3. Report & AI
                report = await self.data.synthesize_market_report(symbol, tf)
                
                # Specialized Strategy Context for AI
                strat = self.strategy._get_strategy(symbol)
                strat_meta = {
                    "name": strat.__class__.__name__,
                    "rules": strat.__doc__ or "Standard hybrid technical rules."
                }
                
                # RESCUE PLAN: Increased jitter (0-20s) to distribute AI requests
                import random
                jitter = random.uniform(0, 20)
                await asyncio.sleep(jitter)

                ai_start = time.perf_counter()
                ai_response = await self.ai.analyze(report, strategy_metadata=strat_meta)
                ai_latency_ms = round((time.perf_counter() - ai_start) * 1000, 1)

            # 4. Strategy Final Decision
            decision = self.strategy.evaluate(df, ai_response, current_price, symbol=symbol)
            log.info(f"[{symbol}][{tf}] Decision: {decision.signal} | Conf: {decision.confidence} | Strat: {strat_meta['name']}")

            # 5. Risk + Execution
            lot_size = 0.0
            was_traded = False
            order_result = None

            # -- Dynamic Exit Check --
            # If we have a position and AI signal is opposite, close it now
            await self.executor.close_opposite_if_signal_reverses(symbol, ai_response.get("signal", "HOLD"))
            
            # EXPLICIT EXIT SIGNAL
            if decision.signal == "EXIT":
                from config import MAGIC_NUMBER
                if existing_pos:
                    for p in existing_pos:
                        if p.magic == MAGIC_NUMBER:
                            log.info(f"[AI EXIT: Closing {symbol} position {p.ticket}]")
                            await self.executor.close_position(p.ticket, symbol, float(p.volume))
                            # self._last_closed_trades[symbol] = time.time() # Handled by sync loop
                
                # RECORD CYCLE before returning
                total_ms = round((time.perf_counter() - cycle_start) * 1000, 1)
                await self.db.record_cycle(
                    symbol, tf, report, ai_response, decision.signal, 
                    0.0, False, total_ms, DRY_RUN
                )
                return 

            if decision.signal != "HOLD":
                # Enforcement: Only open NEW trades if we don't already have one
                import MetaTrader5 as mt5
                from config import MAGIC_NUMBER
                current_positions = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: mt5.positions_get(symbol=symbol)
                )
                our_pos = [p for p in (current_positions or []) if p.magic == MAGIC_NUMBER]
                
                if our_pos:
                    log.debug(f"Position already exists for {symbol}. Skipping new {decision.signal} entry.")
                    # Still record cycle but don't place order
                    total_ms = round((time.perf_counter() - cycle_start) * 1000, 1)
                    await self.db.record_cycle(
                        symbol, tf, report, ai_response, "SKIP (Exist)", 
                        0.0, False, total_ms, DRY_RUN
                    )
                    return

                try:
                    await self.risk.check_all(symbol, decision.signal, decision.confidence)
                    si = await self.data.get_symbol_info_for_risk(symbol)
                    ai_conf = ai_response.get("confidence_score", 0.0)
                    atr = float(df["atr"].iloc[-1])
                    lot_size = await self.risk.calculate_lot_size(
                        symbol,
                        float(ai_response.get("entry_params", {}).get("stop_loss", 0.0)),
                        current_price,
                        ai_conf,
                        si, # Assuming si is the correct object, not si_dict
                        atr=atr
                    )
                    if lot_size > 0:
                        # FALLBACK PRICE: If AI suggested 0.0, use current mid price
                        final_entry_price = decision.entry_price
                        if final_entry_price <= 0:
                            final_entry_price = current_price

                        # Include votes and reasoning in entry_params for notifications
                        exec_params = {
                            "suggested_price": final_entry_price,
                            "stop_loss": decision.stop_loss,
                            "take_profit": decision.take_profit,
                            "reasoning": decision.reason,
                            "votes": ai_response.get("votes")
                        }

                        order_result = await self.executor.place_order(
                            symbol, decision.signal, 
                            exec_params,
                            lot_size, comment=f"AIBot_{tf}"
                        )
                        was_traded = order_result.get("success", False)

                except HaltTradingError:
                    self._halt = True
                    log.critical("Halt triggered")
                except RiskVetoError as e:
                    log.warning(f"[{symbol}][{tf}] Risk veto: {str(e)}")
                except Exception as e:
                    log.error(f"[{symbol}][{tf}] Execution error: {str(e)}")

            # 6. Audit
            total_ms = round((time.perf_counter() - cycle_start) * 1000, 1)
            log_id = await self.db.record_cycle(
                symbol, tf, report, ai_response, decision.signal, 
                lot_size, was_traded, total_ms, DRY_RUN
            )

            if was_traded and not DRY_RUN and order_result:
                ticket = order_result["ticket"]
                self._active_tickets.add(ticket)
                await self.db.record_trade_open(
                    ticket, symbol, decision.signal, lot_size,
                    decision.entry_price, decision.stop_loss, decision.take_profit, log_id
                )

            log.info(f"[{symbol}][{tf}] Cycle complete | ms={total_ms} | signal={decision.signal} | traded={was_traded}")

        except Exception as e:
            log.error(f"[{tf}] Cycle failed: {str(e)}", exc_info=True)

    async def _trailing_stop_loop(self) -> None:
        """Runs every 10s to manage profitable trailing stops."""
        while self._running:
            await asyncio.sleep(10)
            if self._halt: continue
            for symbol in self.symbols:
                try:
                    # Fetch ATR from M15 for trailing logic
                    df = await self.data.get_ohlc(symbol, "M15")
                    if df.empty: continue
                    df = self.data.add_indicators(df)
                    atr = float(df["atr"].iloc[-1])
                    await self.executor.manage_trailing_stop(symbol, atr)
                    await self.executor.manage_tactical_profit_locks(symbol)
                except Exception as e:
                    log.error(f"Trailing stop error for {symbol}", error=str(e))

    async def _position_sync_loop(self) -> None:
        """Detects trades closed by TP, SL, or manual intervention."""
        while self._running:
            await asyncio.sleep(15) # Check every 15s
            try:
                import MetaTrader5 as mt5
                current_pos = await self.client.get_open_positions()
                current_tickets = {p.ticket for p in current_pos}

                # Find tickets that disappeared
                closed_tickets = self._active_tickets - current_tickets
                for ticket in closed_tickets:
                    # Find which symbol this was (history can help)
                    from datetime import datetime, timedelta
                    end = datetime.now()
                    start = end - timedelta(minutes=60)
                    deals = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: mt5.history_deals_get(start, end)
                    )
                    
                    symbol = "UNKNOWN"
                    vol = 0.0
                    if deals:
                        for d in deals:
                            if d.position_id == ticket:
                                symbol = d.symbol
                                vol = d.volume
                                break
                    
                    log.info(f"Detected auto-closure of ticket {ticket}", symbol=symbol)
                    # Use a specialized close reporter that handles the AI post-mortem
                    close_time = await self.executor.handle_external_closure(ticket, symbol, vol)
                    if symbol != "UNKNOWN":
                        self._last_closed_trades[symbol] = close_time or time.time()
                    self._active_tickets.remove(ticket)

                # Re-sync in case of external opens (rare) or missed updates
                self._active_tickets = current_tickets
            except Exception as e:
                log.error(f"Sync loop error: {str(e)}")

def _install_shutdown_handler(bot: MultiTimeframeBot, loop: asyncio.AbstractEventLoop) -> None:
    def _handler():
        log.info("Shutdown signal received")
        bot._running = False
        for task in asyncio.all_tasks(loop): task.cancel()

    if sys.platform != "win32":
        loop.add_signal_handler(os_signal.SIGINT, _handler)
        loop.add_signal_handler(os_signal.SIGTERM, _handler)

async def main() -> None:
    bot = MultiTimeframeBot(SYMBOLS, TIMEFRAMES)
    loop = asyncio.get_running_loop()
    _install_shutdown_handler(bot, loop)
    try:
        await bot.start()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt")
    except MT5ConnectionError as e:
        log.critical(f"MT5 Connection Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
