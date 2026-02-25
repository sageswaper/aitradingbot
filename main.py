"""
main.py — Async orchestrator for the AI Trading Bot.

Flow per M15 cycle:
  1. Wait until exact M15 candle close
  2. Fetch OHLC + synthesize Market Situation Report
  3. Send report to AIBrain for analysis (non-blocking)
  4. Run RiskManager checks
  5. Run Strategy evaluation
  6. Place order via ExecutionEngine (no-op in DRY_RUN)
  7. Record everything to AuditDB

Parallel tasks:
  - MT5 heartbeat monitor (10s interval)
  - Trailing stop updater (every 60s)
  - Main M15 analysis loop
"""
from __future__ import annotations

import asyncio
import signal as os_signal
import sys
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from config import SYMBOLS, TIMEFRAMES, DRY_RUN, MT5_TIMEFRAME_MAP

# Use the first one for the legacy single-asset main.py
SYMBOL = SYMBOLS[0]
TIMEFRAME = TIMEFRAMES[0]
from logger import get_logger
from mt5_client import MT5Client, MT5ConnectionError
from data_manager import DataManager
from ai_brain import AIBrain
from risk_manager import RiskManager, HaltTradingError, RiskVetoError
from execution_engine import ExecutionEngine
from strategy import TrendMeanReversionStrategy
from database import AuditDB

log = get_logger("main")

# ────────────────────────────────────────────────────────────────
# Timing helpers
# ────────────────────────────────────────────────────────────────
_TIMEFRAME_SECONDS: dict[str, int] = {
    "M1": 60, "M2": 120, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}


def _seconds_to_next_candle(timeframe: str) -> float:
    """Calculate seconds until the next candle close (+ 2s buffer)."""
    tf_secs = _TIMEFRAME_SECONDS.get(timeframe, 900)
    now_ts = time.time()
    elapsed = now_ts % tf_secs
    wait = tf_secs - elapsed + 2   # +2s buffer for broker delivery lag
    return wait


# ────────────────────────────────────────────────────────────────
# TradingBot
# ────────────────────────────────────────────────────────────────
class TradingBot:
    def __init__(self) -> None:
        self.client    = MT5Client()
        self.data      = DataManager(self.client)
        self.ai        = AIBrain()
        self.risk      = RiskManager(self.client)
        self.executor  = ExecutionEngine(self.client)
        self.strategy  = TrendMeanReversionStrategy()
        self.db        = AuditDB()
        self._running  = False
        self._halt     = False    # set True on HaltTradingError

    # ── Lifecycle ─────────────────────────────────────────────────

    async def start(self) -> None:
        log.info(
            "AI Trading Bot starting",
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            dry_run=DRY_RUN,
        )
        await self.db.initialize()
        await self.client.connect()
        self.client.start_heartbeat()
        self._running = True

        if DRY_RUN:
            log.warning("DRY RUN MODE ACTIVE — no real orders will be placed")

        try:
            await asyncio.gather(
                self._main_loop(),
                self._trailing_stop_loop(),
            )
        except asyncio.CancelledError:
            log.info("Bot tasks cancelled — shutting down cleanly")
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        self._running = False
        await self.client.disconnect()
        await self.ai.close()
        log.info("Bot shutdown complete")

    # ── Main Analysis Loop ────────────────────────────────────────

    async def _main_loop(self) -> None:
        log.info("Main loop started — waiting for first candle close")

        while self._running:
            if self._halt:
                log.critical("TRADING HALTED — drawdown limit breached. Bot idle.")
                await asyncio.sleep(60)
                continue

            # Wait until the next candle close
            wait_secs = _seconds_to_next_candle(TIMEFRAME)
            candle_time = datetime.now(tz=timezone.utc) + timedelta(seconds=wait_secs)
            log.info(
                f"Next {TIMEFRAME} candle close",
                wait_s=round(wait_secs, 1),
                scheduled_at=candle_time.strftime("%H:%M:%S UTC"),
            )
            await asyncio.sleep(wait_secs)

            await self._run_analysis_cycle()

    async def _run_analysis_cycle(self) -> None:
        """Execute one full analysis + decision cycle."""
        cycle_start = time.perf_counter()
        log.info("─" * 60)
        log.info(f"Analysis cycle starting for {SYMBOL} {TIMEFRAME}")

        # 1. Synthesize market report
        try:
            report = await self.data.synthesize_market_report(SYMBOL, TIMEFRAME)
        except Exception as exc:
            log.error("Failed to synthesize market report", error=str(exc))
            return

        # 2. AI analysis (concurrent — doesn't block heartbeat)
        ai_start = time.perf_counter()
        ai_response = await self.ai.analyze(report)
        ai_latency_ms = round((time.perf_counter() - ai_start) * 1000, 1)

        # 3. Get latest data for strategy
        try:
            df = await self.data.get_ohlc(SYMBOL, TIMEFRAME)
            df = self.data.add_indicators(df)
            bid, ask = self.client.get_current_price(SYMBOL)
            current_price = (bid + ask) / 2
        except Exception as exc:
            log.error("Failed to fetch indicator data", error=str(exc))
            return

        # 4. Strategy evaluation
        decision = self.strategy.evaluate(df, ai_response, current_price)
        log.info(
            "Strategy decision",
            signal=decision.signal,
            reason=decision.reason,
            confidence=decision.confidence,
        )

        # 5. Risk checks + lot sizing
        lot_size = 0.0
        was_traded = False
        order_result: Optional[dict] = None

        if decision.signal != "HOLD":
            try:
                self.risk.check_all(SYMBOL, decision.signal, decision.confidence)

                # Get symbol info for lot sizing
                si = await self.data.get_symbol_info_for_risk(SYMBOL)
                lot_size = self.risk.calculate_lot_size(
                    symbol=SYMBOL,
                    sl_price=decision.stop_loss,
                    entry_price=decision.entry_price,
                    confidence=decision.confidence,
                    symbol_info=si,
                )

                if lot_size > 0:
                    # 6. Execute order
                    order_result = await self.executor.place_order(
                        symbol=SYMBOL,
                        signal=decision.signal,
                        entry_params={
                            "suggested_price": decision.entry_price,
                            "stop_loss":       decision.stop_loss,
                            "take_profit":     decision.take_profit,
                        },
                        lot=lot_size,
                    )
                    was_traded = order_result.get("success", False)

                    if was_traded and not DRY_RUN:
                        ticket = order_result.get("ticket", 0)
                        log.info(
                            "Trade placed",
                            ticket=ticket,
                            signal=decision.signal,
                            lot=lot_size,
                        )

            except HaltTradingError:
                self._halt = True
                log.critical("Global halt triggered — bot entering idle mode")

            except RiskVetoError as exc:
                log.warning("Trade vetoed by risk manager", reason=str(exc))

            except Exception as exc:
                log.error("Unexpected error during execution", error=str(exc), exc_info=True)

        # 7. Audit log
        total_latency_ms = round((time.perf_counter() - cycle_start) * 1000, 1)
        log_id = await self.db.record_cycle(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            market_report=report,
            ai_response=ai_response,
            strategy_signal=decision.signal,
            lot_size=lot_size,
            was_traded=was_traded,
            latency_ms=total_latency_ms,
            dry_run=DRY_RUN,
        )

        # Record trade in DB if placed
        if was_traded and not DRY_RUN and order_result:
            ticket = order_result.get("ticket", 0)
            await self.db.record_trade_open(
                ticket=ticket,
                symbol=SYMBOL,
                signal=decision.signal,
                lot=lot_size,
                entry_price=decision.entry_price,
                stop_loss=decision.stop_loss,
                take_profit=decision.take_profit,
                analysis_log_id=log_id,
            )

        log.info(
            "Cycle complete",
            total_ms=total_latency_ms,
            ai_ms=ai_latency_ms,
            signal=decision.signal,
            traded=was_traded,
        )

    # ── Trailing Stop Loop ────────────────────────────────────────

    async def _trailing_stop_loop(self) -> None:
        """Update trailing stops on open positions every 60s."""
        while self._running:
            await asyncio.sleep(60)
            if self._halt:
                continue
            try:
                df = await self.data.get_ohlc(SYMBOL, TIMEFRAME, bars=20)
                df = self.data.add_indicators(df)
                atr = float(df["atr"].iloc[-1])
                await self.executor.manage_trailing_stop(SYMBOL, atr)
            except Exception as exc:
                log.debug("Trailing stop update skipped", error=str(exc))


# ────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────
def _install_shutdown_handler(bot: TradingBot, loop: asyncio.AbstractEventLoop) -> None:
    """Gracefully cancel all tasks on SIGINT/SIGTERM."""
    def _handler():
        log.info("Shutdown signal received — cancelling tasks")
        bot._running = False
        for task in asyncio.all_tasks(loop):
            task.cancel()

    if sys.platform != "win32":
        loop.add_signal_handler(os_signal.SIGINT, _handler)
        loop.add_signal_handler(os_signal.SIGTERM, _handler)
    else:
        # Windows: KeyboardInterrupt is caught by asyncio.run()
        pass


async def main() -> None:
    bot = TradingBot()
    loop = asyncio.get_running_loop()
    _install_shutdown_handler(bot, loop)
    try:
        await bot.start()
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt — shutting down")
    except MT5ConnectionError as exc:
        log.critical("Fatal MT5 connection error", error=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
