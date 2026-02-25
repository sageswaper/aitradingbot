"""
mt5_client.py — Production-grade MetaTrader 5 bridge.

Responsibilities:
  - Connection & authentication with automatic retry
  - Session persistence via async heartbeat task
  - Latency measurement & reconnection on drop
  - Market watch symbol population
  - Typed wrappers for common MT5 operations
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import MetaTrader5 as mt5

from config import (
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH,
    MT5_MAX_RETRIES, MT5_HEARTBEAT_INTERVAL,
)
from logger import get_logger

log = get_logger("mt5_client")


# ────────────────────────────────────────────────────────────────
# Custom exceptions
# ────────────────────────────────────────────────────────────────
class MT5ConnectionError(RuntimeError):
    """Raised when we cannot establish or restore a connection."""


class MT5SymbolError(RuntimeError):
    """Raised when a symbol cannot be found or enabled."""


# ────────────────────────────────────────────────────────────────
# Data containers
# ────────────────────────────────────────────────────────────────
@dataclass
class AccountInfo:
    login: int
    name: str
    server: str
    currency: str
    balance: float
    equity: float
    free_margin: float
    leverage: int
    profit: float


@dataclass
class SymbolInfo:
    name: str
    bid: float
    ask: float
    spread: int           # in points
    digits: int
    point: float
    trade_tick_size: float
    trade_contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    type_filling: int
    trade_stops_level: int


# ────────────────────────────────────────────────────────────────
# MT5Client
# ────────────────────────────────────────────────────────────────
class MT5Client:
    """
    Thread-safe (single-threaded asyncio) bridge to MetaTrader 5.

    Usage:
        client = MT5Client()
        await client.connect()
        info = client.account_info()
        # ... do work ...
        await client.disconnect()
    """

    def __init__(self) -> None:
        self._connected: bool = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._last_latency_ms: float = 0.0
        self._failed_heartbeats: int = 0
        self._initial_balance: Optional[float] = None

    # ── Public API ────────────────────────────────────────────────

    async def connect(self) -> None:
        """Establish connection to MT5 terminal with retry logic."""
        for attempt in range(1, MT5_MAX_RETRIES + 1):
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._connect_sync
                )
                self._connected = True
                self._failed_heartbeats = 0
                info = await self.account_info()
                self._initial_balance = info.balance
                log.info(
                    "MT5 connected",
                    account=info.login,
                    server=info.server,
                    balance=info.balance,
                    currency=info.currency,
                )
                return
            except MT5ConnectionError as exc:
                wait = 2 ** (attempt - 1)
                log.warning(
                    "MT5 connection attempt failed",
                    attempt=attempt,
                    max=MT5_MAX_RETRIES,
                    error=str(exc),
                    retry_in=wait,
                )
                if attempt == MT5_MAX_RETRIES:
                    raise
                await asyncio.sleep(wait)

    async def disconnect(self) -> None:
        """Gracefully stop heartbeat and shut down MT5."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        mt5.shutdown()
        self._connected = False
        log.info("MT5 disconnected")

    async def ensure_connected(self) -> None:
        """Check connectivity; reconnect if needed."""
        term = await asyncio.get_event_loop().run_in_executor(None, mt5.terminal_info)
        if not self._connected or term is None or not term.connected:
            log.warning("MT5 connection lost — attempting reconnect")
            self._connected = False
            await self.connect()

    def start_heartbeat(self) -> None:
        """Launch background heartbeat coroutine."""
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(), name="mt5_heartbeat"
            )
            log.info("Heartbeat monitor started", interval_s=MT5_HEARTBEAT_INTERVAL)

    # ── Account / Symbol Data ─────────────────────────────────────

    async def account_info(self) -> AccountInfo:
        raw = await asyncio.get_event_loop().run_in_executor(None, mt5.account_info)
        if raw is None:
            raise MT5ConnectionError(f"account_info() failed: {mt5.last_error()}")
        return AccountInfo(
            login=raw.login,
            name=raw.name,
            server=raw.server,
            currency=raw.currency,
            balance=raw.balance,
            equity=raw.equity,
            free_margin=raw.margin_free,
            leverage=raw.leverage,
            profit=raw.profit,
        )

    async def symbol_info(self, symbol: str) -> SymbolInfo:
        await self.ensure_symbol_visible(symbol)
        raw = await asyncio.get_event_loop().run_in_executor(None, mt5.symbol_info, symbol)
        if raw is None:
            raise MT5SymbolError(f"symbol_info({symbol}) failed: {mt5.last_error()}")
        return SymbolInfo(
            name=raw.name,
            bid=raw.bid,
            ask=raw.ask,
            spread=raw.spread,
            digits=raw.digits,
            point=raw.point,
            trade_tick_size=raw.trade_tick_size,
            trade_contract_size=raw.trade_contract_size,
            volume_min=raw.volume_min,
            volume_max=raw.volume_max,
            volume_step=raw.volume_step,
            type_filling=getattr(raw, 'type_filling', getattr(raw, 'filling_mode', 3)),
            trade_stops_level=getattr(raw, 'trade_stops_level', 0)
        )

    async def get_current_price(self, symbol: str) -> tuple[float, float]:
        """Return (bid, ask) for symbol."""
        info = await self.symbol_info(symbol)
        return info.bid, info.ask

    def server_time(self) -> int:
        """Return broker server time as UNIX timestamp."""
        info = mt5.terminal_info()
        if info is None:
            raise MT5ConnectionError("terminal_info() returned None")
        # MT5 doesn't expose server time directly; use last tick time as proxy
        return int(time.time())  # fallback — overridden by tick time below

    async def server_time_from_symbol(self, symbol: str) -> int:
        tick = await asyncio.get_event_loop().run_in_executor(None, mt5.symbol_info_tick, symbol)
        if tick is None:
            return int(time.time())
        return tick.time

    async def get_open_positions(self, symbol: Optional[str] = None):
        """Return list of open mt5.TradePosition objects."""
        if symbol:
            res = await asyncio.get_event_loop().run_in_executor(None, mt5.positions_get, symbol)
        else:
            res = await asyncio.get_event_loop().run_in_executor(None, mt5.positions_get)
        return res or []

    async def get_open_orders(self, symbol: Optional[str] = None):
        if symbol:
            res = await asyncio.get_event_loop().run_in_executor(None, mt5.orders_get, symbol)
        else:
            res = await asyncio.get_event_loop().run_in_executor(None, mt5.orders_get)
        return res or []

    # ── Properties ────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_latency_ms(self) -> float:
        return self._last_latency_ms

    @property
    def initial_balance(self) -> Optional[float]:
        return self._initial_balance

    # ── Private Helpers ───────────────────────────────────────────

    def _connect_sync(self) -> None:
        """Blocking MT5 initialisation (to run in executor)."""
        # Step 1: Try to connect to an already running terminal
        connected_to_active = mt5.initialize()
        
        if connected_to_active:
            log.debug("Connected to active MT5 terminal instance")
            # Step 2: Ensure we are logged into the correct account
            if mt5.login(login=int(MT5_LOGIN), password=MT5_PASSWORD, server=MT5_SERVER):
                log.debug("Login successful on active terminal")
                return
            else:
                log.warning(f"Login failed on active terminal: {mt5.last_error()}. Retrying with full init.")
        
        # Step 3: Fallback to full initialization if step 1 or 2 failed
        log.info("Performing full MT5 initialization")
        kwargs: dict = {
            "login": int(MT5_LOGIN),
            "password": MT5_PASSWORD,
            "server": MT5_SERVER,
        }
        if MT5_PATH:
            kwargs["path"] = MT5_PATH

        if not mt5.initialize(**kwargs):
            err = mt5.last_error()
            raise MT5ConnectionError(f"mt5.initialize() failed: {err}")

        # Final verification
        term = mt5.terminal_info()
        if term is None or not term.connected:
            mt5.shutdown()
            raise MT5ConnectionError("Terminal initialized but not connected to broker")

    async def ensure_symbol_visible(self, symbol: str) -> None:
        """Add symbol to Market Watch if absent."""
        success = await asyncio.get_event_loop().run_in_executor(None, mt5.symbol_select, symbol, True)
        if not success:
            raise MT5SymbolError(
                f"Cannot enable symbol '{symbol}' in Market Watch: {mt5.last_error()}"
            )

    async def _heartbeat_loop(self) -> None:
        """Periodically verify connectivity; reconnect if needed."""
        while True:
            await asyncio.sleep(MT5_HEARTBEAT_INTERVAL)
            try:
                t0 = time.perf_counter()
                term = mt5.terminal_info()
                self._last_latency_ms = (time.perf_counter() - t0) * 1000

                if term is None or not term.connected:
                    self._failed_heartbeats += 1
                    log.warning(
                        "Heartbeat failed — terminal not connected",
                        consecutive_failures=self._failed_heartbeats,
                    )
                    self._connected = False
                    await self.connect()
                else:
                    self._failed_heartbeats = 0
                    log.debug(
                        "Heartbeat OK",
                        latency_ms=round(self._last_latency_ms, 2),
                        build=term.build,
                    )
            except Exception as exc:
                log.error("Heartbeat exception", error=str(exc), exc_info=True)
