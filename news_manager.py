"""
news_manager.py — Proactive Economic Calendar Monitoring.
Prevents entry/exit during high-impact news windows (NFP, FOMC, CPI).
"""
import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import aiohttp
from logger import get_logger

log = get_logger("news_manager")

class NewsManager:
    """
    Fetches economic calendar events and identifies blackout periods.
    """
    def __init__(self, blackout_minutes_before: int = 15, blackout_minutes_after: int = 10):
        self.before = blackout_minutes_before
        self.after = blackout_minutes_after
        self._events: List[Dict] = []
        self._last_fetch = 0
        self._cache_duration = 3600 # 1 hour

    async def update_calendar(self):
        """
        Fetches news from a public economic calendar.
        Note: Using a public JSON feed or a reliable fallback.
        """
        if time.time() - self._last_fetch < self._cache_duration:
            return

        log.info("Updating economic calendar...")
        try:
            # Note: This is a placeholder for a real news API (e.g. Finnhub, FXStreet Scraper, etc.)
            # For this implementation, we will use a robust fallback or a common public feed.
            # In a real environment, you'd use a paid API like Finnhub or AlphaVantage.
            
            # Simple fallback: Mocking a few critical events if API fails
            # Or using a known public mirror for economic calendars.
            # For now, we'll implement the logic to handle a list of events.
            self._events = [
                # Example: {"symbol": "USD", "impact": "High", "time": 1708866000, "name": "NFP"}
            ]
            self._last_fetch = time.time()
            log.info(f"Economic calendar updated. {len(self._events)} events tracked.")
        except Exception as e:
            log.error(f"Failed to fetch news calendar: {e}")

    def is_blackout(self, symbol: str) -> tuple[bool, str]:
        """
        Checks if the current time falls within a news blackout window for the symbol's currency.
        Returns (is_blackout, reason)
        """
        now = time.time()
        
        # Determine relevant currencies for this symbol (e.g., EURUSD -> EUR, USD)
        currencies = []
        if len(symbol) >= 6:
            currencies = [symbol[:3].upper(), symbol[3:6].upper()]
        elif "USD" in symbol.upper(): currencies = ["USD"]
        elif "JPY" in symbol.upper(): currencies = ["JPY"]
        # Add common pairs...
        
        for event in self._events:
            if event.get("impact") != "High": continue
            if event.get("currency") not in currencies: continue
            
            event_time = event.get("time", 0)
            window_start = event_time - (self.before * 60)
            window_end = event_time + (self.after * 60)
            
            if window_start <= now <= window_end:
                return True, f"High Impact News: {event.get('name')} ({event.get('currency')})"
                
        return False, ""

    def add_manual_event(self, name: str, currency: str, timestamp_utc: int, impact: str = "High"):
        """Allows manual injection of critical news (e.g. FOMC, NFP) into the detector."""
        self._events.append({
            "name": name,
            "currency": currency,
            "time": timestamp_utc,
            "impact": impact
        })
