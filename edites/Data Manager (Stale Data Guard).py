# ... existing code ...
class DataManager:
    def __init__(self, client: MT5Client) -> None:
        self._client = client
        self._timeframe_map = {
            "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }

    def _is_market_dead_zone(self) -> bool:
        """Prop-Firm Safety: Block rollover time (23:55 to 00:05 UTC) due to insane spreads and no liquidity."""
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        if now.hour == 23 and now.minute >= 55: return True
        if now.hour == 0 and now.minute <= 5: return True
        return False

    async def get_ohlc(self, symbol: str, timeframe: str, count: int = 100) -> pd.DataFrame:
        if self._is_market_dead_zone():
            log.warning(f"Market Rollover Dead-Zone active. Blocking data fetch for {symbol}.")
            return pd.DataFrame() # Return empty DF to force strategy to HOLD

        tf_val = self._timeframe_map.get(timeframe)
# ... existing code ...
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        
        # 🚨 STALE DATA GUARD 🚨
        # Check if the last candle is too old (broker disconnect/lag detection)
        import time
        last_candle_time = df['time'].iloc[-1].timestamp()
        current_time = time.time()
        tf_seconds = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800, "H1": 3600}.get(timeframe, 3600)
        
        if (current_time - last_candle_time) > (tf_seconds * 2.5):
            log.error(f"STALE DATA DETECTED for {symbol} [{timeframe}]. Broker lagging! Gap: {current_time - last_candle_time:.1f}s")
            return pd.DataFrame() # Force HOLD

        return df

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
# ... existing code ...