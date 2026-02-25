import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()

if not mt5.initialize(
    login=int(os.getenv("MT5_LOGIN")),
    password=os.getenv("MT5_PASSWORD"),
    server=os.getenv("MT5_SERVER"),
    path=os.getenv("MT5_PATH")
):
    print("Failed")
else:
    symbols = mt5.symbols_get()
    t_symbols = [s.name for s in symbols if s.name.endswith("-T")]
    print(f"Symbols found: {', '.join(t_symbols[:50])}")
    if len(t_symbols) > 50:
        print(f"... and {len(t_symbols)-50} more")
    mt5.shutdown()
