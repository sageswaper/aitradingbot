import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()

def list_symbols():
    if not mt5.initialize(
        login=int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
        path=os.getenv("MT5_PATH")
    ):
        print(f"Init failed: {mt5.last_error()}")
        return

    symbols = mt5.symbols_get()
    print(f"Total symbols: {len(symbols)}")
    
    # Priority search for EURUSD
    matches = []
    for s in symbols:
        if "EURUSD" in s.name:
            matches.append(s.name)
    
    if matches:
        print(f"Found matches: {matches}")
    else:
        print("No EURUSD matches found. Listing first 10 symbols:")
        for s in symbols[:10]:
            print(f"- {s.name}")
            
    mt5.shutdown()

if __name__ == "__main__":
    list_symbols()
