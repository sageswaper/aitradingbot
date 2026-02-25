import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()

def get_symbols():
    if not mt5.initialize(
        login=int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
        path=os.getenv("MT5_PATH")
    ):
        print("MT5 Init Failed")
        return

    symbols = mt5.symbols_get()
    if symbols is None:
        print("No symbols found")
    else:
        # Filter for typical trading pairs (Forex, Metals, Crude)
        # We look for symbols that have 'EUR', 'USD', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'NZD', 'GOLD', 'SILVER', 'BRENT', 'XAU', 'XAG'
        all_names = [s.name for s in symbols]
        print(", ".join(all_names))

    mt5.shutdown()

if __name__ == "__main__":
    get_symbols()
