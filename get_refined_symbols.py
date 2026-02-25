import MetaTrader5 as mt5
import os
from dotenv import load_dotenv

load_dotenv()

def get_refined_symbols():
    if not mt5.initialize(
        login=int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER"),
        path=os.getenv("MT5_PATH")
    ):
        return

    symbols = mt5.symbols_get()
    if symbols:
        # Filter: Exclude stocks (usually contain # or are very short/long with dot)
        # Include anything that looks like Forex (6 chars + -T) or major commodities
        trading_symbols = []
        for s in symbols:
            name = s.name
            if name.startswith("#"): continue
            
            # Categories: Forex, Metals, Indices, Commodities
            if any(x in name for x in ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD", "GOLD", "SILVER", "BRENT", "NGAS", "OIL", "XAU", "XAG", "100", "30", "40"]):
                trading_symbols.append(name)
        
        print(", ".join(trading_symbols))

    mt5.shutdown()

if __name__ == "__main__":
    get_refined_symbols()
