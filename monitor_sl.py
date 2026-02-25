import MetaTrader5 as mt5
import time
import os
from dotenv import load_dotenv

load_dotenv()

def monitor():
    mt5.initialize(
        path=os.getenv("MT5_PATH"),
        login=int(os.getenv("MT5_LOGIN")),
        password=os.getenv("MT5_PASSWORD"),
        server=os.getenv("MT5_SERVER")
    )
    
    print("Monitoring SL/TP for 10s...")
    for _ in range(5):
        positions = mt5.positions_get()
        if positions:
            for p in positions:
                tick = mt5.symbol_info_tick(p.symbol)
                bid = tick.bid if tick else 0
                ask = tick.ask if tick else 0
                print(f"[{p.symbol}] Price: {bid}/{ask} | SL: {p.sl} | TP: {p.tp} | Profit: {p.profit}")
        else:
            print("No positions")
        time.sleep(2)
    
    mt5.shutdown()

if __name__ == "__main__":
    monitor()
