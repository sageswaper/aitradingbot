import asyncio
import sys
import os
import MetaTrader5 as mt5

# Add current directory to path
sys.path.append(os.getcwd())

from mt5_client import MT5Client
from execution_engine import ExecutionEngine

async def main():
    client = MT5Client()
    await client.connect()
    engine = ExecutionEngine(client)
    
    print("Fetching active positions to close...")
    positions = mt5.positions_get(symbol="EURUSD-T")
    if positions:
        for pos in positions:
            if pos.comment == "PROOFOFLIFE_TEST":
                print(f"Closing Ticket: {pos.ticket}...")
                success = await engine.close_position(pos.ticket, "EURUSD-T", pos.volume, "CLEANUP")
                if success:
                    print(f"Ticket {pos.ticket} closed successfully.")
                else:
                    print(f"Failed to close ticket {pos.ticket}.")
    else:
        print("No PROOFOFLIFE_TEST positions found.")
        
    await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
