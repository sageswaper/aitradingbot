import MetaTrader5 as mt5
import time

MAGIC_NUMBER = 20260224

def close_position(ticket, symbol, volume, type):
    close_type = mt5.ORDER_TYPE_SELL if type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = mt5.symbol_info_tick(symbol).bid if type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": close_type,
        "position": ticket,
        "price": price,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "Cleanup duplicates",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Failed to close {ticket}: {result.comment}")
    else:
        print(f"Closed duplicate {ticket} for {symbol}")

def main():
    if not mt5.initialize():
        print("MT5 Init failed")
        return

    positions = mt5.positions_get()
    if not positions:
        print("No positions found")
        return

    # Group by symbol
    symbol_groups = {}
    for p in positions:
        if p.magic == MAGIC_NUMBER:
            if p.symbol not in symbol_groups:
                symbol_groups[p.symbol] = []
            symbol_groups[p.symbol].append(p)

    for symbol, pos_list in symbol_groups.items():
        if len(pos_list) > 1:
            print(f"Found {len(pos_list)} positions for {symbol}. Keeping oldest.")
            # Sort by time (keep oldest)
            pos_list.sort(key=lambda x: x.time)
            to_close = pos_list[1:] # All except first
            for p in to_close:
                close_position(p.ticket, p.symbol, p.volume, p.type)

    mt5.shutdown()

if __name__ == "__main__":
    main()
