import aiohttp
from typing import Optional
from logger import get_logger
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

log = get_logger("telegram_notifier")

class TelegramNotifier:
    """Sends asynchronous notifications to Telegram."""
    
    @staticmethod
    async def send_message(text: str) -> bool:
        """Send a simple text message to the configured chat."""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            log.warning("Telegram credentials not configured; skipping notification.")
            return False

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML"
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 200:
                        log.info("Telegram notification sent successfully.")
                        return True
                    else:
                        error_text = await resp.text()
                        log.error("Telegram API error", status=resp.status, detail=error_text)
                        return False
        except Exception as e:
            log.error("Failed to send Telegram notification", error=str(e))
            return False

    @staticmethod
    async def notify_trade_open(
        symbol: str, 
        signal: str, 
        lot: float, 
        price: float, 
        sl: float, 
        tp: float, 
        timeframe: str,
        reasoning: str, 
        votes: Optional[dict] = None,
        ticket: int = 0
    ):
        """Send alert for a newly opened trade with professional template."""
        from config import RISK_HIGH_CONFIDENCE # or whatever risk pct is used
        emoji = "🟢" if signal == "BUY" else "🔴"
        
        # Format votes if available
        votes_str = ""
        if votes:
            votes_str = f"\n🗳️ <b>Ensemble Votes:</b> BUY:{votes.get('BUY',0)} SELL:{votes.get('SELL',0)} HOLD:{votes.get('HOLD',0)}"

        message = (
            f"{emoji} <b>NEW POSITION EXECUTED: {signal} {symbol} [{timeframe}]</b>\n\n"
            f"💰 <b>Real Fill Price:</b> {price}\n"
            f"🛡️ <b>Stop Loss:</b> {sl}\n"
            f"🎯 <b>Take Profit:</b> {tp}\n"
            f"📊 <b>Executed Lot Size:</b> {lot}\n"
            f"🤖 <b>AI Signal:</b> {signal}\n"
            f"{votes_str}\n\n"
            f"🧠 <b>Tactical Reasoning:</b>\n<i>{reasoning}</i>\n\n"
            f"🎫 <b>MT5 Ticket:</b> {ticket}"
        )
        await TelegramNotifier.send_message(message)

    @staticmethod
    async def notify_error(message: str):
        """Send a dedicated 🚨 ERROR alert to Telegram."""
        await TelegramNotifier.send_message(f"🚨 <b>SYSTEM ERROR</b>\n\n{message}")

    @staticmethod
    async def notify_trade_close(
        symbol: str, 
        ticket: int, 
        profit: float, 
        detail: str, 
        lot: float = 0.0,
        entry_price: float = 0.0,
        close_price: float = 0.0,
        ai_explanation: str = ""
    ):
        """Send alert for a closed position with detailed stats and AI post-mortem."""
        emoji = "💰" if profit >= 0 else "📉"
        status = "بربح" if profit >= 0 else "بخسارة"
        
        # Calculate percentage if possible
        pct_str = ""
        if entry_price > 0:
            diff = close_price - entry_price
            # Adjustment for SELL trades
            # We don't easily know if it's BUY or SELL here without passing it, 
            # but we can infer from profit and price diff.
            # Simplified: just showing the absolute price movement %
            pct = (abs(diff) / entry_price) * 100
            pct_str = f" ({pct:.2f}%)"

        message = (
            f"{emoji} <b>تم إغلاق الصفقة بنجاح</b>\n\n"
            f"📈 <b>الرمز:</b> {symbol}\n"
            f"🎫 <b>العملية:</b> #{ticket}\n"
            f"⚖️ <b>الكمية:</b> {lot} Lot\n"
            f"💵 <b>النتيجة:</b> {status} <b>{profit:.2f} USD</b>{pct_str}\n\n"
            f"📍 <b>سعر الدخول:</b> {entry_price}\n"
            f"🏁 <b>سعر الإغلاق:</b> {close_price}\n"
            f"ℹ️ <b>السبب:</b> {detail}\n\n"
            f"🧠 <b>تحليل فريق الـ AI (Post-Mortem):</b>\n"
            f"<i>{ai_explanation}</i>\n\n"
            f"✅ <i>تم تحديث المحفظة.</i>"
        )
        await TelegramNotifier.send_message(message)
