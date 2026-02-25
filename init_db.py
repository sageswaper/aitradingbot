import asyncio
import aiosqlite
from database import AuditDB

async def test_init():
    db = AuditDB()
    print(f"Initializing DB at {db._path}...")
    await db.initialize()
    
    async with aiosqlite.connect(db._path) as conn:
        async with conn.execute("SELECT name FROM sqlite_master WHERE type='table'") as cursor:
            tables = await cursor.fetchall()
            print("Tables found:", tables)

if __name__ == "__main__":
    asyncio.run(test_init())
