import os
import asyncpg
from pathlib import Path
from dotenv import load_dotenv

# Force load .env from project root regardless of where script runs from
ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

# Debug print to verify values loaded
LAKEBASE_HOST = os.getenv("LAKEBASE_HOST", "NOT_SET")
print(f"[db.py] LAKEBASE_HOST = {LAKEBASE_HOST}")
print(f"[db.py] LAKEBASE_PORT = {os.getenv('LAKEBASE_PORT', 'NOT_SET')}")
print(f"[db.py] LAKEBASE_USER = {os.getenv('LAKEBASE_USER', 'NOT_SET')}")

_pool = None

async def get_pool():
    global _pool
    if _pool is None:
        host     = os.getenv("LAKEBASE_HOST")
        database = os.getenv("LAKEBASE_DB", "postgres")
        user     = os.getenv("LAKEBASE_USER")
        password = os.getenv("LAKEBASE_PASSWORD")
        port     = int(os.getenv("LAKEBASE_PORT", 6543))

        if not host or host == "NOT_SET":
            raise ValueError(
                "LAKEBASE_HOST not found! "
                "Check .env file exists in project root."
            )

        print(f"[db.py] Connecting to {host}:{port} as {user}")

        _pool = await asyncpg.create_pool(
            host                 = host,
            database             = database,
            user                 = user,
            password             = password,
            ssl                  = "require",
            port                 = port,
            min_size             = 2,
            max_size             = 10,
            statement_cache_size = 0  # required for Supabase PgBouncer
        )
        print(f"✅ Database pool created: {host}")
    return _pool

async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        print("Database pool closed")