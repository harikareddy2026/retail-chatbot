import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from mcp_servers.shared.auth import APIKeyMiddleware
from mcp_servers.shared.db import get_pool, close_pool

# Rate limiter — 30 requests per minute per IP
limiter = Limiter(key_func=get_remote_address,
                  default_limits=["30/minute"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs at startup and shutdown"""
    # STARTUP: create the database pool
    app.state.pool = await get_pool()
    print("✅ Inventory MCP server ready on port 8001")
    yield   # server runs here
    # SHUTDOWN: close the pool
    await close_pool()
    print("Inventory MCP server stopped")

# Create FastAPI app
app = FastAPI(
    title="Inventory MCP Server",
    description="Checks stock levels by SKU",
    lifespan=lifespan
)

# Add middleware (runs on EVERY request in this order)
app.add_middleware(SlowAPIMiddleware)   # rate limiting
app.add_middleware(APIKeyMiddleware)    # API key check
app.state.limiter = limiter

@app.get("/health")
async def health():
    """Health check — no auth needed. Docker uses this."""
    return {"status": "ok", "service": "inventory", "port": 8001}

@app.get("/inventory/{sku}")
@limiter.limit("30/minute")
async def get_inventory(sku: str, request: Request):
    """
    Check stock level for a product SKU.
    Returns: sku, stock_qty, product_name, status
    """
    try:
        # Borrow a connection from the pool
        async with request.app.state.pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT sku, stock_qty, product_name
                   FROM products
                   WHERE sku = $1""",
                sku
            )

        if not row:
            return {
                "sku": sku,
                "stock": 0,
                "product": None,
                "status": "not_found",
                "message": f"SKU {sku} not found"
            }

        return {
            "sku":     row["sku"],
            "stock":   row["stock_qty"],
            "product": row["product_name"],
            "status":  "ok"
        }

    except Exception as e:
        # ARCHITECT FIX: always return dict, never raise
        return {
            "sku":     sku,
            "stock":   None,
            "status":  "error",
            "message": str(e)
        }