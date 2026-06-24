import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from mcp_servers.shared.auth import APIKeyMiddleware
from mcp_servers.shared.db import get_pool, close_pool

limiter = Limiter(key_func=get_remote_address,
                  default_limits=["30/minute"])

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = await get_pool()
    print("✅ Orders MCP server ready on port 8002")
    yield
    await close_pool()

app = FastAPI(title="Orders MCP Server", lifespan=lifespan)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(APIKeyMiddleware)
app.state.limiter = limiter

@app.get("/health")
async def health():
    return {"status": "ok", "service": "orders", "port": 8002}

@app.get("/orders/{merchant_id}")
@limiter.limit("30/minute")
async def get_orders(merchant_id: str, request: Request):
    """
    Get recent orders for a merchant.
    Returns: last 20 orders sorted by newest first.
    """
    try:
        async with request.app.state.pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT order_id, product_id, quantity,
                          status, total, created_at
                   FROM orders
                   WHERE merchant_id = $1
                   ORDER BY created_at DESC
                   LIMIT 20""",
                merchant_id
            )

        # Convert rows to list of dicts
        orders = []
        for r in rows:
            orders.append({
                "order_id":   r["order_id"],
                "product_id": r["product_id"],
                "quantity":   r["quantity"],
                "status":     r["status"],
                "total":      float(r["total"] or 0),
                "created_at": str(r["created_at"])
            })

        return {
            "merchant_id": merchant_id,
            "orders":      orders,
            "count":       len(orders),
            "status":      "ok"
        }

    except Exception as e:
        return {
            "merchant_id": merchant_id,
            "orders":      [],
            "status":      "error",
            "message":     str(e)
        }