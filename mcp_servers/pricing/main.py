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
    print("✅ Pricing MCP server ready on port 8003")
    yield
    await close_pool()

app = FastAPI(title="Pricing MCP Server", lifespan=lifespan)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(APIKeyMiddleware)
app.state.limiter = limiter

@app.get("/health")
async def health():
    return {"status": "ok", "service": "pricing", "port": 8003}

@app.get("/pricing/{product_id}")
@limiter.limit("30/minute")
async def get_pricing(product_id: str, request: Request):
    """
    Get price for a specific product.
    Returns: product_id, product_name, price, status
    """
    try:
        async with request.app.state.pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT product_id, product_name, price
                   FROM products
                   WHERE product_id = $1""",
                product_id
            )

        if not row:
            return {
                "product_id": product_id,
                "price":      None,
                "status":     "not_found",
                "message":    f"Product {product_id} not found"
            }

        return {
            "product_id": row["product_id"],
            "product":    row["product_name"],
            "price":      float(row["price"] or 0),
            "status":     "ok"
        }

    except Exception as e:
        return {
            "product_id": product_id,
            "price":      None,
            "status":     "error",
            "message":    str(e)
        }