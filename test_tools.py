"""
Unit tests for all 4 LangChain tools.
Tests: happy path, not found, and server down scenarios.
Run: python test_tools.py
(All 3 Day 3 MCP servers must be running first)
"""
from tools.mcp_tools import (
    search_products, check_inventory,
    get_order_status, get_pricing, ALL_TOOLS
)

print("=" * 55)
print("TESTING ALL 4 LANGCHAIN TOOLS")
print("=" * 55)

# ── Test 1: search_products ──────────────────────────
print("\n--- Tool 1: search_products ---")

result = search_products.invoke({
    "query":       "what is the return policy",
    "merchant_id": "M001"
})
print(f"Query: 'what is the return policy' for M001")
print(f"Result preview: {result[:150]}...")

# Verify it returns string content, not error
if "unavailable" in result.lower() or "error" in result.lower():
    print("⚠ search_products returned error — check Supabase connection")
else:
    print("✅ search_products working!")

# ── Test 2: check_inventory ──────────────────────────
print("\n--- Tool 2: check_inventory ---")

result = check_inventory.invoke({"sku": "SKU-001"})
print(f"Inventory for SKU-001: {result}")

if result.get("status") == "ok":
    print(f"✅ check_inventory working! Stock: {result.get('stock')}")
elif result.get("status") == "not_found":
    print("⚠ SKU-001 not found — check your products table has data")
elif result.get("status") == "unavailable":
    print("❌ Inventory MCP server not running on port 8001")
else:
    print(f"⚠ Unexpected: {result}")

# ── Test 3: get_order_status ─────────────────────────
print("\n--- Tool 3: get_order_status ---")

result = get_order_status.invoke({"merchant_id": "M001"})
print(f"Orders for M001: status={result.get('status')}, "
      f"count={result.get('count', 0)}")

if result.get("status") == "ok":
    print(f"✅ get_order_status working! Orders: {result.get('count')}")
elif result.get("status") == "unavailable":
    print("❌ Orders MCP server not running on port 8002")

# ── Test 4: get_pricing ──────────────────────────────
print("\n--- Tool 4: get_pricing ---")

result = get_pricing.invoke({"product_id": "P001"})
print(f"Pricing for P001: {result}")

if result.get("status") == "ok":
    print(f"✅ get_pricing working! Price: ${result.get('price')}")
elif result.get("status") == "unavailable":
    print("❌ Pricing MCP server not running on port 8003")

# ── Test 5: error boundary (server down) ────────────
print("\n--- Test 5: error boundary (wrong port) ---")

import httpx, os
from langchain_core.tools import tool

@tool
def test_error_boundary(sku: str) -> dict:
    """Test tool that calls wrong port to simulate server down."""
    try:
        r = httpx.get(f"http://localhost:9999/inventory/{sku}",
                      headers={"X-API-Key": os.getenv("MCP_API_KEY")},
                      timeout=2.0)
        return r.json()
    except Exception as e:
        return {"status": "unavailable", "message": str(e)}

result = test_error_boundary.invoke({"sku": "SKU-001"})
if result.get("status") == "unavailable":
    print("✅ Error boundary working! Server down returns dict not crash")
else:
    print(f"❌ Error boundary broken: {result}")

# ── Test 6: ALL_TOOLS export ─────────────────────────
print("\n--- Test 6: ALL_TOOLS list ---")
print(f"Tools available: {[t.name for t in ALL_TOOLS]}")
assert len(ALL_TOOLS) == 4, "Should have exactly 4 tools"
print("✅ ALL_TOOLS has 4 tools — ready for LangGraph agent")

print("\n" + "=" * 55)
print("ALL TESTS COMPLETE")
print("=" * 55)

print("\n--- Tool schema check (what the LLM sees) ---")
for tool in ALL_TOOLS:
    print(f"\nTool: {tool.name}")
    print(f"Description: {tool.description[:100]}...")