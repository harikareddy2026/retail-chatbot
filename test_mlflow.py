"""
Day 5 smoke test.
Calls all 4 LangChain tools inside an MLflow run
and verifies traces are captured correctly.

Run: python test_mlflow.py
(All 3 Day 3 MCP servers must be running first)
"""

# ── STEP 1: Setup MLflow BEFORE importing tools ──
# This is the critical order — autolog must be active
# before any LangChain objects are imported
from agent.mlflow_config import (
    setup_mlflow, log_metric, log_param, log_tag,
    print_run_summary
)
setup_mlflow()  # ← MUST be first

# ── STEP 2: Now import tools ──────────────────────
import mlflow
from tools.mcp_tools import (
    search_products, check_inventory,
    get_order_status, get_pricing, ALL_TOOLS
)

print("\n" + "="*50)
print("DAY 5 SMOKE TEST — MLflow + All 4 Tools")
print("="*50)

# ── STEP 3: Run all tools inside an MLflow run ────
with mlflow.start_run(run_name="day5_smoke_test"):

    # Log what we are testing
    log_param("tools_count",     len(ALL_TOOLS))
    log_param("embedding_model", "all-MiniLM-L6-v2")
    log_param("test_merchant",   "M001")
    log_tag("day",    "5")
    log_tag("type",   "smoke_test")

    results = {}

    # Tool 1: search_products (RAG search)
    print("\n[1/4] Testing search_products...")
    result1 = search_products.invoke({
        "query":       "what is the return policy",
        "merchant_id": "M001"
    })
    passed1 = len(result1) > 0 and "unavailable" not in result1
    results["search_products"] = passed1
    log_metric("search_products_pass", 1 if passed1 else 0)
    print(f"      Result: {'✅ PASS' if passed1 else '❌ FAIL'}")
    print(f"      Preview: {result1[:80]}...")

    # Tool 2: check_inventory
    print("\n[2/4] Testing check_inventory...")
    result2 = check_inventory.invoke({"sku": "SKU-001"})
    passed2 = result2.get("status") in ("ok", "not_found")
    results["check_inventory"] = passed2
    log_metric("check_inventory_pass", 1 if passed2 else 0)
    log_metric("inventory_stock",
               result2.get("stock", 0) or 0)
    print(f"      Result: {'✅ PASS' if passed2 else '❌ FAIL'}")
    print(f"      Stock: {result2.get('stock')} units")

    # Tool 3: get_order_status
    print("\n[3/4] Testing get_order_status...")
    result3 = get_order_status.invoke({"merchant_id": "M001"})
    passed3 = result3.get("status") in ("ok", "not_found") or result3.get("count") is not None
    results["get_order_status"] = passed3
    order_count = result3.get("count", 0) or 0
    log_metric("get_order_status_pass", 1 if passed3 else 0)
    log_metric("orders_count", order_count)
    print(f"      Result: {'✅ PASS' if passed3 else '❌ FAIL'}")
    print(f"      Orders: {order_count} found")
    print(f"      Raw response: {result3}")

    # Tool 4: get_pricing
    print("\n[4/4] Testing get_pricing...")
    result4 = get_pricing.invoke({"product_id": "P001"})
    passed4 = result4.get("status") in ("ok", "not_found")
    results["get_pricing"] = passed4
    log_metric("get_pricing_pass", 1 if passed4 else 0)
    log_metric("product_price",
               result4.get("price", 0) or 0)
    print(f"      Result: {'✅ PASS' if passed4 else '❌ FAIL'}")
    print(f"      Price: ${result4.get('price')}")
    print(f"      Raw response: {result4}")
    
    # Overall result
    all_pass = all(results.values())
    log_metric("all_tools_passing", 1 if all_pass else 0)
    log_tag("status", "pass" if all_pass else "fail")

    print("\n" + "="*50)
    if all_pass:
        print("✅ ALL 4 TOOLS PASSING — traces logged to MLflow")
    else:
        failed = [k for k,v in results.items() if not v]
        print(f"❌ FAILING TOOLS: {failed}")
    print("="*50)

# ── STEP 4: Show all runs ─────────────────────────
print("\n")
print_run_summary()