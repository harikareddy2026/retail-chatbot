import httpx
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("MCP_API_KEY")
HEADERS = {"X-API-Key": API_KEY}

def test_all_servers():
    print("Testing all 3 MCP servers...\n")

    # Test Inventory
    r = httpx.get("http://localhost:8001/inventory/SKU-001",
                  headers=HEADERS, timeout=5.0)
    print(f"Inventory SKU-001: {r.status_code}")
    print(f"  Response: {r.json()}\n")

    # Test bad key → should be 401
    r_bad = httpx.get("http://localhost:8001/inventory/SKU-001",
                      headers={"X-API-Key": "wrong"}, timeout=5.0)
    print(f"Bad key test: {r_bad.status_code} (should be 401)\n")

    # Test Orders
    r = httpx.get("http://localhost:8002/orders/M001",
                  headers=HEADERS, timeout=5.0)
    data = r.json()
    print(f"Orders M001: {r.status_code}")
    print(f"  Count: {data.get('count')} orders\n")

    # Test Pricing
    r = httpx.get("http://localhost:8003/pricing/P001",
                  headers=HEADERS, timeout=5.0)
    print(f"Pricing P001: {r.status_code}")
    print(f"  Response: {r.json()}\n")

    # Test SKU not found
    r = httpx.get("http://localhost:8001/inventory/SKU-FAKE",
                  headers=HEADERS, timeout=5.0)
    print(f"Not found test: {r.status_code}")
    print(f"  Response: {r.json()}")
    print("\n✅ All tests complete!")

test_all_servers()