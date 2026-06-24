"""
LangChain @tool wrappers for the retail chatbot AI agent.
These 4 tools are what the LangGraph supervisor calls
when answering merchant questions.

IMPORTANT: The docstring of each @tool is what the LLM reads
to decide WHEN to call that tool. Write them carefully.
"""
import os
import httpx
import psycopg2
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────────
# API key for calling Day 3 MCP servers
MCP_API_KEY = os.getenv("MCP_API_KEY")
if not MCP_API_KEY:
    raise ValueError("MCP_API_KEY not found in .env!")

# Headers sent with every MCP server request
HEADERS = {
    "X-API-Key": MCP_API_KEY,
    "Content-Type": "application/json"
}

# Supabase connection for pgvector similarity search
SUPABASE_CONN = {
    "host": os.getenv("LAKEBASE_HOST"),
    "dbname": os.getenv("LAKEBASE_DB","postgres"),
    "user": os.getenv("LAKEBASE_USER"),
    "password": os.getenv("LAKEBASE_PASSWORD"),
    "sslmode": "require",
    "port": int(os.getenv("LAKEBASE_PORT",6543))
}

# Embedding model (same as Day 2 — must match!)
# Loads once when this file is imported
print("Loading embedding model for search_products tool...")
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded — 384 dimensions")

def embed_query(text: str) -> list:
    vec = EMBED_MODEL.encode(text,normalize_embeddings=True)
    python_list = vec.tolist()
    return str(python_list)

# ── TOOL 1: search_products ─────────────────────────────
@tool
def search_products(query: str,merchant_id:str) -> str:
    """
    Search the retail product catalog for a specific merchant.

    Use this tool when the merchant asks about:
    - Product details, descriptions, or specifications
    - Return policies or refund rules
    - What products are available in their catalog
    - Product categories or features
    - FAQs about products

    Do NOT use this for live stock levels (use check_inventory),
    order history (use get_order_status), or prices (use get_pricing).

    Input:
        query: the merchant's question in natural language
        merchant_id: the merchant identifier like M001, M002, M003

    Returns:
        Top 5 most relevant product information chunks as text
    """
    try:
        # Step 1: embed the query into a 384-dim vector
        query_vec_str = embed_query(query)

        # Step 2: connect to Supabase and search pgvector
        conn = psycopg2.connect(**SUPABASE_CONN)
        cur = conn.cursor()

        # Set HNSW search quality (same as Day 2)
        cur.execute("SET hnsw.ef_search = 100;")

        # Step 3: find top 5 most similar chunks
        # WHERE merchant_id = %s is the merchant isolation filter
        cur.execute("""
            SELECT chunk_text,1-(embedding <=> %s::vector) AS similarity 
            FROM retail_embeddings 
            WHERE merchant_id = %s 
            ORDER BY embedding <=> %s::vector 
            LIMIT 5 
        """,(query_vec_str,merchant_id,query_vec_str))

        results = cur.fetchall()
        cur.close()
        conn.close()
        
        if not results:
           return f"No relevant product information found for merchant {merchant_id}."
        
        # Step 4: format results as readable text for the LLM
        formatted = []
        for i, (chunk_text,similarity) in enumerate(results):
            formatted.append(f"[Result {i+1}] (relevance: {similarity:.2f})\n{chunk_text}")
        
        return "\n\n".join(formatted)
    
    except Exception as e:
        # ARCHITECT FIX: never raise — always return a string
        return f"Product search temporarily unavailable: {str(e)}"


        # ── TOOL 2: check_inventory ─────────────────────────────
@tool
def check_inventory(sku: str) -> dict:
    """
    Check the current stock level for a specific product SKU.

    Use this tool when the merchant asks about:
    - Stock availability for a specific SKU
    - How many units are in stock
    - Whether a product is available
    - Inventory count for a specific item

    Input:
        sku: the product SKU code, for example SKU-001 or SKU-042

    Returns:
        Dictionary with sku, stock quantity, product name, and status
    """
    try:
        response = httpx.get(
            f"http://localhost:8001/inventory/{sku}",
            headers=HEADERS,
            timeout=5.0  # fail fast if server is down
        )
        return response.json()

    except httpx.TimeoutException:
        # ARCHITECT FIX: return error dict, never raise
        return {
            "sku":     sku,
            "stock":   None,
            "status":  "unavailable",
            "message": "Inventory service timed out. Please try again."
        }
    except Exception as e:
        return {
            "sku":     sku,
            "stock":   None,
            "status":  "error",
            "message": str(e)
        }


# ── TOOL 3: get_order_status ────────────────────────────
@tool
def get_order_status(merchant_id: str) -> dict:
    """
    Retrieve recent orders and order status for the current merchant.

    Use this tool when the merchant asks about:
    - Recent orders or order history
    - Which orders are pending, shipped, or delivered
    - Order counts or totals
    - Delivery status of their orders
    - Transactions in a time period

    Input:
        merchant_id: the merchant identifier like M001, M002, M003

    Returns:
        Dictionary with list of last 20 orders and their statuses
    """
    try:
        response = httpx.get(
            f"http://localhost:8002/orders/{merchant_id}",
            headers=HEADERS,
            timeout=5.0
        )
        return response.json()

    except httpx.TimeoutException:
        return {
            "merchant_id": merchant_id,
            "orders":      [],
            "status":      "unavailable",
            "message":     "Orders service timed out. Please try again."
        }
    except Exception as e:
        return {
            "merchant_id": merchant_id,
            "orders":      [],
            "status":      "error",
            "message":     str(e)
        }


# ── TOOL 4: get_pricing ─────────────────────────────────
@tool
def get_pricing(product_id: str) -> dict:
    """
    Get the current price for a specific product.

    Use this tool when the merchant asks about:
    - Current price of a specific product
    - How much a product costs
    - Pricing for a product ID like P001, P042, P123

    Input:
        product_id: the product identifier like P001, P042

    Returns:
        Dictionary with product_id, product name, price, and status
    """
    try:
        response = httpx.get(
            f"http://localhost:8003/pricing/{product_id}",
            headers=HEADERS,
            timeout=5.0
        )
        return response.json()

    except httpx.TimeoutException:
        return {
            "product_id": product_id,
            "price":      None,
            "status":     "unavailable",
            "message":    "Pricing service timed out. Please try again."
        }
    except Exception as e:
        return {
            "product_id": product_id,
            "price":      None,
            "status":     "error",
            "message":    str(e)
        }


# ── Export all tools ─────────────────────────────────────
# The LangGraph agent imports this list to know all available tools
ALL_TOOLS = [
    search_products,
    check_inventory,
    get_order_status,
    get_pricing
]

print(f"✅ {len(ALL_TOOLS)} tools ready: "
      f"{[t.name for t in ALL_TOOLS]}")