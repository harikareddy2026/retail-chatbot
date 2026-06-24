"""
LangGraph agent for the retail merchant chatbot.
Day 6: classifier + supervisor + RAG worker.
Day 7: adds action + data + escalation workers.
"""
import os
import re
import json
import sys
sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.mlflow_config import setup_mlflow
from tools.mcp_tools import (
    search_products,
    check_inventory,
    get_order_status,
    get_pricing
)

# Setup MLflow FIRST before any LangChain imports
setup_mlflow()

# ── LLM Setup ─────────────────────────────────────
# Try AWS Bedrock first, fall back to keyword routing
LLM_AVAILABLE = False
llm = None

try:
    from langchain_aws import ChatBedrock
    llm = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        region_name="us-east-1",
        model_kwargs={"max_tokens": 500}
    )
    # Quick test
    test = llm.invoke([HumanMessage(content="hi")])
    LLM_AVAILABLE = True
    print("✅ AWS Bedrock LLM available")
except Exception as e:
    print(f"⚠ Bedrock not available ({str(e)[:50]})")
    print("  Using keyword-based routing fallback")
    LLM_AVAILABLE = False


# ── Classifier Node ───────────────────────────────
# Runs FIRST on every message.
# Blocks non-retail queries before they reach supervisor.

# Find RETAIL_KEYWORDS and add these words:
RETAIL_KEYWORDS = [
    "product", "return", "policy", "refund", "stock",
    "inventory", "order", "price", "pricing", "cost",
    "sku", "available", "ship", "deliver", "catalog",
    "description", "item", "sell", "buy", "purchase",
    "revenue", "total", "how many", "count", "analytics",
    "summary", "breakdown", "top", "best", "popular",
    # ADD THESE:
    "tell me", "about", "describe", "what is", "yoga",
    "sunglasses", "jacket", "roller", "pants", "bottle",
    "earbuds", "bands", "cap", "shoes", "running"
]

# Find BLOCK_KEYWORDS and add:
BLOCK_KEYWORDS = [
    "ignore", "forget", "pretend", "jailbreak",
    "weather", "news", "recipe", "movie", "music",
    # ADD THESE:
    "poem", "write me", "song", "story", "joke",
    "rhyme", "fiction", "essay", "write a"
]

async def classifier_node(state: AgentState) -> dict:
    """
    Checks if the query is a valid retail question.
    If not → sets query_blocked=True → escalation.
    """
    last_msg = state["messages"][-1].content.lower()

    # Check for prompt injection attempts
    for kw in BLOCK_KEYWORDS:
        if kw in last_msg:
            print(f"  [classifier] BLOCKED — contains '{kw}'")
            return {"query_blocked": True,
                    "active_worker": "escalation"}

    if LLM_AVAILABLE:
        # Use LLM for accurate classification
        prompt = f"""Is this a valid retail merchant query
about products, inventory, orders, or pricing?
Query: "{state['messages'][-1].content}"
Answer only: yes or no"""
        response = await llm.ainvoke(
            [HumanMessage(content=prompt)])
        blocked = "no" in response.content.lower()
    else:
        # Keyword fallback — check if any retail word present
        blocked = not any(
            kw in last_msg for kw in RETAIL_KEYWORDS)

    if blocked:
        print(f"  [classifier] BLOCKED — not a retail query")
    else:
        print(f"  [classifier] PASSED — routing to supervisor")

    return {"query_blocked": blocked,
            "active_worker": "escalation" if blocked else ""}


# ── Supervisor Node ───────────────────────────────
# Routes valid queries to the correct worker.
# Uses structured output + regex fallback.

SUPERVISOR_PROMPT = """You are a supervisor routing
retail merchant queries to the correct worker.

Workers:
- rag_worker: product info, descriptions, return
  policies, FAQs, catalog questions
- action_worker: inventory stock levels, order
  status, current pricing
- data_worker: analytics, counts, totals,
  revenue reports
- escalation: anything NOT about products,
  orders, inventory, or pricing

Few-shot examples:
"What is the return policy?" → rag_worker
"Is SKU-001 in stock?" → action_worker
"How many orders last month?" → data_worker
"What is the weather?" → escalation
"Tell me about product P003" → rag_worker
"Check stock for blue shoes" → action_worker

Reply ONLY with valid JSON:
{"worker": "worker_name", "reason": "one sentence"}"""

def parse_routing(text: str) -> str:
    """
    Parse supervisor response with 3-layer fallback.
    Layer 1: direct JSON parse
    Layer 2: regex extract worker name
    Layer 3: default to escalation
    """
    valid = {"rag_worker", "action_worker",
             "data_worker", "escalation"}

    # Layer 1: try direct JSON parse
    try:
        data = json.loads(text)
        worker = data.get("worker", "")
        if worker in valid:
            return worker
    except Exception:
        pass

    # Layer 2: regex fallback
    match = re.search(
        r'"worker"\s*:\s*"(\w+)"', text)
    if match:
        worker = match.group(1)
        if worker in valid:
            return worker

    # Layer 3: safe default
    print("  [supervisor] parse failed — defaulting to escalation")
    return "escalation"

async def supervisor_node(state: AgentState) -> dict:
    """Routes the query to the correct worker."""
    last_msg = state["messages"][-1].content

    if LLM_AVAILABLE:
        messages = [
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(content=last_msg)
        ]
        response = await llm.ainvoke(messages)
        worker = parse_routing(response.content)
    else:
        # Keyword-based routing fallback
        msg = last_msg.lower()
        if any(w in msg for w in
           ["how many", "total", "revenue", "count",
           "breakdown", "analytics", "summary",
           "top", "best", "popular", "earning"]):
           worker = "data_worker"

        elif any(w in msg for w in
           ["stock", "inventory", "sku", "available",
           "price", "cost", "pricing", "how much",
           "order status", "shipped", "delivered",
           "my orders", "recent orders"]):
            worker = "action_worker"

        elif any(w in msg for w in
           ["product", "policy", "return", "refund",
           "description", "what", "tell me", "about",
           "catalog", "sell"]):
           worker = "rag_worker"

        else:
           worker = "escalation"

    print(f"  [supervisor] → {worker}")
    return {"active_worker": worker}

# ── RAG Worker Node ───────────────────────────────
# Searches pgvector for relevant chunks
# then generates an answer using the LLM.

async def rag_worker_node(state: AgentState) -> dict:
    """Answers product/policy questions using pgvector."""
    try:
        last_msg   = state["messages"][-1].content
        merchant_id = state["merchant_id"]

        # Step 1: search pgvector for relevant chunks
        context = search_products.invoke({
            "query":       last_msg,
            "merchant_id": merchant_id
        })

        # Step 2: generate answer from context
        if LLM_AVAILABLE:
            prompt = f"""You are a helpful retail assistant
for merchant {merchant_id}.

Answer the merchant's question using ONLY the
context below. Be concise — 2-3 sentences maximum.
If the context does not contain the answer, say so.

Context:
{context}

Question: {last_msg}"""
            response = await llm.ainvoke(
                [HumanMessage(content=prompt)])
            answer = response.content
        else:
            # Return raw context if no LLM
            # answer = f"Here is relevant information:\n{context[:500]}"
                # Strip chunk headers and pipe separators
            # Turn raw context into a readable sentence
                # Extract first result's text only, keep original vocabulary
            lines = context.split("\n")
            first_result = ""
            for line in lines:
             if line.strip() and not line.strip().startswith("[Result") \
                   and not line.strip().startswith("relevance"):
                first_result = line.strip()
                break
            answer = first_result if first_result else context[:300]
        
        print(f"  [rag_worker] answered for {merchant_id}")
        return {
            "messages":    [AIMessage(content=answer)],
            "error_state": False
        }

    except Exception as e:
        # ARCHITECT FIX: never crash — return graceful message
        print(f"  [rag_worker] ERROR: {e}")
        return {
            "messages": [AIMessage(
                content="I am having trouble searching "
                        "products right now. Please try again."
            )],
            "error_state": True
        }


# ── Escalation Node ────────────────────────────────
async def escalation_node(state: AgentState) -> dict:
    """Handles off-topic or blocked queries."""
    import mlflow

    if state.get("query_blocked"):
        msg = ("I can only help with product questions, "
               "inventory checks, order status, and pricing. "
               "Could you rephrase your question?")
    else:
        msg = ("I am not sure how to help with that. "
               "I can assist with products, inventory, "
               "orders, and pricing.")

    # Log escalation to MLflow
    try:
        mlflow.log_metric("escalations", 1)
    except Exception:
        pass

    print(f"  [escalation] sent fallback response")
    return {
        "messages": [AIMessage(content=msg)],
        "error_state": False
    }

# ── Action Worker Node ────────────────────────────
async def action_worker_node(state: AgentState) -> dict:
    """
    Handles live data queries by calling MCP servers.
    Detects which tool to call from message keywords.
    """
    try:
        last_msg    = state["messages"][-1].content
        merchant_id = state["merchant_id"]
        msg_lower   = last_msg.lower()

        result    = None
        tool_used = "none"

        # ── Which tool to call ─────────────────────
        if any(w in msg_lower for w in
               ["stock", "inventory", "sku",
                "available", "units"]):
            import re
            sku_match = re.search(
                r'sku[-\s]?(\w+)', msg_lower, re.I)
            sku  = f"SKU-{sku_match.group(1).upper()}" \
                   if sku_match else "SKU-001"
            result    = check_inventory.invoke({"sku": sku})
            tool_used = f"check_inventory({sku})"

        elif any(w in msg_lower for w in
                 ["order", "delivery", "shipped",
                  "pending", "delivered", "transaction"]):
            result    = get_order_status.invoke(
                {"merchant_id": merchant_id})
            tool_used = f"get_order_status({merchant_id})"

        elif any(w in msg_lower for w in
                 ["price", "cost", "pricing",
                  "how much", "amount"]):
            import re
            pid_match = re.search(r'p(\d+)', msg_lower, re.I)
            pid  = f"P{pid_match.group(1)}" \
                   if pid_match else "P001"
            result    = get_pricing.invoke({"product_id": pid})
            tool_used = f"get_pricing({pid})"

        else:
            result    = get_order_status.invoke(
                {"merchant_id": merchant_id})
            tool_used = f"get_order_status({merchant_id})"

        print(f"  [action_worker] called {tool_used}")

        # ── Format result ──────────────────────────
        if result and result.get("status") == "ok":
            answer = (
                f"Here is the information you requested:\n\n"
                f"{json.dumps(result, indent=2, default=str)}"
            )
        elif result and result.get("status") == "not_found":
            answer = (
                "I could not find that data. "
                "Please check the ID and try again."
            )
        elif result and result.get("status") in \
                ("error", "unavailable"):
            return {
                "messages": [AIMessage(
                    content="The service is temporarily "
                            "unavailable. Please try again."
                )],
                "error_state": True
            }
        else:
            answer = str(result)

        return {
            "messages":    [AIMessage(content=answer)],
            "error_state": False
        }

    except Exception as e:
        print(f"  [action_worker] ERROR: {e}")
        return {
            "messages": [AIMessage(
                content="I am having trouble fetching "
                        "that data. Please try again."
            )],
            "error_state": True
        }

# ── Data Worker Node ──────────────────────────────
async def data_worker_node(state: AgentState) -> dict:
    """
    Handles analytics by running SQL directly on Supabase.
    For COUNT, SUM, AVG, GROUP BY queries.
    """
    try:
        import psycopg2
        from pathlib import Path
        from dotenv import load_dotenv
        load_dotenv(
            Path(__file__).parent.parent / ".env")

        last_msg    = state["messages"][-1].content
        merchant_id = state["merchant_id"]
        msg_lower   = last_msg.lower()

        conn = psycopg2.connect(
            host    = os.getenv("LAKEBASE_HOST"),
            dbname  = os.getenv("LAKEBASE_DB", "postgres"),
            user    = os.getenv("LAKEBASE_USER"),
            password= os.getenv("LAKEBASE_PASSWORD"),
            sslmode = "require",
            port    = int(os.getenv("LAKEBASE_PORT", 6543))
        )
        cur = conn.cursor()

        # ── Which SQL to run ───────────────────────
        if any(w in msg_lower for w in
               ["revenue", "total", "earned",
                "how much", "sales"]):
            cur.execute("""
                SELECT COUNT(*)              AS total_orders,
                       COALESCE(SUM(total),0) AS revenue,
                       COALESCE(AVG(total),0) AS avg_order
                FROM orders
                WHERE merchant_id = %s
            """, (merchant_id,))
            row = cur.fetchone()
            answer = (
                f"Analytics for {merchant_id}:\n"
                f"  Total orders:  {row[0]}\n"
                f"  Total revenue: ${float(row[1]):.2f}\n"
                f"  Avg order:     ${float(row[2]):.2f}"
            )

        elif any(w in msg_lower for w in
                 ["status", "breakdown", "pending",
                  "shipped", "delivered"]):
            cur.execute("""
                SELECT status, COUNT(*) AS cnt
                FROM orders
                WHERE merchant_id = %s
                GROUP BY status
                ORDER BY cnt DESC
            """, (merchant_id,))
            rows = cur.fetchall()
            lines = [f"  {r[0]}: {r[1]} orders"
                     for r in rows]
            answer = (
                f"Order status breakdown for "
                f"{merchant_id}:\n" + "\n".join(lines)
            )

        elif any(w in msg_lower for w in
                 ["top", "best", "popular",
                  "product", "selling"]):
            cur.execute("""
                SELECT product_id,
                       COUNT(*)   AS order_count,
                       SUM(total) AS total_revenue
                FROM orders
                WHERE merchant_id = %s
                GROUP BY product_id
                ORDER BY order_count DESC
                LIMIT 5
            """, (merchant_id,))
            rows = cur.fetchall()
            lines = [
                f"  {r[0]}: {r[1]} orders "
                f"(${float(r[2]):.2f})"
                for r in rows
            ]
            answer = (
                f"Top 5 products for {merchant_id}:\n"
                + "\n".join(lines)
            )

        else:
            cur.execute("""
                SELECT COUNT(*),
                       COALESCE(SUM(total), 0)
                FROM orders WHERE merchant_id = %s
            """, (merchant_id,))
            row = cur.fetchone()
            answer = (
                f"Summary for {merchant_id}: "
                f"{row[0]} orders, "
                f"${float(row[1]):.2f} revenue"
            )

        cur.close()
        conn.close()
        print(f"  [data_worker] SQL for {merchant_id}")

        return {
            "messages":    [AIMessage(content=answer)],
            "error_state": False
        }

    except Exception as e:
        print(f"  [data_worker] ERROR: {e}")
        return {
            "messages": [AIMessage(
                content="I am having trouble running "
                        "analytics right now. "
                        "Please try again."
            )],
            "error_state": True
        }
    

# ── Routing Functions ─────────────────────────────
# These decide which edge to follow after each node.

def route_after_classifier(state: AgentState) -> str:
    """After classifier: blocked → escalation, else → supervisor."""
    if state.get("query_blocked"):
        return "escalation"
    return "supervisor"

def route_after_supervisor(state: AgentState) -> str:
    """After supervisor: route to chosen worker."""
    return state.get("active_worker", "escalation")


# ── AsyncPostgresSaver Setup ───────────────────────
# Saves conversation state to Supabase after every node.
# Uses native PG password — no OAuth token expiry.

def get_checkpointer_conn_string() -> str:
    """Build psycopg-compatible connection string for AsyncPostgresSaver."""
    host     = os.getenv("LAKEBASE_HOST")
    user     = os.getenv("LAKEBASE_USER")
    password = os.getenv("LAKEBASE_PASSWORD")
    db       = os.getenv("LAKEBASE_DB", "postgres")
    port     = os.getenv("LAKEBASE_PORT", "6543")

    # psycopg uses postgresql:// not postgresql+asyncpg://
    # sslmode=require not ssl=require
    return (
        f"postgresql://{user}:{password}"
        f"@{host}:{port}/{db}?sslmode=require"
    )


# ── Build App Function ─────────────────────────────
async def build_app():
    """Build LangGraph app with all 6 nodes."""
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()
    print("✅ MemorySaver ready")

    graph = StateGraph(AgentState)

    # ── Add all 6 nodes ────────────────────────────
    graph.add_node("classifier",    classifier_node)
    graph.add_node("supervisor",    supervisor_node)
    graph.add_node("rag_worker",    rag_worker_node)
    graph.add_node("action_worker", action_worker_node)
    graph.add_node("data_worker",   data_worker_node)
    graph.add_node("escalation",    escalation_node)

    # ── Edges ──────────────────────────────────────
    graph.add_edge(START, "classifier")

    graph.add_conditional_edges(
        "classifier",
        route_after_classifier,
        {
            "supervisor": "supervisor",
            "escalation": "escalation"
        }
    )

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "rag_worker":    "rag_worker",
            "action_worker": "action_worker",
            "data_worker":   "data_worker",
            "escalation":    "escalation"
        }
    )

    graph.add_edge("rag_worker",    END)
    graph.add_edge("action_worker", END)
    graph.add_edge("data_worker",   END)
    graph.add_edge("escalation",    END)

    app = graph.compile(checkpointer=checkpointer)
    print("✅ LangGraph app compiled — 6 nodes wired")
    return app