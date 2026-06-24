"""
Day 9 — Streamlit UI for retail merchant AI chatbot.
Server-side merchant auth from user_merchant_map.
Agent called asynchronously via asyncio.run().
"""
import sys
import os
import asyncio
import psycopg2
import streamlit as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from agent.mlflow_config import setup_mlflow

load_dotenv()

# ── Page config (must be first Streamlit call) ─────
st.set_page_config(
    page_title="Retail Merchant Assistant",
    page_icon="🛒",
    layout="centered"
)

# ── Initialize session state ───────────────────────
# These persist across re-runs until the browser tab closes
if "logged_in"   not in st.session_state:
    st.session_state.logged_in   = False
if "merchant_id" not in st.session_state:
    st.session_state.merchant_id = None
if "email"       not in st.session_state:
    st.session_state.email       = None
if "messages"    not in st.session_state:
    st.session_state.messages    = []
if "thread_id"   not in st.session_state:
    st.session_state.thread_id   = None
if "app"         not in st.session_state:
    st.session_state.app         = None


# ── Supabase connection helper ─────────────────────
def get_db_conn():
    """Get a fresh psycopg2 connection to Supabase."""
    return psycopg2.connect(
        host     = os.getenv("LAKEBASE_HOST"),
        dbname   = os.getenv("LAKEBASE_DB", "postgres"),
        user     = os.getenv("LAKEBASE_USER"),
        password = os.getenv("LAKEBASE_PASSWORD"),
        sslmode  = "require",
        port     = int(os.getenv("LAKEBASE_PORT", 6543))
    )


# ── ARCHITECT FIX: Server-side merchant auth ───────
def lookup_merchant_id(email: str):
    """
    Look up merchant_id from user_merchant_map.
    NEVER trust client-supplied merchant_id.
    Returns merchant_id string or None if not found.
    """
    try:
        conn = get_db_conn()
        cur  = conn.cursor()
        cur.execute(
            "SELECT merchant_id FROM user_merchant_map "
            "WHERE email = %s",
            (email.strip().lower(),)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row[0] if row else None
    except Exception as e:
        st.error(f"Database error: {e}")
        return None


# ── Build and cache the LangGraph app ─────────────
@st.cache_resource
def get_agent_app():
    """
    Build the LangGraph app once and cache it.
    st.cache_resource means this runs only on first load
    and the same app object is reused for all users.
    """
    from agent.graph import build_app
    setup_mlflow()
    return asyncio.run(build_app())


# ── Call agent asynchronously ──────────────────────
def call_agent(user_message: str) -> dict:
    """
    Send a message to the LangGraph agent.
    Returns the result state dict.
    """
    app         = st.session_state.app
    merchant_id = st.session_state.merchant_id
    thread_id   = st.session_state.thread_id

    state = {
        "messages":      [HumanMessage(user_message)],
        "merchant_id":   merchant_id,
        "active_worker": "",
        "query_blocked": False,
        "error_state":   False
    }
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 10
    }

    # asyncio.run() runs the async agent from sync Streamlit
    result = asyncio.run(
        app.ainvoke(state, config=config)
    )
    return result


# ══════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════

# ── LOGIN PAGE ─────────────────────────────────────
if not st.session_state.logged_in:
    st.title("Retail Merchant Assistant")
    st.markdown("Sign in with your merchant email to continue.")

    with st.form("login_form"):
        email    = st.text_input(
            "Email address",
            placeholder="demo@test.com"
        )
        submitted = st.form_submit_button(
            "Sign in", use_container_width=True)

    if submitted:
        if not email:
            st.error("Please enter your email address.")
        else:
            with st.spinner("Verifying credentials..."):
                merchant_id = lookup_merchant_id(email)

            if merchant_id:
                # Login success — store in session state
                st.session_state.logged_in   = True
                st.session_state.merchant_id = merchant_id
                st.session_state.email       = email.lower()
                # Unique thread_id per session
                st.session_state.thread_id   = (
                    f"{merchant_id}-"
                    f"{email.replace('@','_').replace('.','_')}"
                )
                # Load the agent
                with st.spinner("Loading agent..."):
                    st.session_state.app = get_agent_app()

                st.success(
                    f"Welcome! Signed in as {merchant_id}")
                st.rerun()
            else:
                st.error(
                    "Email not found. "
                    "Try demo@test.com"
                )

# ── CHAT PAGE ──────────────────────────────────────
else:
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("Retail Merchant Assistant")
        st.caption(
            f"Signed in as {st.session_state.email} "
            f"· Merchant: {st.session_state.merchant_id}"
        )
    with col2:
        if st.button("Sign out"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.divider()

    # ── Display conversation history ────────────────
    for msg in st.session_state.messages:
        role    = msg["role"]
        content = msg["content"]
        worker  = msg.get("worker", "")

        with st.chat_message(role):
            st.write(content)
            # Show reasoning expander for assistant messages
            if role == "assistant" and worker:
                with st.expander("Agent reasoning", expanded=False):
                    st.caption(f"Worker: `{worker}`")
                    if msg.get("error_state"):
                        st.caption("Status: error")
                    else:
                        st.caption("Status: ok")

    # ── Suggested questions (first message only) ────
    if not st.session_state.messages:
        st.info(
            "Try asking: "
            "'What is the return policy?' · "
            "'Is SKU-001 in stock?' · "
            "'What is my total revenue?'"
        )

    # ── Chat input ──────────────────────────────────
    user_input = st.chat_input(
        "Ask about products, inventory, orders, or pricing..."
    )

    if user_input:
        # Show user message immediately
        with st.chat_message("user"):
            st.write(user_input)

        # Add to history
        st.session_state.messages.append({
            "role":    "user",
            "content": user_input
        })

        # Call agent with spinner
        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                try:
                    result = call_agent(user_input)

                    # Extract answer and metadata
                    answer = result["messages"][-1].content
                    worker = result.get("active_worker", "")
                    if result.get("query_blocked"):
                        worker = "escalation"
                    error  = result.get("error_state", False)

                    # Display answer
                    st.write(answer)

                    # Reasoning expander
                    with st.expander(
                            "Agent reasoning", expanded=False):
                        st.caption(f"Worker: `{worker}`")
                        st.caption(
                            f"Merchant: "
                            f"`{st.session_state.merchant_id}`"
                        )
                        st.caption(
                            f"Blocked: "
                            f"`{result.get('query_blocked')}`"
                        )
                        if error:
                            st.caption(
                                "Status: service error")

                    # Add to history
                    st.session_state.messages.append({
                        "role":        "assistant",
                        "content":     answer,
                        "worker":      worker,
                        "error_state": error
                    })

                except Exception as e:
                    err_msg = (
                        "I encountered an error. "
                        "Please try again."
                    )
                    st.write(err_msg)
                    st.caption(f"Error detail: {str(e)[:100]}")
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": err_msg,
                        "worker":  "error"
                    })