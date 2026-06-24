"""
AgentState — the shared state that flows through every
LangGraph node. Every node reads from this dict and
writes back to it.
"""
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    State shared across all agent nodes.

    messages: full conversation history.
              add_messages means new messages are APPENDED
              not overwritten — so history is preserved.

    merchant_id: which merchant is chatting (e.g. "M001").
                 Set at login from user_merchant_map.
                 Never trust client-supplied value.

    active_worker: which worker the supervisor chose.
                   Set by supervisor_node.
                   Used for logging and debugging.

    query_blocked: True if classifier blocked the query.
                   If True, skip supervisor and go
                   directly to escalation.

    error_state: True if any worker tool call failed.
                 Used to log error metrics to MLflow.
    """
    messages:       Annotated[list, add_messages]
    merchant_id:    str
    active_worker:  str
    query_blocked:  bool
    error_state:    bool