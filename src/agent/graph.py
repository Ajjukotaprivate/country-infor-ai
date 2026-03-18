"""
LangGraph StateGraph
"""

import os
from langgraph.graph import END, StateGraph

from src.agent.state import AgentState
from src.agent.nodes import (
    intent_node,
    router_node,
    route_fn,
    tool_node,
    synthesis_node,
    reject_node,
    error_node,
)
from src.core.config import get_settings

# Setup LangSmith tracing
def setup_tracing():
    cfg = get_settings()
    if cfg.langchain_api_key:
        os.environ.setdefault("LANGCHAIN_API_KEY", cfg.langchain_api_key)
        os.environ.setdefault("LANGCHAIN_TRACING_V2", cfg.langchain_tracing_v2)
        os.environ.setdefault("LANGCHAIN_PROJECT", cfg.langchain_project)
        os.environ.setdefault("LANGCHAIN_ENDPOINT", cfg.langchain_endpoint)

setup_tracing()

def build_graph():
    builder = StateGraph(AgentState)

    # Add all our nodes
    builder.add_node("intent", intent_node)
    builder.add_node("router", router_node)
    builder.add_node("tool", tool_node)
    builder.add_node("synthesis", synthesis_node)
    builder.add_node("reject", reject_node)
    builder.add_node("error", error_node)

    # Kick things off with the intent parser
    builder.set_entry_point("intent")
    
    # Route based on what the intent was
    builder.add_edge("intent", "router")
    builder.add_conditional_edges(
        "router",
        route_fn,
        {
            "tool": "tool",
            "reject": "reject",
        },
    )

    # Main flow
    builder.add_edge("tool", "synthesis")
    builder.add_edge("synthesis", END)
    
    # Edge cases exit early
    builder.add_edge("reject", END)
    builder.add_edge("error", END)

    return builder.compile()

# Global agent instance to import elsewhere
agent = build_graph()

