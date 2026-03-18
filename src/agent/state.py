"""
LangGraph AgentState + all Pydantic schemas
"""
from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


# Intent Node output schema
class IntentResult(BaseModel):
    """
    Validated output from the Intent extraction node.
    """
    country_name: str = Field(
        default="",
        description="Normalised English country name extracted from the user query. "
                    "Empty string if the query is off-topic or ambiguous.",
    )
    requested_fields: list[str] = Field(
        default_factory=list,
        description="List of data fields the user is asking about, e.g. "
                    "['capital', 'population', 'currency', 'languages', 'area'].",
    )
    is_off_topic: bool = Field(
        default=False,
        description="True if the user's question is NOT about a country "
                    "(e.g. asking for a recipe, math problem, etc.).",
    )


# LangGraph State
class AgentState(TypedDict, total=False):
    """
    The single source of truth passed through every node in the graph.
    """
    messages: list[BaseMessage]
    intent: Optional[IntentResult]
    api_data: Optional[dict[str, Any]]
    answer: Optional[str]
    error: Optional[str]
