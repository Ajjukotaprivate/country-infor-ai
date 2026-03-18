"""
Token-based sliding window for LangGraph message history.
"""
from __future__ import annotations

import logging
from typing import Sequence, Any

import tiktoken
from langchain_core.messages import BaseMessage, SystemMessage

logger = logging.getLogger(__name__)

_ENC = tiktoken.get_encoding("cl100k_base")

def _count_tokens(messages: Sequence[Any], encoding_name: str = "cl100k_base") -> int:
    """Count total tokens across all messages using tiktoken."""
    total = 0
    for msg in messages:
        # 4 tokens overhead per message (role + separators)
        
        # In LangGraph Studio, messages might be passed as raw dictionaries
        if isinstance(msg, dict):
            content = str(msg.get("content", ""))
        else:
            content = str(msg.content)
        total += 4 + len(_ENC.encode(content))
    return total


def trim_messages(
    messages: list[Any],
    max_tokens: int = 3000,
    encoding_name: str = "cl100k_base",
) -> list[Any]:
    """
    Trim a message list to stay within max_tokens.

    Strategy:
      1. keep the SystemMessage (index 0 if present).
      2. Drop oldest non-system messages until under budget.
      3. Returns the trimmed list.
    """
    if not messages:
        return messages

    # Separate system message from conversation history
    # Handle both BaseMessage objects and raw dicts from LangGraph Studio
    system_msgs = []
    convo_msgs = []
    for m in messages:
        if isinstance(m, SystemMessage):
            system_msgs.append(m)
        elif isinstance(m, dict) and m.get("type") == "system":
            system_msgs.append(m)
        else:
            convo_msgs.append(m)

    # If already under budget, return as is
    if _count_tokens(messages, encoding_name) <= max_tokens:
        return messages

    logger.info("trimming history (over %d budget)", max_tokens)

    # Greedily drop from the front of conversation history
    while convo_msgs and _count_tokens(system_msgs + convo_msgs, encoding_name) > max_tokens:
        dropped = convo_msgs.pop(0)
        logger.debug("dropped message: %s", str(dropped.content)[:60])

    trimmed = system_msgs + convo_msgs
    logger.info(
        "trimmed to %d messages / %d tokens",
        len(trimmed),
        _count_tokens(trimmed, encoding_name),
    )
    return trimmed
