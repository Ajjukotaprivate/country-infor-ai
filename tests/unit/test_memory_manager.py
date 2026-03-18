"""
Unit tests for the token-based memory trimming utility.
No LLM calls, these are pure logic tests.
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.core.memory_manager import _count_tokens, trim_messages


def make_messages():
    return [
        SystemMessage(content="You are a country assistant."),
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there! How can I help?"),
        HumanMessage(content="What is the capital of France?"),
        AIMessage(content="The capital of France is Paris."),
    ]


class TestCountTokens:
    def test_empty_returns_zero(self):
        assert _count_tokens([]) == 0

    def test_single_message_positive(self):
        msgs = [HumanMessage(content="Hello")]
        assert _count_tokens(msgs) > 0

    def test_more_content_more_tokens(self):
        short = [HumanMessage(content="Hi")]
        long = [HumanMessage(content="What is the capital of France and its population?")]
        assert _count_tokens(long) > _count_tokens(short)


class TestTrimMessages:
    def test_under_budget_unchanged(self):
        msgs = make_messages()
        result = trim_messages(msgs, max_tokens=10000)
        assert result == msgs

    def test_system_message_preserved(self):
        msgs = make_messages()
        # Very tight budget so we should still keep SystemMessage
        result = trim_messages(msgs, max_tokens=20)
        system_msgs = [m for m in result if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 1

    def test_empty_input(self):
        assert trim_messages([]) == []

    def test_trimmed_fits_budget(self):
        msgs = make_messages()
        budget = 50
        result = trim_messages(msgs, max_tokens=budget)
        assert _count_tokens(result) <= budget

    def test_order_preserved_after_trim(self):
        msgs = make_messages()
        result = trim_messages(msgs, max_tokens=100)
        # System message should always be first
        if isinstance(result[0], SystemMessage):
            assert True  # OK
        else:
            pytest.fail("SystemMessage not first after trim")
