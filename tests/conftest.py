"""
tests/conftest.py
─────────────────
Shared pytest configuration and fixtures.
"""
import pytest


# Mark all tests in tests/integration/ as requiring live API access
def pytest_collection_modifyitems(config, items):
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(
                pytest.mark.skipif(
                    not _has_openai_key(),
                    reason="OPENAI_API_KEY not set, skipping integration tests",
                )
            )


def _has_openai_key() -> bool:
    import os
    from pathlib import Path
    # Check env var or .env file
    if os.environ.get("OPENAI_API_KEY"):
        return True
    env_file = Path(".env")
    if env_file.exists():
        return "OPENAI_API_KEY" in env_file.read_text()
    return False
