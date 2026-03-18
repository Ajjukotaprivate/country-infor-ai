"""
Async REST Countries API client.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

from src.core.config import get_settings

logger = logging.getLogger(__name__)


# Custom exceptions
class CountryNotFoundError(Exception):
    """Raised when the REST Countries API returns 404."""
    def __init__(self, country_name: str) -> None:
        super().__init__(f"Country not found: '{country_name}'")
        self.country_name = country_name


class APIUnavailableError(Exception):
    """Raised when the REST Countries API returns 5xx or is unreachable."""
    def __init__(self, detail: str = "") -> None:
        super().__init__(f"REST Countries API unavailable. {detail}")


# Simple TTL cache
class _TTLCache:
    """Thread-safe(ish) in-memory TTL cache for country API responses."""

    def __init__(self, ttl_seconds: int) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        timestamp, value = entry
        if time.monotonic() - timestamp > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (time.monotonic(), value)


# API Client
class CountryAPIClient:
    """
    Async client for https://restcountries.com/v3.1/name/{country}

    Example Usage:
    client = CountryAPIClient()
    data = await client.fetch("Germany")
    """

    def __init__(self) -> None:
        cfg = get_settings()
        self._base_url = cfg.countries_api_base
        self._timeout = cfg.api_timeout
        self._max_retries = cfg.api_max_retries
        self._cache = _TTLCache(ttl_seconds=cfg.cache_ttl_seconds)

    async def fetch(self, country_name: str) -> dict[str, Any]:
        """
        Fetch country data from the REST Countries API.

        Returns the first match as a dict.
        Raises CountryNotFoundError or APIUnavailableError on failure.
        """
        cache_key = country_name.lower().strip()

        # check cache
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("cache hit for '%s'", country_name)
            return cached

        logger.info("fetching data for '%s'", country_name)
        url = f"{self._base_url}/name/{country_name}"

        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(url)

                if response.status_code == 404:
                    raise CountryNotFoundError(country_name)

                if response.status_code >= 500:
                    raise APIUnavailableError(f"HTTP {response.status_code}")

                response.raise_for_status()
                response_data = response.json()
                
                # Check for an exact match first to handle cases like "United States" 
                target_name = country_name.lower().strip()
                data = None
                for item in response_data:
                    name_info = item.get("name", {})
                    common = name_info.get("common", "").lower()
                    official = name_info.get("official", "").lower()
                    if common == target_name or official == target_name:
                        data = item
                        break
                
                # Fallback to the first element if no exact match is found
                if data is None:
                    data = response_data[0]
                    
                self._cache.set(cache_key, data)
                logger.info("fetched and cached data for '%s'", country_name)
                return data

            except CountryNotFoundError:
                raise

            except httpx.TimeoutException as exc:
                last_exc = exc
                logger.warning(
                    "timeout on attempt %d/%d for '%s'",
                    attempt + 1, self._max_retries + 1, country_name,
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))  # backoff

            except httpx.HTTPError as exc:
                last_exc = exc
                logger.error("http error: %s", exc)
                break

        raise APIUnavailableError(str(last_exc))
