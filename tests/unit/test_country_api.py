"""
Production-ready unit tests for CountryAPIClient.

Mocking strategy:
  we mock ONLY the HTTP transport layer (httpx.AsyncClient).
  real Settings load from .env, making fake credentials unnecessary.
  no live network calls, no LLM calls.


"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.tools.country_api import CountryAPIClient, CountryNotFoundError, APIUnavailableError


# Sample fixture data (mirrors the real REST Countries API schema)
GERMANY_DATA = {
    "name": {"common": "Germany", "official": "Federal Republic of Germany"},
    "capital": ["Berlin"],
    "population": 83240525,
    "currencies": {"EUR": {"name": "Euro", "symbol": "€"}},
    "languages": {"deu": "German"},
    "region": "Europe",
    "area": 357114.0,
    "flags": {"svg": "https://flagcdn.com/de.svg"},
}


def _make_response(status_code: int, data=None):
    """Helper: build a minimal mock httpx Response."""
    mock = MagicMock()
    mock.status_code = status_code
    if data is not None:
        mock.json.return_value = data
    mock.raise_for_status = MagicMock()
    return mock


@pytest.fixture
def client():
    """ CountryAPIClient that uses Settings from .env."""
    return CountryAPIClient()


class TestSuccessPath:
    @pytest.mark.asyncio
    async def test_returns_first_match_from_api(self, client):
        """Should parse and return the first result from the API JSON array."""
        mock_resp = _make_response(200, [GERMANY_DATA])

        with patch("httpx.AsyncClient") as mock_http:
            mock_http.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_resp
            )
            data = await client.fetch("Germany")

        assert data["population"] == 83240525
        assert data["capital"] == ["Berlin"]
        assert data["name"]["common"] == "Germany"

    @pytest.mark.asyncio
    async def test_fetch_populates_cache(self, client):
        """After a successful fetch, the result should be in the TTL cache."""
        mock_resp = _make_response(200, [GERMANY_DATA])

        with patch("httpx.AsyncClient") as mock_http:
            mock_http.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_resp
            )
            await client.fetch("Germany")

        cached = client._cache.get("germany")
        assert cached is not None
        assert cached["population"] == 83240525


class TestCaching:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_http(self, client):
        """Second fetch for the same country must NOT make a second HTTP call."""
        mock_resp = _make_response(200, [GERMANY_DATA])

        with patch("httpx.AsyncClient") as mock_http:
            get_mock = AsyncMock(return_value=mock_resp)
            mock_http.return_value.__aenter__.return_value.get = get_mock

            first  = await client.fetch("Germany")
            second = await client.fetch("germany")  # different case → same cache key

        assert get_mock.call_count == 1, "HTTP GET should only fire once due to cache"
        assert first == second

    @pytest.mark.asyncio
    async def test_different_countries_both_fetched(self, client):
        """Two distinct countries should each trigger one HTTP call."""
        japan_data = {**GERMANY_DATA, "name": {"common": "Japan"}, "capital": ["Tokyo"]}
        mock_de  = _make_response(200, [GERMANY_DATA])
        mock_jp  = _make_response(200, [japan_data])

        with patch("httpx.AsyncClient") as mock_http:
            get_mock = AsyncMock(side_effect=[mock_de, mock_jp])
            mock_http.return_value.__aenter__.return_value.get = get_mock

            await client.fetch("Germany")
            await client.fetch("Japan")

        assert get_mock.call_count == 2


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_404_raises_country_not_found(self, client):
        """REST Countries returns 404 for unknown names → typed exception."""
        mock_resp = _make_response(404)

        with patch("httpx.AsyncClient") as mock_http:
            mock_http.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_resp
            )
            with pytest.raises(CountryNotFoundError) as exc_info:
                await client.fetch("Xyzlandia")

        assert "Xyzlandia" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_500_raises_api_unavailable(self, client):
        """A 5xx response should raise APIUnavailableError."""
        mock_resp = _make_response(500)

        with patch("httpx.AsyncClient") as mock_http:
            mock_http.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_resp
            )
            with pytest.raises(APIUnavailableError):
                await client.fetch("Germany")

    @pytest.mark.asyncio
    async def test_timeout_retries_then_raises(self, client):
        """
        A timeout should retry once (per api_max_retries=1 in config)
        then raise APIUnavailableError.
        """
        import httpx as _httpx

        with patch("httpx.AsyncClient") as mock_http:
            get_mock = AsyncMock(side_effect=_httpx.TimeoutException("timed out"))
            mock_http.return_value.__aenter__.return_value.get = get_mock

            with pytest.raises(APIUnavailableError):
                await client.fetch("Germany")

        # Should have tried 2 times: initial + 1 retry
        assert get_mock.call_count == 2

    @pytest.mark.asyncio
    async def test_country_not_found_not_cached(self, client):
        """A 404 result should NOT be cached since the user may fix the name next time."""
        mock_resp = _make_response(404)

        with patch("httpx.AsyncClient") as mock_http:
            mock_http.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_resp
            )
            with pytest.raises(CountryNotFoundError):
                await client.fetch("BadName")

        assert client._cache.get("badname") is None
