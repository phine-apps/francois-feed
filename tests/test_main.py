# Copyright (c) 2026 phine-apps
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

"""Unit tests for main.py.

Covers pure/testable functions:
  - extract_previous_items()
  - RSSGenerationError
  - create_http_session()
  - URL replacement logic (via _restore_urls helper extracted inline)
"""

import pytest
import requests

from google.genai.errors import APIError
from main import (
    RSSGenerationError,
    _search_one_query,
    create_http_session,
    execute_with_retry,
    extract_previous_items,
    get_previous_rss_content,
    plan_search_queries,
)

# ---------------------------------------------------------------------------
# Shared fixtures / sample data
# ---------------------------------------------------------------------------

RSS_WITH_ITEMS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>[AI] Some AI news</title>
      <link>https://example.com/ai-news</link>
    </item>
    <item>
      <title>[Tech] Python update</title>
      <link>https://example.com/python</link>
    </item>
  </channel>
</rss>
"""

RSS_NO_ITEMS = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Empty Feed</title>
  </channel>
</rss>
"""

MALFORMED_XML = "<<this is not valid XML>>"


# ---------------------------------------------------------------------------
# extract_previous_items()
# ---------------------------------------------------------------------------


class TestExtractPreviousItems:
    def test_returns_items_from_valid_rss(self):
        items = extract_previous_items(RSS_WITH_ITEMS)
        assert len(items) == 2
        assert items[0] == {
            "title": "[AI] Some AI news",
            "link": "https://example.com/ai-news",
        }
        assert items[1] == {
            "title": "[Tech] Python update",
            "link": "https://example.com/python",
        }

    def test_returns_empty_list_for_rss_without_items(self):
        assert extract_previous_items(RSS_NO_ITEMS) == []

    def test_returns_empty_list_for_empty_string(self):
        assert extract_previous_items("") == []

    def test_returns_empty_list_for_malformed_xml(self):
        # Should not raise; logs a warning and returns []
        assert extract_previous_items(MALFORMED_XML) == []

    def test_item_without_link_is_skipped(self):
        rss = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item><title>No link</title></item>
    <item><title>Has link</title><link>https://example.com</link></item>
  </channel>
</rss>
"""
        items = extract_previous_items(rss)
        assert len(items) == 1
        assert items[0]["title"] == "Has link"

    def test_item_without_title_is_skipped(self):
        rss = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item><link>https://example.com/no-title</link></item>
    <item><title>Has title</title><link>https://example.com/with-title</link></item>
  </channel>
</rss>
"""
        items = extract_previous_items(rss)
        assert len(items) == 1
        assert items[0]["link"] == "https://example.com/with-title"

    def test_empty_title_element_defaults_to_empty_string(self):
        rss = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <item>
      <title/>
      <link>https://example.com</link>
    </item>
  </channel>
</rss>
"""
        items = extract_previous_items(rss)
        assert items == [{"title": "", "link": "https://example.com"}]

    def test_nested_items_all_found(self):
        # Items inside deeply nested channels should still be found via .//item
        rss = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <channel>
      <item><title>Nested</title><link>https://example.com/nested</link></item>
    </channel>
  </channel>
</rss>
"""
        items = extract_previous_items(rss)
        assert len(items) == 1
        assert items[0]["title"] == "Nested"


# ---------------------------------------------------------------------------
# RSSGenerationError
# ---------------------------------------------------------------------------


class TestRSSGenerationError:
    def test_is_exception_subclass(self):
        assert issubclass(RSSGenerationError, Exception)

    def test_can_be_raised_and_caught_as_exception(self):
        with pytest.raises(Exception):
            raise RSSGenerationError("any message")

    def test_can_be_caught_by_specific_type(self):
        with pytest.raises(RSSGenerationError, match="test error"):
            raise RSSGenerationError("test error")

    def test_message_is_preserved(self):
        err = RSSGenerationError("something went wrong")
        assert str(err) == "something went wrong"

    def test_does_not_catch_unrelated_exceptions(self):
        # RSSGenerationError should NOT catch generic ValueError
        with pytest.raises(ValueError):
            try:
                raise ValueError("not an RSS error")
            except RSSGenerationError:
                pass  # should not reach here


# ---------------------------------------------------------------------------
# create_http_session()
# ---------------------------------------------------------------------------


class TestCreateHttpSession:
    def test_returns_session_instance(self):
        session = create_http_session()
        assert isinstance(session, requests.Session)

    def test_https_adapter_is_mounted(self):
        session = create_http_session()
        # The session should have an adapter for https://
        adapter = session.get_adapter("https://example.com")
        assert isinstance(adapter, requests.adapters.HTTPAdapter)

    def test_retry_is_configured_on_adapter(self):
        from urllib3.util.retry import Retry

        session = create_http_session()
        adapter = session.get_adapter("https://example.com")
        assert isinstance(adapter.max_retries, Retry)
        assert adapter.max_retries.total == 3


# ---------------------------------------------------------------------------
# URL replacement logic (inline, mirrors what generate_rss_content does)
# ---------------------------------------------------------------------------


class TestUrlRestoration:
    """Tests for the ID→URL replacement logic used in generate_rss_content."""

    @staticmethod
    def _restore(content: str, url_map: dict[str, str]) -> str:
        """Mirrors the restoration logic in generate_rss_content."""
        for source_id in sorted(url_map.keys(), key=len, reverse=True):
            content = content.replace(source_id, url_map[source_id])
        return content

    def test_single_id_replaced(self):
        content = "<link>REF_ID_1</link>"
        url_map = {"REF_ID_1": "https://example.com/a"}
        result = self._restore(content, url_map)
        assert result == "<link>https://example.com/a</link>"

    def test_longer_id_replaced_before_shorter_to_avoid_partial_match(self):
        # REF_ID_10 must not be partially matched by REF_ID_1
        content = "<link>REF_ID_10</link><guid>REF_ID_1</guid>"
        url_map = {
            "REF_ID_1": "https://example.com/one",
            "REF_ID_10": "https://example.com/ten",
        }
        result = self._restore(content, url_map)
        assert "https://example.com/ten" in result
        assert "https://example.com/one" in result
        # REF_ID_10 must NOT become "https://example.com/one0"
        assert "https://example.com/one0" not in result

    def test_multiple_ids_all_replaced(self):
        content = "REF_ID_1 REF_ID_2 REF_ID_3"
        url_map = {
            "REF_ID_1": "https://a.com",
            "REF_ID_2": "https://b.com",
            "REF_ID_3": "https://c.com",
        }
        result = self._restore(content, url_map)
        assert result == "https://a.com https://b.com https://c.com"

    def test_unknown_ids_left_unchanged(self):
        content = "<link>REF_ID_99</link>"
        url_map = {"REF_ID_1": "https://example.com"}
        result = self._restore(content, url_map)
        assert "REF_ID_99" in result


# ---------------------------------------------------------------------------
# execute_with_retry()
# ---------------------------------------------------------------------------


class TestExecuteWithRetry:
    def test_succeeds_immediately(self):
        call_count = 0

        def dummy_func(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        res = execute_with_retry(dummy_func, 5)
        assert res == 10
        assert call_count == 1

    def test_retries_on_429_api_error_and_succeeds(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("time.sleep", sleeps.append)

        call_count = 0

        def dummy_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                err = APIError(429, {"error": "RESOURCE_EXHAUSTED"})
                raise err
            return "success"

        res = execute_with_retry(
            dummy_func, max_retries=3, initial_backoff=0.01, backoff_factor=1.0
        )
        assert res == "success"
        assert call_count == 3
        assert len(sleeps) == 2

    def test_retries_on_string_matching_rate_limit_and_succeeds(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("time.sleep", sleeps.append)

        call_count = 0

        def dummy_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("API returned 429 Resource Exhausted")
            return "success"

        res = execute_with_retry(
            dummy_func, max_retries=2, initial_backoff=0.01, backoff_factor=1.0
        )
        assert res == "success"
        assert call_count == 2
        assert len(sleeps) == 1

    def test_raises_unrelated_errors_immediately(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("time.sleep", sleeps.append)

        call_count = 0

        def dummy_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Something else went wrong")

        with pytest.raises(ValueError, match="Something else went wrong"):
            execute_with_retry(
                dummy_func, max_retries=3, initial_backoff=0.01, backoff_factor=1.0
            )
        assert call_count == 1
        assert len(sleeps) == 0

    def test_raises_after_max_retries(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("time.sleep", sleeps.append)

        call_count = 0

        def dummy_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("429 resource exhausted error")

        with pytest.raises(ValueError, match="429 resource exhausted error"):
            execute_with_retry(
                dummy_func, max_retries=3, initial_backoff=0.01, backoff_factor=1.0
            )
        assert call_count == 3
        assert len(sleeps) == 2


# ---------------------------------------------------------------------------
# plan_search_queries()
# ---------------------------------------------------------------------------


class TestPlanSearchQueries:
    def test_plan_search_queries_success(self):
        class MockResponse:
            def __init__(self, text):
                self.text = text

        class MockModels:
            def generate_content(self, model, contents, config=None):
                return MockResponse('["query1", "query2"]')

        class MockClient:
            models = MockModels()

        client = MockClient()
        queries = plan_search_queries(client, "test instruction", "gemini-model")
        assert queries == ["query1", "query2"]

    def test_plan_search_queries_failure_returns_empty(self):
        class MockModels:
            def generate_content(self, model, contents, config=None):
                raise ValueError("API error")

        class MockClient:
            models = MockModels()

        client = MockClient()
        queries = plan_search_queries(client, "test instruction", "gemini-model")
        assert queries == []


# ---------------------------------------------------------------------------
# _search_one_query()
# ---------------------------------------------------------------------------


class TestSearchOneQuery:
    def test_search_one_query_success(self):
        class MockWeb:
            def __init__(self, uri, title):
                self.uri = uri
                self.title = title

        class MockChunk:
            def __init__(self, uri, title):
                self.web = MockWeb(uri, title)

        class MockMetadata:
            def __init__(self, chunks):
                self.grounding_chunks = chunks

        class MockCandidate:
            def __init__(self, chunks):
                self.grounding_metadata = MockMetadata(chunks)

        class MockResponse:
            def __init__(self, text, chunks):
                self.text = text
                self.candidates = [MockCandidate(chunks)]

        class MockModels:
            def generate_content(self, model, contents, config=None):
                return MockResponse(
                    "Summary text", [MockChunk("https://example.com/1", "Title 1")]
                )

        class MockClient:
            models = MockModels()

        client = MockClient()
        results, summary = _search_one_query(
            client, "test query", "model", "2026-07-19"
        )
        assert results == [("https://example.com/1", "Title 1")]
        assert summary == "Summary text"


# ---------------------------------------------------------------------------
# get_previous_rss_content()
# ---------------------------------------------------------------------------


class TestGetPreviousRssContent:
    def test_get_from_local_file_exists(self, tmp_path):
        f = tmp_path / "old.xml"
        f.write_text("old content", encoding="utf-8")
        content = get_previous_rss_content(gist_id=None, filepath=str(f))
        assert content == "old content"

    def test_get_from_local_file_not_exists(self, tmp_path):
        f = tmp_path / "doesnotexist.xml"
        content = get_previous_rss_content(gist_id=None, filepath=str(f))
        assert content is None

    def test_get_from_gist_success(self, monkeypatch):
        monkeypatch.setenv("GH_TOKEN", "mock_token")

        class MockResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"files": {"my_rss.xml": {"content": "gist content"}}}

        class MockSession:
            def get(self, url, headers=None, timeout=None):
                return MockResponse()

        monkeypatch.setattr("main.create_http_session", MockSession)

        content = get_previous_rss_content(gist_id="12345", filepath=None)
        assert content == "gist content"

    def test_get_from_gist_no_token(self, monkeypatch):
        monkeypatch.delenv("GH_TOKEN", raising=False)
        content = get_previous_rss_content(gist_id="12345", filepath=None)
        assert content is None
