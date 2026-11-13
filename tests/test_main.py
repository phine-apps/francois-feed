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

import re
import xml.etree.ElementTree as ET
import xml.sax.saxutils
from typing import Any

import pytest
import requests
from google.genai.errors import APIError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from main import (
    RSSGenerationError,
    _sanitize_xml_ampersands,
    _search_one_query,
    annotate_text_with_supports,
    create_http_session,
    ensure_xml_namespaces,
    execute_with_retry,
    extract_og_image,
    extract_previous_items,
    generate_rss_content,
    get_previous_rss_content,
    inject_channel_metadata,
    is_safe_url,
    main,
    merge_rss_items,
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
        assert isinstance(adapter, HTTPAdapter)

    def test_retry_is_configured_on_adapter(self):
        session = create_http_session()
        adapter = session.get_adapter("https://example.com")
        assert isinstance(adapter, HTTPAdapter)
        assert isinstance(adapter.max_retries, Retry)
        assert adapter.max_retries.total == 3

    def test_http_adapter_is_mounted_with_retry(self):
        session = create_http_session()
        adapter = session.get_adapter("http://example.com")
        assert isinstance(adapter, HTTPAdapter)
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
            escaped_url = xml.sax.saxutils.escape(
                url_map[source_id], entities={'"': "&quot;"}
            )
            content = content.replace(source_id, escaped_url)
        content = re.sub(
            r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)",
            "&amp;",
            content,
        )
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

    def test_url_with_ampersand_is_escaped_and_parses_valid_xml(self):
        content = '<rss version="2.0"><channel><item><link>REF_ID_1</link></item></channel></rss>'
        url_map = {"REF_ID_1": "https://example.com/search?q=test&page=2"}
        result = self._restore(content, url_map)
        assert "https://example.com/search?q=test&amp;page=2" in result
        root = ET.fromstring(result)
        link = root.find(".//item/link")
        assert link is not None
        assert link.text == "https://example.com/search?q=test&page=2"

    def test_stray_ampersand_in_title_is_sanitized(self):
        content = '<rss version="2.0"><channel><item><title>Tom & Jerry</title><link>REF_ID_1</link></item></channel></rss>'
        url_map = {"REF_ID_1": "https://example.com"}
        result = self._restore(content, url_map)
        root = ET.fromstring(result)
        title = root.find(".//item/title")
        assert title is not None
        assert title.text == "Tom & Jerry"


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

    def test_retries_on_503_api_error_and_succeeds(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("time.sleep", sleeps.append)

        call_count = 0

        def dummy_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                err = APIError(
                    503,
                    {
                        "error": {
                            "code": 503,
                            "message": "This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.",
                            "status": "UNAVAILABLE",
                        }
                    },
                )
                raise err
            return "success"

        res = execute_with_retry(
            dummy_func, max_retries=3, initial_backoff=0.01, backoff_factor=1.0
        )
        assert res == "success"
        assert call_count == 3
        assert len(sleeps) == 2

    def test_retries_on_string_matching_503_unavailable_and_succeeds(self, monkeypatch):
        sleeps = []
        monkeypatch.setattr("time.sleep", sleeps.append)

        call_count = 0

        def dummy_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError(
                    "503 UNAVAILABLE - model currently experiencing high demand"
                )
            return "success"

        res = execute_with_retry(
            dummy_func, max_retries=2, initial_backoff=0.01, backoff_factor=1.0
        )
        assert res == "success"
        assert call_count == 2
        assert len(sleeps) == 1


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

        client: Any = MockClient()
        queries = plan_search_queries(client, "test instruction", "gemini-model")
        assert queries == ["query1", "query2"]

    def test_plan_search_queries_handles_conversational_preamble(self):
        class MockResponse:
            def __init__(self, text):
                self.text = text

        class MockModels:
            def generate_content(self, model, contents, config=None):
                return MockResponse(
                    'Here are the queries:\n["query1", "query2"]\nHope this helps!'
                )

        class MockClient:
            models = MockModels()

        client: Any = MockClient()
        queries = plan_search_queries(client, "test instruction", "gemini-model")
        assert queries == ["query1", "query2"]

    def test_plan_search_queries_failure_returns_empty(self):
        class MockModels:
            def generate_content(self, model, contents, config=None):
                raise ValueError("API error")

        class MockClient:
            models = MockModels()

        client: Any = MockClient()
        queries = plan_search_queries(client, "test instruction", "gemini-model")
        assert queries == []


# ---------------------------------------------------------------------------
# _search_one_query()
# ---------------------------------------------------------------------------


class TestAnnotateTextWithSupports:
    def test_empty_or_none_inputs_return_original_text(self):
        assert annotate_text_with_supports("", None, 0) == ""
        assert annotate_text_with_supports("hello", None, 1) == "hello"
        assert annotate_text_with_supports("hello", [], 1) == "hello"
        assert annotate_text_with_supports("hello", [Any], 0) == "hello"

    def test_valid_supports_annotated_in_order(self):
        class MockSeg:
            def __init__(self, s, e, t):
                self.start_index = s
                self.end_index = e
                self.text = t

        class MockSup:
            def __init__(self, seg, idxs):
                self.segment = seg
                self.grounding_chunk_indices = idxs

        text = "Story A happened. Story B happened."
        supports = [
            MockSup(MockSeg(0, 17, "Story A happened."), [0]),
            MockSup(MockSeg(18, 35, "Story B happened."), [1]),
        ]
        result = annotate_text_with_supports(text, supports, 2)
        assert result == "Story A happened. [CHUNK_0] Story B happened. [CHUNK_1]"

    def test_japanese_unicode_offsets(self):
        class MockSeg:
            def __init__(self, s, e, t):
                self.start_index = s
                self.end_index = e
                self.text = t

        class MockSup:
            def __init__(self, seg, idxs):
                self.segment = seg
                self.grounding_chunk_indices = idxs

        text = "OpenAIが発表。Googleが公開。"
        supports = [
            MockSup(MockSeg(0, 8, "OpenAIが発表。"), [0]),
            MockSup(MockSeg(8, 18, "Googleが公開。"), [1]),
        ]
        result = annotate_text_with_supports(text, supports, 2)
        assert result == "OpenAIが発表。 [CHUNK_0]Googleが公開。 [CHUNK_1]"

    def test_repeated_japanese_phrases_annotated_separately(self):
        class MockSeg:
            def __init__(self, s, e, t):
                self.start_index = s
                self.end_index = e
                self.text = t

        class MockSup:
            def __init__(self, seg, idxs):
                self.segment = seg
                self.grounding_chunk_indices = idxs

        text = "AIが進化。さらにAIが進化。"
        # Even with approximate offsets, repeated segments should match sequentially
        supports = [
            MockSup(MockSeg(0, 1, "AIが進化。"), [0]),
            MockSup(MockSeg(2, 3, "AIが進化。"), [1]),
        ]
        result = annotate_text_with_supports(text, supports, 2)
        assert result == "AIが進化。 [CHUNK_0]さらにAIが進化。 [CHUNK_1]"

    def test_japanese_utf8_byte_offsets_without_seg_text(self):
        class MockSeg:
            def __init__(self, s, e, t):
                self.start_index = s
                self.end_index = e
                self.text = t

        class MockSup:
            def __init__(self, seg, idxs):
                self.segment = seg
                self.grounding_chunk_indices = idxs

        # "あいうえお" is 5 characters (15 UTF-8 bytes); "あい" is 6 UTF-8 bytes
        text = "あいうえお"
        supports = [
            MockSup(MockSeg(0, 6, None), [0]),
            # Invalid mid-character byte offset (4) when seg_text is None should be ignored
            MockSup(MockSeg(0, 4, None), [1]),
        ]
        result = annotate_text_with_supports(text, supports, 2)
        assert result == "あい [CHUNK_0]うえお"

    def test_out_of_range_chunk_indices_ignored(self):
        class MockSeg:
            def __init__(self, s, e, t):
                self.start_index = s
                self.end_index = e
                self.text = t

        class MockSup:
            def __init__(self, seg, idxs):
                self.segment = seg
                self.grounding_chunk_indices = idxs

        text = "Valid story."
        supports = [MockSup(MockSeg(0, 12, "Valid story."), [5, -1])]
        result = annotate_text_with_supports(text, supports, 2)
        assert result == "Valid story."


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

        client: Any = MockClient()
        results, summary = _search_one_query(
            client, "test query", "model", "2026-07-19"
        )
        assert results == [("https://example.com/1", "Title 1")]
        assert summary == "Summary text"

    def test_search_one_query_with_grounding_supports(self):
        class MockWeb:
            def __init__(self, uri, title):
                self.uri = uri
                self.title = title

        class MockChunk:
            def __init__(self, uri, title):
                self.web = MockWeb(uri, title)

        class MockSeg:
            def __init__(self, s, e, t):
                self.start_index = s
                self.end_index = e
                self.text = t

        class MockSup:
            def __init__(self, seg, idxs):
                self.segment = seg
                self.grounding_chunk_indices = idxs

        class MockMetadata:
            def __init__(self, chunks, supports):
                self.grounding_chunks = chunks
                self.grounding_supports = supports

        class MockCandidate:
            def __init__(self, chunks, supports):
                self.grounding_metadata = MockMetadata(chunks, supports)

        class MockResponse:
            def __init__(self, text, chunks, supports):
                self.text = text
                self.candidates = [MockCandidate(chunks, supports)]

        class MockModels:
            def generate_content(self, model, contents, config=None):
                chunks = [MockChunk("https://example.com/item", "Item Title")]
                supports = [MockSup(MockSeg(0, 12, "Summary text"), [0])]
                return MockResponse("Summary text", chunks, supports)

        class MockClient:
            models = MockModels()

        client: Any = MockClient()
        results, summary = _search_one_query(
            client, "test query", "model", "2026-07-19"
        )
        assert results == [("https://example.com/item", "Item Title")]
        assert summary == "Summary text [CHUNK_0]"

    def test_search_one_query_remaps_indices_when_empty_chunk_skipped(self):
        class MockWeb:
            def __init__(self, uri, title):
                self.uri = uri
                self.title = title

        class MockChunk:
            def __init__(self, uri, title):
                self.web = MockWeb(uri, title) if uri is not None else None

        class MockSeg:
            def __init__(self, s, e, t):
                self.start_index = s
                self.end_index = e
                self.text = t

        class MockSup:
            def __init__(self, seg, idxs):
                self.segment = seg
                self.grounding_chunk_indices = idxs

        class MockMetadata:
            def __init__(self, chunks, supports):
                self.grounding_chunks = chunks
                self.grounding_supports = supports

        class MockCandidate:
            def __init__(self, chunks, supports):
                self.grounding_metadata = MockMetadata(chunks, supports)

        class MockResponse:
            def __init__(self, text, chunks, supports):
                self.text = text
                self.candidates = [MockCandidate(chunks, supports)]

        class MockModels:
            def generate_content(self, model, contents, config=None):
                chunks = [
                    MockChunk(None, "Empty Chunk"),
                    MockChunk("https://example.com/valid", "Valid Title"),
                ]
                # Original chunk index 1 corresponds to filtered results index 0
                supports = [MockSup(MockSeg(0, 12, "Summary text"), [0, 1])]
                return MockResponse("Summary text", chunks, supports)

        class MockClient:
            models = MockModels()

        client: Any = MockClient()
        results, summary = _search_one_query(
            client, "test query", "model", "2026-07-19"
        )
        assert results == [("https://example.com/valid", "Valid Title")]
        assert summary == "Summary text [CHUNK_0]"


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


# ---------------------------------------------------------------------------
# is_safe_url()
# ---------------------------------------------------------------------------


class TestIsSafeUrl:
    def test_public_http_and_https_are_safe(self):
        assert is_safe_url("https://example.com/news") is True
        assert is_safe_url("http://example.org/test") is True

    def test_localhost_and_loopback_are_rejected(self):
        assert is_safe_url("http://localhost:8080") is False
        assert is_safe_url("http://127.0.0.1:8000") is False
        assert is_safe_url("http://127.0.0.2") is False

    def test_cloud_metadata_and_internal_domains_are_rejected(self):
        assert is_safe_url("http://169.254.169.254/latest/meta-data") is False
        assert is_safe_url("http://metadata.google.internal/computeMetadata") is False
        assert is_safe_url("http://service.internal") is False
        assert is_safe_url("http://host.local") is False

    def test_non_http_schemes_are_rejected(self):
        assert is_safe_url("ftp://example.com") is False
        assert is_safe_url("file:///etc/passwd") is False
        assert is_safe_url("javascript:alert(1)") is False
        assert is_safe_url("") is False


# ---------------------------------------------------------------------------
# generate_rss_content()
# ---------------------------------------------------------------------------


class TestGenerateRssContent:
    def test_malformed_xml_logs_snippet_and_raises(self, monkeypatch, caplog):
        malformed = (
            '<rss version="2.0"><channel>\n'
            "<title>Feed</title>\n"
            "<item><title>Item 1</title><link>REF_ID_1</link><description>bad <tag></description></item>\n"
            "</channel></rss>"
        )

        class MockResponse:
            def __init__(self, text):
                self.text = text

        class MockModels:
            def generate_content(self, model, contents, config=None):
                return MockResponse(malformed)

        class MockClient:
            def __init__(self, api_key):
                self.models = MockModels()

        monkeypatch.setattr("main.Client", MockClient)
        monkeypatch.setattr("main.plan_search_queries", lambda *args, **kwargs: ["q1"])
        monkeypatch.setattr(
            "main._search_one_query",
            lambda *args, **kwargs: ([("https://example.com", "Title")], "Summary"),
        )

        import logging

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RSSGenerationError, match="Generated XML is malformed"):
                generate_rss_content(
                    api_key="mock_key",
                    instruction="mock prompt",
                )

        assert "Malformed XML snippet around line" in caplog.text
        assert "bad <tag>" in caplog.text

    def test_bracketed_ids_in_links_and_citations_in_descriptions_cleaned(
        self, monkeypatch
    ):
        xml_template = (
            '<rss version="2.0"><channel>\n'
            "<title>Feed</title>\n"
            "<item>\n"
            "  <title>Item 1</title>\n"
            "  <link>[REF_ID_1]</link>\n"
            '  <guid isPermaLink="true">[REF_ID_1]</guid>\n'
            "  <description>Story summary with [REF_ID_1] citation and [cite: 1].</description>\n"
            "</item>\n"
            "</channel></rss>"
        )

        class MockResponse:
            def __init__(self, text):
                self.text = text

        captured_prompt = {}

        class MockModels:
            def generate_content(self, model, contents, config=None):
                captured_prompt["contents"] = contents
                return MockResponse(xml_template)

        class MockClient:
            def __init__(self, api_key):
                self.models = MockModels()

        monkeypatch.setattr("main.Client", MockClient)
        monkeypatch.setattr(
            "main.plan_search_queries", lambda *args, **kwargs: ["q_topic1"]
        )
        monkeypatch.setattr(
            "main._search_one_query",
            lambda *args, **kwargs: (
                [("https://example.com/target", "Target Article")],
                "Target summary text [CHUNK_0]",
            ),
        )

        result = generate_rss_content(
            api_key="mock_key",
            instruction="mock prompt",
        )

        root = ET.fromstring(result)
        item = root.find(".//item")
        assert item is not None
        link = item.findtext("link")
        guid = item.findtext("guid")
        desc = item.findtext("description")

        assert link == "https://example.com/target"
        assert guid == "https://example.com/target"
        assert desc is not None
        assert "[REF_ID_1]" not in desc
        assert "[cite:" not in desc
        assert "Story summary with citation and ." in desc or "citation" in desc

        prompt_str = captured_prompt.get("contents", "")
        assert "### Search Topic: q_topic1" in prompt_str
        assert "ID: REF_ID_1 | Title: Target Article" in prompt_str
        assert "Target summary text [REF_ID_1]" in prompt_str


# ---------------------------------------------------------------------------
# Feeder compatibility extensions
# ---------------------------------------------------------------------------


class TestEnsureXmlNamespaces:
    def test_adds_missing_namespaces_when_prefixes_used(self):
        xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<rss version="2.0"><channel><item>'
            "<content:encoded><![CDATA[<p>Hello</p>]]></content:encoded>"
            "<dc:creator>Example</dc:creator>"
            "</item></channel></rss>"
        )
        result = ensure_xml_namespaces(xml)
        assert 'xmlns:content="http://purl.org/rss/1.0/modules/content/"' in result
        assert 'xmlns:dc="http://purl.org/dc/elements/1.1/"' in result
        root = ET.fromstring(result)
        assert root.tag == "rss"

    def test_does_not_duplicate_existing_namespace(self):
        xml = (
            '<rss xmlns:content="http://purl.org/rss/1.0/modules/content/" '
            'version="2.0"><channel><item>'
            "<content:encoded>Hi</content:encoded>"
            "</item></channel></rss>"
        )
        result = ensure_xml_namespaces(xml)
        assert result.count("xmlns:content=") == 1


class TestSanitizeXmlAmpersands:
    def test_escapes_outside_cdata_and_preserves_inside_cdata(self):
        xml = (
            "<item><title>A & B</title>"
            "<content:encoded><![CDATA[<p>A & B <a href='https://x.com?a=1&b=2'>link</a></p>]]></content:encoded>"
            "</item>"
        )
        result = _sanitize_xml_ampersands(xml)
        assert "<title>A &amp; B</title>" in result
        assert (
            "<![CDATA[<p>A & B <a href='https://x.com?a=1&b=2'>link</a></p>]]>"
            in result
        )


class TestInjectChannelMetadata:
    def test_injects_image_and_atom_self_link(self):
        result = inject_channel_metadata(
            RSS_WITH_ITEMS,
            feed_image_url="https://example.com/icon.png",
            feed_self_url="https://example.com/feed.xml",
        )
        root = ET.fromstring(result)
        channel = root.find("channel")
        assert channel is not None
        image = channel.find("image")
        assert image is not None
        assert image.findtext("url") == "https://example.com/icon.png"
        assert image.findtext("title") == "Test Feed"
        atom_links = channel.findall("{http://www.w3.org/2005/Atom}link")
        assert len(atom_links) == 1
        assert atom_links[0].get("rel") == "self"
        assert atom_links[0].get("href") == "https://example.com/feed.xml"

    def test_ignores_unsafe_urls(self):
        result = inject_channel_metadata(
            RSS_WITH_ITEMS,
            feed_image_url="http://127.0.0.1/icon.png",
            feed_self_url="file:///etc/passwd",
        )
        root = ET.fromstring(result)
        assert root.find(".//image") is None
        assert root.find(".//{http://www.w3.org/2005/Atom}link") is None


class TestMergeRssItems:
    def test_merges_non_duplicate_items_respecting_max_items(self):
        new_xml = """\
<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>[AI] New story</title>
      <link>https://example.com/new</link>
      <guid isPermaLink="true">https://example.com/new</guid>
    </item>
    <item>
      <title>[AI] Some AI news</title>
      <link>https://example.com/ai-news</link>
      <guid isPermaLink="true">https://example.com/ai-news</guid>
    </item>
  </channel>
</rss>
"""
        merged = merge_rss_items(new_xml, RSS_WITH_ITEMS, max_items=3)
        root = ET.fromstring(merged)
        items = root.findall(".//item")
        assert len(items) == 3
        links = [it.findtext("link") for it in items]
        assert links == [
            "https://example.com/new",
            "https://example.com/ai-news",
            "https://example.com/python",
        ]

    def test_returns_new_xml_when_previous_malformed(self):
        assert (
            merge_rss_items(RSS_WITH_ITEMS, MALFORMED_XML, max_items=10)
            == RSS_WITH_ITEMS
        )


class TestExtractOgImageFallback:
    def test_falls_back_to_twitter_image(self):
        closed = []

        class MockResp:
            encoding = "utf-8"
            url = "https://example.com/post"
            history = []

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size=16384):
                yield b'<html><head><meta name="twitter:image" content="https://example.com/tw.jpg"></head></html>'

            def close(self):
                closed.append(True)

        class MockSession:
            def get(self, *args, **kwargs):
                return MockResp()

        url = extract_og_image("https://example.com/post", MockSession())  # type: ignore[arg-type]
        assert url == "https://example.com/tw.jpg"
        assert closed == [True]

    def test_rejects_unsafe_redirect_target_and_closes_response(self):
        closed = []

        class MockResp:
            encoding = "utf-8"
            url = "http://127.0.0.1:8080/internal"
            history = []

            def raise_for_status(self):
                return None

            def iter_content(self, chunk_size=16384):
                yield b'<html><head><meta property="og:image" content="https://example.com/og.jpg"></head></html>'

            def close(self):
                closed.append(True)

        class MockSession:
            def get(self, *args, **kwargs):
                return MockResp()

        url = extract_og_image("https://example.com/redirect-to-local", MockSession())  # type: ignore[arg-type]
        assert url is None
        assert closed == [True]


class TestMainGistIdEnvFallback:
    def test_uses_gist_id_from_env_when_cli_flag_omitted(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "mock_key")
        monkeypatch.setenv("RSS_CONFIG_PROMPT", "mock prompt")
        monkeypatch.setenv("GIST_ID", "env_gist_123")
        monkeypatch.setattr("sys.argv", ["main.py"])

        fetched_gist_ids: list[str | None] = []
        updated_gists: list[tuple[str, str]] = []

        def mock_get_prev(gist_id, filepath):
            fetched_gist_ids.append(gist_id)
            return None

        def mock_update_gist(gist_id, content):
            updated_gists.append((gist_id, content))

        monkeypatch.setattr("main.get_previous_rss_content", mock_get_prev)
        monkeypatch.setattr(
            "main.generate_rss_content", lambda *args, **kwargs: RSS_WITH_ITEMS
        )
        monkeypatch.setattr("main.update_gist", mock_update_gist)

        main()

        assert fetched_gist_ids == ["env_gist_123"]
        assert updated_gists == [("env_gist_123", RSS_WITH_ITEMS)]
