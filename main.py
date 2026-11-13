# Copyright (c) 2026 phine-apps
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

import argparse
import ipaddress
import json
import logging
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
import xml.sax.saxutils
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Callable, TypeVar
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from google.genai import Client
from google.genai.errors import APIError
from google.genai.types import GenerateContentResponse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class RSSGenerationError(Exception):
    """Raised when RSS content generation fails for any reason.

    Using a custom exception instead of sys.exit() keeps the generation
    logic independently testable and allows callers to handle errors
    gracefully without coupling to process exit semantics.
    """


# Timeout (seconds) applied to all outbound HTTP requests
_REQUEST_TIMEOUT = 30

# Default Gemini models — all overridable via environment variables.
# GEMINI_SEARCH_MODEL: used for query planning (Step 0) and web search (Step 1).
# GEMINI_GEN_MODEL:    used for RSS XML generation (Step 2).
# GEMINI_MODEL:        legacy single-model override; used as fallback for both above.
_DEFAULT_SEARCH_MODEL = "gemini-2.5-flash"
_DEFAULT_GEN_MODEL = "gemini-3.8-flash"

# RSS media, atom, content, and Dublin Core namespace URIs
_MEDIA_NS = "http://search.yahoo.com/mrss/"
_ATOM_NS = "http://www.w3.org/2005/Atom"
_CONTENT_NS = "http://purl.org/rss/1.0/modules/content/"
_DC_NS = "http://purl.org/dc/elements/1.1/"

_XML_NAMESPACES: dict[str, str] = {
    "media": _MEDIA_NS,
    "atom": _ATOM_NS,
    "content": _CONTENT_NS,
    "dc": _DC_NS,
}
for _prefix, _uri in _XML_NAMESPACES.items():
    ET.register_namespace(_prefix, _uri)

# Regexes to extract og:image (primary) and twitter:image (fallback) from HTML
_OG_IMAGE_RE = re.compile(
    r"<meta[^>]+?(?:"
    r"property=[\"']og:image[\"'][^>]+?content=[\"']([^\"']+)[\"']"
    r"|content=[\"']([^\"']+)[\"'][^>]+?property=[\"']og:image[\"']"
    r")",
    re.IGNORECASE | re.DOTALL,
)
_TWITTER_IMAGE_RE = re.compile(
    r"<meta[^>]+?(?:"
    r"(?:name|property)=[\"']twitter:image[\"'][^>]+?content=[\"']([^\"']+)[\"']"
    r"|content=[\"']([^\"']+)[\"'][^>]+?(?:name|property)=[\"']twitter:image[\"']"
    r")",
    re.IGNORECASE | re.DOTALL,
)


def create_http_session() -> requests.Session:
    """Creates a requests.Session with automatic retry and exponential backoff.

    Retries up to 3 times on transient server errors (5xx) and rate
    limiting (429), with a 1-second base backoff factor between attempts.

    Returns:
        requests.Session: A configured session ready for use.
    """
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "PATCH"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


# Load environment variables from .env file
load_dotenv()

# Apply timezone from environment if configured
if hasattr(time, "tzset") and "TZ" in os.environ:
    time.tzset()


def is_safe_url(url: str) -> bool:
    """Validates that a URL is a safe public HTTP/HTTPS URL, mitigating SSRF risks.

    Rejects non-HTTP(S) schemes, localhost, private/loopback/link-local IP addresses,
    and internal domain names.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if safe, False otherwise.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in ("http", "https"):
            return False
        host = parsed.hostname
        if not host:
            return False
        host_lower = host.lower()
        if host_lower in (
            "localhost",
            "metadata.google.internal",
        ) or host_lower.endswith((".local", ".internal", ".localhost")):
            return False
        try:
            ip = ipaddress.ip_address(host)
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_unspecified
            ):
                return False
        except ValueError:
            pass
        return True
    except Exception:
        return False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


T = TypeVar("T")


def execute_with_retry(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
    backoff_factor: float = 2.0,
    **kwargs: Any,
) -> T:
    """Executes a function (usually a Gemini API call) with exponential backoff on transient errors (429, 503, 5xx, etc.).

    Args:
        func: The callable to run.
        max_retries (int): Maximum number of attempts.
        initial_backoff (float): The initial delay in seconds.
        backoff_factor (float): Multiplier applied to the backoff delay.
        *args, **kwargs: Arguments passed to the function.

    Returns:
        The return value of func.
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            is_transient = False
            if isinstance(e, APIError):
                if e.code in (429, 500, 502, 503, 504):
                    is_transient = True
                elif getattr(e, "status", None) in (
                    "RESOURCE_EXHAUSTED",
                    "UNAVAILABLE",
                    "INTERNAL",
                    "DEADLINE_EXCEEDED",
                ):
                    is_transient = True
            if not is_transient:
                err_msg = str(e).lower()
                transient_keywords = (
                    "429",
                    "503",
                    "500",
                    "502",
                    "504",
                    "resource_exhausted",
                    "quota",
                    "unavailable",
                    "high demand",
                    "internal server error",
                    "temporarily unavailable",
                )
                if any(kw in err_msg for kw in transient_keywords):
                    is_transient = True

            if is_transient and attempt < max_retries - 1:
                sleep_time = (
                    initial_backoff * (backoff_factor**attempt)
                ) + random.uniform(0, 1.0)
                logger.warning(
                    f"Gemini API transient error encountered. "
                    f"Retrying in {sleep_time:.2f} seconds (attempt {attempt + 1}/{max_retries}). Error: {e}"
                )
                time.sleep(sleep_time)
            else:
                raise e

    raise RuntimeError("Unreachable")


def update_gist(gist_id: str, content: str) -> None:
    """Updates a GitHub Gist with the provided content.

    This function updates the specified Gist with the new content provided.
    It requires the `GH_TOKEN` environment variable to be set.

    Args:
        gist_id (str): The ID of the Gist to update.
        content (str): The new content to write to the Gist.

    Raises:
        SystemExit: If the GH_TOKEN environment variable is not set or if the request fails.
    """
    token = os.environ.get("GH_TOKEN")
    if not token:
        logger.error("GH_TOKEN environment variable is not set.")
        sys.exit(1)

    headers = {"Authorization": f"token {token}"}
    data = {"files": {"my_rss.xml": {"content": content}}}

    session = create_http_session()
    try:
        response = session.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers=headers,
            json=data,
            timeout=_REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        logger.info("Successfully updated Gist")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to update Gist: {e}")
        if hasattr(e, "response") and e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        sys.exit(1)


def get_previous_rss_content(gist_id: str | None, filepath: str | None) -> str | None:
    """Retrieves the previous RSS content from a Gist or a local file.

    Args:
        gist_id (str | None): The ID of the Gist to fetch from.
        filepath (str | None): The path to the local file to read from.

    Returns:
        str | None: The RSS content if successfully retrieved, None otherwise.
    """
    if gist_id:
        token = os.environ.get("GH_TOKEN")
        if not token:
            logger.warning("GH_TOKEN not set, cannot fetch previous Gist content.")
            return None

        headers = {"Authorization": f"token {token}"}
        session = create_http_session()
        try:
            response = session.get(
                f"https://api.github.com/gists/{gist_id}",
                headers=headers,
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            gist_data = response.json()
            if "my_rss.xml" in gist_data.get("files", {}):
                return gist_data["files"]["my_rss.xml"].get("content")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to fetch previous Gist content: {e}")
            return None

    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, encoding="utf-8") as f:
                return f.read()
        except OSError as e:
            logger.warning(f"Failed to read previous RSS from file: {e}")
            return None

    return None


def extract_previous_items(rss_content: str) -> list[dict[str, str]]:
    """Parses an RSS XML string and extracts item titles and links.

    Args:
        rss_content (str): The RSS XML content to parse.

    Returns:
        list[dict[str, str]]: A list of dictionaries, each containing 'title' and 'link'.
    """
    if not rss_content:
        return []

    try:
        root = ET.fromstring(rss_content)
        items = []
        for item in root.findall(".//item"):
            title = item.find("title")
            link = item.find("link")
            if title is not None and link is not None:
                items.append({"title": title.text or "", "link": link.text or ""})
        return items
    except ET.ParseError as e:
        logger.warning(f"Failed to parse previous RSS content: {e}")
        return []


def plan_search_queries(
    client: Client,
    instruction: str,
    model: str,
) -> list[str]:
    """Step 0: Uses an LLM to derive targeted search queries from the RSS instruction.

    Reads the RSS_CONFIG_PROMPT and generates a small set of focused Google
    search queries — one per content category. This allows the search phase
    (Step 1) to run category-specific sequential searches rather than a single
    broad query, and automatically adapts when the prompt changes.

    Args:
        client (Client): Authenticated Gemini API client.
        instruction (str): The RSS configuration prompt (RSS_CONFIG_PROMPT).
        model (str): The Gemini model to use for planning.

    Returns:
        list[str]: A list of 3-6 search query strings. Returns [] on failure.
    """
    planning_prompt = (
        "You are a search query planner for an RSS feed generator.\n"
        "Read the RSS configuration below and produce 3 to 6 specific Google "
        "search queries that together would cover all the required content categories.\n"
        "Rules:\n"
        "- Output ONLY a valid JSON array of strings. No explanation, no markdown.\n"
        "- Each query must be concrete and targeted (include site: filters where helpful).\n"
        "- English queries are preferred for global news; Japanese is fine for local topics.\n"
        "- Vocabulary / word-of-the-day categories do NOT need a search query "
        "(the model generates them from its own knowledge).\n\n"
        f"RSS Configuration:\n{instruction}"
    )
    try:
        response: GenerateContentResponse = execute_with_retry(
            client.models.generate_content,
            model=model,
            contents=planning_prompt,
        )
        text = (response.text or "").strip()
        # Strip accidental markdown fences
        text = re.sub(r"```(?:json)?\n?|```", "", text).strip()
        # Resiliently extract JSON array if wrapped in explanatory text
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and start < end:
            text = text[start : end + 1]
        queries: list[str] = json.loads(text)
        if not isinstance(queries, list) or not all(
            isinstance(q, str) for q in queries
        ):
            raise ValueError("Parsed result is not a list of strings.")
        logger.info(f"Step 0: planned {len(queries)} search queries.")
        return queries
    except Exception as e:
        logger.warning(
            f"Step 0 query planning failed ({e}); will fall back to single search."
        )
        return []


def annotate_text_with_supports(
    text: str,
    supports: list[Any] | None,
    num_chunks: int,
) -> str:
    """Annotates text with [CHUNK_idx] markers using GroundingSupport metadata.

    Inserts source chunk index tags at the end of referenced text segments,
    allowing downstream generation to precisely map summary claims to source URLs.

    Args:
        text (str): The raw generated text.
        supports (list[Any] | None): Grounding supports from Gemini API.
        num_chunks (int): Total number of grounding chunks available.

    Returns:
        str: Text annotated with [CHUNK_X] markers, or original text if no supports.
    """
    if not text or not supports or num_chunks <= 0:
        return text

    valid_supports: list[tuple[int, list[int]]] = []
    for s in supports:
        segment = getattr(s, "segment", None)
        chunk_indices = getattr(s, "grounding_chunk_indices", None)
        if not segment or not chunk_indices:
            continue

        valid_indices = [
            idx
            for idx in chunk_indices
            if isinstance(idx, int) and 0 <= idx < num_chunks
        ]
        if not valid_indices:
            continue

        seg_text = getattr(segment, "text", None)
        start_idx = getattr(segment, "start_index", None)
        end_idx = getattr(segment, "end_index", None)

        actual_end: int | None = None
        if (
            isinstance(start_idx, int)
            and isinstance(end_idx, int)
            and 0 <= start_idx < end_idx <= len(text)
        ):
            if seg_text is None or text[start_idx:end_idx] == seg_text:
                actual_end = end_idx

        if actual_end is None and seg_text and seg_text in text:
            pos = text.find(seg_text)
            if pos != -1:
                actual_end = pos + len(seg_text)

        if actual_end is not None:
            valid_supports.append((actual_end, valid_indices))

    if not valid_supports:
        return text

    # Sort descending by offset so insertion does not alter earlier offsets
    valid_supports.sort(key=lambda item: item[0], reverse=True)

    annotated = text
    for end_offset, indices in valid_supports:
        tag = " " + " ".join(f"[CHUNK_{idx}]" for idx in indices)
        annotated = annotated[:end_offset] + tag + annotated[end_offset:]

    return annotated


def _search_one_query(
    client: Client,
    query: str,
    model: str,
    current_date: str,
) -> tuple[list[tuple[str, str]], str]:
    """Runs a single Google-grounded search and returns URL/title pairs and a summary.

    Args:
        client (Client): Authenticated Gemini API client.
        query (str): The search query string.
        model (str): The Gemini model to use.
        current_date (str): ISO date string injected for recency context.

    Returns:
        tuple: ([(url, title), ...], summary_text)
    """
    prompt = (
        f"Search for the latest information about: {query}\n"
        f"Current date: {current_date}.\n"
        "Summarize the most relevant results from the last 24 hours."
    )
    response: GenerateContentResponse = client.models.generate_content(
        model=model,
        contents=prompt,
        config={"tools": [{"google_search": {}}]},
    )
    results: list[tuple[str, str]] = []
    candidates = response.candidates
    metadata = candidates[0].grounding_metadata if candidates else None
    chunks = getattr(metadata, "grounding_chunks", None) if metadata else None
    if chunks:
        for chunk in chunks:
            if chunk.web and chunk.web.uri:
                results.append((chunk.web.uri, chunk.web.title or "No Title"))

    raw_summary = response.text or ""
    supports = getattr(metadata, "grounding_supports", None) if metadata else None
    summary = annotate_text_with_supports(raw_summary, supports, len(results))
    return results, summary


def extract_og_image(url: str, session: requests.Session) -> str | None:
    """Fetches a URL and extracts the og:image meta tag value.

    Validates URL safety before fetching to mitigate SSRF and bounds response
    reading to 512KB to prevent memory exhaustion.

    Args:
        url (str): The page URL to inspect.
        session (requests.Session): HTTP session with retry logic.

    Returns:
        str | None: The OG image URL, or None if not found or on error.
    """
    if not is_safe_url(url):
        logger.debug(f"Skipping unsafe or private URL for og:image: {url}")
        return None

    try:
        resp = session.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0 (compatible; FrancoisFeed/1.0)"},
            allow_redirects=True,
            stream=True,
        )
        resp.raise_for_status()

        # Safely read up to 512KB of the head content to extract og:image
        byte_chunks: list[bytes] = []
        total_bytes = 0
        max_bytes = 512 * 1024
        for chunk in resp.iter_content(chunk_size=16384):
            if chunk:
                byte_chunks.append(chunk)
                total_bytes += len(chunk)
                if total_bytes >= max_bytes:
                    break
        text = b"".join(byte_chunks).decode(resp.encoding or "utf-8", errors="replace")

        for pattern in (_OG_IMAGE_RE, _TWITTER_IMAGE_RE):
            match = pattern.search(text)
            if match:
                img_url = match.group(1) or match.group(2)
                if img_url and is_safe_url(img_url):
                    return img_url
    except Exception:
        pass
    return None


def ensure_xml_namespaces(xml_content: str) -> str:
    """Ensures used XML namespace prefixes are declared on the root <rss> element.

    Args:
        xml_content (str): The RSS XML string.

    Returns:
        str: XML string with missing xmlns declarations added to <rss>.
    """
    missing_attrs: list[str] = []
    for prefix, uri in _XML_NAMESPACES.items():
        if f"<{prefix}:" in xml_content and f"xmlns:{prefix}=" not in xml_content:
            missing_attrs.append(f'xmlns:{prefix}="{uri}"')

    if not missing_attrs:
        return xml_content

    attrs_str = " ".join(missing_attrs)
    return re.sub(
        r"<rss(\s+|>)",
        rf"<rss {attrs_str}\1",
        xml_content,
        count=1,
    )


def _sanitize_xml_ampersands(content: str) -> str:
    """Escapes stray unescaped ampersands outside CDATA sections.

    Args:
        content (str): Raw XML string potentially containing CDATA blocks.

    Returns:
        str: XML string with stray ampersands escaped outside CDATA.
    """
    parts = re.split(r"(<!\[CDATA\[.*?\]\]>)", content, flags=re.DOTALL)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(
            r"&(?!amp;|lt;|gt;|quot;|apos;|#\d+;|#x[0-9a-fA-F]+;)",
            "&amp;",
            parts[i],
        )
    return "".join(parts)


def inject_channel_metadata(
    xml_content: str,
    feed_image_url: str | None = None,
    feed_self_url: str | None = None,
) -> str:
    """Injects optional channel-level metadata (<image> and <atom:link rel="self">).

    Args:
        xml_content (str): Valid RSS 2.0 XML string.
        feed_image_url (str | None): Optional public URL for the feed icon.
        feed_self_url (str | None): Optional canonical self URL for the feed.

    Returns:
        str: Updated RSS 2.0 XML string.
    """
    if not feed_image_url and not feed_self_url:
        return xml_content

    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        logger.warning(f"Could not parse XML for channel metadata injection: {e}")
        return xml_content

    channel = root.find("channel")
    if channel is None:
        return xml_content

    if feed_self_url and is_safe_url(feed_self_url):
        has_self_link = any(
            el.get("rel") == "self" for el in channel.findall(f"{{{_ATOM_NS}}}link")
        )
        if not has_self_link:
            atom_link = ET.SubElement(channel, f"{{{_ATOM_NS}}}link")
            atom_link.set("href", feed_self_url)
            atom_link.set("rel", "self")
            atom_link.set("type", "application/rss+xml")

    if feed_image_url and is_safe_url(feed_image_url):
        if channel.find("image") is None:
            title_el = channel.find("title")
            link_el = channel.find("link")
            image_el = ET.SubElement(channel, "image")
            url_sub = ET.SubElement(image_el, "url")
            url_sub.text = feed_image_url
            title_sub = ET.SubElement(image_el, "title")
            title_sub.text = (
                title_el.text if title_el is not None and title_el.text else "RSS Feed"
            )
            link_sub = ET.SubElement(image_el, "link")
            link_sub.text = (
                link_el.text
                if link_el is not None and link_el.text
                else feed_self_url or feed_image_url
            )

    serialized = ET.tostring(root, encoding="unicode")
    serialized = ensure_xml_namespaces(serialized)
    if not serialized.startswith("<?xml"):
        serialized = '<?xml version="1.0" encoding="UTF-8"?>\n' + serialized
    return serialized


def merge_rss_items(
    new_xml: str,
    previous_xml: str | None,
    max_items: int = 50,
) -> str:
    """Merges previous RSS <item> entries into newly generated RSS XML.

    Preserves newly generated items at the top and appends non-duplicate
    items from previous_xml up to max_items total entries.

    Args:
        new_xml (str): Newly generated valid RSS 2.0 XML string.
        previous_xml (str | None): Previous RSS 2.0 XML string, if any.
        max_items (int): Maximum number of <item> elements to retain.

    Returns:
        str: Merged RSS 2.0 XML string, or new_xml if merging is skipped/fails.
    """
    if not previous_xml or max_items <= 0:
        return new_xml

    try:
        new_root = ET.fromstring(new_xml)
        prev_root = ET.fromstring(previous_xml)
    except ET.ParseError as e:
        logger.warning(f"Could not parse XML for feed merge: {e}")
        return new_xml

    new_channel = new_root.find("channel")
    if new_channel is None:
        return new_xml

    current_items = new_channel.findall("item")
    if len(current_items) >= max_items:
        return new_xml

    def _item_key(item: ET.Element) -> str:
        guid_el = item.find("guid")
        if guid_el is not None and guid_el.text and guid_el.text.strip():
            return guid_el.text.strip()
        link_el = item.find("link")
        if link_el is not None and link_el.text and link_el.text.strip():
            return link_el.text.strip()
        title_el = item.find("title")
        return title_el.text.strip() if title_el is not None and title_el.text else ""

    seen_keys = {k for k in (_item_key(it) for it in current_items) if k}
    added_count = 0
    for prev_item in prev_root.findall(".//item"):
        if len(current_items) + added_count >= max_items:
            break
        key = _item_key(prev_item)
        if key and key in seen_keys:
            continue
        if key:
            seen_keys.add(key)
        new_channel.append(prev_item)
        added_count += 1

    if added_count == 0:
        return new_xml

    serialized = ET.tostring(new_root, encoding="unicode")
    serialized = ensure_xml_namespaces(serialized)
    if not serialized.startswith("<?xml"):
        serialized = '<?xml version="1.0" encoding="UTF-8"?>\n' + serialized
    logger.info(
        f"Merged {added_count} previous items into feed (total: {len(current_items) + added_count})."
    )
    return serialized


def add_media_thumbnails(xml_content: str) -> str:
    """Adds <media:thumbnail> elements to RSS items by fetching og:image URLs.

    Parses the RSS XML, fetches each item's link URL in parallel (up to 5
    workers), extracts the og:image meta tag, and embeds a thumbnail element.
    Items for which fetching fails are left unchanged (graceful degradation).

    Args:
        xml_content (str): A valid RSS 2.0 XML string.

    Returns:
        str: The RSS XML with <media:thumbnail> elements added where available.
    """
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        logger.warning(f"Could not parse XML for thumbnail injection: {e}")
        return xml_content

    items = root.findall(".//item")
    if not items:
        return xml_content

    # Gather (item_element, url) pairs
    item_url_pairs: list[tuple[ET.Element, str]] = []
    for item in items:
        link_el = item.find("link")
        if link_el is not None and link_el.text:
            item_url_pairs.append((item, link_el.text))

    # Fetch OG images in parallel with thread-isolated HTTP sessions
    def _fetch(pair: tuple[ET.Element, str]) -> tuple[ET.Element, str | None]:
        element, url = pair
        thread_session = create_http_session()
        try:
            return element, extract_og_image(url, thread_session)
        finally:
            thread_session.close()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_fetch, pair): pair for pair in item_url_pairs}
        for future in as_completed(futures):
            element, og_url = future.result()
            if og_url:
                thumb = ET.SubElement(element, f"{{{_MEDIA_NS}}}thumbnail")
                thumb.set("url", og_url)

    # Serialize and ensure the media namespace is declared on the root <rss> element
    serialized = ET.tostring(root, encoding="unicode")
    if "xmlns:media" not in serialized:
        serialized = re.sub(
            r"<rss(\s+|>)",
            rf'<rss xmlns:media="{_MEDIA_NS}"\1',
            serialized,
            count=1,
        )
    if not serialized.startswith("<?xml"):
        serialized = '<?xml version="1.0" encoding="UTF-8"?>\n' + serialized
    logger.info("Added media:thumbnail elements where og:image was available.")
    return serialized


def generate_rss_content(
    api_key: str,
    instruction: str,
    previous_items: list[dict[str, str]] | None = None,
    gen_model: str = _DEFAULT_GEN_MODEL,
    search_model: str = _DEFAULT_SEARCH_MODEL,
    enable_thumbnails: bool = False,
    feed_image_url: str | None = None,
    feed_self_url: str | None = None,
) -> str:
    """Generates RSS content using a 4-step pipeline.

    Step 0 — Query Planning: An LLM reads the instruction and generates
        focused search queries (one per content category). This decouples
        the search strategy from the code — when the prompt changes, the
        search queries adapt automatically.
    Step 1 — Sequential Search: Queries are executed sequentially with a
        brief delay to mitigate API rate limits, using Google Search grounding.
        Results are merged into a single deduplicated URL map with global
        REF_ID assignments.
    Step 2 — Generation: The gen_model produces RSS 2.0 XML from the
        combined context, using ID placeholders instead of real URLs.
    Step 3 — Post-processing: IDs are restored to real URLs, citation
        markers are cleaned up, optional OG thumbnails are added, and the
        final XML is validated for well-formedness.

    Args:
        api_key (str): Gemini API key.
        instruction (str): The RSS configuration prompt (RSS_CONFIG_PROMPT).
        previous_items (list[dict[str, str]] | None): Items from the previous
            feed, used for semantic deduplication.
        gen_model (str): Model name for Step 2 generation.
        search_model (str): Model name for Steps 0 and 1.
        enable_thumbnails (bool): If True, fetch og:image and embed thumbnails.

    Returns:
        str: Validated RSS 2.0 XML with real URLs restored.
    """
    try:
        client: Client = Client(api_key=api_key)
        current_date = datetime.now().strftime("%Y-%m-%d")

        # --------------------------------------------------------------
        # Step 0: Query Planning
        # --------------------------------------------------------------
        logger.info("Step 0: Planning search queries from instruction...")
        queries = plan_search_queries(client, instruction, search_model)

        # Fallback: if planning fails, use a single broad search
        if not queries:
            queries = [
                f"Search for the latest news and information for: {instruction}. "
                f"Current date: {current_date}. "
                "Provide a summary of the most relevant news from the last 24 hours or today."
            ]

        # --------------------------------------------------------------
        # Step 1: Sequential Search
        # --------------------------------------------------------------
        logger.info(f"Step 1: Running {len(queries)} search queries sequentially...")

        query_data: list[dict[str, Any]] = []
        all_url_title_pairs: list[tuple[str, str]] = []

        quota_exhausted = False
        for i, q in enumerate(queries):
            if i > 0:
                time.sleep(1.0)
            try:
                url_title_pairs, summary = execute_with_retry(
                    _search_one_query, client, q, search_model, current_date
                )
                query_data.append(
                    {
                        "query": q,
                        "pairs": url_title_pairs,
                        "summary": summary,
                    }
                )
                all_url_title_pairs.extend(url_title_pairs)
                logger.info(f"  Query '{q[:60]}': {len(url_title_pairs)} sources")
            except Exception as e:
                err_msg = str(e).lower()
                if "quota" in err_msg or "billing" in err_msg:
                    quota_exhausted = True
                logger.warning(f"  Search failed for query '{q[:60]}': {e}")

        # Deduplicate URLs while preserving insertion order
        seen_urls: set[str] = set()
        unique_pairs: list[tuple[str, str]] = []
        for url, title in all_url_title_pairs:
            if url not in seen_urls:
                seen_urls.add(url)
                unique_pairs.append((url, title))

        if not unique_pairs:
            if quota_exhausted:
                raise RSSGenerationError(
                    "Gemini Google Search Grounding quota has been exhausted. "
                    "Please check your plan/billing details in Google AI Studio or wait for the daily quota to reset."
                )
            raise RSSGenerationError(
                "No search results found with valid URLs from any query."
            )

        # Resolve redirect URLs in parallel to obtain clean, direct links in the RSS feed
        logger.info("Resolving search grounding redirect URLs...")
        resolved_pairs: list[tuple[str, str]] = []

        def _resolve(pair: tuple[str, str]) -> tuple[str, str]:
            url, title = pair
            if "grounding-api-redirect" in url:
                thread_session = create_http_session()
                try:
                    r = thread_session.head(url, allow_redirects=True, timeout=5)
                    if r.status_code < 400:
                        return r.url, title
                    r = thread_session.get(
                        url, allow_redirects=True, timeout=5, stream=True
                    )
                    return r.url, title
                except Exception as e:
                    logger.debug(f"Failed to resolve redirect for {url}: {e}")
                finally:
                    thread_session.close()
            return url, title

        with ThreadPoolExecutor(max_workers=10) as executor:
            resolved_pairs = list(executor.map(_resolve, unique_pairs))

        # Assign global REF_IDs to deduplicated sources
        url_map: dict[str, str] = {}
        raw_url_to_id: dict[str, str] = {}
        master_sources_context: list[str] = []
        for i, (orig_pair, resolved_pair) in enumerate(
            zip(unique_pairs, resolved_pairs), start=1
        ):
            source_id = f"REF_ID_{i}"
            resolved_url, resolved_title = resolved_pair
            orig_url, _ = orig_pair
            url_map[source_id] = resolved_url
            raw_url_to_id[orig_url] = source_id
            raw_url_to_id[resolved_url] = source_id
            master_sources_context.append(f"ID: {source_id}\nTitle: {resolved_title}")

        # Build topic-grouped context with mapped REF_IDs
        category_sections: list[str] = []
        for q_item in query_data:
            q_str = q_item["query"]
            q_pairs = q_item["pairs"]
            q_summary = q_item["summary"]

            # Map chunk tags [CHUNK_0] in q_summary to [REF_ID_X]
            for chunk_idx, (p_url, _) in enumerate(q_pairs):
                ref_id = raw_url_to_id.get(p_url)
                if ref_id:
                    q_summary = q_summary.replace(f"[CHUNK_{chunk_idx}]", f"[{ref_id}]")
            # Clean up any leftover unmapped [CHUNK_\d+]
            q_summary = re.sub(r"\[CHUNK_\d+\]", "", q_summary)

            # Query-specific sources
            q_sources_lines: list[str] = []
            seen_q_ids: set[str] = set()
            for p_url, p_title in q_pairs:
                ref_id = raw_url_to_id.get(p_url)
                if ref_id and ref_id not in seen_q_ids:
                    seen_q_ids.add(ref_id)
                    q_sources_lines.append(f"  - ID: {ref_id} | Title: {p_title}")

            sources_text = "\n".join(q_sources_lines) if q_sources_lines else "  (None)"

            section = (
                f"### Search Topic: {q_str}\n"
                f"Available Sources for this topic:\n{sources_text}\n\n"
                f"Context & Findings:\n{q_summary}"
            )
            category_sections.append(section)

        combined_context = "\n\n---\n\n".join(category_sections)
        master_context_str = "\n---\n".join(master_sources_context)
        logger.info(
            f"Step 1 complete: {len(url_map)} unique sources across all queries."
        )

        # --------------------------------------------------------------
        # Step 2: Generation
        # --------------------------------------------------------------
        logger.info(f"Step 2: Generating RSS with {gen_model}...")

        # Build deduplication instruction (CON-2: no fictional "DDB" reference)
        dedup_instr = ""
        if previous_items:
            prev_items_str = "\n".join(
                [
                    f"- Title: {item['title']}\n  Link: {item['link']}"
                    for item in previous_items
                ]
            )
            dedup_instr = (
                "\n### Deduplication Rules (System-Provided)\n"
                "The following items were published in the PREVIOUS feed update "
                "and are provided automatically by the system. "
                "This IS the deduplication reference — do NOT include the same "
                "story again unless there is genuinely new information:\n"
                f"{prev_items_str}\n\n"
                "1. If a story in the current context describes the SAME event as "
                "an item above, SKIP it. Apply semantic understanding: same core "
                "news = same story, even if the headline or source differs.\n"
                "2. If a previously-covered URL now contains NEW, MORE RECENT "
                "information not covered before, you MAY include it as a new entry.\n"
            )

        generation_prompt = (
            f"User Requirement: {instruction}\n\n"
            f"Search Context by Topic:\n{combined_context}\n\n"
            f"All Available Sources (ID-masked Reference):\n{master_context_str}\n\n"
            f"{dedup_instr}\n"
            "Task: Generate a valid RSS 2.0 XML feed based on the requirements and search context above.\n"
            f'Declare xmlns:content="{_CONTENT_NS}" and xmlns:dc="{_DC_NS}" on the root <rss version="2.0"> element.\n'
            "Rules:\n"
            '1. For any item based on the Search Context, you MUST use the exact ID (e.g., REF_ID_1) in BOTH the <link> and <guid isPermaLink="true"> tags. Do NOT use real URLs for these items.\n'
            "2. Precise Source Matching: Each news story in the Search Context is annotated with its corresponding Source ID (e.g., [REF_ID_X]) and grouped under its specific Search Topic. You MUST match each news item with its EXACT corresponding Source ID from that topic. NEVER cross-assign Source IDs between different topics or unrelated articles.\n"
            "3. For items generated from your own knowledge (such as daily English vocabulary, quotes, or general knowledge items not found in the Search Context), you MUST use real, appropriate public URLs (e.g., a standard dictionary URL for vocabulary items like https://dictionary.cambridge.org/dictionary/english/<word>, or a search URL like https://www.google.com/search?q=weather+in+<city> (replacing <city> with the lowercase English name of the target city, e.g., 'tokyo') for weather). NEVER reuse or map these to any of the REF_ID_X placeholders.\n"
            "4. Add a category prefix in brackets to each <title> (e.g., '[Weather]', '[AI]', '[News]', '[Book]', '[Quote]'), AND include a matching <category> element (e.g., <category>AI</category>) in each <item>.\n"
            "5. Include a <dc:creator> element in each <item> indicating the source publication name (or 'Francois Feed AI' for self-generated knowledge items).\n"
            "6. For <pubDate>, use the specific publication time if mentioned in the context. "
            "If not mentioned, assign a realistic, varied time within the last 24 hours so that items are not all identical.\n"
            "7. If no relevant sources or NEW updates are available for a specific topic, SKIP it entirely. "
            "NEVER generate entries like 'no news found' or 'No new announcements or updates were confirmed'.\n"
            "8. If a total number of items is requested (e.g., '30 items'), you MUST fulfill this by adding more items from the categories that DO have valid sources. "
            "Prioritize depth in active news topics over including empty/update-only entries for quiet topics.\n"
            "9. Deduplication (Internal): If multiple search results describe the same news story within this current run, merge them or choose the best one.\n"
            "10. Keep <title> and <description> as plain text without HTML tags; escape '<' and '>' as '&lt;' and '&gt;'. Do NOT output citation brackets like [REF_ID_X] or [cite: ...] in <title>, <description>, or <content:encoded>.\n"
            "11. Include <content:encoded><![CDATA[...]]></content:encoded> in each <item> with clean structured HTML (<p>, <ul>, <li>, <strong>) providing a rich, readable digest for offline RSS readers like Feeder.\n"
            "12. Return ONLY raw XML. No markdown, no explanations."
        )

        rss_response: GenerateContentResponse = execute_with_retry(
            client.models.generate_content,
            model=gen_model,
            contents=generation_prompt,
            config={
                "system_instruction": (
                    "You are an RSS feed generation engine. "
                    "You strictly use the provided Source IDs in <link> tags for search-based items, and real public URLs for self-generated knowledge items (like vocabulary). "
                    "Return ONLY standard RSS 2.0 XML data. Markdown is strictly prohibited."
                )
            },
        )

        if not rss_response.text:
            raise RSSGenerationError(
                "Gemini API returned empty response during generation phase."
            )

        # --------------------------------------------------------------
        # Step 3: Post-processing
        # --------------------------------------------------------------
        content = rss_response.text

        # Remove markdown code fences if present
        content = re.sub(r"```xml\n?|```", "", content).strip()

        # Normalize any bracketed IDs inside link/guid tags: <link>[REF_ID_1]</link> -> <link>REF_ID_1</link>
        content = re.sub(
            r"(<(?:link|guid)[^>]*>)\s*\[(REF_ID_\d+)\]\s*(</(?:link|guid)>)",
            r"\1\2\3",
            content,
        )

        # Strip bracketed citation tags from text content before URL substitution
        content = re.sub(r" ?\[cite:.*?\]", "", content)
        content = re.sub(r" ?\[REF_ID_\d+\]", "", content)

        # Restore real URLs with XML entity escaping (sort by length descending to avoid partial matches)
        for source_id in sorted(url_map.keys(), key=len, reverse=True):
            raw_url = url_map[source_id]
            escaped_url = xml.sax.saxutils.escape(raw_url, entities={'"': "&quot;"})
            content = content.replace(source_id, escaped_url)

        # Sanitize any stray unescaped ampersands outside CDATA to prevent ParseError
        content = _sanitize_xml_ampersands(content)

        # Ensure any used namespace prefixes (content:, dc:, atom:, media:) are declared
        content = ensure_xml_namespaces(content)

        # Optional: add <media:thumbnail> from og:image / twitter:image (CON-4)
        if enable_thumbnails:
            logger.info("Step 3: Fetching OG images for media:thumbnail elements...")
            content = add_media_thumbnails(content)

        # Optional: inject channel-level <image> and <atom:link rel="self">
        if feed_image_url or feed_self_url:
            content = inject_channel_metadata(
                content,
                feed_image_url=feed_image_url,
                feed_self_url=feed_self_url,
            )

        # Validate well-formedness (BUG-1)
        try:
            ET.fromstring(content)
        except ET.ParseError as e:
            lines = content.splitlines()
            line_num = e.position[0] if getattr(e, "position", None) else None
            if line_num and 1 <= line_num <= len(lines):
                start = max(1, line_num - 2)
                end = min(len(lines), line_num + 2)
                snippet = "\n".join(
                    f"  {idx:4d}: {lines[idx - 1]}" for idx in range(start, end + 1)
                )
                logger.error(
                    f"Malformed XML snippet around line {line_num}:\n{snippet}"
                )
            raise RSSGenerationError(f"Generated XML is malformed: {e}") from e

        return content

    except RSSGenerationError:
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during content generation: {e}")
        raise RSSGenerationError(str(e)) from e


def save_to_file(content: str, filepath: str) -> None:
    """Saves the provided content to a file.

    Args:
        content (str): The content to save.
        filepath (str): The path to the output file.

    Raises:
        SystemExit: If a file I/O error occurs.
    """
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Successfully wrote RSS to {filepath}")
    except IOError as e:
        logger.error(f"Failed to write to file {filepath}: {e}")
        sys.exit(1)


def main() -> None:
    """Main function to generate RSS feed.

    Parses command-line arguments, retrieves environment variables,
    generates RSS content using Gemini API, and handles output (stdout, file, or Gist).
    """
    parser = argparse.ArgumentParser(description="Generate RSS feed using Gemini API.")
    parser.add_argument("-o", "--output", help="Output to a file")
    parser.add_argument("-g", "--gist", help="Update a GitHub Gist with the given ID")
    parser.add_argument("--no-dedup", action="store_true", help="Disable deduplication")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge newly generated items with previous feed items",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum number of items to retain when --merge is enabled (default: 50)",
    )
    args = parser.parse_args()

    instruction: str | None = os.environ.get("RSS_CONFIG_PROMPT")
    api_key: str | None = os.environ.get("GEMINI_API_KEY")
    system_tz = datetime.now().astimezone().tzname() or "Unknown"
    timezone_info = os.environ.get("TZ") or f"{system_tz} (system default)"

    # Model selection: specific env vars take priority; legacy GEMINI_MODEL is the fallback.
    legacy_model = os.environ.get("GEMINI_MODEL", "")
    search_model: str = (
        os.environ.get("GEMINI_SEARCH_MODEL") or legacy_model or _DEFAULT_SEARCH_MODEL
    )
    gen_model: str = (
        os.environ.get("GEMINI_GEN_MODEL") or legacy_model or _DEFAULT_GEN_MODEL
    )
    enable_thumbnails = (
        os.environ.get("ENABLE_MEDIA_THUMBNAILS", "false").lower() == "true"
    )
    merge_enabled = (
        args.merge or os.environ.get("ENABLE_FEED_MERGE", "false").lower() == "true"
    )
    max_items_raw = os.environ.get("MAX_FEED_ITEMS", "50")
    try:
        default_max_items = max(1, int(max_items_raw))
    except ValueError:
        default_max_items = 50
    max_items = args.max_items if args.max_items is not None else default_max_items
    feed_image_url = os.environ.get("FEED_IMAGE_URL") or None
    feed_self_url = os.environ.get("FEED_SELF_URL") or None

    logger.info(f"Starting RSS generation at {datetime.now()} (TZ: {timezone_info})")
    logger.info(f"Models — search: {search_model}, generation: {gen_model}")

    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    if not instruction:
        logger.warning("RSS_CONFIG_PROMPT is not set. Using default instruction.")
        instruction = "Generate a generic RSS feed update."

    prev_content: str | None = None
    previous_items: list[dict[str, str]] = []
    if not args.no_dedup or merge_enabled:
        logger.info("Attempting to fetch previous RSS content...")
        prev_content = get_previous_rss_content(args.gist, args.output)
        if prev_content and not args.no_dedup:
            previous_items = extract_previous_items(prev_content)
            logger.info(f"Found {len(previous_items)} previous items to exclude.")
        elif not prev_content:
            logger.info("No previous RSS content found.")

    try:
        rss_content = generate_rss_content(
            api_key,
            instruction,
            previous_items,
            gen_model=gen_model,
            search_model=search_model,
            enable_thumbnails=enable_thumbnails,
            feed_image_url=feed_image_url,
            feed_self_url=feed_self_url,
        )
    except RSSGenerationError as e:
        logger.error(f"RSS generation failed: {e}")
        sys.exit(1)

    if merge_enabled and prev_content:
        rss_content = merge_rss_items(rss_content, prev_content, max_items=max_items)

    if args.output:
        save_to_file(rss_content, args.output)

    if args.gist:
        update_gist(args.gist, rss_content)

    if not args.output and not args.gist:
        print(rss_content)


if __name__ == "__main__":
    main()
