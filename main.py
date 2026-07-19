# Copyright (c) 2026 phine-apps
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

import argparse
import json
import logging
import os
import random
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

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
_DEFAULT_GEN_MODEL = "gemini-3.5-flash"

# RSS media and atom namespace URIs
_MEDIA_NS = "http://search.yahoo.com/mrss/"
_ATOM_NS = "http://www.w3.org/2005/Atom"
ET.register_namespace("media", _MEDIA_NS)
ET.register_namespace("atom", _ATOM_NS)

# Regex to extract og:image from HTML — handles both attribute orderings
_OG_IMAGE_RE = re.compile(
    r"<meta[^>]+?(?:"
    r"property=[\"']og:image[\"'][^>]+?content=[\"']([^\"']+)[\"']"
    r"|content=[\"']([^\"']+)[\"'][^>]+?property=[\"']og:image[\"']"
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def execute_with_retry(
    func,
    *args,
    max_retries=5,
    initial_backoff=2.0,
    backoff_factor=2.0,
    **kwargs,
):
    """Executes a function (usually a Gemini API call) with exponential backoff on 429/RESOURCE_EXHAUSTED errors.

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
            is_429 = False
            if isinstance(e, APIError):
                if e.code == 429:
                    is_429 = True
            if not is_429:
                err_msg = str(e).lower()
                if (
                    "429" in err_msg
                    or "resource_exhausted" in err_msg
                    or "quota" in err_msg
                ):
                    is_429 = True

            if is_429 and attempt < max_retries - 1:
                sleep_time = (
                    initial_backoff * (backoff_factor**attempt)
                ) + random.uniform(0, 1.0)
                logger.warning(
                    f"Gemini API rate limited (429/RESOURCE_EXHAUSTED). "
                    f"Retrying in {sleep_time:.2f} seconds (attempt {attempt + 1}/{max_retries}). Error: {e}"
                )
                time.sleep(sleep_time)
            else:
                raise e


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
    (Step 1) to run category-specific parallel searches rather than a single
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
    if metadata and metadata.grounding_chunks:
        for chunk in metadata.grounding_chunks:
            if chunk.web and chunk.web.uri:
                results.append((chunk.web.uri, chunk.web.title or "No Title"))
    return results, response.text or ""


def extract_og_image(url: str, session: requests.Session) -> str | None:
    """Fetches a URL and extracts the og:image meta tag value.

    Args:
        url (str): The page URL to inspect.
        session (requests.Session): HTTP session with retry logic.

    Returns:
        str | None: The OG image URL, or None if not found or on error.
    """
    try:
        resp = session.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0 (compatible; FrancoisFeed/1.0)"},
            allow_redirects=True,
        )
        resp.raise_for_status()
        match = _OG_IMAGE_RE.search(resp.text)
        if match:
            return match.group(1) or match.group(2)
    except Exception:
        pass
    return None


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

    session = create_http_session()

    # Gather (item_element, url) pairs
    item_url_pairs: list[tuple[ET.Element, str]] = []
    for item in items:
        link_el = item.find("link")
        if link_el is not None and link_el.text:
            item_url_pairs.append((item, link_el.text))

    # Fetch OG images in parallel
    def _fetch(pair: tuple[ET.Element, str]) -> tuple[ET.Element, str | None]:
        element, url = pair
        return element, extract_og_image(url, session)

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
        serialized = serialized.replace(
            "<rss ",
            f'<rss xmlns:media="{_MEDIA_NS}" ',
            1,
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
) -> str:
    """Generates RSS content using a 4-step pipeline.

    Step 0 — Query Planning: An LLM reads the instruction and generates
        focused search queries (one per content category). This decouples
        the search strategy from the code — when the prompt changes, the
        search queries adapt automatically.
    Step 1 — Parallel Search: All queries are executed concurrently using
        Google Search grounding. Results are merged into a single deduplicated
        URL map with global REF_ID assignments.
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

        all_url_title_pairs: list[tuple[str, str]] = []
        all_summaries: list[str] = []

        quota_exhausted = False
        for i, q in enumerate(queries):
            if i > 0:
                time.sleep(1.0)
            try:
                url_title_pairs, summary = execute_with_retry(
                    _search_one_query, client, q, search_model, current_date
                )
                all_url_title_pairs.extend(url_title_pairs)
                if summary:
                    all_summaries.append(summary)
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
        session = create_http_session()

        def _resolve(pair: tuple[str, str]) -> tuple[str, str]:
            url, title = pair
            if "grounding-api-redirect" in url:
                try:
                    r = session.head(url, allow_redirects=True, timeout=5)
                    if r.status_code < 400:
                        return r.url, title
                    r = session.get(url, allow_redirects=True, timeout=5)
                    return r.url, title
                except Exception as e:
                    logger.debug(f"Failed to resolve redirect for {url}: {e}")
            return url, title

        with ThreadPoolExecutor(max_workers=10) as executor:
            resolved_pairs = list(executor.map(_resolve, unique_pairs))

        # Assign global REF_IDs to deduplicated sources
        url_map: dict[str, str] = {}
        sources_context: list[str] = []
        for i, (url, title) in enumerate(resolved_pairs, start=1):
            source_id = f"REF_ID_{i}"
            url_map[source_id] = url
            sources_context.append(f"ID: {source_id}\nTitle: {title}")

        combined_summary = "\n\n---\n\n".join(all_summaries)
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

        context_str = "\n---\n".join(sources_context)

        generation_prompt = (
            f"User Requirement: {instruction}\n\n"
            f"Search Context:\n{combined_summary}\n\n"
            f"Available Sources (ID-masked):\n{context_str}\n\n"
            f"{dedup_instr}\n"
            "Task: Generate a valid RSS 2.0 XML feed based on the requirements and context above.\n"
            "Rules:\n"
            '1. For any item based on the Search Context, you MUST use the exact ID (e.g., REF_ID_1) in BOTH the <link> and <guid isPermaLink="true"> tags. Do NOT use real URLs for these items.\n'
            "2. For items generated from your own knowledge (such as daily English vocabulary, quotes, or general knowledge items not found in the Search Context), you MUST use real, appropriate public URLs (e.g., a standard dictionary URL for vocabulary items like https://dictionary.cambridge.org/dictionary/english/<word>, or a search URL like https://www.google.com/search?q=weather+in+<city> (replacing <city> with the lowercase English name of the target city, e.g., 'tokyo') for weather). NEVER reuse or map these to any of the REF_ID_X placeholders.\n"
            "3. Add a category prefix in brackets to each <title> (e.g., '[Weather]', '[AI]', '[News]', '[Book]', '[Quote]').\n"
            "4. For <pubDate>, use the specific publication time if mentioned in the context. "
            "If not mentioned, assign a realistic, varied time within the last 24 hours so that items are not all identical.\n"
            "5. If no relevant sources or NEW updates are available for a specific topic, SKIP it entirely. "
            "NEVER generate entries like 'no news found' or 'No new announcements or updates were confirmed'.\n"
            "6. If a total number of items is requested (e.g., '30 items'), you MUST fulfill this by adding more items from the categories that DO have valid sources. "
            "Prioritize depth in active news topics over including empty/update-only entries for quiet topics.\n"
            "7. Deduplication (Internal): If multiple search results describe the same news story within this current run, merge them or choose the best one.\n"
            "8. Return ONLY raw XML. No markdown, no explanations."
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

        # Restore real URLs (sort by length descending to avoid partial matches)
        for source_id in sorted(url_map.keys(), key=len, reverse=True):
            content = content.replace(source_id, url_map[source_id])

        # Remove citation markers like [cite: ...]
        content = re.sub(r" ?\[cite:.*?\]", "", content)

        # Optional: add <media:thumbnail> from og:image (CON-4)
        if enable_thumbnails:
            logger.info("Step 3: Fetching OG images for media:thumbnail elements...")
            content = add_media_thumbnails(content)

        # Validate well-formedness (BUG-1)
        try:
            ET.fromstring(content)
        except ET.ParseError as e:
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
    args = parser.parse_args()

    instruction: str | None = os.environ.get("RSS_CONFIG_PROMPT")
    api_key: str | None = os.environ.get("GEMINI_API_KEY")
    timezone_env = os.environ.get("TZ", "UTC (not set)")

    # Model selection: specific env vars take priority; legacy GEMINI_MODEL is the fallback.
    legacy_model = os.environ.get("GEMINI_MODEL", "")
    search_model: str = os.environ.get(
        "GEMINI_SEARCH_MODEL", legacy_model or _DEFAULT_SEARCH_MODEL
    )
    gen_model: str = os.environ.get(
        "GEMINI_GEN_MODEL", legacy_model or _DEFAULT_GEN_MODEL
    )
    enable_thumbnails = (
        os.environ.get("ENABLE_MEDIA_THUMBNAILS", "false").lower() == "true"
    )

    logger.info(f"Starting RSS generation at {datetime.now()} (TZ: {timezone_env})")
    logger.info(f"Models — search: {search_model}, generation: {gen_model}")

    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    if not instruction:
        logger.warning("RSS_CONFIG_PROMPT is not set. Using default instruction.")
        instruction = "Generate a generic RSS feed update."

    previous_items: list[dict[str, str]] = []
    if not args.no_dedup:
        logger.info("Attempting to fetch previous RSS content for deduplication...")
        prev_content = get_previous_rss_content(args.gist, args.output)
        if prev_content:
            previous_items = extract_previous_items(prev_content)
            logger.info(f"Found {len(previous_items)} previous items to exclude.")
        else:
            logger.info("No previous RSS content found for deduplication.")

    try:
        rss_content = generate_rss_content(
            api_key,
            instruction,
            previous_items,
            gen_model=gen_model,
            search_model=search_model,
            enable_thumbnails=enable_thumbnails,
        )
    except RSSGenerationError as e:
        logger.error(f"RSS generation failed: {e}")
        sys.exit(1)

    if args.output:
        save_to_file(rss_content, args.output)

    if args.gist:
        update_gist(args.gist, rss_content)

    if not args.output and not args.gist:
        print(rss_content)


if __name__ == "__main__":
    main()
