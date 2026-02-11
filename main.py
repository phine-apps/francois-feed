# Copyright (c) 2026 phine-apps
# This software is released under the MIT License.
# http://opensource.org/licenses/mit-license.php

import argparse
import logging
import os
import sys
from datetime import datetime

import re
import requests
from dotenv import load_dotenv
from google.genai import Client
from google.genai.types import GenerateContentResponse

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


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

    try:
        response = requests.patch(
            f"https://api.github.com/gists/{gist_id}", headers=headers, json=data
        )
        response.raise_for_status()
        logger.info("Successfully updated Gist")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to update Gist: {e}")
        if hasattr(e, "response") and e.response is not None:
            logger.error(f"Response content: {e.response.text}")
        sys.exit(1)


def generate_rss_content(api_key: str, instruction: str) -> str:
    """Generates RSS content using the Gemini API with URL masking to prevent hallucinations.

    The process follows three steps:
    1. Search Phase: Get search grounding results and extract real URLs.
    2. Generation Phase: Ask Gemini to generate RSS using IDs instead of URLs.
    3. Post-processing: Replace IDs with the mapped real URLs in the output.

    Args:
        api_key (str): The Gemini API key.
        instruction (str): The instruction prompt for the model.

    Returns:
        str: The generated RSS XML content with real URLs restored.
    """
    try:
        client: Client = Client(api_key=api_key)

        # Step 1: Search Phase - Get search grounding results
        logger.info(f"Performing search to gather sources for: {instruction}")
        current_date = datetime.now().strftime("%Y-%m-%d")
        search_prompt = (
            f"Search for the latest news and information for: {instruction}. "
            f"Current date: {current_date}. "
            f"Provide a summary of the most relevant news from the last 24 hours or today."
        )
        search_response: GenerateContentResponse = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=search_prompt,
            config={"tools": [{"google_search": {}}]},
        )

        # Extract sources and map to IDs
        url_map = {}
        sources_context = []
        metadata = search_response.candidates[0].grounding_metadata

        if metadata and metadata.grounding_chunks:
            for i, chunk in enumerate(metadata.grounding_chunks):
                if chunk.web:
                    source_id = f"REF_ID_{i + 1}"
                    url = chunk.web.uri
                    title = chunk.web.title or "No Title"

                    url_map[source_id] = url
                    sources_context.append(f"ID: {source_id}\nTitle: {title}")

        if not url_map:
            logger.error(
                "No search results found with valid URLs from grounding metadata."
            )
            sys.exit(1)

        logger.info(f"Found {len(url_map)} sources. Generating RSS with ID masking...")

        # Step 2: Generation Phase - Generate RSS using IDs
        context_str = "\n---\n".join(sources_context)
        search_summary = search_response.text
        generation_prompt = (
            f"User Requirement: {instruction}\n\n"
            f"Detailed Search Context:\n{search_summary}\n\n"
            f"Available Sources (ID-masked):\n{context_str}\n\n"
            "Task: Generate a valid RSS 2.0 XML feed based on the requirements and context above.\n"
            "Rules:\n"
            '1. You MUST use the exact ID (e.g., REF_ID_1) in BOTH the <link> and <guid isPermaLink="true"> tags for each item.\n'
            "2. DO NOT include real URLs. ONLY use the provided IDs.\n"
            "3. Add a category prefix in brackets to each <title> (e.g., '[Weather]', '[AI]', '[News]', '[Book]', '[Quote]').\n"
            "4. For <pubDate>, use the specific publication time if mentioned in the context. "
            "If not mentioned, assign a realistic, varied time within the last 24 hours so that items are not all identical.\n"
            "5. If no relevant sources or NEW updates are available for a specific topic, SKIP it entirely. "
            "NEVER generate entries like 'no news found' or 'No new announcements or updates were confirmed'.\n"
            "6. If a total number of items is requested (e.g., '30 items'), you MUST fulfill this by adding more items from the categories that DO have valid sources. "
            "Prioritize depth in active news topics over including empty/update-only entries for quiet topics.\n"
            "7. Deduplication: If multiple sources describe the same news story, merge them into a single entry or choose the most relevant one. DO NOT include the same story multiple times, even in different categories.\n"
            "8. Return ONLY raw XML. No markdown, no explanations."
        )

        rss_response: GenerateContentResponse = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=generation_prompt,
            config={
                "system_instruction": (
                    "You are an RSS feed generation engine. "
                    "You strictly use the provided Source IDs in <link> tags to ensure data integrity. "
                    "Return ONLY standard RSS 2.0 XML data. Markdown is strictly prohibited."
                )
            },
        )

        if not rss_response.text:
            logger.error("Gemini API returned empty response during generation phase.")
            sys.exit(1)

        # Step 3: Post-processing - Restore real URLs
        content = rss_response.text

        # Cleanup: remove markdown code blocks if present
        content = re.sub(r"```xml\n?|```", "", content).strip()

        # Restore URLs by replacing IDs
        # Sort IDs by length descending to prevent partial replacement (e.g., REF_ID_1 matching REF_ID_11)
        for source_id in sorted(url_map.keys(), key=len, reverse=True):
            url = url_map[source_id]
            content = content.replace(source_id, url)

        # Remove citation markers like [cite: ...]
        content = re.sub(r" ?\[cite:.*?\]", "", content)

        return content

    except Exception as e:
        logger.error(f"An error occurred during content generation: {e}")
        sys.exit(1)


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


def main():
    """Main function to generate RSS feed.

    Parses command-line arguments, retrieves environment variables,
    generates RSS content using Gemini API, and handles output (stdout, file, or Gist).
    """
    parser = argparse.ArgumentParser(description="Generate RSS feed using Gemini API.")
    parser.add_argument("-o", "--output", help="Output to a file")
    parser.add_argument("-g", "--gist", help="Update a GitHub Gist with the given ID")
    args = parser.parse_args()

    instruction: str | None = os.environ.get("RSS_CONFIG_PROMPT")
    api_key: str | None = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)

    if not instruction:
        logger.warning("RSS_CONFIG_PROMPT is not set. Using default instruction.")
        instruction = "Generate a generic RSS feed update."

    rss_content = generate_rss_content(api_key, instruction)

    if args.output:
        save_to_file(rss_content, args.output)

    if args.gist:
        update_gist(args.gist, rss_content)

    if not args.output and not args.gist:
        print(rss_content)


if __name__ == "__main__":
    main()
