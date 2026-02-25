# francois-feed

A lightweight RSS feed generator powered by Gemini API. It generates an RSS 2.0 feed based on a custom prompt and can update a GitHub Gist or save to a local file.

## Features

- **AI-Powered**: Uses Gemini 2.5 Flash to generate relevant and structured RSS content.
- **Flexible Output**: Supports stdout, local file output, and GitHub Gist updates.
- **Fast & Modern**: Built with Python and managed by `uv` for high performance and reproducible environments.
- **CI/CD Ready**: Integrated with GitHub Actions for daily automated updates.

## Setup

### Prerequisites

- [uv](https://github.com/astral-sh/uv) (recommended)
- Gemini API Key ([get one here](https://aistudio.google.com/app/apikey))
- [GitHub Gist](https://gist.github.com/) (if using Gist update feature)
- GitHub Token (Personal Access Token with `gist` scope)

> [!NOTE]
> This project runs comfortably within the **Gemini API Free Tier**. For detailed pricing and quota information, please refer to the [Gemini API Pricing](https://ai.google.dev/pricing) page.

### Installation

```bash
git clone https://github.com/phine-apps/francois-feed.git
cd francois-feed
uv sync
```

### Environment Variables

Set the following environment variables:

- `GEMINI_API_KEY`: Your Gemini API key.
- `RSS_CONFIG_PROMPT`: Instructions for the AI on what content to generate.
- `GH_TOKEN`: GitHub Personal Access Token with `gist` scope (required for `-g`).

### Configuration Example

For the `RSS_CONFIG_PROMPT` environment variable, you can use a prompt like this:

```text
Generate an RSS feed covering:
1. Global News: Top 3 international headlines (e.g., from Reuters or AP).
2. Tech Trends: Latest updates in Web Development and open-source projects.
3. Productivity: A practical tip for remote work or time management.
4. Science & Environment: Brief news on space exploration or climate research.

Format: Use professional tone. Summaries should be 200-300 characters long.
```

## Usage

Run the script using `uv`:

```bash
# Preview to stdout
uv run main.py

# Output to a local file
uv run main.py -o daily.rss

# Update a GitHub Gist
uv run main.py -g your_gist_id_here
```

> [!TIP]
> To get a Gist ID, create a new Gist at [gist.github.com](https://gist.github.com/) with a file named `my_rss.xml` containing some dummy content (e.g., "dummy"). The ID is the alphanumeric string at the end of the Gist's URL.
>
> You can also set environment variables in a `.env` file for local development. `uv` will automatically load them if configured, or you can use `export $(cat .env | xargs)` before running.

### CLI Options

- `-h, --help`: Show help message.
- `-o, --output OUTPUT`: Path to save the generated RSS xml.
- `-g, --gist GIST`: GitHub Gist ID to update.
- `--no-dedup`: Disable deduplication against previous results.

### GitHub Actions (Automation)

The project includes a workflow in `.github/workflows/daily_rss.yml` that runs daily. To use it, configure the following:

#### Secrets

The following should be configured as **GitHub Secrets**:

- `GEMINI_API_KEY`
- `RSS_CONFIG_PROMPT`
- `GH_TOKEN`
- `GIST_ID` (Passed to the script via the `-g` flag in the workflow)

#### Variables

- `SCHEDULE_HOUR_JST`: The hour (0-23) in Japan Standard Time at which you want the feed to be generated.

## Adding to RSS Readers

The following RSS readers have been verified to work correctly with Gist's raw URLs:

- **Feeder** (Android)
- **FocusReader** (Android)
- **NetNewsWire** (macOS)

1.  **Get the Raw URL**: Open your Gist and click the "Raw" button.
2.  **Remove Commit Hash**: The URL will look like `.../raw/long-hash/my_rss.xml`. Remove the `long-hash/` part to get the latest version.
    - **Correct Persistent URL**: `https://gist.githubusercontent.com/user/id/raw/my_rss.xml`
3.  **Add to Reader**: Open your app, search for this URL, and add it.

> [!NOTE]
> Some RSS readers may fail to read this URL directly. If you encounter issues, consider using one of the recommended apps above.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
