# Inspect Context.dev

Context.dev web search for Inspect AI evaluations.

This package provides an Inspect tool through its public extension API. It does
not replace or change the built-in `inspect_ai.tool.web_search` providers.

## Install

Requires Python 3.11+ and Inspect AI 0.3.290+.

```sh
uv pip install .
export CONTEXT_DEV_API_KEY="your-api-key"
```

Get an API key from [Context.dev](https://context.dev/signup). Searches consume
Context.dev credits, including additional credits for returned highlights; see
[current pricing](https://docs.context.dev/account/credits).

## Use in an evaluation

```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, use_tools
from inspect_context_dev import web_search


@task
def research():
    return Task(
        dataset=[Sample(input="Find the current Inspect AI web search documentation.")],
        solver=[
            use_tools(
                web_search(num_results=3, include_domains=["inspect.aisi.org.uk"])
            ),
            generate(),
        ],
    )
```

Run with `inspect eval research.py --model <your-model>`. The model needs its
usual credentials as well as `CONTEXT_DEV_API_KEY` for search. The package
registers `inspect_context_dev/web_search` through the `inspect_ai` entry point,
so Inspect can record and reconstruct the tool in evaluation logs.

The model supplies only `query`. Configure search behavior when creating the
tool:

| Option | Default | Meaning |
| --- | --- | --- |
| `num_results` | `10` | Maximum results, from 1 to 100. Fewer may be returned. |
| `max_characters` | `2000` | Passage or description characters per result. |
| `include_domains` | `None` | Restrict results to these domains, up to 100. |
| `exclude_domains` | `None` | Exclude these domains, up to 100. |
| `freshness` | `None` | `last_24_hours`, `last_week`, `last_month`, or `last_year`. |
| `max_connections` | `10` | Concurrent searches across an Inspect evaluation. |
| `timeout` | `60` | Total seconds, including concurrency waits and retries. |

Each result is a `ContentText` block with a `UrlCitation` preserving the source
URL, title, and returned passage. Successful highlights contain page passages.
Titles and URLs also appear in the text, including for model APIs that do not
forward citation metadata from tool results.
When highlights fail or are empty, the result explicitly labels the fallback
as a search description; it does not claim that the page was read successfully.
Only successful page highlights populate citation `cited_text`; description
fallbacks retain the URL and title without quoting unread page content.
Partial responses are labeled, and malformed responses fail instead of becoming
empty search results.

Searches retry connection failures, HTTP 408, 429, and 5xx responses up to five
attempts within the total deadline. Authentication, permission, billing, and
other 4xx errors are not retried. Unresolved HTTP failures fail the evaluation
sample; invalid queries are reported to the model as a `ToolError`. There is no
fallback to another search service. Each call closes its HTTP client, including
when canceled. The API key is sent in an Authorization header.

This is not a Google CSE compatibility layer: Google-specific pagination,
ranking, image search, or engine configuration are not emulated. See the
[Context.dev search reference](https://docs.context.dev/api-reference/web-scraping/search).

## Development

```sh
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest -m "not live"
uv build
```

The offline tests exercise HTTP behavior on asyncio and Trio and an Inspect
evaluation with a deterministic model. They do not establish live search quality.
To run the real Context.dev API through an Inspect evaluation and save a
repeatable `.eval` log plus `live-result.json`:

```sh
uv run pytest -m live tests/test_live.py --basetemp=/tmp/inspect-context-live
```

The live test requires `CONTEXT_DEV_API_KEY`, consumes credits, and does not
require a paid model API. Inspect's deterministic model issues a real search
tool call; the results and citations come from Context.dev.
