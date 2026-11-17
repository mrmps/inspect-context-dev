import math
import os
from typing import Literal

import anyio
import httpx
from inspect_ai.tool import ContentText, Tool, ToolError, ToolResult, UrlCitation, tool
from inspect_ai.util import concurrency
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

_ENDPOINT = "https://api.context.dev/v1/web/search"


class _Highlights(BaseModel):
    code: str
    highlights: list[str] | None = None


class _Result(BaseModel):
    url: str
    title: str | None = None
    description: str | None = None
    highlights: _Highlights


class _Response(BaseModel):
    results: list[_Result]
    partial: bool = False


def _should_retry(error: BaseException) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return status in (408, 429) or 500 <= status < 600
    return isinstance(error, httpx.TransportError)


@tool(parallel=True)
def web_search(
    num_results: int = 10,
    *,
    max_characters: int = 2000,
    include_domains: list[str] | None = None,
    exclude_domains: list[str] | None = None,
    freshness: Literal["last_24_hours", "last_week", "last_month", "last_year"]
    | None = None,
    max_connections: int = 10,
    timeout: float = 60,
) -> Tool:
    """Search the web with Context.dev and return source passages with citations.

    Requires CONTEXT_DEV_API_KEY. Uses Inspect's public tool extension API;
    it does not configure the built-in web_search provider list.

    Args:
        num_results: Maximum results to return, from 1 to 100.
        max_characters: Maximum passage or description characters per result,
            from 100 to 20000.
        include_domains: Restrict results to these domains (up to 100).
        exclude_domains: Exclude these domains (up to 100).
        freshness: Restrict results by publication recency.
        max_connections: Maximum concurrent Context.dev searches.
        timeout: Total execution deadline in seconds, including retries and
            waiting for a concurrency slot. Each HTTP connect, read, write,
            and pool phase is limited to at most 30 seconds. Client cleanup
            may take up to 5 additional seconds.

    Returns:
        An Inspect tool accepting a search query. Successful page reads return
        cited highlights; unavailable highlights are labeled with their status
        and return the search description instead. HTTP and invalid-response
        errors fail the sample rather than being presented as empty results.
        Deadline expiry becomes Inspect's timeout tool error, allowing the
        model to continue the sample.
    """
    if not 1 <= num_results <= 100:
        raise ValueError("num_results must be between 1 and 100")
    if not 100 <= max_characters <= 20000:
        raise ValueError("max_characters must be between 100 and 20000")
    if max_connections < 1:
        raise ValueError("max_connections must be positive")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    for domains in (include_domains, exclude_domains):
        if domains is not None and len(domains) > 100:
            raise ValueError("Domain filters may contain at most 100 domains")

    options: dict[str, object] = {
        "numResults": num_results,
        "highlightsOptions": {"enabled": True, "maxCharacters": max_characters},
        "descriptionMaxCharacters": max_characters,
    }
    if include_domains is not None:
        options["includeDomains"] = list(include_domains)
    if exclude_domains is not None:
        options["excludeDomains"] = list(exclude_domains)
    if freshness is not None:
        options["freshness"] = freshness

    async def execute(query: str) -> ToolResult:
        """Search the web for relevant passages and cite their original URLs.

        Args:
            query: Search query, from 1 to 500 characters. Supports search
                operators such as site:, quoted phrases, and OR.
        """
        if not query.strip() or len(query) > 500:
            raise ToolError("Search query must contain 1 to 500 characters")
        api_key = os.environ.get("CONTEXT_DEV_API_KEY")
        if not api_key:
            raise RuntimeError("Set CONTEXT_DEV_API_KEY to use Context.dev web search")

        with anyio.fail_after(timeout):
            async with concurrency("context_dev_web_search", max_connections):
                client = httpx.AsyncClient(timeout=min(timeout, 30))
                try:
                    async for attempt in AsyncRetrying(
                        wait=wait_exponential_jitter(),
                        stop=stop_after_attempt(5),
                        retry=retry_if_exception(_should_retry),
                        reraise=True,
                    ):
                        with attempt:
                            response = await client.post(
                                _ENDPOINT,
                                headers={"Authorization": f"Bearer {api_key}"},
                                json={"query": query, **options},
                            )
                            response.raise_for_status()
                    search = _Response.model_validate(response.json())
                finally:
                    with anyio.move_on_after(5, shield=True):
                        await client.aclose()

        if not search.results:
            return (
                "Search returned a partial response with no results."
                if search.partial
                else "No search results found."
            )

        content: list[ContentText] = []
        if search.partial:
            content.append(ContentText(text="Search returned partial results."))
        for result in search.results[:num_results]:
            highlights = result.highlights
            passages = "\n\n".join(highlights.highlights or [])
            if highlights.code == "SUCCESS" and passages.strip():
                passage = passages[:max_characters]
                text = passage
                cited_text = passage
            else:
                passage = (result.description or "")[:max_characters]
                cited_text = None
                reason = (
                    highlights.code
                    if highlights.code != "SUCCESS"
                    else "no passages returned"
                )
                text = (
                    f"Search description (highlights unavailable: {reason}):\n{passage}"
                )
            content.append(
                ContentText(
                    text=f"{result.title or result.url}\n{result.url}\n\n{text}",
                    citations=[
                        UrlCitation(
                            url=result.url,
                            title=result.title,
                            cited_text=cited_text,
                        )
                    ],
                )
            )
        return list(content)

    return execute
