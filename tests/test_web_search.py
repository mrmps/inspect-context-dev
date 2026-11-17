import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import httpx
import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageTool, ModelOutput, get_model
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import ContentText, ToolDef, ToolError, UrlCitation

from inspect_context_dev import web_search


@pytest.fixture(params=["asyncio", "trio"])
def anyio_backend(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(autouse=True)
def api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "test-context-secret")


@pytest.fixture
def install_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[httpx.AsyncBaseTransport], list[httpx.AsyncClient]]:
    client_class = httpx.AsyncClient
    clients: list[httpx.AsyncClient] = []

    def install(transport: httpx.AsyncBaseTransport) -> list[httpx.AsyncClient]:
        def client(**kwargs: Any) -> httpx.AsyncClient:
            instance = client_class(transport=transport, **kwargs)
            clients.append(instance)
            return instance

        monkeypatch.setattr("inspect_context_dev._search.httpx.AsyncClient", client)
        return clients

    return install


def response_body() -> dict[str, Any]:
    return {
        "query": "test query",
        "request_id": "12345678-1234-1234-1234-123456789012",
        "results": [
            {
                "title": "Primary source",
                "url": "https://example.org/primary",
                "description": "Search description.",
                "highlights": {"code": "SUCCESS", "highlights": ["Source passage."]},
            },
            {
                "title": "Unread page",
                "url": "https://example.org/unread",
                "description": "Search-only information.",
                "highlights": {"code": "WEBSITE_ACCESS_ERROR", "highlights": None},
            },
        ],
    }


def test_tool_schema_and_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    definition = ToolDef(web_search())
    assert definition.name == "web_search"
    assert set(definition.parameters.properties) == {"query"}
    monkeypatch.delenv("CONTEXT_DEV_API_KEY")
    with pytest.raises(RuntimeError, match="CONTEXT_DEV_API_KEY"):
        web_search()


@pytest.mark.parametrize(
    "options",
    [
        {"num_results": 0},
        {"num_results": 101},
        {"max_characters": 0},
        {"max_connections": 0},
        {"timeout": 0},
        {"timeout": float("nan")},
    ],
)
def test_invalid_configuration(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        web_search(**options)


@pytest.mark.anyio
async def test_request_content_citations_and_cleanup(
    install_transport: Callable[[httpx.AsyncBaseTransport], list[httpx.AsyncClient]],
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=response_body())

    clients = install_transport(httpx.MockTransport(handler))
    search = web_search(num_results=2, include_domains=["example.org"], max_connections=1)
    result = await search("a&b=c#d")
    assert isinstance(result, list)
    assert result[0] == ContentText(
        text="Source passage.",
        citations=[
            UrlCitation(
                title="Primary source",
                url="https://example.org/primary",
                cited_text="Source passage.",
            )
        ],
    )
    assert isinstance(result[1], ContentText)
    assert "Search-only information." in result[1].text
    assert "WEBSITE_ACCESS_ERROR" in result[1].text
    assert "Search description" in result[1].text
    request = requests[0]
    assert str(request.url) == "https://api.context.dev/v1/web/search"
    assert request.headers["Authorization"] == "Bearer test-context-secret"
    assert json.loads(request.content) == {
        "query": "a&b=c#d",
        "numResults": 2,
        "highlightsOptions": {"enabled": True, "maxCharacters": 2000},
        "descriptionMaxCharacters": 2000,
        "includeDomains": ["example.org"],
    }
    assert all(client.is_closed for client in clients)


@pytest.mark.anyio
async def test_empty_and_partial_results(
    install_transport: Callable[[httpx.AsyncBaseTransport], list[httpx.AsyncClient]],
) -> None:
    install_transport(httpx.MockTransport(lambda _: httpx.Response(200, json={
        "results": [], "partial": False,
    })))
    assert await web_search()("unknown query") == "No search results found."
    install_transport(httpx.MockTransport(lambda _: httpx.Response(200, json={
        "results": [], "partial": True,
    })))
    result = await web_search()("unknown query")
    assert isinstance(result, str)
    assert "partial" in result.lower()


@pytest.mark.anyio
async def test_untrusted_response_is_not_empty_success(
    install_transport: Callable[[httpx.AsyncBaseTransport], list[httpx.AsyncClient]],
) -> None:
    install_transport(httpx.MockTransport(lambda _: httpx.Response(200, json={
        "message": "upstream failure",
    })))
    with pytest.raises(ValueError):
        await web_search()("test query")


@pytest.mark.anyio
async def test_invalid_query_does_not_reach_api(
    install_transport: Callable[[httpx.AsyncBaseTransport], list[httpx.AsyncClient]],
) -> None:
    def unexpected(request: httpx.Request) -> httpx.Response:
        pytest.fail("invalid query must not reach the API")

    install_transport(httpx.MockTransport(unexpected))
    search = web_search()
    for query in ["", " " * 4, "x" * 501]:
        with pytest.raises(ToolError):
            await search(query)


@pytest.mark.anyio
@pytest.mark.parametrize("status", [400, 401, 402, 403])
async def test_permanent_errors_fail_without_retry(
    status: int,
    install_transport: Callable[[httpx.AsyncBaseTransport], list[httpx.AsyncClient]],
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, json={"error_code": "TEST_ERROR"})

    clients = install_transport(httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        await web_search()("test query")
    assert len(requests) == 1
    assert all(client.is_closed for client in clients)


@pytest.mark.anyio
async def test_transient_retry_preserves_request_and_secret(
    install_transport: Callable[[httpx.AsyncBaseTransport], list[httpx.AsyncClient]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json=response_body())

    install_transport(httpx.MockTransport(handler))
    assert isinstance(await web_search()("test query"), list)
    assert len(requests) == 2
    assert requests[0].content == requests[1].content
    assert "test-context-secret" not in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["network", "rate_limit", "timeout"])
async def test_retryable_failures_recover(
    failure: str,
    install_transport: Callable[[httpx.AsyncBaseTransport], list[httpx.AsyncClient]],
) -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            if failure == "network":
                raise httpx.ConnectError("connection refused", request=request)
            return httpx.Response(429 if failure == "rate_limit" else 408)
        return httpx.Response(200, json=response_body())

    install_transport(httpx.MockTransport(handler))
    assert isinstance(await web_search()("test query"), list)
    assert attempts == 2


@pytest.mark.anyio
async def test_output_bounds_and_empty_highlights(
    install_transport: Callable[[httpx.AsyncBaseTransport], list[httpx.AsyncClient]],
) -> None:
    response = response_body()
    response["results"][0]["highlights"]["highlights"] = ["A" * 200, "B" * 200]
    response["results"][1]["highlights"] = {"code": "SUCCESS", "highlights": []}
    install_transport(httpx.MockTransport(lambda _: httpx.Response(200, json=response)))
    result = await web_search(num_results=2, max_characters=100)("test query")
    assert isinstance(result, list)
    assert isinstance(result[0], ContentText)
    assert result[0].text == "A" * 100
    assert isinstance(result[1], ContentText)
    assert "Search description" in result[1].text
    assert "Search-only information." in result[1].text


class BlockingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.started = anyio.Event()
        self.active = 0
        self.peak = 0
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.active += 1
        self.peak = max(self.active, self.peak)
        self.started.set()
        try:
            await anyio.sleep_forever()
        finally:
            self.active -= 1

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.anyio
async def test_cancellation_and_concurrency_cleanup(
    install_transport: Callable[[httpx.AsyncBaseTransport], list[httpx.AsyncClient]],
) -> None:
    transport = BlockingTransport()
    clients = install_transport(transport)
    search = web_search(max_connections=1)
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(search, "first query")
        await transport.started.wait()
        tasks.start_soon(search, "second query")
        tasks.cancel_scope.cancel()
    assert transport.peak == 1
    assert transport.active == 0
    assert transport.closed
    assert all(client.is_closed for client in clients)


@pytest.mark.anyio
async def test_total_timeout_closes_client(
    install_transport: Callable[[httpx.AsyncBaseTransport], list[httpx.AsyncClient]],
) -> None:
    transport = BlockingTransport()
    clients = install_transport(transport)
    with pytest.raises(TimeoutError):
        await web_search(timeout=0.05)("test query")
    assert transport.closed
    assert all(client.is_closed for client in clients)


def test_eval_log_round_trip(
    install_transport: Callable[[httpx.AsyncBaseTransport], list[httpx.AsyncClient]],
    tmp_path: Path,
) -> None:
    install_transport(httpx.MockTransport(lambda _: httpx.Response(200, json=response_body())))
    model = get_model("mockllm/model", custom_outputs=[
        ModelOutput.for_tool_call("mockllm/model", "web_search", {"query": "test query"}),
        ModelOutput.from_content("mockllm/model", "Found the cited primary source."),
    ])
    task = Task(
        dataset=[Sample(input="Find a source.")],
        solver=[use_tools(web_search()), generate()],
    )
    log = eval(task, model=model, log_dir=str(tmp_path), display="none")[0]
    assert log.status == "success"
    assert log.location is not None
    restored = read_eval_log(log.location)
    assert restored.plan is not None
    assert restored.samples is not None
    tools = [message for message in restored.samples[0].messages if isinstance(message, ChatMessageTool)]
    assert len(tools) == 1
    assert tools[0].error is None
    assert "Source passage." in tools[0].text
    assert "inspect_context_dev/web_search" in json.dumps(
        restored.plan.model_dump(mode="json")
    )

