import json
import os
from pathlib import Path

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageTool, ModelOutput, get_model
from inspect_ai.solver import generate, use_tools
from inspect_ai.tool import ContentText

from inspect_context_dev import web_search


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("CONTEXT_DEV_API_KEY"), reason="CONTEXT_DEV_API_KEY is not set"
)
def test_live_eval_and_saved_citations(tmp_path: Path) -> None:
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.for_tool_call(
                "mockllm/model",
                "web_search",
                {
                    "query": "Inspect AI web search tools",
                },
            ),
            ModelOutput.from_content("mockllm/model", "Search completed."),
        ],
    )
    task = Task(
        dataset=[Sample(input="Find Inspect's web search documentation.")],
        solver=[
            use_tools(
                web_search(num_results=1, include_domains=["inspect.aisi.org.uk"])
            ),
            generate(),
        ],
    )
    log = eval(task, model=model, log_dir=str(tmp_path), display="none")[0]
    assert log.status == "success"
    assert log.location is not None
    restored = read_eval_log(log.location)
    assert restored.samples
    tool_messages = [
        m for m in restored.samples[0].messages if isinstance(m, ChatMessageTool)
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0].error is None
    content = tool_messages[0].content
    assert isinstance(content, list)
    assert isinstance(content[0], ContentText)
    assert content[0].citations
    citation = content[0].citations[0]
    assert citation.type == "url"
    assert "inspect.aisi.org.uk" in citation.url
    assert citation.cited_text
    artifact = {
        "eval_log": log.location,
        "status": restored.status,
        "tool_result": [block.model_dump(mode="json") for block in content],
    }
    (tmp_path / "live-result.json").write_text(json.dumps(artifact, indent=2))
