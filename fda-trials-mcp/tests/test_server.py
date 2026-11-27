"""Tests for FastMCP server tool and prompt registration."""

import httpx
import respx

from fda_trials_mcp.openfda import OPENFDA_BASE_URL
from fda_trials_mcp.server import mcp


async def test_tools_and_prompts_registered() -> None:
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    assert tool_names == {
        "search_drug_labels",
        "search_adverse_events",
        "get_top_adverse_reactions",
        "search_drug_recalls",
        "search_clinical_trials",
        "get_clinical_trial",
    }

    prompts = await mcp.list_prompts()
    assert {p.name for p in prompts} == {"drug_safety_brief"}


@respx.mock
async def test_call_tool_via_mcp() -> None:
    respx.get(f"{OPENFDA_BASE_URL}/label.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "meta": {"results": {"total": 1}},
                "results": [
                    {
                        "set_id": "id-1",
                        "openfda": {"brand_name": ["ADVIL"], "generic_name": ["IBUPROFEN"]},
                        "indications_and_usage": ["Temporary relief of minor aches and pains."],
                    }
                ],
            },
        )
    )

    result = await mcp.call_tool("search_drug_labels", {"drug_name": "ibuprofen", "limit": 1})
    # FastMCP returns a tuple (content_list, structured_dict) or content_list depending on version
    text = str(result)
    assert "ADVIL" in text
    assert "IBUPROFEN" in text
