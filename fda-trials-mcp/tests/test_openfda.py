"""Tests for the openFDA async client."""

import httpx
import pytest
import respx

from fda_trials_mcp.openfda import (
    OPENFDA_BASE_URL,
    OpenFDAError,
    get_top_adverse_reactions,
    search_adverse_events,
    search_drug_labels,
    search_drug_recalls,
)


@respx.mock
async def test_search_drug_labels_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FDA_API_KEY", "test-key-123")
    route = respx.get(f"{OPENFDA_BASE_URL}/label.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "meta": {"results": {"total": 1}},
                "results": [
                    {
                        "set_id": "abc-123",
                        "effective_time": "20240101",
                        "openfda": {
                            "brand_name": ["GLUCOPHAGE"],
                            "generic_name": ["METFORMIN HYDROCHLORIDE"],
                            "manufacturer_name": ["Bristol-Myers Squibb"],
                            "route": ["ORAL"],
                        },
                        "indications_and_usage": ["Treatment of type 2 diabetes mellitus."],
                        "boxed_warning": ["LACTIC ACIDOSIS warning."],
                        "contraindications": ["Severe renal impairment."],
                    }
                ],
            },
        )
    )

    result = await search_drug_labels("metformin", limit=2)
    assert route.called
    raw_url = str(route.calls[0].request.url)
    assert "%2BOR%2B" not in raw_url
    assert "+OR+" in raw_url
    assert route.calls[0].request.url.params["api_key"] == "test-key-123"
    assert result["total_matches"] == 1
    assert len(result["results"]) == 1
    label = result["results"][0]
    assert label["brand_names"] == ["GLUCOPHAGE"]
    assert label["boxed_warning"] == "LACTIC ACIDOSIS warning."


@respx.mock
async def test_search_drug_labels_404_returns_empty() -> None:
    respx.get(f"{OPENFDA_BASE_URL}/label.json").mock(
        return_value=httpx.Response(404, json={"error": {"code": "NOT_FOUND"}})
    )

    result = await search_drug_labels("nonexistentdrugxyz")
    assert result["total_matches"] == 0
    assert result["results"] == []


async def test_search_drug_labels_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        await search_drug_labels("   ")


@respx.mock
async def test_search_adverse_events_success() -> None:
    route = respx.get(f"{OPENFDA_BASE_URL}/event.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "meta": {"results": {"total": 42}},
                "results": [
                    {
                        "safetyreportid": "10001",
                        "receivedate": "20250115",
                        "serious": "1",
                        "seriousnesshospitalization": "1",
                        "patient": {
                            "patientonsetage": "64",
                            "patientsex": "2",
                            "reaction": [{"reactionmeddrapt": "Nausea"}],
                            "drug": [
                                {
                                    "medicinalproduct": "OZEMPIC",
                                    "drugindication": "TYPE 2 DIABETES",
                                    "drugcharacterization": "1",
                                }
                            ],
                        },
                    }
                ],
            },
        )
    )

    result = await search_adverse_events("ozempic", reaction="Nausea", serious_only=True)
    assert "%2BAND%2B" not in str(route.calls[0].request.url)
    assert "+AND+" in str(route.calls[0].request.url)
    assert result["total_matches"] == 42
    event = result["results"][0]
    assert event["serious"] is True
    assert event["seriousness_flags"] == ["hospitalization"]
    assert event["patient_sex"] == "Female"
    assert event["reactions"] == ["Nausea"]
    assert event["drugs"][0]["role"] == "Suspect"


@respx.mock
async def test_search_adverse_events_rate_limit_raises() -> None:
    respx.get(f"{OPENFDA_BASE_URL}/event.json").mock(
        return_value=httpx.Response(429, text="Rate limit exceeded")
    )

    with pytest.raises(OpenFDAError, match="rate limit"):
        await search_adverse_events("aspirin")


@respx.mock
async def test_get_top_adverse_reactions_success() -> None:
    respx.get(f"{OPENFDA_BASE_URL}/event.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "results": [
                    {"term": "NAUSEA", "count": 1200},
                    {"term": "FATIGUE", "count": 850},
                ]
            },
        )
    )

    result = await get_top_adverse_reactions("semaglutide", limit=2, serious_only=True)
    assert result["serious_only"] is True
    assert result["reactions"] == [
        {"reaction": "NAUSEA", "count": 1200},
        {"reaction": "FATIGUE", "count": 850},
    ]


@respx.mock
async def test_search_drug_recalls_success() -> None:
    route = respx.get(f"{OPENFDA_BASE_URL}/enforcement.json").mock(
        return_value=httpx.Response(
            200,
            json={
                "meta": {"results": {"total": 1}},
                "results": [
                    {
                        "recall_number": "D-0001-2025",
                        "status": "Ongoing",
                        "classification": "Class I",
                        "recalling_firm": "Example Pharma",
                        "recall_initiation_date": "20250201",
                        "voluntary_mandated": "Voluntary: Firm initiated",
                        "product_description": "Ibuprofen Tablets USP 800 mg",
                        "reason_for_recall": "Impurity detected",
                        "distribution_pattern": "Nationwide",
                    }
                ],
            },
        )
    )

    result = await search_drug_recalls(drug_name="Ibuprofen 800 mg Tablets", classification="Class I")
    assert route.called
    assert "product_description:(Ibuprofen AND 800 AND mg AND Tablets)" in route.calls[0].request.url.params["search"]
    assert result["total_matches"] == 1
    assert result["results"][0]["classification"] == "Class I"


@respx.mock
async def test_search_drug_labels_multi_word_fallback() -> None:
    route = respx.get(f"{OPENFDA_BASE_URL}/label.json").mock(
        side_effect=[
            httpx.Response(404, json={"error": {"code": "NOT_FOUND"}}),
            httpx.Response(
                200,
                json={
                    "meta": {"results": {"total": 10}},
                    "results": [
                        {
                            "set_id": "ibu-800",
                            "openfda": {"generic_name": ["IBUPROFEN"]},
                            "indications_and_usage": ["Relief of pain."],
                        }
                    ],
                },
            ),
        ]
    )

    result = await search_drug_labels("Ibuprofen 800 mg", limit=1)
    assert route.call_count == 2
    assert result["total_matches"] == 10
    assert result["results"][0]["set_id"] == "ibu-800"


async def test_search_drug_recalls_requires_filter() -> None:
    with pytest.raises(ValueError, match="at least one filter"):
        await search_drug_recalls()
