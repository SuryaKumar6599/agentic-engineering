"""Async client for the openFDA Drug API (labels, adverse events, recalls)."""

from __future__ import annotations

import os
import re
from typing import Any

import httpx

OPENFDA_BASE_URL = "https://api.fda.gov/drug"
DEFAULT_TIMEOUT = 15.0


class OpenFDAError(Exception):
    """Raised when an openFDA API request fails."""


def _quote_term(value: str) -> str:
    """Escape quotes and wrap a search term in double quotes for openFDA Lucene queries."""
    cleaned = value.strip().replace('"', "")
    return f'"{cleaned}"'


def _tokenize(value: str) -> list[str]:
    """Extract alphanumeric search tokens from a free-text query."""
    return re.findall(r"[A-Za-z0-9]+", value)


def _token_group(field: str, value: str) -> str:
    """Build an AND-joined field query so word order or punctuation differences still match."""
    tokens = _tokenize(value)
    if not tokens:
        return f"{field}:{_quote_term(value)}"
    if len(tokens) == 1:
        return f"{field}:{tokens[0]}"
    return f"{field}:(" + " AND ".join(tokens) + ")"


def _truncate(text: str | None, max_chars: int = 1200) -> str | None:
    """Truncate long regulatory text blocks to keep LLM context usage reasonable."""
    if not text:
        return None
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _first_text(value: Any, max_chars: int = 1200) -> str | None:
    """Extract and truncate the first string from an openFDA list-or-string field."""
    if isinstance(value, list):
        if not value:
            return None
        return _truncate(str(value[0]), max_chars=max_chars)
    if isinstance(value, str):
        return _truncate(value, max_chars=max_chars)
    return None


async def _get_json(
    endpoint: str,
    params: dict[str, Any],
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any] | None:
    """Perform a GET request against openFDA, returning None on 404 (no matches)."""
    api_key = os.environ.get("FDA_API_KEY")
    query_params = {k: v for k, v in params.items() if v is not None}
    if api_key:
        query_params["api_key"] = api_key

    url = f"{OPENFDA_BASE_URL}/{endpoint}.json"

    async def _do_request(c: httpx.AsyncClient) -> dict[str, Any] | None:
        try:
            resp = await c.get(url, params=query_params)
        except httpx.TimeoutException as exc:
            raise OpenFDAError(f"openFDA request timed out for {endpoint}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise OpenFDAError(f"openFDA network error for {endpoint}: {exc}") from exc

        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            raise OpenFDAError(
                "openFDA rate limit exceeded (HTTP 429). Set FDA_API_KEY to raise the daily limit."
            )
        if resp.status_code >= 400:
            detail = resp.text[:300]
            raise OpenFDAError(f"openFDA API error (HTTP {resp.status_code}): {detail}")

        return resp.json()

    if client is not None:
        return await _do_request(client)

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as new_client:
        return await _do_request(new_client)


async def search_drug_labels(
    drug_name: str,
    limit: int = 3,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Search FDA drug product labels by brand, generic, or substance name."""
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string.")
    limit = max(1, min(limit, 10))
    term = _quote_term(drug_name)
    search_query = (
        f"openfda.brand_name:{term} OR openfda.generic_name:{term} OR openfda.substance_name:{term}"
    )
    data = await _get_json(
        "label",
        {"search": search_query, "limit": limit},
        client=client,
    )
    if not data:
        tokens = _tokenize(drug_name)
        if len(tokens) > 1:
            lead = tokens[0]
            rest = " AND ".join(tokens[1:])
            fallback_query = (
                f"(openfda.brand_name:{lead} OR openfda.generic_name:{lead} "
                f"OR openfda.substance_name:{lead}) AND ({rest})"
            )
            data = await _get_json(
                "label",
                {"search": fallback_query, "limit": limit},
                client=client,
            )
    if not data:
        return {"query": drug_name, "total_matches": 0, "results": []}

    total = data.get("meta", {}).get("results", {}).get("total", 0)
    results = []
    for item in data.get("results", []):
        openfda = item.get("openfda", {})
        results.append(
            {
                "set_id": item.get("set_id"),
                "effective_time": item.get("effective_time"),
                "brand_names": openfda.get("brand_name", []),
                "generic_names": openfda.get("generic_name", []),
                "manufacturer_names": openfda.get("manufacturer_name", []),
                "routes": openfda.get("route", []),
                "indications_and_usage": _first_text(item.get("indications_and_usage"), 1200),
                "boxed_warning": _first_text(item.get("boxed_warning"), 1000),
                "contraindications": _first_text(item.get("contraindications"), 1000),
                "dosage_and_administration": _first_text(
                    item.get("dosage_and_administration"), 1000
                ),
                "warnings_and_cautions": _first_text(item.get("warnings_and_cautions"), 1000),
                "adverse_reactions": _first_text(item.get("adverse_reactions"), 1000),
                "drug_interactions": _first_text(item.get("drug_interactions"), 1000),
            }
        )

    return {"query": drug_name, "total_matches": total, "results": results}


async def search_adverse_events(
    drug_name: str,
    reaction: str | None = None,
    serious_only: bool = False,
    limit: int = 5,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Search FAERS adverse event reports for a drug."""
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string.")
    limit = max(1, min(limit, 20))

    clauses = [f"patient.drug.medicinalproduct:{_quote_term(drug_name)}"]
    if reaction and reaction.strip():
        clauses.append(f"patient.reaction.reactionmeddrapt:{_quote_term(reaction)}")
    if serious_only:
        clauses.append("serious:1")

    search_query = " AND ".join(clauses)
    data = await _get_json(
        "event",
        {"search": search_query, "limit": limit, "sort": "receivedate:desc"},
        client=client,
    )
    if not data:
        return {"query": drug_name, "total_matches": 0, "results": []}

    total = data.get("meta", {}).get("results", {}).get("total", 0)
    sex_map = {"0": "Unknown", "1": "Male", "2": "Female"}
    results = []
    for item in data.get("results", []):
        patient = item.get("patient", {})
        serious_flags = [
            flag
            for flag, key in (
                ("death", "seriousnessdeath"),
                ("life_threatening", "seriousnesslifethreatening"),
                ("hospitalization", "seriousnesshospitalization"),
                ("disability", "seriousnessdisabling"),
                ("congenital_anomaly", "seriousnesscongenitalanomali"),
                ("other", "seriousnessother"),
            )
            if str(item.get(key, "")) == "1"
        ]
        drugs = [
            {
                "name": d.get("medicinalproduct"),
                "indication": d.get("drugindication"),
                "role": {"1": "Suspect", "2": "Concomitant", "3": "Interacting"}.get(
                    str(d.get("drugcharacterization", "")), "Unknown"
                ),
            }
            for d in patient.get("drug", [])[:10]
        ]
        reactions = [
            r.get("reactionmeddrapt")
            for r in patient.get("reaction", [])
            if r.get("reactionmeddrapt")
        ]
        results.append(
            {
                "safety_report_id": item.get("safetyreportid"),
                "receive_date": item.get("receivedate"),
                "serious": str(item.get("serious", "")) == "1",
                "seriousness_flags": serious_flags,
                "patient_age": patient.get("patientonsetage"),
                "patient_sex": sex_map.get(str(patient.get("patientsex", "")), "Unknown"),
                "reactions": reactions,
                "drugs": drugs,
            }
        )

    return {"query": drug_name, "total_matches": total, "results": results}


async def get_top_adverse_reactions(
    drug_name: str,
    limit: int = 15,
    serious_only: bool = False,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Return the most frequently reported MedDRA adverse reactions for a drug in FAERS."""
    if not drug_name or not drug_name.strip():
        raise ValueError("drug_name must be a non-empty string.")
    limit = max(1, min(limit, 50))

    clauses = [f"patient.drug.medicinalproduct:{_quote_term(drug_name)}"]
    if serious_only:
        clauses.append("serious:1")

    data = await _get_json(
        "event",
        {
            "search": " AND ".join(clauses),
            "count": "patient.reaction.reactionmeddrapt.exact",
        },
        client=client,
    )
    if not data:
        return {"query": drug_name, "serious_only": serious_only, "reactions": []}

    raw_counts = data.get("results", [])[:limit]
    reactions = [{"reaction": item.get("term"), "count": item.get("count")} for item in raw_counts]
    return {"query": drug_name, "serious_only": serious_only, "reactions": reactions}


async def search_drug_recalls(
    drug_name: str | None = None,
    recalling_firm: str | None = None,
    classification: str | None = None,
    limit: int = 5,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Search FDA drug recall/enforcement reports."""
    limit = max(1, min(limit, 20))
    clauses = []
    if drug_name and drug_name.strip():
        clauses.append(_token_group("product_description", drug_name))
    if recalling_firm and recalling_firm.strip():
        clauses.append(_token_group("recalling_firm", recalling_firm))
    if classification and classification.strip():
        clauses.append(f"classification:{_quote_term(classification)}")

    if not clauses:
        raise ValueError(
            "Provide at least one filter: drug_name, recalling_firm, or classification."
        )

    data = await _get_json(
        "enforcement",
        {
            "search": " AND ".join(clauses),
            "limit": limit,
            "sort": "recall_initiation_date:desc",
        },
        client=client,
    )
    if not data:
        return {"total_matches": 0, "results": []}

    total = data.get("meta", {}).get("results", {}).get("total", 0)
    results = [
        {
            "recall_number": item.get("recall_number"),
            "status": item.get("status"),
            "classification": item.get("classification"),
            "recalling_firm": item.get("recalling_firm"),
            "recall_initiation_date": item.get("recall_initiation_date"),
            "voluntary_mandated": item.get("voluntary_mandated"),
            "product_description": _truncate(item.get("product_description"), 500),
            "reason_for_recall": _truncate(item.get("reason_for_recall"), 500),
            "distribution_pattern": _truncate(item.get("distribution_pattern"), 300),
        }
        for item in data.get("results", [])
    ]
    return {"total_matches": total, "results": results}
