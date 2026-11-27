"""Async client for the ClinicalTrials.gov REST API v2."""

from __future__ import annotations

import re
from typing import Any

import httpx

CT_BASE_URL = "https://clinicaltrials.gov/api/v2"
DEFAULT_TIMEOUT = 15.0
NCT_PATTERN = re.compile(r"^NCT\d{8}$", re.IGNORECASE)

SEARCH_FIELDS = ",".join(
    [
        "NCTId",
        "BriefTitle",
        "OverallStatus",
        "Phase",
        "Condition",
        "InterventionName",
        "LeadSponsorName",
        "BriefSummary",
        "StartDate",
        "CompletionDate",
    ]
)


class ClinicalTrialsError(Exception):
    """Raised when a ClinicalTrials.gov API request fails."""


def _truncate(text: str | None, max_chars: int = 800) -> str | None:
    if not text:
        return None
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


async def _get_json(
    path: str,
    params: dict[str, Any] | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Perform a GET request against ClinicalTrials.gov API v2."""
    url = f"{CT_BASE_URL}{path}"
    query_params = {k: v for k, v in (params or {}).items() if v is not None}

    async def _do_request(c: httpx.AsyncClient) -> dict[str, Any]:
        try:
            resp = await c.get(url, params=query_params)
        except httpx.TimeoutException as exc:
            raise ClinicalTrialsError(f"ClinicalTrials.gov request timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ClinicalTrialsError(f"ClinicalTrials.gov network error: {exc}") from exc

        if resp.status_code == 404:
            raise ClinicalTrialsError(f"Resource not found on ClinicalTrials.gov: {path}")
        if resp.status_code >= 400:
            raise ClinicalTrialsError(
                f"ClinicalTrials.gov API error (HTTP {resp.status_code}): {resp.text[:300]}"
            )
        return resp.json()

    if client is not None:
        return await _do_request(client)

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as new_client:
        return await _do_request(new_client)


async def search_clinical_trials(
    condition: str | None = None,
    intervention: str | None = None,
    status: str | None = None,
    phase: str | None = None,
    sponsor: str | None = None,
    limit: int = 5,
    page_token: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Search ClinicalTrials.gov v2 studies with optional filters."""
    if not any([condition, intervention, status, phase, sponsor]):
        raise ValueError(
            "Provide at least one search filter (condition, intervention, status, phase, or sponsor)."
        )

    limit = max(1, min(limit, 25))
    params: dict[str, Any] = {
        "pageSize": limit,
        "countTotal": "true",
        "fields": SEARCH_FIELDS,
    }
    if condition and condition.strip():
        params["query.cond"] = condition.strip()
    if intervention and intervention.strip():
        params["query.intr"] = intervention.strip()
    if status and status.strip():
        params["filter.overallStatus"] = status.strip().upper()
    if phase and phase.strip():
        params["query.term"] = f"AREA[Phase]{phase.strip().upper()}"
    if sponsor and sponsor.strip():
        params["query.spons"] = sponsor.strip()
    if page_token and page_token.strip():
        params["pageToken"] = page_token.strip()

    data = await _get_json("/studies", params=params, client=client)

    studies = []
    for raw in data.get("studies", []):
        proto = raw.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        stat = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        cond = proto.get("conditionsModule", {})
        arms = proto.get("armsInterventionsModule", {})
        spons = proto.get("sponsorCollaboratorsModule", {})
        desc = proto.get("descriptionModule", {})

        interventions = [
            i.get("name") for i in arms.get("interventions", []) if i.get("name")
        ]
        nct_id = ident.get("nctId")
        studies.append(
            {
                "nct_id": nct_id,
                "brief_title": ident.get("briefTitle"),
                "overall_status": stat.get("overallStatus"),
                "phases": design.get("phases", []),
                "conditions": cond.get("conditions", []),
                "interventions": interventions,
                "lead_sponsor": spons.get("leadSponsor", {}).get("name"),
                "start_date": stat.get("startDateStruct", {}).get("date"),
                "completion_date": stat.get("completionDateStruct", {}).get("date"),
                "brief_summary": _truncate(desc.get("briefSummary"), 500),
                "url": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else None,
            }
        )

    return {
        "total_count": data.get("totalCount", len(studies)),
        "next_page_token": data.get("nextPageToken"),
        "studies": studies,
    }


async def get_clinical_trial(
    nct_id: str,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Fetch detailed protocol information for a single clinical trial by NCT ID."""
    cleaned_id = nct_id.strip().upper()
    if not NCT_PATTERN.match(cleaned_id):
        raise ValueError(
            f"Invalid NCT ID '{nct_id}'. Expected format like 'NCT04280705'."
        )

    raw = await _get_json(f"/studies/{cleaned_id}", client=client)
    proto = raw.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    stat = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    cond = proto.get("conditionsModule", {})
    arms = proto.get("armsInterventionsModule", {})
    spons = proto.get("sponsorCollaboratorsModule", {})
    desc = proto.get("descriptionModule", {})
    elig = proto.get("eligibilityModule", {})
    outcomes = proto.get("outcomesModule", {})
    contacts = proto.get("contactsLocationsModule", {})

    locations = contacts.get("locations", [])
    sample_locations = [
        {
            "facility": loc.get("facility"),
            "city": loc.get("city"),
            "state": loc.get("state"),
            "country": loc.get("country"),
            "status": loc.get("status"),
        }
        for loc in locations[:5]
    ]

    return {
        "nct_id": ident.get("nctId", cleaned_id),
        "brief_title": ident.get("briefTitle"),
        "official_title": ident.get("officialTitle"),
        "overall_status": stat.get("overallStatus"),
        "study_type": design.get("studyType"),
        "phases": design.get("phases", []),
        "enrollment": design.get("enrollmentInfo", {}),
        "conditions": cond.get("conditions", []),
        "interventions": [
            {
                "type": i.get("type"),
                "name": i.get("name"),
                "description": _truncate(i.get("description"), 300),
            }
            for i in arms.get("interventions", [])
        ],
        "lead_sponsor": spons.get("leadSponsor", {}).get("name"),
        "collaborators": [c.get("name") for c in spons.get("collaborators", []) if c.get("name")],
        "start_date": stat.get("startDateStruct", {}).get("date"),
        "completion_date": stat.get("completionDateStruct", {}).get("date"),
        "brief_summary": _truncate(desc.get("briefSummary"), 1200),
        "eligibility": {
            "sex": elig.get("sex"),
            "minimum_age": elig.get("minimumAge"),
            "maximum_age": elig.get("maximumAge"),
            "healthy_volunteers": elig.get("healthyVolunteers"),
            "criteria": _truncate(elig.get("eligibilityCriteria"), 1500),
        },
        "primary_outcomes": [
            {
                "measure": o.get("measure"),
                "time_frame": o.get("timeFrame"),
            }
            for o in outcomes.get("primaryOutcomes", [])[:5]
        ],
        "location_count": len(locations),
        "sample_locations": sample_locations,
        "url": f"https://clinicaltrials.gov/study/{cleaned_id}",
    }
