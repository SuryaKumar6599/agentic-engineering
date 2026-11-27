"""Tests for the ClinicalTrials.gov v2 async client."""

import httpx
import pytest
import respx

from fda_trials_mcp.clinical_trials import (
    CT_BASE_URL,
    ClinicalTrialsError,
    get_clinical_trial,
    search_clinical_trials,
)


@respx.mock
async def test_search_clinical_trials_success() -> None:
    route = respx.get(f"{CT_BASE_URL}/studies").mock(
        return_value=httpx.Response(
            200,
            json={
                "totalCount": 12,
                "nextPageToken": "tok-2",
                "studies": [
                    {
                        "protocolSection": {
                            "identificationModule": {
                                "nctId": "NCT04280705",
                                "briefTitle": "Adaptive COVID-19 Treatment Trial",
                            },
                            "statusModule": {
                                "overallStatus": "COMPLETED",
                                "startDateStruct": {"date": "2020-02-21"},
                                "completionDateStruct": {"date": "2020-05-21"},
                            },
                            "designModule": {"phases": ["PHASE3"]},
                            "conditionsModule": {"conditions": ["COVID-19"]},
                            "armsInterventionsModule": {
                                "interventions": [{"name": "Remdesivir"}]
                            },
                            "sponsorCollaboratorsModule": {
                                "leadSponsor": {"name": "NIAID"}
                            },
                            "descriptionModule": {
                                "briefSummary": "A multicenter trial evaluating remdesivir."
                            },
                        }
                    }
                ],
            },
        )
    )

    result = await search_clinical_trials(
        condition="COVID-19",
        intervention="Remdesivir",
        status="COMPLETED",
        phase="PHASE3",
        limit=5,
    )
    assert route.called
    assert result["total_count"] == 12
    assert result["next_page_token"] == "tok-2"
    study = result["studies"][0]
    assert study["nct_id"] == "NCT04280705"
    assert study["phases"] == ["PHASE3"]
    assert study["url"] == "https://clinicaltrials.gov/study/NCT04280705"


async def test_search_clinical_trials_requires_filter() -> None:
    with pytest.raises(ValueError, match="at least one search filter"):
        await search_clinical_trials()


@respx.mock
async def test_get_clinical_trial_success() -> None:
    respx.get(f"{CT_BASE_URL}/studies/NCT04280705").mock(
        return_value=httpx.Response(
            200,
            json={
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT04280705",
                        "briefTitle": "Adaptive COVID-19 Treatment Trial",
                        "officialTitle": "A Multicenter, Adaptive, Randomized Trial",
                    },
                    "statusModule": {"overallStatus": "COMPLETED"},
                    "designModule": {
                        "studyType": "INTERVENTIONAL",
                        "phases": ["PHASE3"],
                        "enrollmentInfo": {"count": 1062, "type": "ACTUAL"},
                    },
                    "conditionsModule": {"conditions": ["COVID-19"]},
                    "armsInterventionsModule": {
                        "interventions": [
                            {"type": "DRUG", "name": "Remdesivir", "description": "IV daily"}
                        ]
                    },
                    "sponsorCollaboratorsModule": {
                        "leadSponsor": {"name": "NIAID"},
                        "collaborators": [{"name": "Gilead Sciences"}],
                    },
                    "descriptionModule": {"briefSummary": "Summary text."},
                    "eligibilityModule": {
                        "sex": "ALL",
                        "minimumAge": "18 Years",
                        "healthyVolunteers": False,
                        "eligibilityCriteria": "Inclusion: Adults >= 18.",
                    },
                    "outcomesModule": {
                        "primaryOutcomes": [
                            {"measure": "Time to recovery", "timeFrame": "Day 29"}
                        ]
                    },
                    "contactsLocationsModule": {
                        "locations": [
                            {
                                "facility": "NIH Clinical Center",
                                "city": "Bethesda",
                                "state": "Maryland",
                                "country": "United States",
                            }
                        ]
                    },
                }
            },
        )
    )

    trial = await get_clinical_trial("nct04280705")
    assert trial["nct_id"] == "NCT04280705"
    assert trial["location_count"] == 1
    assert trial["eligibility"]["minimum_age"] == "18 Years"
    assert trial["primary_outcomes"][0]["measure"] == "Time to recovery"


async def test_get_clinical_trial_invalid_nct_id() -> None:
    with pytest.raises(ValueError, match="Invalid NCT ID"):
        await get_clinical_trial("INVALID123")


@respx.mock
async def test_get_clinical_trial_404_raises() -> None:
    respx.get(f"{CT_BASE_URL}/studies/NCT99999999").mock(
        return_value=httpx.Response(404, json={"message": "Not found"})
    )

    with pytest.raises(ClinicalTrialsError, match="not found"):
        await get_clinical_trial("NCT99999999")
