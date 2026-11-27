"""MCP server exposing openFDA (drug labels, adverse events, recalls) and ClinicalTrials.gov v2."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from . import clinical_trials, openfda

mcp = MCPServer(
    "fda-trials-mcp",
    instructions=(
        "Query real-time FDA drug labels, FAERS adverse event reports, FDA drug recalls, "
        "and ClinicalTrials.gov v2 studies. Prefer concise queries and use get_clinical_trial "
        "for full eligibility criteria after finding relevant NCT IDs."
    ),
)

READ_ONLY_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=True,
)


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
async def search_drug_labels(drug_name: str, limit: int = 3) -> dict[str, Any]:
    """Search FDA drug product labels by brand, generic, or active substance name.

    Returns structured labeling sections: indications, boxed warnings, contraindications,
    dosage & administration, warnings, adverse reactions, and drug interactions.

    Args:
        drug_name: Brand, generic, or active ingredient name (e.g. "metformin", "Ozempic").
        limit: Maximum number of matching labels to return (1-10, default 3).
    """
    return await openfda.search_drug_labels(drug_name=drug_name, limit=limit)


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
async def search_adverse_events(
    drug_name: str,
    reaction: str | None = None,
    serious_only: bool = False,
    limit: int = 5,
) -> dict[str, Any]:
    """Search FDA Adverse Event Reporting System (FAERS) case reports for a drug.

    Args:
        drug_name: Drug product name (e.g. "ibuprofen", "semaglutide").
        reaction: Optional MedDRA reaction term to filter by (e.g. "Nausea").
        serious_only: If True, return only reports marked as serious (death, hospitalization, etc.).
        limit: Maximum number of case reports to return (1-20, default 5).
    """
    return await openfda.search_adverse_events(
        drug_name=drug_name,
        reaction=reaction,
        serious_only=serious_only,
        limit=limit,
    )


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
async def get_top_adverse_reactions(
    drug_name: str,
    limit: int = 15,
    serious_only: bool = False,
) -> dict[str, Any]:
    """Get the most frequently reported MedDRA adverse reactions for a drug in FAERS.

    Args:
        drug_name: Drug product name (e.g. "atorvastatin", "lisinopril").
        limit: Number of top reaction terms to return (1-50, default 15).
        serious_only: If True, count reactions only across serious reports.
    """
    return await openfda.get_top_adverse_reactions(
        drug_name=drug_name,
        limit=limit,
        serious_only=serious_only,
    )


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
async def search_drug_recalls(
    drug_name: str | None = None,
    recalling_firm: str | None = None,
    classification: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search FDA drug recall and enforcement reports.

    Provide at least one filter (drug_name, recalling_firm, or classification).

    Args:
        drug_name: Keyword in product description (e.g. "valsartan", "metformin").
        recalling_firm: Name of the recalling company.
        classification: FDA recall class: "Class I", "Class II", or "Class III".
        limit: Maximum number of recall reports to return (1-20, default 5).
    """
    return await openfda.search_drug_recalls(
        drug_name=drug_name,
        recalling_firm=recalling_firm,
        classification=classification,
        limit=limit,
    )


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
async def search_clinical_trials(
    condition: str | None = None,
    intervention: str | None = None,
    status: str | None = None,
    phase: str | None = None,
    sponsor: str | None = None,
    limit: int = 5,
    page_token: str | None = None,
) -> dict[str, Any]:
    """Search ClinicalTrials.gov v2 studies by condition, intervention, status, phase, or sponsor.

    Provide at least one search filter. Returns compact study summaries with NCT IDs.

    Args:
        condition: Disease or condition (e.g. "type 2 diabetes", "glioblastoma").
        intervention: Drug, therapy, or device (e.g. "pembrolizumab", "tirzepatide").
        status: Recruitment status (e.g. "RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED").
        phase: Trial phase (e.g. "PHASE1", "PHASE2", "PHASE3", "PHASE4").
        sponsor: Lead sponsor organization name.
        limit: Maximum number of studies per page (1-25, default 5).
        page_token: Pagination token from a previous search response.
    """
    return await clinical_trials.search_clinical_trials(
        condition=condition,
        intervention=intervention,
        status=status,
        phase=phase,
        sponsor=sponsor,
        limit=limit,
        page_token=page_token,
    )


@mcp.tool(annotations=READ_ONLY_ANNOTATIONS)
async def get_clinical_trial(nct_id: str) -> dict[str, Any]:
    """Fetch full protocol details for a specific ClinicalTrials.gov study by NCT ID.

    Includes eligibility criteria, primary outcomes, interventions, enrollment, and sample locations.

    Args:
        nct_id: ClinicalTrials.gov identifier (e.g. "NCT04280705").
    """
    return await clinical_trials.get_clinical_trial(nct_id=nct_id)


@mcp.prompt()
def drug_safety_brief(drug_name: str) -> str:
    """Generate a structured regulatory & clinical trial brief for a drug."""
    return (
        f"Create a concise regulatory, safety, and clinical trial brief for '{drug_name}' using the "
        f"fda-trials-mcp tools:\n"
        f"1. Call `search_drug_labels` for '{drug_name}' to summarize approved indications, boxed "
        f"warnings, and key contraindications.\n"
        f"2. Call `get_top_adverse_reactions` for '{drug_name}' (serious_only=True) to list the top "
        f"reported serious FAERS reactions.\n"
        f"3. Call `search_drug_recalls` for '{drug_name}' to check recent enforcement/recall history.\n"
        f"4. Call `search_clinical_trials` with intervention='{drug_name}' and status='RECRUITING' "
        f"to highlight active recruiting trials.\n"
        f"Note that FAERS reports represent spontaneous submissions and do not establish causality."
    )


def main() -> None:
    """Entry point for the fda-trials-mcp CLI."""
    mcp.run()


if __name__ == "__main__":
    main()
