from __future__ import annotations

from contextlib import asynccontextmanager
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import db
from backend.agents import (
    ClinicalAssessmentAgent,
    MedicalReviewAgent,
    QCAgent,
    run_intake_pipeline,
)
from backend.eval_suite import run_evaluation
from backend.llm_client import get_ollama_status
from backend.models import (
    AuditTrailEntry,
    CaseStatus,
    FieldUpdateRequest,
    HITLDecisionRequest,
    IngestRequest,
    PVCase,
    WorkflowStage,
)
from backend.rbac import (
    DEMO_USERS,
    EnterpriseRole,
    EnterpriseUser,
    build_user,
    get_current_user,
    issue_token,
    require_permission,
)

OPENFDA_SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "openfda_sample.json"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    if len(db.list_cases()) == 0 and OPENFDA_SAMPLE_PATH.exists():
        raw = json.loads(OPENFDA_SAMPLE_PATH.read_text())
        records = raw.get("results", raw) if isinstance(raw, dict) else raw
        for rec in records[:6]:
            run_intake_pipeline(source_type="Regulatory / FAERS", raw_json=rec)
    yield


app = FastAPI(
    title="Pharmacovigilance Multi-Agent Case-Processing API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    username: str
    password: str = "pv-demo-password"


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/llm/status")
def llm_status() -> Dict[str, Any]:
    return get_ollama_status()


@app.get("/api/auth/users")
def list_demo_users() -> List[EnterpriseUser]:
    return [build_user(u) for u in DEMO_USERS]


@app.post("/api/auth/login")
def login(req: LoginRequest) -> Dict[str, Any]:
    if req.username not in DEMO_USERS or DEMO_USERS[req.username]["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = issue_token(req.username)
    user = build_user(req.username)
    return {"token": token, "user": user.model_dump()}


@app.get("/api/auth/me")
def get_me(user: EnterpriseUser = Depends(get_current_user)) -> EnterpriseUser:
    return user


@app.get("/api/stats")
def get_stats(user: EnterpriseUser = Depends(get_current_user)) -> Dict[str, Any]:
    require_permission(user, "view_cases")
    cases = db.list_cases()
    by_stage: Dict[str, int] = {s.value: 0 for s in WorkflowStage}
    by_priority: Dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "STANDARD": 0, "LOW": 0}
    by_status: Dict[str, int] = {}
    for c in cases:
        by_stage[c.current_stage.value] = by_stage.get(c.current_stage.value, 0) + 1
        by_priority[c.priority.value] = by_priority.get(c.priority.value, 0) + 1
        by_status[c.status.value] = by_status.get(c.status.value, 0) + 1
    urgent_7day = sum(1 for c in cases if c.regulatory_deadline_days <= 7 and c.status != CaseStatus.CLOSED_SUBMITTED)
    closed_count = sum(1 for c in cases if c.status == CaseStatus.CLOSED_SUBMITTED)
    return {
        "total_cases": len(cases),
        "urgent_7day_sla": urgent_7day,
        "closed_submitted": closed_count,
        "by_stage": by_stage,
        "by_priority": by_priority,
        "by_status": by_status,
    }


@app.get("/api/cases", response_model=List[PVCase])
def list_cases(
    stage: Optional[str] = None,
    priority: Optional[str] = None,
    status: Optional[str] = None,
    q: Optional[str] = None,
    user: EnterpriseUser = Depends(get_current_user),
) -> List[PVCase]:
    require_permission(user, "view_cases")
    cases = db.list_cases()
    if stage:
        cases = [c for c in cases if c.current_stage.value == stage]
    if priority:
        cases = [c for c in cases if c.priority.value == priority]
    if status:
        cases = [c for c in cases if c.status.value == status]
    if q:
        ql = q.lower()
        cases = [
            c for c in cases
            if ql in c.case_id.lower()
            or (c.safetyreportid and ql in c.safetyreportid.lower())
            or any(ql in d.name.lower() for d in c.structured_data.suspect_drugs)
            or any(ql in m.pt.lower() for m in c.meddra_codings)
        ]
    # Strip heavy raw_report blob in list view (full raw_report is returned on GET /api/cases/{case_id})
    return [c.model_copy(update={"raw_report": {}}) for c in cases]


@app.get("/api/cases/{case_id}")
def get_case_detail(case_id: str, user: EnterpriseUser = Depends(get_current_user)) -> Dict[str, Any]:
    require_permission(user, "view_cases")
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    audit = db.get_audit_trail(case_id=case_id)
    return {"case": case.model_dump(), "audit_trail": [a.model_dump() for a in audit]}


@app.post("/api/cases/ingest", response_model=PVCase)
def ingest_case(req: IngestRequest, user: EnterpriseUser = Depends(get_current_user)) -> PVCase:
    require_permission(user, "ingest_cases")
    if not req.raw_json and not (req.raw_text and req.raw_text.strip()):
        raise HTTPException(status_code=400, detail="Provide raw_text or raw_json")
    return run_intake_pipeline(
        source_type=req.source_type,
        raw_json=req.raw_json,
        raw_text=req.raw_text,
    )


@app.post("/api/cases/ingest-openfda")
def ingest_from_openfda(
    drug_name: str = "aspirin",
    limit: int = 3,
    user: EnterpriseUser = Depends(get_current_user),
) -> Dict[str, Any]:
    require_permission(user, "ingest_cases")
    """Fetch public openFDA FAERS reports (falls back to local sample if offline) and run pipeline."""
    limit = max(1, min(limit, 10))
    records: List[Dict[str, Any]] = []
    source_used = "openFDA Live API"
    api_404 = False

    try:
        url = "https://api.fda.gov/drug/event.json"
        params = {
            "search": f"patient.drug.medicinalproduct:{drug_name}+AND+_exists_:patient.reaction.reactionmeddrapt",
            "limit": limit,
        }
        resp = httpx.get(url, params=params, timeout=10.0)
        if resp.status_code == 200:
            records = resp.json().get("results", [])
        elif resp.status_code == 404:
            api_404 = True
    except Exception:
        records = []

    if api_404 and not records:
        raise HTTPException(
            status_code=404,
            detail=f"No openFDA FAERS records found for '{drug_name}'. Check spelling (e.g., 'cetirizine', 'aspirin', 'metformin').",
        )

    if not records and OPENFDA_SAMPLE_PATH.exists():
        source_used = "Cached public openFDA FAERS sample"
        raw = json.loads(OPENFDA_SAMPLE_PATH.read_text())
        all_recs = raw.get("results", raw) if isinstance(raw, dict) else raw
        records = all_recs[:limit]

    if not records:
        raise HTTPException(status_code=502, detail="Unable to fetch openFDA records")

    # For batch ingestion (limit > 1), use fast deterministic pipeline (<1s) so UI never hangs;
    # single-case ingestion (limit == 1) and Re-assess/Medical Review use live Ollama LLMs.
    use_llm = limit == 1
    created = [run_intake_pipeline(source_type="Regulatory / FAERS", raw_json=r, use_llm=use_llm) for r in records]
    return {"source": source_used, "ingested_count": len(created), "case_ids": [c.case_id for c in created]}


@app.post("/api/cases/ingest-openfda-device")
def ingest_openfda_device(
    device_name: str = "insulin pump",
    limit: int = 2,
    user: EnterpriseUser = Depends(get_current_user),
) -> Dict[str, Any]:
    require_permission(user, "ingest_cases")
    """Fetch public openFDA MAUDE medical device adverse event reports."""
    limit = max(1, min(limit, 10))
    url = "https://api.fda.gov/device/event.json"
    params = {"search": f"device.generic_name:{device_name}", "limit": limit}
    try:
        resp = httpx.get(url, params=params, timeout=10.0)
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"No openFDA MAUDE device records found for '{device_name}'.")
        resp.raise_for_status()
        records = resp.json().get("results", [])
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"openFDA Device API error: {exc}")

    created = [run_intake_pipeline(source_type="Regulatory / FDA MAUDE (Device)", raw_json=r, use_llm=(limit == 1)) for r in records]
    return {"source": "openFDA Device MAUDE API", "ingested_count": len(created), "case_ids": [c.case_id for c in created]}


@app.post("/api/cases/ingest-pubmed")
def ingest_pubmed(
    query: str = "pembrolizumab hepatotoxicity case report",
    limit: int = 2,
    user: EnterpriseUser = Depends(get_current_user),
) -> Dict[str, Any]:
    require_permission(user, "ingest_cases")
    """Fetch public PubMed literature case reports via NCBI E-utilities."""
    import xml.etree.ElementTree as ET

    limit = max(1, min(limit, 5))
    try:
        esearch = httpx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": limit, "retmode": "json"},
            timeout=10.0,
        )
        esearch.raise_for_status()
        pmids = esearch.json().get("esearchresult", {}).get("idlist", [])
        if not pmids:
            raise HTTPException(status_code=404, detail=f"No PubMed articles found for '{query}'.")

        efetch = httpx.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"},
            timeout=12.0,
        )
        efetch.raise_for_status()
        root = ET.fromstring(efetch.text)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"PubMed E-utilities error: {exc}")

    created_ids: List[str] = []
    first_word = query.split()[0] if query.split() else ""
    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID") or "UNK"
        title = "".join(article.find(".//ArticleTitle").itertext()) if article.find(".//ArticleTitle") is not None else ""
        abs_parts = ["".join(el.itertext()) for el in article.findall(".//AbstractText")]
        abstract = " ".join(abs_parts)
        pub_year = article.findtext(".//JournalIssue/PubDate/Year") or ""
        payload = {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "pub_date": pub_year,
            "query_drug": first_word,
            "raw_text": f"{title}. {abstract}",
        }
        c = run_intake_pipeline(source_type="Literature (PubMed)", raw_json=payload, use_llm=(limit == 1))
        created_ids.append(c.case_id)

    return {"source": "PubMed Literature (NCBI E-utilities)", "ingested_count": len(created_ids), "case_ids": created_ids}


@app.post("/api/cases/ingest-clinicaltrials")
def ingest_clinicaltrials(
    query: str = "semaglutide",
    limit: int = 2,
    user: EnterpriseUser = Depends(get_current_user),
) -> Dict[str, Any]:
    require_permission(user, "ingest_cases")
    """Fetch public ClinicalTrials.gov studies with reported adverse events."""
    limit = max(1, min(limit, 5))
    try:
        resp = httpx.get(
            "https://clinicaltrials.gov/api/v2/studies",
            params={"query.term": query, "filter.advanced": "AREA[HasResults]true", "pageSize": limit},
            timeout=12.0,
        )
        resp.raise_for_status()
        studies = resp.json().get("studies", [])
        if not studies:
            raise HTTPException(status_code=404, detail=f"No ClinicalTrials.gov studies with results found for '{query}'.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ClinicalTrials.gov API error: {exc}")

    created_ids: List[str] = []
    for st in studies:
        proto = st.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        nct_id = ident.get("nctId", "NCT-UNK")
        title = ident.get("briefTitle", "")
        arms = proto.get("armsInterventionsModule", {})
        interventions = [i.get("name", "") for i in arms.get("interventions", []) if i.get("name")]

        ae_mod = st.get("resultsSection", {}).get("adverseEventsModule", {})
        serious_ev = [e.get("term", "") for e in ae_mod.get("seriousEvents", [])[:4] if e.get("term")]
        other_ev = [e.get("term", "") for e in ae_mod.get("otherEvents", [])[:3] if e.get("term")]
        ae_terms = serious_ev or other_ev

        payload = {
            "nct_id": nct_id,
            "title": title,
            "interventions": interventions or [query],
            "adverse_events": ae_terms,
            "serious": "1" if serious_ev else "2",
            "ae_summary": f"Serious events reported: {', '.join(serious_ev)}. Other events: {', '.join(other_ev)}.",
            "raw_text": f"Clinical trial {nct_id} ({title}) evaluating {', '.join(interventions or [query])}. Reported adverse events: {', '.join(ae_terms)}.",
        }
        c = run_intake_pipeline(source_type="Clinical Trial (ClinicalTrials.gov)", raw_json=payload, use_llm=(limit == 1))
        created_ids.append(c.case_id)

    return {"source": "ClinicalTrials.gov API v2", "ingested_count": len(created_ids), "case_ids": created_ids}


@app.patch("/api/cases/{case_id}/fields", response_model=PVCase)
def update_case_fields(
    case_id: str,
    req: FieldUpdateRequest,
    user: EnterpriseUser = Depends(get_current_user),
) -> PVCase:
    require_permission(user, "edit_case_fields")
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    now = db.utc_now()
    changes: List[tuple[str, str, str]] = []

    if req.structured_data is not None:
        changes.append(("structured_data", case.structured_data.model_dump_json(), req.structured_data.model_dump_json()))
        case.structured_data = req.structured_data
    if req.meddra_codings is not None:
        old_pts = ", ".join(m.pt for m in case.meddra_codings)
        new_pts = ", ".join(m.pt for m in req.meddra_codings)
        changes.append(("meddra_codings", old_pts, new_pts))
        case.meddra_codings = req.meddra_codings
    if req.seriousness is not None:
        changes.append(("seriousness", str(case.seriousness.is_serious), str(req.seriousness.is_serious)))
        case.seriousness = req.seriousness
    if req.causality is not None:
        changes.append(("causality", case.causality.who_umc_category, req.causality.who_umc_category))
        case.causality = req.causality
    if req.listedness is not None:
        changes.append(("listedness", case.listedness.status, req.listedness.status))
        case.listedness = req.listedness
    if req.narrative_text is not None and req.narrative_text != case.narrative_text:
        changes.append(("narrative_text", case.narrative_text[:120], req.narrative_text[:120]))
        case.narrative_text = req.narrative_text

    case.updated_at = now
    db.save_case(case)

    for field_name, old_val, new_val in changes:
        db.append_audit(
            AuditTrailEntry(
                case_id=case_id,
                timestamp=now,
                actor=f"{user.full_name} ({user.role.value})",
                actor_type="HUMAN",
                stage=case.current_stage.value,
                action="FIELD_UPDATED",
                field_name=field_name,
                old_value=old_val,
                new_value=new_val,
                rationale=req.rationale,
            )
        )
    return case


@app.post("/api/cases/{case_id}/hitl", response_model=PVCase)
def hitl_decision(
    case_id: str,
    req: HITLDecisionRequest,
    user: EnterpriseUser = Depends(get_current_user),
) -> PVCase:
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    # Enforce Enterprise Separation of Duties per Stage
    if req.decision == "RETURN_FOR_CORRECTION":
        require_permission(user, "return_for_correction")
    elif req.decision == "REJECT_INVALID":
        require_permission(user, "reject_invalid")
    elif req.decision == "APPROVE":
        if case.current_stage in (WorkflowStage.CASE_PROCESSING, WorkflowStage.IDENTIFICATION, WorkflowStage.DUPLICATE_CHECK):
            require_permission(user, "submit_to_qc")
        elif case.current_stage == WorkflowStage.QUALITY_CONTROL:
            require_permission(user, "approve_qc")
        elif case.current_stage == WorkflowStage.MEDICAL_REVIEW:
            require_permission(user, "approve_medical")
        elif case.current_stage == WorkflowStage.PROJECT_LEAD_REVIEW:
            require_permission(user, "approve_lead")
    elif req.decision == "CLOSE_FINALIZE":
        require_permission(user, "approve_lead")

    now = db.utc_now()
    old_stage = case.current_stage.value
    old_status = case.status.value

    if req.decision == "RETURN_FOR_CORRECTION":
        case.current_stage = WorkflowStage.CASE_PROCESSING
        case.status = CaseStatus.REWORK_REQUIRED
    elif req.decision == "REJECT_INVALID":
        case.current_stage = WorkflowStage.IDENTIFICATION
        case.status = CaseStatus.INVALID_CASE
    elif req.decision == "APPROVE":
        if case.current_stage in (WorkflowStage.CASE_PROCESSING, WorkflowStage.IDENTIFICATION, WorkflowStage.DUPLICATE_CHECK):
            qc_agent = QCAgent()
            qc_res, qc_trace = qc_agent.run(case)
            case.qc_checklist = qc_res
            case.agent_traces.append(qc_trace)
            case.current_stage = WorkflowStage.QUALITY_CONTROL
            case.status = CaseStatus.AWAITING_QC
        elif case.current_stage == WorkflowStage.QUALITY_CONTROL:
            med_agent = MedicalReviewAgent()
            med_res, med_trace = med_agent.run(case)
            case.medical_review = med_res
            case.agent_traces.append(med_trace)
            case.current_stage = WorkflowStage.MEDICAL_REVIEW
            case.status = CaseStatus.AWAITING_MEDICAL_REVIEW
        elif case.current_stage == WorkflowStage.MEDICAL_REVIEW:
            case.current_stage = WorkflowStage.PROJECT_LEAD_REVIEW
            case.status = CaseStatus.AWAITING_LEAD_APPROVAL
        elif case.current_stage == WorkflowStage.PROJECT_LEAD_REVIEW:
            case.current_stage = WorkflowStage.CASE_CLOSURE
            case.status = CaseStatus.CLOSED_SUBMITTED
            case.regulatory_submission_id = f"E2B-R3-{uuid.uuid4().hex[:8].upper()}"
    elif req.decision == "CLOSE_FINALIZE":
        case.current_stage = WorkflowStage.CASE_CLOSURE
        case.status = CaseStatus.CLOSED_SUBMITTED
        if not case.regulatory_submission_id:
            case.regulatory_submission_id = f"E2B-R3-{uuid.uuid4().hex[:8].upper()}"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown decision: {req.decision}")

    case.updated_at = now
    db.save_case(case)

    db.append_audit(
        AuditTrailEntry(
            case_id=case_id,
            timestamp=now,
            actor=f"{user.full_name} ({user.role.value})",
            actor_type="HUMAN",
            stage=case.current_stage.value,
            action=f"HITL_{req.decision}",
            field_name="workflow_stage",
            old_value=f"{old_stage} [{old_status}]",
            new_value=f"{case.current_stage.value} [{case.status.value}]",
            rationale=req.comments or "Human-in-the-loop review decision",
        )
    )
    return case


@app.get("/api/audit-trail", response_model=List[AuditTrailEntry])
def global_audit_trail(
    case_id: Optional[str] = None,
    limit: int = 200,
    user: EnterpriseUser = Depends(get_current_user),
) -> List[AuditTrailEntry]:
    require_permission(user, "view_audit")
    return db.get_audit_trail(case_id=case_id, limit=limit)


@app.get("/api/eval/run")
def run_eval_suite(user: EnterpriseUser = Depends(get_current_user)) -> Dict[str, Any]:
    require_permission(user, "run_evals")
    return run_evaluation()


@app.post("/api/cases/{case_id}/reassess", response_model=PVCase)
def reassess_case(case_id: str, user: EnterpriseUser = Depends(get_current_user)) -> PVCase:
    require_permission(user, "edit_case_fields")
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    assess_agent = ClinicalAssessmentAgent()
    seriousness, causality, listedness, narrative, trace5 = assess_agent.run(
        case.raw_report, case.structured_data, case.meddra_codings, use_llm=True, deep_reasoning=True
    )
    case.seriousness = seriousness
    case.causality = causality
    case.listedness = listedness
    case.narrative_text = narrative
    case.agent_traces.append(trace5)

    qc_agent = QCAgent()
    qc_res, trace6 = qc_agent.run(case)
    case.qc_checklist = qc_res
    case.agent_traces.append(trace6)
    case.updated_at = db.utc_now()
    db.save_case(case)

    db.append_audit(
        AuditTrailEntry(
            case_id=case_id,
            timestamp=case.updated_at,
            actor=f"{user.full_name} ({user.role.value})",
            actor_type="HUMAN",
            stage=case.current_stage.value,
            action="REASSESS_TRIGGERED",
            rationale="Re-ran ClinicalAssessmentAgent and QCAgent after case edits.",
        )
    )
    return case


@app.get("/api/cases/{case_id}/export-e2b")
def export_e2b_xml(case_id: str, user: EnterpriseUser = Depends(get_current_user)) -> Response:
    require_permission(user, "view_cases")
    case = db.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    from xml.sax.saxutils import escape

    drugs_xml = "\n".join(
        f"      <drug><characterization>{escape(d.role)}</characterization><medicinalproduct>{escape(d.name)}</medicinalproduct></drug>"
        for d in (case.structured_data.suspect_drugs + case.structured_data.concomitant_drugs)
    )
    reactions_xml = "\n".join(
        f"      <reaction><pt>{escape(m.pt)}</pt><soc>{escape(m.soc)}</soc><meddraversion>{escape(m.meddra_version)}</meddraversion></reaction>"
        for m in case.meddra_codings
    )
    xml_body = f"""<?xml version="1.0" encoding="UTF-8"?>
<ichicsr lang="en">
  <safetyreport>
    <safetyreportid>{escape(case.regulatory_submission_id or case.case_id)}</safetyreportid>
    <caseid>{escape(case.case_id)}</caseid>
    <serious>{'1' if case.seriousness.is_serious else '2'}</serious>
    <causality>{escape(case.causality.who_umc_category)}</causality>
    <listedness>{escape(case.listedness.status)}</listedness>
    <patient>
      <age>{escape(case.structured_data.patient_age or 'Unknown')}</age>
      <sex>{escape(case.structured_data.patient_sex or 'Unknown')}</sex>
{drugs_xml}
{reactions_xml}
    </patient>
    <narrative>{escape(case.narrative_text)}</narrative>
  </safetyreport>
</ichicsr>
"""
    return Response(
        content=xml_body,
        media_type="application/xml",
        headers={"Content-Disposition": f'attachment; filename="{case.case_id}-e2b.xml"'},
    )


FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(str(candidate))
        return FileResponse(str(FRONTEND_DIST / "index.html"))
