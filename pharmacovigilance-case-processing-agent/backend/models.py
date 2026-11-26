from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class WorkflowStage(str, Enum):
    INTAKE = "1_CASE_INTAKE"
    IDENTIFICATION = "2_CASE_IDENTIFICATION"
    TRIAGE = "3_TRIAGE"
    DUPLICATE_CHECK = "4_DUPLICATE_CHECK"
    CASE_PROCESSING = "5_CASE_PROCESSING"
    QUALITY_CONTROL = "6_QUALITY_CONTROL"
    MEDICAL_REVIEW = "7_MEDICAL_REVIEW"
    PROJECT_LEAD_REVIEW = "8_PROJECT_LEAD_REVIEW"
    CASE_CLOSURE = "9_CASE_CLOSURE"


class CaseStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    AWAITING_QC = "AWAITING_QC"
    AWAITING_MEDICAL_REVIEW = "AWAITING_MEDICAL_REVIEW"
    AWAITING_LEAD_APPROVAL = "AWAITING_LEAD_APPROVAL"
    REWORK_REQUIRED = "REWORK_REQUIRED"
    INVALID_CASE = "INVALID_CASE"
    DUPLICATE_CLOSED = "DUPLICATE_CLOSED"
    CLOSED_SUBMITTED = "CLOSED_SUBMITTED"


class PriorityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    STANDARD = "STANDARD"
    LOW = "LOW"


class DuplicateStatus(str, Enum):
    NEW = "NEW"
    FOLLOW_UP = "FOLLOW_UP"
    DUPLICATE = "DUPLICATE"


class ValidityCheck(BaseModel):
    is_valid: bool
    identifiable_patient: bool
    identifiable_reporter: bool
    suspect_product: bool
    adverse_event: bool
    missing_criteria: List[str] = Field(default_factory=list)
    rationale: str = ""


class TriageAssessment(BaseModel):
    priority: PriorityLevel
    is_serious: bool
    report_type: str  # Spontaneous, Literature, Clinical Trial, Regulatory
    source_channel: str  # HCP, Patient/Consumer, Literature, Clinical Trial, FAERS
    regulatory_timeline_days: int  # 7, 15, or 90
    sla_label: str
    rationale: str = ""


class DuplicateCheckResult(BaseModel):
    status: DuplicateStatus
    matched_case_id: Optional[str] = None
    similarity_score: float = 0.0
    match_reasons: List[str] = Field(default_factory=list)


class DrugEntry(BaseModel):
    name: str
    role: str = "Suspect"  # Suspect, Concomitant, Interacting
    dose: Optional[str] = None
    route: Optional[str] = None
    indication: Optional[str] = None
    start_date: Optional[str] = None
    action_taken: Optional[str] = None


class StructuredCaseData(BaseModel):
    safety_report_id: Optional[str] = None
    receive_date: Optional[str] = None
    country: Optional[str] = None
    patient_age: Optional[str] = None
    patient_sex: Optional[str] = None
    patient_weight_kg: Optional[str] = None
    reporter_qualification: Optional[str] = None
    reporter_country: Optional[str] = None
    suspect_drugs: List[DrugEntry] = Field(default_factory=list)
    concomitant_drugs: List[DrugEntry] = Field(default_factory=list)
    adverse_events_raw: List[str] = Field(default_factory=list)
    medical_history: List[str] = Field(default_factory=list)
    lab_findings: List[str] = Field(default_factory=list)
    reaction_outcome: Optional[str] = None


class MedDRACoding(BaseModel):
    verbatim_term: str
    llt: str
    pt: str
    soc: str
    meddra_version: str = "29.0"
    confidence: float
    accepted: bool = True
    coding_rationale: str = ""


class SeriousnessAssessment(BaseModel):
    is_serious: bool
    death: bool = False
    life_threatening: bool = False
    hospitalization: bool = False
    disability: bool = False
    congenital_anomaly: bool = False
    other_medically_important: bool = False
    rationale: str = ""


class CausalityAssessment(BaseModel):
    who_umc_category: str  # Certain, Probable/Likely, Possible, Unlikely, Unassessable
    naranjo_score: int
    naranjo_category: str
    dechallenge_positive: Optional[bool] = None
    rechallenge_positive: Optional[bool] = None
    temporal_relationship: bool = True
    alternative_causes_present: bool = False
    rationale: str = ""


class ListednessAssessment(BaseModel):
    status: str  # Listed / Expected, Unlisted / Unexpected, Indeterminate
    reference_document: str = "Core Company Data Sheet (CCDS) / FDA USPI"
    matched_label_terms: List[str] = Field(default_factory=list)
    rationale: str = ""


class QCChecklist(BaseModel):
    passed: bool
    data_entry_accurate: bool
    completeness_ok: bool
    coding_accurate: bool
    assessments_consistent: bool
    narrative_consistent: bool
    sop_compliant: bool
    issues: List[str] = Field(default_factory=list)


class MedicalReviewSummary(BaseModel):
    clinically_sound: bool
    clinical_notes: str
    recommended_action: str  # APPROVE_TO_LEAD or RETURN_FOR_CORRECTION
    flagged_items: List[str] = Field(default_factory=list)


class AgentTraceStep(BaseModel):
    agent_name: str
    stage: WorkflowStage
    timestamp: str
    duration_ms: int
    status: str
    summary: str
    details: Dict[str, Any] = Field(default_factory=dict)


class AuditTrailEntry(BaseModel):
    id: Optional[int] = None
    case_id: str
    timestamp: str
    actor: str
    actor_type: str  # AGENT or HUMAN
    stage: str
    action: str
    field_name: Optional[str] = None
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    rationale: Optional[str] = None


class PVCase(BaseModel):
    case_id: str
    safetyreportid: Optional[str] = None
    source_type: str
    current_stage: WorkflowStage
    status: CaseStatus
    priority: PriorityLevel
    regulatory_deadline_days: int
    duplicate_status: DuplicateStatus
    parent_case_id: Optional[str] = None
    raw_report: Dict[str, Any]
    validity: ValidityCheck
    triage: TriageAssessment
    duplicate_check: DuplicateCheckResult
    structured_data: StructuredCaseData
    meddra_codings: List[MedDRACoding] = Field(default_factory=list)
    seriousness: SeriousnessAssessment
    causality: CausalityAssessment
    listedness: ListednessAssessment
    narrative_text: str
    qc_checklist: Optional[QCChecklist] = None
    medical_review: Optional[MedicalReviewSummary] = None
    regulatory_submission_id: Optional[str] = None
    agent_traces: List[AgentTraceStep] = Field(default_factory=list)
    created_at: str
    updated_at: str


class IngestRequest(BaseModel):
    source_type: str = "Spontaneous (HCP)"
    raw_text: Optional[str] = None
    raw_json: Optional[Dict[str, Any]] = None


class FieldUpdateRequest(BaseModel):
    actor: str = "Case Processor"
    rationale: str = "Manual data entry / correction"
    structured_data: Optional[StructuredCaseData] = None
    meddra_codings: Optional[List[MedDRACoding]] = None
    seriousness: Optional[SeriousnessAssessment] = None
    causality: Optional[CausalityAssessment] = None
    listedness: Optional[ListednessAssessment] = None
    narrative_text: Optional[str] = None


class HITLDecisionRequest(BaseModel):
    reviewer_name: str
    reviewer_role: str  # QC_REVIEWER, MEDICAL_REVIEWER, PROJECT_LEAD, CASE_PROCESSOR
    decision: str  # APPROVE, RETURN_FOR_CORRECTION, REJECT_INVALID, CLOSE_FINALIZE
    comments: str = ""
