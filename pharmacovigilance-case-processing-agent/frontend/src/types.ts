export type WorkflowStage =
  | '1_CASE_INTAKE'
  | '2_CASE_IDENTIFICATION'
  | '3_TRIAGE'
  | '4_DUPLICATE_CHECK'
  | '5_CASE_PROCESSING'
  | '6_QUALITY_CONTROL'
  | '7_MEDICAL_REVIEW'
  | '8_PROJECT_LEAD_REVIEW'
  | '9_CASE_CLOSURE';

export const STAGE_LABELS: { key: WorkflowStage; short: string; full: string; role: string }[] = [
  { key: '1_CASE_INTAKE', short: '1. Intake', full: '1. Case Intake', role: 'IntakeAgent' },
  { key: '2_CASE_IDENTIFICATION', short: '2. Identification', full: '2. Case Identification', role: 'CaseValidityAgent' },
  { key: '3_TRIAGE', short: '3. Triage', full: '3. Triage & SLA', role: 'TriageAgent' },
  { key: '4_DUPLICATE_CHECK', short: '4. Duplicate', full: '4. Duplicate Check', role: 'DuplicateCheckAgent' },
  { key: '5_CASE_PROCESSING', short: '5. Processing', full: '5. Case Processing', role: 'Case Processor' },
  { key: '6_QUALITY_CONTROL', short: '6. QC', full: '6. Quality Control', role: 'QC Reviewer' },
  { key: '7_MEDICAL_REVIEW', short: '7. Medical', full: '7. Medical Review (MD)', role: 'Medical Reviewer (MD)' },
  { key: '8_PROJECT_LEAD_REVIEW', short: '8. Lead', full: '8. Project Lead Approval', role: 'Project Lead' },
  { key: '9_CASE_CLOSURE', short: '9. Closure', full: '9. Case Closure', role: 'Finalized / E2B' },
];

export interface DrugEntry {
  name: string;
  role: string;
  dose?: string | null;
  route?: string | null;
  indication?: string | null;
  start_date?: string | null;
  action_taken?: string | null;
}

export interface MedDRACoding {
  verbatim_term: string;
  llt: string;
  pt: string;
  soc: string;
  meddra_version: string;
  confidence: number;
  accepted: boolean;
  coding_rationale: string;
}

export interface PVCase {
  case_id: string;
  safetyreportid?: string | null;
  source_type: string;
  current_stage: WorkflowStage;
  status: string;
  priority: 'CRITICAL' | 'HIGH' | 'STANDARD' | 'LOW';
  regulatory_deadline_days: number;
  duplicate_status: 'NEW' | 'FOLLOW_UP' | 'DUPLICATE';
  parent_case_id?: string | null;
  raw_report: Record<string, unknown>;
  validity: {
    is_valid: boolean;
    identifiable_patient: boolean;
    identifiable_reporter: boolean;
    suspect_product: boolean;
    adverse_event: boolean;
    missing_criteria: string[];
    rationale: string;
  };
  triage: {
    priority: string;
    is_serious: boolean;
    report_type: string;
    source_channel: string;
    regulatory_timeline_days: number;
    sla_label: string;
    rationale: string;
  };
  duplicate_check: {
    status: string;
    matched_case_id?: string | null;
    similarity_score: number;
    match_reasons: string[];
  };
  structured_data: {
    safety_report_id?: string | null;
    receive_date?: string | null;
    country?: string | null;
    patient_age?: string | null;
    patient_sex?: string | null;
    patient_weight_kg?: string | null;
    reporter_qualification?: string | null;
    reporter_country?: string | null;
    suspect_drugs: DrugEntry[];
    concomitant_drugs: DrugEntry[];
    adverse_events_raw: string[];
    medical_history: string[];
    lab_findings: string[];
    reaction_outcome?: string | null;
  };
  meddra_codings: MedDRACoding[];
  seriousness: {
    is_serious: boolean;
    death: boolean;
    life_threatening: boolean;
    hospitalization: boolean;
    disability: boolean;
    congenital_anomaly: boolean;
    other_medically_important: boolean;
    rationale: string;
  };
  causality: {
    who_umc_category: string;
    naranjo_score: number;
    naranjo_category: string;
    dechallenge_positive?: boolean | null;
    rechallenge_positive?: boolean | null;
    temporal_relationship: boolean;
    alternative_causes_present: boolean;
    rationale: string;
  };
  listedness: {
    status: string;
    reference_document: string;
    matched_label_terms: string[];
    rationale: string;
  };
  narrative_text: string;
  qc_checklist?: {
    passed: boolean;
    data_entry_accurate: boolean;
    completeness_ok: boolean;
    coding_accurate: boolean;
    assessments_consistent: boolean;
    narrative_consistent: boolean;
    sop_compliant: boolean;
    issues: string[];
  } | null;
  medical_review?: {
    clinically_sound: boolean;
    clinical_notes: string;
    recommended_action: string;
    flagged_items: string[];
  } | null;
  regulatory_submission_id?: string | null;
  agent_traces: {
    agent_name: string;
    stage: WorkflowStage;
    timestamp: string;
    duration_ms: number;
    status: string;
    summary: string;
    details: Record<string, unknown>;
  }[];
  created_at: string;
  updated_at: string;
}

export interface AuditTrailEntry {
  id?: number;
  case_id: string;
  timestamp: string;
  actor: string;
  actor_type: 'AGENT' | 'HUMAN';
  stage: string;
  action: string;
  field_name?: string | null;
  old_value?: string | null;
  new_value?: string | null;
  rationale?: string | null;
}

export interface EvalResult {
  total_cases: number;
  synthetic_cases: number;
  openfda_cases: number;
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  per_field: {
    field: string;
    precision: number;
    recall: number;
    f1: number;
    exact_match: number;
    tp: number;
    fp: number;
    fn: number;
    support: number;
  }[];
  cases: {
    id: string;
    source_type: string;
    validity_match: boolean;
    meddra_pt_match: boolean;
    seriousness_match: boolean;
    pred_pts: string[];
    gold_pts: string[];
  }[];
}
