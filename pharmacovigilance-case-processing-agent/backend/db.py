from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.models import (
    AgentTraceStep,
    AuditTrailEntry,
    CaseStatus,
    CausalityAssessment,
    DuplicateCheckResult,
    DuplicateStatus,
    ListednessAssessment,
    MedDRACoding,
    MedicalReviewSummary,
    PVCase,
    PriorityLevel,
    QCChecklist,
    SeriousnessAssessment,
    StructuredCaseData,
    TriageAssessment,
    ValidityCheck,
    WorkflowStage,
)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "pv_safety.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            safetyreportid TEXT,
            source_type TEXT NOT NULL,
            current_stage TEXT NOT NULL,
            status TEXT NOT NULL,
            priority TEXT NOT NULL,
            regulatory_deadline_days INTEGER NOT NULL,
            duplicate_status TEXT NOT NULL,
            parent_case_id TEXT,
            raw_report_json TEXT NOT NULL,
            validity_json TEXT NOT NULL,
            triage_json TEXT NOT NULL,
            duplicate_check_json TEXT NOT NULL,
            structured_data_json TEXT NOT NULL,
            meddra_codings_json TEXT NOT NULL,
            seriousness_json TEXT NOT NULL,
            causality_json TEXT NOT NULL,
            listedness_json TEXT NOT NULL,
            narrative_text TEXT NOT NULL,
            qc_checklist_json TEXT,
            medical_review_json TEXT,
            regulatory_submission_id TEXT,
            agent_traces_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_trail (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            actor TEXT NOT NULL,
            actor_type TEXT NOT NULL,
            stage TEXT NOT NULL,
            action TEXT NOT NULL,
            field_name TEXT,
            old_value TEXT,
            new_value TEXT,
            rationale TEXT
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_audit_case ON audit_trail(case_id)")
    conn.commit()
    conn.close()


def _row_to_case(row: sqlite3.Row) -> PVCase:
    return PVCase(
        case_id=row["case_id"],
        safetyreportid=row["safetyreportid"],
        source_type=row["source_type"],
        current_stage=WorkflowStage(row["current_stage"]),
        status=CaseStatus(row["status"]),
        priority=PriorityLevel(row["priority"]),
        regulatory_deadline_days=row["regulatory_deadline_days"],
        duplicate_status=DuplicateStatus(row["duplicate_status"]),
        parent_case_id=row["parent_case_id"],
        raw_report=json.loads(row["raw_report_json"]),
        validity=ValidityCheck(**json.loads(row["validity_json"])),
        triage=TriageAssessment(**json.loads(row["triage_json"])),
        duplicate_check=DuplicateCheckResult(**json.loads(row["duplicate_check_json"])),
        structured_data=StructuredCaseData(**json.loads(row["structured_data_json"])),
        meddra_codings=[MedDRACoding(**m) for m in json.loads(row["meddra_codings_json"])],
        seriousness=SeriousnessAssessment(**json.loads(row["seriousness_json"])),
        causality=CausalityAssessment(**json.loads(row["causality_json"])),
        listedness=ListednessAssessment(**json.loads(row["listedness_json"])),
        narrative_text=row["narrative_text"],
        qc_checklist=QCChecklist(**json.loads(row["qc_checklist_json"])) if row["qc_checklist_json"] else None,
        medical_review=MedicalReviewSummary(**json.loads(row["medical_review_json"])) if row["medical_review_json"] else None,
        regulatory_submission_id=row["regulatory_submission_id"],
        agent_traces=[AgentTraceStep(**t) for t in json.loads(row["agent_traces_json"])],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def save_case(case: PVCase) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO cases (
            case_id, safetyreportid, source_type, current_stage, status, priority,
            regulatory_deadline_days, duplicate_status, parent_case_id,
            raw_report_json, validity_json, triage_json, duplicate_check_json,
            structured_data_json, meddra_codings_json, seriousness_json,
            causality_json, listedness_json, narrative_text,
            qc_checklist_json, medical_review_json, regulatory_submission_id,
            agent_traces_json, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            case.case_id,
            case.safetyreportid,
            case.source_type,
            case.current_stage.value,
            case.status.value,
            case.priority.value,
            case.regulatory_deadline_days,
            case.duplicate_status.value,
            case.parent_case_id,
            json.dumps(case.raw_report),
            case.validity.model_dump_json(),
            case.triage.model_dump_json(),
            case.duplicate_check.model_dump_json(),
            case.structured_data.model_dump_json(),
            json.dumps([m.model_dump() for m in case.meddra_codings]),
            case.seriousness.model_dump_json(),
            case.causality.model_dump_json(),
            case.listedness.model_dump_json(),
            case.narrative_text,
            case.qc_checklist.model_dump_json() if case.qc_checklist else None,
            case.medical_review.model_dump_json() if case.medical_review else None,
            case.regulatory_submission_id,
            json.dumps([t.model_dump() for t in case.agent_traces]),
            case.created_at,
            case.updated_at,
        ),
    )
    conn.commit()
    conn.close()


def get_case(case_id: str) -> Optional[PVCase]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
    row = cur.fetchone()
    conn.close()
    return _row_to_case(row) if row else None


def list_cases() -> List[PVCase]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM cases ORDER BY updated_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [_row_to_case(r) for r in rows]


def append_audit(entry: AuditTrailEntry) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO audit_trail (
            case_id, timestamp, actor, actor_type, stage, action,
            field_name, old_value, new_value, rationale
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entry.case_id,
            entry.timestamp,
            entry.actor,
            entry.actor_type,
            entry.stage,
            entry.action,
            entry.field_name,
            entry.old_value,
            entry.new_value,
            entry.rationale,
        ),
    )
    conn.commit()
    conn.close()


def get_audit_trail(case_id: Optional[str] = None, limit: int = 200) -> List[AuditTrailEntry]:
    conn = get_conn()
    cur = conn.cursor()
    if case_id:
        cur.execute(
            "SELECT * FROM audit_trail WHERE case_id = ? ORDER BY id DESC LIMIT ?",
            (case_id, limit),
        )
    else:
        cur.execute("SELECT * FROM audit_trail ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [
        AuditTrailEntry(
            id=r["id"],
            case_id=r["case_id"],
            timestamp=r["timestamp"],
            actor=r["actor"],
            actor_type=r["actor_type"],
            stage=r["stage"],
            action=r["action"],
            field_name=r["field_name"],
            old_value=r["old_value"],
            new_value=r["new_value"],
            rationale=r["rationale"],
        )
        for r in rows
    ]
