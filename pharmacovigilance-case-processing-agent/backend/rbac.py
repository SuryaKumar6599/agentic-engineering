from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from enum import Enum
from typing import Dict, List, Optional, Set

from fastapi import Header, HTTPException
from pydantic import BaseModel

_TOKEN_SECRET = os.environ.get("PV_TOKEN_SECRET", "pv-enterprise-local-hmac-secret-2026").encode("utf-8")


class EnterpriseRole(str, Enum):
    CASE_PROCESSOR = "CASE_PROCESSOR"
    QC_REVIEWER = "QC_REVIEWER"
    MEDICAL_REVIEWER = "MEDICAL_REVIEWER"
    PROJECT_LEAD = "PROJECT_LEAD"
    ADMIN = "ADMIN"


ROLE_PERMISSIONS: Dict[EnterpriseRole, Set[str]] = {
    EnterpriseRole.CASE_PROCESSOR: {
        "view_cases",
        "view_audit",
        "ingest_cases",
        "edit_case_fields",
        "submit_to_qc",
    },
    EnterpriseRole.QC_REVIEWER: {
        "view_cases",
        "view_audit",
        "approve_qc",
        "return_for_correction",
    },
    EnterpriseRole.MEDICAL_REVIEWER: {
        "view_cases",
        "view_audit",
        "approve_medical",
        "return_for_correction",
        "reject_invalid",
    },
    EnterpriseRole.PROJECT_LEAD: {
        "view_cases",
        "view_audit",
        "approve_lead",
        "return_for_correction",
        "run_evals",
    },
    EnterpriseRole.ADMIN: {
        "view_cases",
        "view_audit",
        "ingest_cases",
        "edit_case_fields",
        "submit_to_qc",
        "approve_qc",
        "approve_medical",
        "approve_lead",
        "return_for_correction",
        "reject_invalid",
        "run_evals",
    },
}


class EnterpriseUser(BaseModel):
    username: str
    full_name: str
    email: str
    role: EnterpriseRole
    department: str
    permissions: List[str]


DEMO_USERS: Dict[str, Dict[str, str]] = {
    "jane.processor": {
        "password": "pv-demo-password",
        "full_name": "Jane Miller",
        "email": "jane.miller@pv-enterprise.org",
        "role": EnterpriseRole.CASE_PROCESSOR.value,
        "department": "Case Intake & Processing",
    },
    "sam.qc": {
        "password": "pv-demo-password",
        "full_name": "Samira Khan",
        "email": "samira.khan@pv-enterprise.org",
        "role": EnterpriseRole.QC_REVIEWER.value,
        "department": "Safety Quality Control",
    },
    "dr.patel": {
        "password": "pv-demo-password",
        "full_name": "Dr. Arjun Patel, MD",
        "email": "arjun.patel@pv-enterprise.org",
        "role": EnterpriseRole.MEDICAL_REVIEWER.value,
        "department": "Medical Safety Review",
    },
    "alex.lead": {
        "password": "pv-demo-password",
        "full_name": "Alex Rivera",
        "email": "alex.rivera@pv-enterprise.org",
        "role": EnterpriseRole.PROJECT_LEAD.value,
        "department": "Pharmacovigilance Operations",
    },
    "admin.pv": {
        "password": "pv-demo-password",
        "full_name": "Elena Vance",
        "email": "elena.vance@pv-enterprise.org",
        "role": EnterpriseRole.ADMIN.value,
        "department": "PV System Administration",
    },
}


def build_user(username: str) -> EnterpriseUser:
    info = DEMO_USERS[username]
    role = EnterpriseRole(info["role"])
    return EnterpriseUser(
        username=username,
        full_name=info["full_name"],
        email=info["email"],
        role=role,
        department=info["department"],
        permissions=sorted(ROLE_PERMISSIONS[role]),
    )


def issue_token(username: str, ttl_seconds: int = 86400 * 7) -> str:
    payload = {"sub": username, "exp": int(time.time()) + ttl_seconds}
    b64 = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    sig = hmac.new(_TOKEN_SECRET, b64.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{b64}.{sig}"


def _verify_token(token: str) -> EnterpriseUser:
    try:
        b64, sig = token.split(".", 1)
        expected = hmac.new(_TOKEN_SECRET, b64.encode("ascii"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise ValueError("Bad signature")
        payload = json.loads(base64.urlsafe_b64decode(b64.encode("ascii")).decode("utf-8"))
        if int(payload.get("exp", 0)) < int(time.time()):
            raise ValueError("Token expired")
        username = payload.get("sub", "")
        if username not in DEMO_USERS:
            raise ValueError("Unknown user")
        return build_user(username)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")


def get_current_user(
    authorization: Optional[str] = Header(default=None),
) -> EnterpriseUser:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication required. Sign in via /api/auth/login.")
    token = authorization.removeprefix("Bearer ").strip()
    return _verify_token(token)


def require_permission(user: EnterpriseUser, perm: str) -> None:
    if perm not in user.permissions:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied for role {user.role.value}: missing permission '{perm}'.",
        )
