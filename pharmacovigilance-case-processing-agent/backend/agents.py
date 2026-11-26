from __future__ import annotations

import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from backend import db
from backend.llm_client import FAST_MODEL, REASONING_MODEL, ollama_chat_json, ollama_chat_text
from backend.meddra_dict import MEDDRA_TERMS, check_listedness, map_to_meddra
from backend.models import (
    AgentTraceStep,
    AuditTrailEntry,
    CaseStatus,
    CausalityAssessment,
    DrugEntry,
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


def _qual_code_to_label(code: Optional[str]) -> str:
    mapping = {
        "1": "Physician",
        "2": "Pharmacist",
        "3": "Other Health Professional",
        "4": "Lawyer",
        "5": "Consumer or Non-Health Professional",
    }
    if not code:
        return "Unknown"
    return mapping.get(str(code), str(code))


def _sex_code_to_label(code: Optional[str]) -> str:
    mapping = {"1": "Male", "2": "Female", "0": "Unknown"}
    if not code:
        return "Unknown"
    return mapping.get(str(code), str(code))


def _outcome_code_to_label(code: Optional[str]) -> str:
    mapping = {
        "1": "Recovered/Resolved",
        "2": "Recovering/Resolving",
        "3": "Not Recovered/Not Resolved",
        "4": "Recovered/Resolved with Sequelae",
        "5": "Fatal",
        "6": "Unknown",
    }
    if not code:
        return "Unknown"
    return mapping.get(str(code), str(code))


class IntakeAgent:
    """Stage 1: Case Intake — normalizes openFDA JSON or unstructured text reports."""

    name = "IntakeAgent"

    def run(self, source_type: str, raw_json: Optional[Dict[str, Any]], raw_text: Optional[str]) -> Tuple[Dict[str, Any], str, AgentTraceStep]:
        t0 = time.perf_counter()
        if raw_json:
            payload = dict(raw_json)
            qual = payload.get("primarysource", {}).get("qualification")
            inferred_source = source_type
            if "mdr_report_key" in payload or "device" in payload:
                inferred_source = "Regulatory / FDA MAUDE (Device)"
            elif "pmid" in payload:
                inferred_source = "Literature (PubMed)"
            elif "nct_id" in payload:
                inferred_source = "Clinical Trial (ClinicalTrials.gov)"
            elif source_type == "Spontaneous (HCP)" and qual == "5":
                inferred_source = "Spontaneous (Patient/Consumer)"
            elif "literature" in str(payload.get("literaturereference", "")).lower():
                inferred_source = "Literature"
            summary = f"Normalized structured report (id={payload.get('safetyreportid') or payload.get('mdr_report_key') or payload.get('pmid') or payload.get('nct_id') or 'N/A'}) from {inferred_source}."
        else:
            text = (raw_text or "").strip()
            payload = {"raw_text": text, "source_type": source_type}
            inferred_source = source_type
            summary = f"Ingested unstructured narrative report ({len(text)} chars) from {inferred_source}."

        trace = AgentTraceStep(
            agent_name=self.name,
            stage=WorkflowStage.INTAKE,
            timestamp=db.utc_now(),
            duration_ms=max(1, int((time.perf_counter() - t0) * 1000)),
            status="COMPLETED",
            summary=summary,
            details={"source_type": inferred_source},
        )
        return payload, inferred_source, trace


class ExtractionAndCodingAgent:
    """Stage 5a (also supports Stage 2/3/4): Extracts structured fields & suggests MedDRA codings."""

    name = "ExtractionAndCodingAgent"

    def extract_structured(self, payload: Dict[str, Any], use_llm: bool = False) -> StructuredCaseData:
        if "mdr_report_key" in payload or ("device" in payload and "patient" in payload and isinstance(payload.get("patient"), list)):
            devices = payload.get("device", []) or []
            suspect_drugs = [
                DrugEntry(
                    name=(d.get("brand_name") or d.get("generic_name") or "Medical Device").upper(),
                    role="Suspect",
                    indication=d.get("device_report_product_code"),
                )
                for d in devices[:3]
            ]
            mdr_texts = [t.get("text", "") for t in (payload.get("mdr_text") or []) if t.get("text")]
            combined_text = " ".join(mdr_texts)
            text_extracted = self.extract_structured({"raw_text": combined_text}, use_llm=use_llm) if combined_text else StructuredCaseData()
            aes = text_extracted.adverse_events_raw or [payload.get("event_type") or "Device malfunction"]
            return StructuredCaseData(
                safety_report_id=str(payload.get("mdr_report_key") or payload.get("report_number") or ""),
                receive_date=payload.get("date_received"),
                country="US",
                patient_age=text_extracted.patient_age,
                patient_sex=text_extracted.patient_sex or "Unknown",
                reporter_qualification="Other Health Professional" if payload.get("health_professional") == "Y" else "Consumer or Non-Health Professional",
                suspect_drugs=suspect_drugs or text_extracted.suspect_drugs,
                concomitant_drugs=[],
                adverse_events_raw=aes,
                reaction_outcome="Fatal" if payload.get("event_type") == "Death" else "Unknown",
            )

        if "pmid" in payload:
            abstract = f"{payload.get('title', '')}. {payload.get('abstract', '')}"
            text_extracted = self.extract_structured({"raw_text": abstract}, use_llm=use_llm)
            return StructuredCaseData(
                safety_report_id=f"PMID-{payload['pmid']}",
                receive_date=payload.get("pub_date"),
                country=text_extracted.country or "Global",
                patient_age=text_extracted.patient_age,
                patient_sex=text_extracted.patient_sex or "Unknown",
                reporter_qualification="Physician",
                suspect_drugs=text_extracted.suspect_drugs or ([DrugEntry(name=payload["query_drug"].upper(), role="Suspect")] if payload.get("query_drug") else []),
                concomitant_drugs=text_extracted.concomitant_drugs,
                adverse_events_raw=text_extracted.adverse_events_raw or ["Adverse drug reaction"],
                lab_findings=text_extracted.lab_findings,
                reaction_outcome=text_extracted.reaction_outcome,
            )

        if "nct_id" in payload:
            summary_txt = f"{payload.get('title', '')}. {payload.get('ae_summary', '')}"
            text_extracted = self.extract_structured({"raw_text": summary_txt}, use_llm=use_llm)
            interventions = [DrugEntry(name=n.upper(), role="Suspect") for n in (payload.get("interventions") or [])[:3]]
            return StructuredCaseData(
                safety_report_id=payload["nct_id"],
                receive_date=payload.get("update_date"),
                country="US",
                patient_age=text_extracted.patient_age or "Adult",
                patient_sex=text_extracted.patient_sex or "Unknown",
                reporter_qualification="Physician",
                suspect_drugs=interventions or text_extracted.suspect_drugs,
                concomitant_drugs=text_extracted.concomitant_drugs,
                adverse_events_raw=payload.get("adverse_events") or text_extracted.adverse_events_raw or ["Adverse event"],
                reaction_outcome=text_extracted.reaction_outcome,
            )

        if "patient" in payload and isinstance(payload.get("patient"), dict):
            patient = payload.get("patient", {})
            drugs_raw = patient.get("drug", []) or []
            suspect_drugs: List[DrugEntry] = []
            concomitant_drugs: List[DrugEntry] = []
            for d in drugs_raw:
                char = str(d.get("drugcharacterization", "1"))
                role = "Suspect" if char == "1" else ("Interacting" if char == "3" else "Concomitant")
                entry = DrugEntry(
                    name=d.get("medicinalproduct") or "Unknown Product",
                    role=role,
                    dose=d.get("drugdosagetext") or (
                        f"{d.get('drugstructuredosagenumb', '')} {d.get('drugstructuredosageunit', '')}".strip() or None
                    ),
                    route=d.get("drugadministrationroute"),
                    indication=d.get("drugindication"),
                    start_date=d.get("drugstartdate"),
                    action_taken=str(d.get("actiondrug")) if d.get("actiondrug") else None,
                )
                if role == "Suspect":
                    suspect_drugs.append(entry)
                else:
                    concomitant_drugs.append(entry)

            reactions_raw = patient.get("reaction", []) or []
            ae_list = [r.get("reactionmeddrapt") for r in reactions_raw if r.get("reactionmeddrapt")]
            outcomes = [_outcome_code_to_label(r.get("reactionoutcome")) for r in reactions_raw if r.get("reactionoutcome")]

            age_val = patient.get("patientonsetage")
            age_str = f"{age_val} yr" if age_val else None

            return StructuredCaseData(
                safety_report_id=payload.get("safetyreportid"),
                receive_date=payload.get("receivedate") or payload.get("receiptdate"),
                country=payload.get("occurcountry") or payload.get("primarysourcecountry"),
                patient_age=age_str,
                patient_sex=_sex_code_to_label(patient.get("patientsex")),
                patient_weight_kg=str(patient.get("patientweight")) if patient.get("patientweight") else None,
                reporter_qualification=_qual_code_to_label(payload.get("primarysource", {}).get("qualification")),
                reporter_country=payload.get("primarysource", {}).get("reportercountry"),
                suspect_drugs=suspect_drugs,
                concomitant_drugs=concomitant_drugs,
                adverse_events_raw=ae_list,
                medical_history=[d.get("drugindication") for d in drugs_raw if d.get("drugindication")],
                lab_findings=[],
                reaction_outcome=outcomes[0] if outcomes else "Unknown",
            )

        # Unstructured free-text extraction (LLM via Ollama + deterministic fallback)
        text = payload.get("raw_text", "")
        lower = text.lower()

        if use_llm and len(text.strip()) > 15:
            llm_ext = ollama_chat_json(
                system_prompt=(
                    "You are a pharmacovigilance case-intake extraction agent. "
                    "Extract structured ICH E2B fields from the adverse event report as JSON with keys: "
                    "patient_age (e.g. '64 yr' or null), patient_sex ('Male'|'Female'|'Unknown'), "
                    "reporter_qualification ('Physician'|'Pharmacist'|'Other Health Professional'|'Consumer or Non-Health Professional'|'Unknown'), "
                    "suspect_drugs (list of uppercase drug names), concomitant_drugs (list of uppercase drug names), "
                    "adverse_events (list of adverse event terms), lab_findings (list of strings), "
                    "reaction_outcome ('Recovered/Resolved'|'Recovering/Resolving'|'Not Recovered/Not Resolved'|'Fatal'|'Unknown')."
                ),
                user_prompt=text[:3000],
                model=FAST_MODEL,
                timeout=30.0,
            )
            if isinstance(llm_ext, dict) and (llm_ext.get("suspect_drugs") or llm_ext.get("adverse_events")):
                return StructuredCaseData(
                    safety_report_id=None,
                    receive_date=None,
                    country="US",
                    patient_age=llm_ext.get("patient_age"),
                    patient_sex=llm_ext.get("patient_sex") or "Unknown",
                    reporter_qualification=llm_ext.get("reporter_qualification") or "Unknown",
                    suspect_drugs=[DrugEntry(name=str(d).upper(), role="Suspect") for d in (llm_ext.get("suspect_drugs") or [])],
                    concomitant_drugs=[DrugEntry(name=str(d).upper(), role="Concomitant") for d in (llm_ext.get("concomitant_drugs") or [])],
                    adverse_events_raw=[str(a) for a in (llm_ext.get("adverse_events") or [])],
                    medical_history=[],
                    lab_findings=[str(l) for l in (llm_ext.get("lab_findings") or [])],
                    reaction_outcome=llm_ext.get("reaction_outcome") or "Unknown",
                )

        age_match = re.search(r"\b(\d{1,3})\s*(?:year-old|years old|yo|y/o|yr)\b", lower)
        age_str = f"{age_match.group(1)} yr" if age_match else None

        sex_str = "Unknown"
        if re.search(r"\b(?:male|man|boy|gentleman|m)\b", lower) and not re.search(r"\bfemale\b", lower):
            sex_str = "Male"
        elif re.search(r"\b(?:female|woman|girl|lady|f)\b", lower):
            sex_str = "Female"

        reporter = "Unknown"
        if any(w in lower for w in ["physician", "doctor", "dr.", "md", "clinician"]):
            reporter = "Physician"
        elif "pharmacist" in lower:
            reporter = "Pharmacist"
        elif "nurse" in lower:
            reporter = "Other Health Professional"
        elif any(w in lower for w in ["patient", "consumer", "mother", "father", "self-reported"]):
            reporter = "Consumer or Non-Health Professional"

        # Drug extraction from known drug list + contextual regex fallback
        known_drugs = [
            "Pemetrexed", "Aspirin", "Metformin", "Ibuprofen", "Warfarin",
            "Atorvastatin", "Lisinopril", "Pembrolizumab", "Semaglutide",
            "Amoxicillin", "Acetaminophen", "Paracetamol", "Omeprazole", "Clopidogrel",
            "Cetirizine", "Loratadine", "Prednisone", "Azithromycin", "Naproxen",
            "Furosemide", "Insulin", "Sertraline", "Amlodipine",
        ]
        suspect_drugs: List[DrugEntry] = []
        concomitant_drugs: List[DrugEntry] = []

        concomitant_section = ""
        conc_match = re.search(r"concomitant(?:\s+medications?|\s+drugs?)?[:\s]+([^.;]+)", lower)
        if conc_match:
            concomitant_section = conc_match.group(1)

        seen_drugs = set()
        for kd in known_drugs:
            if re.search(rf"\b{re.escape(kd.lower())}\b", lower):
                seen_drugs.add(kd.upper())
                if kd.lower() in concomitant_section:
                    concomitant_drugs.append(DrugEntry(name=kd.upper(), role="Concomitant"))
                else:
                    suspect_drugs.append(DrugEntry(name=kd.upper(), role="Suspect"))

        if not suspect_drugs:
            ctx_patterns = [
                r"(?:starting|taking|administered|treated with|following|after)\s+([A-Z][a-zA-Z0-9-]{2,25})",
            ]
            stop_words = {"the", "patient", "she", "he", "they", "this", "that", "first", "dose", "days", "hospital"}
            for pat in ctx_patterns:
                for m in re.findall(pat, text):
                    if m.lower() not in stop_words and m.upper() not in seen_drugs:
                        seen_drugs.add(m.upper())
                        suspect_drugs.append(DrugEntry(name=m.upper(), role="Suspect"))

        # Reaction extraction by scanning MEDDRA_TERMS synonyms & PTs
        ae_found: List[str] = []
        for pt, info in MEDDRA_TERMS.items():
            matched = False
            if re.search(rf"\b{re.escape(pt.lower())}\b", lower):
                matched = True
            else:
                for syn in info["synonyms"]:  # type: ignore[attr-defined]
                    if re.search(rf"\b{re.escape(syn)}\b", lower):
                        matched = True
                        break
            if matched and pt not in ae_found:
                ae_found.append(pt)

        # Lab findings extraction
        labs: List[str] = []
        for lab_pat in [r"(?:alt|ast|creatinine|platelets|hemoglobin|ck|glucose)\s*(?:was|of|:|=)?\s*\d+[\d.]*\s*[a-zA-Z/%]*", r"elevated liver enzymes", r"low platelet count"]:
            for m in re.findall(lab_pat, lower):
                labs.append(m.strip())

        outcome = "Unknown"
        if any(w in lower for w in ["died", "fatal", "death", "expired"]):
            outcome = "Fatal"
        elif any(w in lower for w in ["recovered", "resolved", "improved"]):
            outcome = "Recovered/Resolved"
        elif any(w in lower for w in ["recovering", "resolving"]):
            outcome = "Recovering/Resolving"

        return StructuredCaseData(
            safety_report_id=None,
            receive_date=None,
            country="US",
            patient_age=age_str,
            patient_sex=sex_str,
            reporter_qualification=reporter,
            suspect_drugs=suspect_drugs,
            concomitant_drugs=concomitant_drugs,
            adverse_events_raw=ae_found,
            medical_history=[],
            lab_findings=labs,
            reaction_outcome=outcome,
        )

    def code_meddra(self, structured: StructuredCaseData, use_llm: bool = False) -> Tuple[List[MedDRACoding], AgentTraceStep]:
        t0 = time.perf_counter()
        codings: List[MedDRACoding] = []
        llm_used = False
        for raw_ae in structured.adverse_events_raw:
            pt, llt, soc, conf, rationale = map_to_meddra(raw_ae)
            if use_llm and conf < 0.75:
                llm_map = ollama_chat_json(
                    system_prompt=(
                        "You are a MedDRA v29.0 medical coding specialist. Map the verbatim adverse event to JSON: "
                        '{"pt": "<MedDRA Preferred Term>", "llt": "<Low Level Term>", "soc": "<System Organ Class>", "rationale": "<brief clinical rationale>"}'
                    ),
                    user_prompt=f"Verbatim adverse event: {raw_ae}",
                    model=FAST_MODEL,
                    timeout=20.0,
                )
                if isinstance(llm_map, dict) and llm_map.get("pt"):
                    pt = str(llm_map["pt"])
                    llt = str(llm_map.get("llt") or pt)
                    soc = str(llm_map.get("soc") or soc)
                    conf = 0.85
                    rationale = f"[{FAST_MODEL}] {llm_map.get('rationale', 'LLM MedDRA PT mapping.')}"
                    llm_used = True
            codings.append(
                MedDRACoding(
                    verbatim_term=raw_ae,
                    llt=llt,
                    pt=pt,
                    soc=soc,
                    confidence=conf,
                    accepted=True,
                    coding_rationale=rationale,
                )
            )
        trace = AgentTraceStep(
            agent_name=self.name,
            stage=WorkflowStage.CASE_PROCESSING,
            timestamp=db.utc_now(),
            duration_ms=max(1, int((time.perf_counter() - t0) * 1000)),
            status="COMPLETED",
            summary=f"Extracted {len(structured.suspect_drugs)} suspect drug(s), {len(structured.concomitant_drugs)} concomitant drug(s), and coded {len(codings)} MedDRA PT(s).",
            details={"meddra_pts": [c.pt for c in codings], "llm_model": FAST_MODEL if (use_llm and llm_used) else "meddra-dict+fuzzy"},
        )
        return codings, trace


class CaseValidityAgent:
    """Stage 2: Case Identification / Initial Assessment (ICH E2D 4 minimum criteria)."""

    name = "CaseValidityAgent"

    def run(self, structured: StructuredCaseData) -> Tuple[ValidityCheck, AgentTraceStep]:
        t0 = time.perf_counter()
        id_patient = bool(
            (structured.patient_age and structured.patient_age != "Unknown")
            or (structured.patient_sex and structured.patient_sex != "Unknown")
            or structured.patient_weight_kg
            or structured.safety_report_id
        )
        id_reporter = bool(structured.reporter_qualification and structured.reporter_qualification != "Unknown")
        has_suspect = len(structured.suspect_drugs) > 0
        has_ae = len(structured.adverse_events_raw) > 0

        missing: List[str] = []
        if not id_patient:
            missing.append("Identifiable Patient")
        if not id_reporter:
            missing.append("Identifiable Reporter")
        if not has_suspect:
            missing.append("Suspect Product")
        if not has_ae:
            missing.append("Adverse Event")

        is_valid = len(missing) == 0
        rationale = (
            "All 4 ICH E2D minimum validity criteria satisfied."
            if is_valid
            else f"Invalid case: missing {', '.join(missing)}."
        )
        validity = ValidityCheck(
            is_valid=is_valid,
            identifiable_patient=id_patient,
            identifiable_reporter=id_reporter,
            suspect_product=has_suspect,
            adverse_event=has_ae,
            missing_criteria=missing,
            rationale=rationale,
        )
        trace = AgentTraceStep(
            agent_name=self.name,
            stage=WorkflowStage.IDENTIFICATION,
            timestamp=db.utc_now(),
            duration_ms=max(1, int((time.perf_counter() - t0) * 1000)),
            status="COMPLETED" if is_valid else "FLAGGED",
            summary=rationale,
            details=validity.model_dump(),
        )
        return validity, trace


class TriageAgent:
    """Stage 3: Triage — seriousness, report type, urgency, and regulatory SLA."""

    name = "TriageAgent"

    def run(self, payload: Dict[str, Any], source_type: str, structured: StructuredCaseData) -> Tuple[TriageAssessment, AgentTraceStep]:
        t0 = time.perf_counter()
        text = str(payload.get("raw_text", "")).lower()

        death = str(payload.get("seriousnessdeath", "")) == "1" or structured.reaction_outcome == "Fatal" or any(w in text for w in ["died", "fatal", "death"])
        life_thr = str(payload.get("seriousnesslifethreatening", "")) == "1" or any(w in text for w in ["life-threatening", "life threatening", "anaphylaxis", "cardiac arrest"])
        hosp = str(payload.get("seriousnesshospitalization", "")) == "1" or any(w in text for w in ["hospitalized", "hospitalisation", "hospitalization", "admitted to icu", "er visit"])
        other_ser = str(payload.get("serious", "")) == "1" or str(payload.get("seriousnessother", "")) == "1" or str(payload.get("seriousnessdisabling", "")) == "1"

        is_serious = death or life_thr or hosp or other_ser

        if death or life_thr:
            priority = PriorityLevel.CRITICAL
            days = 7
            sla = "7-Day Expedited (Fatal / Life-Threatening)"
        elif is_serious:
            priority = PriorityLevel.HIGH
            days = 15
            sla = "15-Day Expedited (Serious)"
        else:
            priority = PriorityLevel.STANDARD
            days = 90
            sla = "90-Day Periodic / Non-Expedited"

        report_type = "Regulatory / FAERS" if "patient" in payload else ("Clinical Trial" if "trial" in source_type.lower() else "Spontaneous")
        rationale = f"Classified as {priority.value} ({'Serious' if is_serious else 'Non-Serious'}) with {sla}."

        triage = TriageAssessment(
            priority=priority,
            is_serious=is_serious,
            report_type=report_type,
            source_channel=source_type,
            regulatory_timeline_days=days,
            sla_label=sla,
            rationale=rationale,
        )
        trace = AgentTraceStep(
            agent_name=self.name,
            stage=WorkflowStage.TRIAGE,
            timestamp=db.utc_now(),
            duration_ms=max(1, int((time.perf_counter() - t0) * 1000)),
            status="COMPLETED",
            summary=rationale,
            details={"priority": priority.value, "sla_days": days},
        )
        return triage, trace


class DuplicateCheckAgent:
    """Stage 4: Duplicate Check against existing safety database cases."""

    name = "DuplicateCheckAgent"

    def run(self, structured: StructuredCaseData, existing_cases: List[PVCase]) -> Tuple[DuplicateCheckResult, AgentTraceStep]:
        t0 = time.perf_counter()
        best_match: Optional[PVCase] = None
        best_score = 0.0
        reasons: List[str] = []
        dup_status = DuplicateStatus.NEW

        in_drugs = {d.name.upper() for d in structured.suspect_drugs}
        in_aes = {a.lower() for a in structured.adverse_events_raw}

        for ex in existing_cases:
            score = 0.0
            cur_reasons: List[str] = []
            if structured.safety_report_id and ex.safetyreportid and structured.safety_report_id == ex.safetyreportid:
                score = 1.0
                cur_reasons.append(f"Exact safetyreportid match ({structured.safety_report_id})")
            else:
                ex_drugs = {d.name.upper() for d in ex.structured_data.suspect_drugs}
                ex_aes = {a.lower() for a in ex.structured_data.adverse_events_raw}
                if structured.patient_age and ex.structured_data.patient_age == structured.patient_age:
                    score += 0.25
                    cur_reasons.append("Matching patient age")
                if structured.patient_sex and ex.structured_data.patient_sex == structured.patient_sex and structured.patient_sex != "Unknown":
                    score += 0.20
                    cur_reasons.append("Matching patient sex")
                if in_drugs & ex_drugs:
                    score += 0.30
                    cur_reasons.append(f"Overlapping suspect drug(s): {', '.join(in_drugs & ex_drugs)}")
                if in_aes & ex_aes:
                    score += 0.25
                    cur_reasons.append(f"Overlapping AE(s): {', '.join(in_aes & ex_aes)}")

            if score > best_score:
                best_score = score
                best_match = ex
                reasons = cur_reasons

        if best_match and best_score >= 0.95:
            dup_status = DuplicateStatus.DUPLICATE
        elif best_match and best_score >= 0.75:
            dup_status = DuplicateStatus.FOLLOW_UP
        else:
            dup_status = DuplicateStatus.NEW
            best_match = None
            reasons = ["No matching existing case found in safety database."]

        res = DuplicateCheckResult(
            status=dup_status,
            matched_case_id=best_match.case_id if best_match else None,
            similarity_score=round(best_score, 2),
            match_reasons=reasons,
        )
        trace = AgentTraceStep(
            agent_name=self.name,
            stage=WorkflowStage.DUPLICATE_CHECK,
            timestamp=db.utc_now(),
            duration_ms=max(1, int((time.perf_counter() - t0) * 1000)),
            status="COMPLETED",
            summary=f"Duplicate check result: {dup_status.value}" + (f" (matched {res.matched_case_id}, score={res.similarity_score})" if res.matched_case_id else ""),
            details=res.model_dump(),
        )
        return res, trace


class ClinicalAssessmentAgent:
    """Stage 5b: Seriousness, Causality (WHO-UMC & Naranjo), Listedness, and Clinical Narrative."""

    name = "ClinicalAssessmentAgent"

    def run(
        self,
        payload: Dict[str, Any],
        structured: StructuredCaseData,
        codings: List[MedDRACoding],
        use_llm: bool = False,
    ) -> Tuple[SeriousnessAssessment, CausalityAssessment, ListednessAssessment, str, AgentTraceStep]:
        t0 = time.perf_counter()
        text = str(payload.get("raw_text", "")).lower()

        death = str(payload.get("seriousnessdeath", "")) == "1" or structured.reaction_outcome == "Fatal" or any(w in text for w in ["died", "fatal", "death"])
        life_thr = str(payload.get("seriousnesslifethreatening", "")) == "1" or any(w in text for w in ["life-threatening", "life threatening", "anaphylaxis", "cardiac arrest"])
        hosp = str(payload.get("seriousnesshospitalization", "")) == "1" or any(w in text for w in ["hospitalized", "hospitalisation", "hospitalization", "admitted"])
        disab = str(payload.get("seriousnessdisabling", "")) == "1" or "disability" in text
        cong = str(payload.get("seriousnesscongenitalanomali", "")) == "1" or "congenital" in text
        other = str(payload.get("seriousnessother", "")) == "1" or any(c.pt in {"Stevens-Johnson syndrome", "Hepatotoxicity", "Acute kidney injury", "Myocardial infarction"} for c in codings)

        is_serious = death or life_thr or hosp or disab or cong or other or str(payload.get("serious", "")) == "1"
        ser_criteria = [
            name for flag, name in [
                (death, "Death"),
                (life_thr, "Life-Threatening"),
                (hosp, "Hospitalization"),
                (disab, "Disability"),
                (cong, "Congenital Anomaly"),
                (other, "Other Medically Important"),
            ] if flag
        ]
        seriousness = SeriousnessAssessment(
            is_serious=is_serious,
            death=death,
            life_threatening=life_thr,
            hospitalization=hosp,
            disability=disab,
            congenital_anomaly=cong,
            other_medically_important=other,
            rationale=f"Serious ({', '.join(ser_criteria)})" if is_serious else "Non-Serious (no ICH E2A seriousness criteria met).",
        )

        # Causality (WHO-UMC + Naranjo)
        dechallenge = None
        rechallenge = None
        if any(d.action_taken == "1" for d in structured.suspect_drugs) or any(w in text for w in ["discontinued", "stopped", "improved after stopping", "dechallenge"]):
            dechallenge = True
        if "rechallenge" in text and "recurred" in text:
            rechallenge = True

        alt_causes = len(structured.concomitant_drugs) > 0 or len(structured.medical_history) > 0
        naranjo = 2  # Temporal relationship (+2)
        if dechallenge:
            naranjo += 1
        if rechallenge:
            naranjo += 2
        if not alt_causes:
            naranjo += 2

        drug_names = [d.name for d in structured.suspect_drugs]
        pts = [c.pt for c in codings]
        listed_status, matched_label, listed_rationale = check_listedness(drug_names, pts)
        if matched_label:
            naranjo += 1  # Previous conclusive reports (+1)

        if naranjo >= 9:
            naranjo_cat, who_cat = "Definite", "Certain"
        elif naranjo >= 5:
            naranjo_cat, who_cat = "Probable", "Probable/Likely"
        elif naranjo >= 1:
            naranjo_cat, who_cat = "Possible", "Possible"
        else:
            naranjo_cat, who_cat = "Doubtful", "Unlikely"

        causality = CausalityAssessment(
            who_umc_category=who_cat,
            naranjo_score=naranjo,
            naranjo_category=naranjo_cat,
            dechallenge_positive=dechallenge,
            rechallenge_positive=rechallenge,
            temporal_relationship=True,
            alternative_causes_present=alt_causes,
            rationale=f"WHO-UMC: {who_cat}; Naranjo Score: {naranjo} ({naranjo_cat}).",
        )

        listedness = ListednessAssessment(
            status=listed_status,
            matched_label_terms=matched_label,
            rationale=listed_rationale,
        )

        # Draft ICH Clinical Narrative
        age_desc = structured.patient_age or "unknown age"
        sex_desc = (structured.patient_sex or "unknown sex").lower()
        drug_desc = ", ".join(
            f"{d.name}" + (f" ({d.dose})" if d.dose else "") + (f" for {d.indication}" if d.indication else "")
            for d in structured.suspect_drugs
        ) or "suspect product"
        ae_desc = ", ".join(f"{c.pt} [SOC: {c.soc}]" for c in codings) or "unspecified adverse event"
        conc_desc = (
            f" Concomitant medications included {', '.join(d.name for d in structured.concomitant_drugs)}."
            if structured.concomitant_drugs
            else ""
        )
        narrative = (
            f"This {structured.reporter_qualification or 'spontaneous'} report from {structured.country or 'unknown country'} "
            f"describes a {age_desc} {sex_desc} patient who experienced {ae_desc} following administration of {drug_desc}.{conc_desc} "
            f"The case was assessed as {seriousness.rationale} "
            f"Outcome was reported as {structured.reaction_outcome or 'Unknown'}. "
            f"Causality was assessed as {who_cat} (Naranjo score: {naranjo}, {naranjo_cat}), and expectedness was classified as {listed_status}."
        )

        llm_model_used = "deterministic-rules"
        if use_llm:
            llm_assess = ollama_chat_json(
                system_prompt=(
                    "You are a senior pharmacovigilance clinical scientist. Given structured case data, "
                    "assess WHO-UMC causality and draft a concise, regulatory-grade ICH E2B clinical narrative. "
                    "Return JSON with keys: who_umc_category ('Certain'|'Probable/Likely'|'Possible'|'Unlikely'|'Unassessable'), "
                    "causality_rationale (string), narrative (string, 3-5 sentences covering patient, suspect drug, reaction, seriousness, outcome, causality)."
                ),
                user_prompt=(
                    f"Patient: {structured.patient_age or 'unknown age'} {structured.patient_sex or 'unknown sex'}, "
                    f"Suspect drugs: {[d.name for d in structured.suspect_drugs]}, "
                    f"Concomitant: {[d.name for d in structured.concomitant_drugs]}, "
                    f"MedDRA PTs: {[c.pt for c in codings]}, "
                    f"Seriousness: {seriousness.rationale}, Listedness: {listed_status}, Outcome: {structured.reaction_outcome}."
                ),
                model=REASONING_MODEL,
                timeout=120.0,
            )
            if isinstance(llm_assess, dict):
                if llm_assess.get("who_umc_category") in {"Certain", "Probable/Likely", "Possible", "Unlikely", "Unassessable"}:
                    causality.who_umc_category = llm_assess["who_umc_category"]
                if llm_assess.get("causality_rationale"):
                    causality.rationale = f"[{REASONING_MODEL}] {llm_assess['causality_rationale']} (Naranjo: {naranjo}, {naranjo_cat})."
                if llm_assess.get("narrative") and len(str(llm_assess["narrative"]).strip()) >= 60:
                    narrative = str(llm_assess["narrative"]).strip()
                llm_model_used = REASONING_MODEL

        trace = AgentTraceStep(
            agent_name=self.name,
            stage=WorkflowStage.CASE_PROCESSING,
            timestamp=db.utc_now(),
            duration_ms=max(1, int((time.perf_counter() - t0) * 1000)),
            status="COMPLETED",
            summary=f"Assessed seriousness ({'Serious' if is_serious else 'Non-Serious'}), causality ({causality.who_umc_category}), listedness ({listed_status}), and drafted ICH narrative.",
            details={"naranjo_score": naranjo, "who_umc": causality.who_umc_category, "listedness": listed_status, "llm_model": llm_model_used},
        )
        return seriousness, causality, listedness, narrative, trace


class QCAgent:
    """Stage 6: Independent Quality Control (QC) verification."""

    name = "QCAgent"

    def run(self, case: PVCase, use_llm: bool = True) -> Tuple[QCChecklist, AgentTraceStep]:
        t0 = time.perf_counter()
        issues: List[str] = []

        data_ok = len(case.structured_data.suspect_drugs) > 0 and len(case.structured_data.adverse_events_raw) > 0
        if not data_ok:
            issues.append("Missing suspect drug or adverse event in structured data.")

        completeness_ok = bool(case.structured_data.patient_sex and case.structured_data.reporter_qualification)
        if not completeness_ok:
            issues.append("Incomplete patient sex or reporter qualification.")

        low_conf = [c.pt for c in case.meddra_codings if c.confidence < 0.70]
        coding_ok = len(case.meddra_codings) > 0 and len(low_conf) == 0
        if low_conf:
            issues.append(f"Low-confidence MedDRA coding requires human verification: {', '.join(low_conf)}.")

        assess_ok = case.triage.is_serious == case.seriousness.is_serious
        if not assess_ok:
            issues.append("Mismatch between Triage seriousness and Case Processing seriousness.")

        narrative_ok = len(case.narrative_text.strip()) >= 80 and all(
            d.name.split()[0].lower() in case.narrative_text.lower() for d in case.structured_data.suspect_drugs[:1]
        )
        if not narrative_ok:
            issues.append("Clinical narrative is too short or omits primary suspect product.")

        llm_model_used = "deterministic-sop"
        if use_llm:
            qc_llm = ollama_chat_json(
                system_prompt=(
                    "You are an independent Pharmacovigilance QC Reviewer. Check if the clinical narrative accurately reflects "
                    "the suspect drugs, MedDRA PTs, and seriousness. Return JSON: {\"consistent\": true|false, \"issues\": [\"...\"]}."
                ),
                user_prompt=(
                    f"Suspect drugs: {[d.name for d in case.structured_data.suspect_drugs]}, "
                    f"MedDRA PTs: {[c.pt for c in case.meddra_codings]}, Serious: {case.seriousness.is_serious}. "
                    f"Narrative: {case.narrative_text}"
                ),
                model=FAST_MODEL,
                timeout=25.0,
            )
            if isinstance(qc_llm, dict):
                llm_model_used = FAST_MODEL
                if qc_llm.get("consistent") is False and isinstance(qc_llm.get("issues"), list):
                    for iss in qc_llm["issues"]:
                        if iss and str(iss) not in issues:
                            issues.append(f"[{FAST_MODEL} QC] {iss}")

        sop_ok = case.validity.is_valid and len(issues) == 0
        checklist = QCChecklist(
            passed=len(issues) == 0,
            data_entry_accurate=data_ok,
            completeness_ok=completeness_ok,
            coding_accurate=coding_ok,
            assessments_consistent=assess_ok,
            narrative_consistent=narrative_ok,
            sop_compliant=sop_ok,
            issues=issues,
        )
        details = checklist.model_dump()
        details["llm_model"] = llm_model_used
        trace = AgentTraceStep(
            agent_name=self.name,
            stage=WorkflowStage.QUALITY_CONTROL,
            timestamp=db.utc_now(),
            duration_ms=max(1, int((time.perf_counter() - t0) * 1000)),
            status="COMPLETED" if checklist.passed else "FLAGGED",
            summary="Automated QC passed all 6 SOP checks; ready for Human QC sign-off." if checklist.passed else f"Automated QC flagged {len(issues)} issue(s) for review.",
            details=details,
        )
        return checklist, trace


class MedicalReviewAgent:
    """Stage 7: Medical Review (MBBS/MD clinical coherence pre-check)."""

    name = "MedicalReviewAgent"

    def run(self, case: PVCase, use_llm: bool = True) -> Tuple[MedicalReviewSummary, AgentTraceStep]:
        t0 = time.perf_counter()
        flagged: List[str] = []
        if case.seriousness.death and case.structured_data.reaction_outcome != "Fatal":
            flagged.append("Death criterion checked, but reaction outcome is not marked Fatal.")
        if case.causality.who_umc_category == "Certain" and case.causality.alternative_causes_present:
            flagged.append("Causality marked Certain despite documented alternative causes/concomitant drugs.")

        llm_model_used = "deterministic-md-rules"
        clinical_notes = ""
        if use_llm:
            md_llm = ollama_chat_json(
                system_prompt=(
                    "You are a Pharmacovigilance Medical Reviewer (MD). Evaluate clinical plausibility of the drug-event pair, "
                    "seriousness, and WHO-UMC causality. Return JSON: {\"clinically_sound\": true|false, \"clinical_notes\": \"...\", \"flagged_items\": []}"
                ),
                user_prompt=(
                    f"Suspect drugs: {[d.name for d in case.structured_data.suspect_drugs]}, "
                    f"MedDRA PTs: {[c.pt for c in case.meddra_codings]}, Serious: {case.seriousness.rationale}, "
                    f"Causality: {case.causality.who_umc_category}, Listedness: {case.listedness.status}. Narrative: {case.narrative_text}"
                ),
                model=REASONING_MODEL,
                timeout=120.0,
            )
            if isinstance(md_llm, dict):
                llm_model_used = REASONING_MODEL
                if md_llm.get("clinical_notes"):
                    clinical_notes = f"[{REASONING_MODEL}] {md_llm['clinical_notes']}"
                if md_llm.get("clinically_sound") is False and isinstance(md_llm.get("flagged_items"), list):
                    for item in md_llm["flagged_items"]:
                        if item and str(item) not in flagged:
                            flagged.append(str(item))

        sound = len(flagged) == 0
        if not clinical_notes:
            clinical_notes = (
                f"Clinical coherence verified for {len(case.meddra_codings)} MedDRA PT(s), "
                f"seriousness={case.seriousness.is_serious}, causality={case.causality.who_umc_category}. Ready for MD sign-off."
                if sound
                else f"Clinical review flagged {len(flagged)} discrepancy(ies) for MD attention."
            )
        summary = MedicalReviewSummary(
            clinically_sound=sound,
            clinical_notes=clinical_notes,
            recommended_action="APPROVE_TO_LEAD" if sound else "RETURN_FOR_CORRECTION",
            flagged_items=flagged,
        )
        details = summary.model_dump()
        details["llm_model"] = llm_model_used
        trace = AgentTraceStep(
            agent_name=self.name,
            stage=WorkflowStage.MEDICAL_REVIEW,
            timestamp=db.utc_now(),
            duration_ms=max(1, int((time.perf_counter() - t0) * 1000)),
            status="COMPLETED" if sound else "FLAGGED",
            summary=summary.clinical_notes,
            details=details,
        )
        return summary, trace


def run_intake_pipeline(
    source_type: str = "Spontaneous (HCP)",
    raw_json: Optional[Dict[str, Any]] = None,
    raw_text: Optional[str] = None,
    use_llm: bool = True,
) -> PVCase:
    """Execute Stages 1 -> 6 of the multi-agent PV pipeline and persist case + audit trail."""
    now = db.utc_now()
    case_id = f"PV-{now[:10].replace('-', '')}-{uuid.uuid4().hex[:6].upper()}"
    traces: List[AgentTraceStep] = []

    # Stage 1: Intake
    intake_agent = IntakeAgent()
    payload, inferred_source, t1 = intake_agent.run(source_type, raw_json, raw_text)
    traces.append(t1)

    # Extract structured fields & MedDRA codings (used across Stages 2-5)
    ext_agent = ExtractionAndCodingAgent()
    structured = ext_agent.extract_structured(payload, use_llm=use_llm)

    # Stage 2: Validity Check
    val_agent = CaseValidityAgent()
    validity, t2 = val_agent.run(structured)
    traces.append(t2)

    # Stage 3: Triage
    triage_agent = TriageAgent()
    triage, t3 = triage_agent.run(payload, inferred_source, structured)
    traces.append(t3)

    # Stage 4: Duplicate Check
    existing_cases = db.list_cases()
    dup_agent = DuplicateCheckAgent()
    dup_res, t4 = dup_agent.run(structured, existing_cases)
    traces.append(t4)

    # Stage 5: Coding & Clinical Assessments + Narrative
    codings, t5a = ext_agent.code_meddra(structured, use_llm=use_llm)
    traces.append(t5a)

    assess_agent = ClinicalAssessmentAgent()
    seriousness, causality, listedness, narrative, t5b = assess_agent.run(payload, structured, codings, use_llm=use_llm)
    traces.append(t5b)

    # Determine stage & status after automated processing
    if not validity.is_valid:
        stage = WorkflowStage.IDENTIFICATION
        status = CaseStatus.INVALID_CASE
    elif dup_res.status == DuplicateStatus.DUPLICATE:
        stage = WorkflowStage.DUPLICATE_CHECK
        status = CaseStatus.DUPLICATE_CLOSED
    else:
        stage = WorkflowStage.QUALITY_CONTROL
        status = CaseStatus.AWAITING_QC

    case = PVCase(
        case_id=case_id,
        safetyreportid=structured.safety_report_id,
        source_type=inferred_source,
        current_stage=stage,
        status=status,
        priority=triage.priority,
        regulatory_deadline_days=triage.regulatory_timeline_days,
        duplicate_status=dup_res.status,
        parent_case_id=dup_res.matched_case_id if dup_res.status == DuplicateStatus.FOLLOW_UP else None,
        raw_report=payload,
        validity=validity,
        triage=triage,
        duplicate_check=dup_res,
        structured_data=structured,
        meddra_codings=codings,
        seriousness=seriousness,
        causality=causality,
        listedness=listedness,
        narrative_text=narrative,
        agent_traces=traces,
        created_at=now,
        updated_at=now,
    )

    # Stage 6: Automated QC Pre-Check (if awaiting QC)
    if status == CaseStatus.AWAITING_QC:
        qc_agent = QCAgent()
        qc_res, t6 = qc_agent.run(case)
        case.qc_checklist = qc_res
        case.agent_traces.append(t6)

    db.save_case(case)

    # If this is a follow-up to an existing case, record the link on the parent case's audit trail
    if case.duplicate_status == DuplicateStatus.FOLLOW_UP and case.parent_case_id:
        db.append_audit(
            AuditTrailEntry(
                case_id=case.parent_case_id,
                timestamp=now,
                actor="DuplicateCheckAgent",
                actor_type="AGENT",
                stage=WorkflowStage.DUPLICATE_CHECK.value,
                action="FOLLOW_UP_LINKED",
                field_name="follow_up_case_id",
                new_value=case.case_id,
                rationale=f"Linked incoming follow-up {case.case_id} (similarity={dup_res.similarity_score}).",
            )
        )

    # Log immutable audit trail for each agent step
    for tr in case.agent_traces:
        db.append_audit(
            AuditTrailEntry(
                case_id=case.case_id,
                timestamp=tr.timestamp,
                actor=tr.agent_name,
                actor_type="AGENT",
                stage=tr.stage.value,
                action="AGENT_EXECUTED",
                rationale=tr.summary,
            )
        )

    return case
