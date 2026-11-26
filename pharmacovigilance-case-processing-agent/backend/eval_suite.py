from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from backend.agents import (
    CaseValidityAgent,
    ClinicalAssessmentAgent,
    ExtractionAndCodingAgent,
    TriageAgent,
)

OPENFDA_SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "openfda_sample.json"

# Synthetic unstructured narrative benchmark cases (never real patient data)
SYNTHETIC_EVAL_CASES: List[Dict[str, Any]] = [
    {
        "id": "SYN-001",
        "source_type": "Spontaneous (HCP)",
        "raw_text": (
            "Physician reported a 64 year-old male patient who developed severe gastrointestinal bleeding and melena "
            "3 days after starting Aspirin 325 mg daily. Concomitant medications: Metformin. "
            "Patient was hospitalized for 4 days and improved after stopping Aspirin."
        ),
        "gold": {
            "case_validity": True,
            "patient_sex": "Male",
            "patient_age": "64 yr",
            "reporter_qualification": "Physician",
            "suspect_drugs": ["ASPIRIN"],
            "concomitant_drugs": ["METFORMIN"],
            "meddra_pts": ["Gastrointestinal haemorrhage"],
            "seriousness_overall": True,
            "seriousness_criteria": ["hospitalization"],
        },
    },
    {
        "id": "SYN-002",
        "source_type": "Spontaneous (HCP)",
        "raw_text": (
            "Dr. Patel reported a 52 year-old female patient treated with Amoxicillin who experienced "
            "anaphylactic shock with throat swelling and hypotension within 20 minutes of the first dose. "
            "This was a life-threatening event requiring emergency epinephrine."
        ),
        "gold": {
            "case_validity": True,
            "patient_sex": "Female",
            "patient_age": "52 yr",
            "reporter_qualification": "Physician",
            "suspect_drugs": ["AMOXICILLIN"],
            "concomitant_drugs": [],
            "meddra_pts": ["Anaphylactic reaction", "Hypotension"],
            "seriousness_overall": True,
            "seriousness_criteria": ["life_threatening"],
        },
    },
    {
        "id": "SYN-003",
        "source_type": "Spontaneous (Pharmacist)",
        "raw_text": (
            "A pharmacist reported a 71 year-old male patient on Atorvastatin who developed muscle pain and "
            "rhabdomyolysis. Concomitant drugs: Lisinopril. The patient was hospitalized and recovered after stopping Atorvastatin."
        ),
        "gold": {
            "case_validity": True,
            "patient_sex": "Male",
            "patient_age": "71 yr",
            "reporter_qualification": "Pharmacist",
            "suspect_drugs": ["ATORVASTATIN"],
            "concomitant_drugs": ["LISINOPRIL"],
            "meddra_pts": ["Myalgia", "Rhabdomyolysis"],
            "seriousness_overall": True,
            "seriousness_criteria": ["hospitalization"],
        },
    },
    {
        "id": "SYN-004",
        "source_type": "Spontaneous (Patient/Consumer)",
        "raw_text": (
            "Patient self-reported that she is a 39 year-old female who experienced mild nausea and headache "
            "after taking Semaglutide. She continued therapy and symptoms resolved."
        ),
        "gold": {
            "case_validity": True,
            "patient_sex": "Female",
            "patient_age": "39 yr",
            "reporter_qualification": "Consumer or Non-Health Professional",
            "suspect_drugs": ["SEMAGLUTIDE"],
            "concomitant_drugs": [],
            "meddra_pts": ["Nausea", "Headache"],
            "seriousness_overall": False,
            "seriousness_criteria": [],
        },
    },
    {
        "id": "SYN-005",
        "source_type": "Spontaneous (HCP)",
        "raw_text": (
            "A 58 year-old female patient died after developing acute liver injury and elevated liver enzymes "
            "while taking Acetaminophen. Reported by a physician."
        ),
        "gold": {
            "case_validity": True,
            "patient_sex": "Female",
            "patient_age": "58 yr",
            "reporter_qualification": "Physician",
            "suspect_drugs": ["ACETAMINOPHEN"],
            "concomitant_drugs": [],
            "meddra_pts": ["Hepatotoxicity", "Alanine aminotransferase increased"],
            "seriousness_overall": True,
            "seriousness_criteria": ["death", "other_medically_important"],
        },
    },
    {
        "id": "SYN-006",
        "source_type": "Spontaneous (Patient/Consumer)",
        "raw_text": (
            "Someone called to say a medication caused a rash, but refused to provide patient age, sex, "
            "initials, or reporter contact details, and could not name the drug."
        ),
        "gold": {
            "case_validity": False,
            "patient_sex": "Unknown",
            "patient_age": None,
            "reporter_qualification": "Unknown",
            "suspect_drugs": [],
            "concomitant_drugs": [],
            "meddra_pts": ["Rash"],
            "seriousness_overall": False,
            "seriousness_criteria": [],
        },
    },
]


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return round(prec, 4), round(rec, 4), round(f1, 4)


def _load_openfda_eval_cases(limit: int = 8) -> List[Dict[str, Any]]:
    if not OPENFDA_SAMPLE_PATH.exists():
        return []
    raw = json.loads(OPENFDA_SAMPLE_PATH.read_text())
    records = raw.get("results", raw) if isinstance(raw, dict) else raw
    eval_items: List[Dict[str, Any]] = []
    sex_map = {"1": "Male", "2": "Female", "0": "Unknown"}
    qual_map = {
        "1": "Physician",
        "2": "Pharmacist",
        "3": "Other Health Professional",
        "4": "Lawyer",
        "5": "Consumer or Non-Health Professional",
    }
    for rec in records[:limit]:
        patient = rec.get("patient", {})
        drugs = patient.get("drug", []) or []
        suspect = [d.get("medicinalproduct", "").upper() for d in drugs if str(d.get("drugcharacterization", "1")) == "1" and d.get("medicinalproduct")]
        concomitant = [d.get("medicinalproduct", "").upper() for d in drugs if str(d.get("drugcharacterization", "1")) != "1" and d.get("medicinalproduct")]
        pts = [r.get("reactionmeddrapt") for r in (patient.get("reaction", []) or []) if r.get("reactionmeddrapt")]
        age = patient.get("patientonsetage")
        sex = sex_map.get(str(patient.get("patientsex", "")), "Unknown")
        qual = qual_map.get(str(rec.get("primarysource", {}).get("qualification", "")), "Unknown")
        ser_criteria = []
        if str(rec.get("seriousnessdeath", "")) == "1":
            ser_criteria.append("death")
        if str(rec.get("seriousnesslifethreatening", "")) == "1":
            ser_criteria.append("life_threatening")
        if str(rec.get("seriousnesshospitalization", "")) == "1":
            ser_criteria.append("hospitalization")
        if str(rec.get("seriousnessdisabling", "")) == "1":
            ser_criteria.append("disability")
        if str(rec.get("seriousnesscongenitalanomali", "")) == "1":
            ser_criteria.append("congenital_anomaly")
        if str(rec.get("seriousnessother", "")) == "1":
            ser_criteria.append("other_medically_important")

        eval_items.append(
            {
                "id": f"FAERS-{rec.get('safetyreportid', 'UNK')}",
                "source_type": "Regulatory / FAERS",
                "raw_json": rec,
                "gold": {
                    "case_validity": bool((age or sex != "Unknown" or rec.get("safetyreportid")) and qual != "Unknown" and suspect and pts),
                    "patient_sex": sex,
                    "patient_age": f"{age} yr" if age else None,
                    "reporter_qualification": qual,
                    "suspect_drugs": suspect,
                    "concomitant_drugs": concomitant,
                    "meddra_pts": pts,
                    "seriousness_overall": str(rec.get("serious", "")) == "1",
                    "seriousness_criteria": ser_criteria,
                },
            }
        )
    return eval_items


def run_evaluation() -> Dict[str, Any]:
    """Run the multi-agent extraction, coding, validity & seriousness pipeline on the eval set and compute per-field Precision & Recall."""
    ext_agent = ExtractionAndCodingAgent()
    val_agent = CaseValidityAgent()
    assess_agent = ClinicalAssessmentAgent()

    dataset = SYNTHETIC_EVAL_CASES + _load_openfda_eval_cases(limit=8)

    counts: Dict[str, Dict[str, int]] = {
        "case_validity": {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "total": 0},
        "patient_sex": {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "total": 0},
        "patient_age": {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "total": 0},
        "reporter_qualification": {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "total": 0},
        "suspect_drugs": {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "total": 0},
        "concomitant_drugs": {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "total": 0},
        "meddra_pts": {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "total": 0},
        "seriousness_overall": {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "total": 0},
        "seriousness_criteria": {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "total": 0},
    }

    case_results: List[Dict[str, Any]] = []

    for item in dataset:
        payload = item.get("raw_json") or {"raw_text": item.get("raw_text", "")}
        gold = item["gold"]

        structured = ext_agent.extract_structured(payload)
        validity, _ = val_agent.run(structured)
        codings, _ = ext_agent.code_meddra(structured)
        seriousness, _, _, _, _ = assess_agent.run(payload, structured, codings)

        # 1. case_validity (binary)
        counts["case_validity"]["total"] += 1
        pred_val, gold_val = validity.is_valid, gold["case_validity"]
        if pred_val == gold_val:
            counts["case_validity"]["exact"] += 1
        if pred_val and gold_val:
            counts["case_validity"]["tp"] += 1
        elif pred_val and not gold_val:
            counts["case_validity"]["fp"] += 1
        elif not pred_val and gold_val:
            counts["case_validity"]["fn"] += 1

        # 2. scalar fields: patient_sex, patient_age, reporter_qualification
        for field_name, pred_v, gold_v in [
            ("patient_sex", structured.patient_sex, gold["patient_sex"]),
            ("patient_age", structured.patient_age, gold["patient_age"]),
            ("reporter_qualification", structured.reporter_qualification, gold["reporter_qualification"]),
        ]:
            counts[field_name]["total"] += 1
            if pred_v == gold_v:
                counts[field_name]["exact"] += 1
            if pred_v is not None and pred_v != "Unknown":
                if pred_v == gold_v:
                    counts[field_name]["tp"] += 1
                else:
                    counts[field_name]["fp"] += 1
                    if gold_v is not None and gold_v != "Unknown":
                        counts[field_name]["fn"] += 1
            else:
                if gold_v is not None and gold_v != "Unknown":
                    counts[field_name]["fn"] += 1

        # 3. set fields: suspect_drugs, concomitant_drugs, meddra_pts, seriousness_criteria
        pred_suspect = {d.name.upper() for d in structured.suspect_drugs}
        gold_suspect = {d.upper() for d in gold["suspect_drugs"]}

        pred_conc = {d.name.upper() for d in structured.concomitant_drugs}
        gold_conc = {d.upper() for d in gold["concomitant_drugs"]}

        pred_pts = {c.pt.lower() for c in codings}
        gold_pts = {p.lower() for p in gold["meddra_pts"]}

        pred_crit: Set[str] = set()
        if seriousness.death:
            pred_crit.add("death")
        if seriousness.life_threatening:
            pred_crit.add("life_threatening")
        if seriousness.hospitalization:
            pred_crit.add("hospitalization")
        if seriousness.disability:
            pred_crit.add("disability")
        if seriousness.congenital_anomaly:
            pred_crit.add("congenital_anomaly")
        if seriousness.other_medically_important:
            pred_crit.add("other_medically_important")
        gold_crit = set(gold["seriousness_criteria"])

        for field_name, pred_set, gold_set in [
            ("suspect_drugs", pred_suspect, gold_suspect),
            ("concomitant_drugs", pred_conc, gold_conc),
            ("meddra_pts", pred_pts, gold_pts),
            ("seriousness_criteria", pred_crit, gold_crit),
        ]:
            counts[field_name]["total"] += 1
            if pred_set == gold_set:
                counts[field_name]["exact"] += 1
            counts[field_name]["tp"] += len(pred_set & gold_set)
            counts[field_name]["fp"] += len(pred_set - gold_set)
            counts[field_name]["fn"] += len(gold_set - pred_set)

        # 4. seriousness_overall (binary)
        counts["seriousness_overall"]["total"] += 1
        if seriousness.is_serious == gold["seriousness_overall"]:
            counts["seriousness_overall"]["exact"] += 1
        if seriousness.is_serious and gold["seriousness_overall"]:
            counts["seriousness_overall"]["tp"] += 1
        elif seriousness.is_serious and not gold["seriousness_overall"]:
            counts["seriousness_overall"]["fp"] += 1
        elif not seriousness.is_serious and gold["seriousness_overall"]:
            counts["seriousness_overall"]["fn"] += 1

        case_results.append(
            {
                "id": item["id"],
                "source_type": item["source_type"],
                "validity_match": pred_val == gold_val,
                "meddra_pt_match": pred_pts == gold_pts,
                "seriousness_match": seriousness.is_serious == gold["seriousness_overall"],
                "pred_pts": sorted(pred_pts),
                "gold_pts": sorted(gold_pts),
            }
        )

    per_field_metrics: List[Dict[str, Any]] = []
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    for field_name, c in counts.items():
        p, r, f1 = _prf(c["tp"], c["fp"], c["fn"])
        acc = round(c["exact"] / c["total"], 4) if c["total"] else 0.0
        macro_p += p
        macro_r += r
        macro_f1 += f1
        per_field_metrics.append(
            {
                "field": field_name,
                "precision": p,
                "recall": r,
                "f1": f1,
                "exact_match": acc,
                "tp": c["tp"],
                "fp": c["fp"],
                "fn": c["fn"],
                "support": c["total"],
            }
        )

    n_fields = len(per_field_metrics)
    return {
        "total_cases": len(dataset),
        "synthetic_cases": len(SYNTHETIC_EVAL_CASES),
        "openfda_cases": len(dataset) - len(SYNTHETIC_EVAL_CASES),
        "macro_precision": round(macro_p / n_fields, 4),
        "macro_recall": round(macro_r / n_fields, 4),
        "macro_f1": round(macro_f1 / n_fields, 4),
        "per_field": per_field_metrics,
        "cases": case_results,
    }
