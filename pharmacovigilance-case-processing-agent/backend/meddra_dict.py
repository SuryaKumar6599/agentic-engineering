from __future__ import annotations

import difflib
import re
from typing import Dict, List, Tuple

# Curated MedDRA v29.0 Preferred Terms (PT), System Organ Class (SOC), and clinical synonyms/LLTs
MEDDRA_TERMS: Dict[str, Dict[str, object]] = {
    "Skin discolouration": {
        "soc": "Skin and subcutaneous tissue disorders",
        "llts": ["Skin discolouration", "Skin discoloration", "Discoloration of skin"],
        "synonyms": ["skin discoloration", "skin discolouration", "darkening of skin", "skin colour change"],
    },
    "Pigmentation disorder": {
        "soc": "Skin and subcutaneous tissue disorders",
        "llts": ["Pigmentation disorder", "Hyperpigmentation", "Skin hyperpigmentation"],
        "synonyms": ["pigmentation", "hyperpigmentation", "melanosis", "pigment change"],
    },
    "Rash": {
        "soc": "Skin and subcutaneous tissue disorders",
        "llts": ["Rash", "Skin rash", "Eruption"],
        "synonyms": ["skin rash", "rash", "red rash", "maculopapular rash", "cutaneous eruption"],
    },
    "Pruritus": {
        "soc": "Skin and subcutaneous tissue disorders",
        "llts": ["Pruritus", "Itching", "Skin itching"],
        "synonyms": ["itching", "itchy skin", "pruritus", "severe itching"],
    },
    "Stevens-Johnson syndrome": {
        "soc": "Skin and subcutaneous tissue disorders",
        "llts": ["Stevens-Johnson syndrome", "SJS"],
        "synonyms": ["stevens-johnson", "sjs", "blistering skin reaction"],
    },
    "Alopecia": {
        "soc": "Skin and subcutaneous tissue disorders",
        "llts": ["Alopecia", "Hair loss"],
        "synonyms": ["hair loss", "alopecia", "thinning hair"],
    },
    "Anaphylactic reaction": {
        "soc": "Immune system disorders",
        "llts": ["Anaphylactic reaction", "Anaphylaxis", "Anaphylactic shock"],
        "synonyms": ["anaphylaxis", "anaphylactic", "severe allergic shock", "throat swelling and hypotension"],
    },
    "Hypersensitivity": {
        "soc": "Immune system disorders",
        "llts": ["Hypersensitivity", "Allergic reaction", "Drug hypersensitivity"],
        "synonyms": ["allergic reaction", "hypersensitivity", "drug allergy"],
    },
    "Nausea": {
        "soc": "Gastrointestinal disorders",
        "llts": ["Nausea", "Feeling sick"],
        "synonyms": ["nausea", "nauseous", "queasy"],
    },
    "Vomiting": {
        "soc": "Gastrointestinal disorders",
        "llts": ["Vomiting", "Emesis"],
        "synonyms": ["vomiting", "emesis", "threw up", "throwing up"],
    },
    "Diarrhoea": {
        "soc": "Gastrointestinal disorders",
        "llts": ["Diarrhoea", "Diarrhea", "Loose stools"],
        "synonyms": ["diarrhea", "diarrhoea", "watery stools", "loose stools"],
    },
    "Gastrointestinal haemorrhage": {
        "soc": "Gastrointestinal disorders",
        "llts": ["Gastrointestinal haemorrhage", "GI bleeding", "Gastrointestinal hemorrhage"],
        "synonyms": ["gi bleed", "gi bleeding", "gastrointestinal bleeding", "gastrointestinal haemorrhage", "gastrointestinal hemorrhage", "melena", "hematemesis"],
    },
    "Pancreatitis": {
        "soc": "Gastrointestinal disorders",
        "llts": ["Pancreatitis", "Acute pancreatitis"],
        "synonyms": ["pancreatitis", "acute pancreatitis", "pancreatic inflammation"],
    },
    "Abdominal pain": {
        "soc": "Gastrointestinal disorders",
        "llts": ["Abdominal pain", "Stomach pain"],
        "synonyms": ["abdominal pain", "stomach pain", "belly pain", "epigastric pain"],
    },
    "Hepatotoxicity": {
        "soc": "Hepatobiliary disorders",
        "llts": ["Hepatotoxicity", "Drug-induced liver injury", "Liver injury"],
        "synonyms": ["hepatotoxicity", "liver injury", "drug-induced liver injury", "dili", "liver toxicity"],
    },
    "Alanine aminotransferase increased": {
        "soc": "Investigations",
        "llts": ["Alanine aminotransferase increased", "ALT increased", "Elevated liver enzymes"],
        "synonyms": ["elevated liver enzymes", "alt increased", "ast increased", "transaminitis", "elevated alt"],
    },
    "Acute kidney injury": {
        "soc": "Renal and urinary disorders",
        "llts": ["Acute kidney injury", "Acute renal failure", "AKI"],
        "synonyms": ["acute kidney injury", "acute renal failure", "aki", "renal failure", "kidney injury"],
    },
    "Myocardial infarction": {
        "soc": "Cardiac disorders",
        "llts": ["Myocardial infarction", "Heart attack", "Acute myocardial infarction"],
        "synonyms": ["heart attack", "myocardial infarction", "mi", "stemi", "nstemi"],
    },
    "Cardiac arrest": {
        "soc": "Cardiac disorders",
        "llts": ["Cardiac arrest", "Cardiopulmonary arrest"],
        "synonyms": ["cardiac arrest", "heart stopped", "asystole"],
    },
    "Atrial fibrillation": {
        "soc": "Cardiac disorders",
        "llts": ["Atrial fibrillation", "AFib"],
        "synonyms": ["atrial fibrillation", "afib", "irregular heartbeat"],
    },
    "Hypotension": {
        "soc": "Vascular disorders",
        "llts": ["Hypotension", "Low blood pressure"],
        "synonyms": ["hypotension", "low blood pressure", "drop in blood pressure"],
    },
    "Hypertension": {
        "soc": "Vascular disorders",
        "llts": ["Hypertension", "High blood pressure"],
        "synonyms": ["hypertension", "high blood pressure", "elevated blood pressure"],
    },
    "Pulmonary embolism": {
        "soc": "Respiratory, thoracic and mediastinal disorders",
        "llts": ["Pulmonary embolism", "PE"],
        "synonyms": ["pulmonary embolism", "pe", "lung clot"],
    },
    "Deep vein thrombosis": {
        "soc": "Vascular disorders",
        "llts": ["Deep vein thrombosis", "DVT"],
        "synonyms": ["deep vein thrombosis", "dvt", "leg blood clot"],
    },
    "Dyspnoea": {
        "soc": "Respiratory, thoracic and mediastinal disorders",
        "llts": ["Dyspnoea", "Dyspnea", "Shortness of breath"],
        "synonyms": ["shortness of breath", "dyspnea", "dyspnoea", "difficulty breathing", "sob"],
    },
    "Pneumonia": {
        "soc": "Infections and infestations",
        "llts": ["Pneumonia", "Lung infection"],
        "synonyms": ["pneumonia", "lung infection"],
    },
    "Sepsis": {
        "soc": "Infections and infestations",
        "llts": ["Sepsis", "Septicaemia"],
        "synonyms": ["sepsis", "septic shock", "septicaemia"],
    },
    "Neutropenia": {
        "soc": "Blood and lymphatic system disorders",
        "llts": ["Neutropenia", "Decreased neutrophil count"],
        "synonyms": ["neutropenia", "low neutrophils", "agranulocytosis"],
    },
    "Thrombocytopenia": {
        "soc": "Blood and lymphatic system disorders",
        "llts": ["Thrombocytopenia", "Low platelet count"],
        "synonyms": ["thrombocytopenia", "low platelet", "low platelets", "decreased platelets"],
    },
    "Anaemia": {
        "soc": "Blood and lymphatic system disorders",
        "llts": ["Anaemia", "Anemia", "Decreased haemoglobin"],
        "synonyms": ["anemia", "anaemia", "low hemoglobin", "low haemoglobin"],
    },
    "Pyrexia": {
        "soc": "General disorders and administration site conditions",
        "llts": ["Pyrexia", "Fever"],
        "synonyms": ["fever", "pyrexia", "high temperature", "febrile"],
    },
    "Fatigue": {
        "soc": "General disorders and administration site conditions",
        "llts": ["Fatigue", "Tiredness"],
        "synonyms": ["fatigue", "tiredness", "exhaustion", "lethargy"],
    },
    "Drug ineffective": {
        "soc": "General disorders and administration site conditions",
        "llts": ["Drug ineffective", "Lack of efficacy"],
        "synonyms": ["drug ineffective", "lack of efficacy", "not working", "treatment failure"],
    },
    "Death": {
        "soc": "General disorders and administration site conditions",
        "llts": ["Death", "Fatal outcome"],
        "synonyms": ["death", "died", "fatal", "passed away", "expired"],
    },
    "Headache": {
        "soc": "Nervous system disorders",
        "llts": ["Headache", "Cephalalgia"],
        "synonyms": ["headache", "migraine", "head pain"],
    },
    "Dizziness": {
        "soc": "Nervous system disorders",
        "llts": ["Dizziness", "Lightheadedness"],
        "synonyms": ["dizziness", "dizzy", "lightheadedness", "vertigo"],
    },
    "Peripheral neuropathy": {
        "soc": "Nervous system disorders",
        "llts": ["Peripheral neuropathy", "Neuropathy peripheral"],
        "synonyms": ["peripheral neuropathy", "numbness and tingling", "neuropathy"],
    },
    "Seizure": {
        "soc": "Nervous system disorders",
        "llts": ["Seizure", "Convulsion"],
        "synonyms": ["seizure", "convulsion", "epileptic fit"],
    },
    "Hyperglycaemia": {
        "soc": "Metabolism and nutrition disorders",
        "llts": ["Hyperglycaemia", "Hyperglycemia", "High blood sugar"],
        "synonyms": ["hyperglycemia", "hyperglycaemia", "high blood sugar", "elevated glucose"],
    },
    "Hypoglycaemia": {
        "soc": "Metabolism and nutrition disorders",
        "llts": ["Hypoglycaemia", "Hypoglycemia", "Low blood sugar"],
        "synonyms": ["hypoglycemia", "hypoglycaemia", "low blood sugar"],
    },
    "Myalgia": {
        "soc": "Musculoskeletal and connective tissue disorders",
        "llts": ["Myalgia", "Muscle pain"],
        "synonyms": ["myalgia", "muscle pain", "muscle ache"],
    },
    "Arthralgia": {
        "soc": "Musculoskeletal and connective tissue disorders",
        "llts": ["Arthralgia", "Joint pain"],
        "synonyms": ["arthralgia", "joint pain", "joint ache"],
    },
    "Rhabdomyolysis": {
        "soc": "Musculoskeletal and connective tissue disorders",
        "llts": ["Rhabdomyolysis", "Muscle breakdown"],
        "synonyms": ["rhabdomyolysis", "elevated ck", "muscle breakdown"],
    },
    "Off label use": {
        "soc": "Injury, poisoning and procedural complications",
        "llts": ["Off label use"],
        "synonyms": ["off label", "off-label use"],
    },
}

# Reference Safety Information (RSI / USPI) expected adverse reactions per drug keyword
REFERENCE_PRODUCT_LABELS: Dict[str, List[str]] = {
    "PEMETREXED": ["Nausea", "Vomiting", "Fatigue", "Neutropenia", "Anaemia", "Thrombocytopenia", "Rash", "Diarrhoea", "Skin discolouration"],
    "ASPIRIN": ["Gastrointestinal haemorrhage", "Nausea", "Vomiting", "Abdominal pain", "Hypersensitivity", "Dyspnoea"],
    "METFORMIN": ["Nausea", "Vomiting", "Diarrhoea", "Abdominal pain", "Hypoglycaemia"],
    "IBUPROFEN": ["Gastrointestinal haemorrhage", "Nausea", "Abdominal pain", "Acute kidney injury", "Rash", "Dizziness"],
    "WARFARIN": ["Gastrointestinal haemorrhage", "Anaemia", "Hypersensitivity"],
    "ATORVASTATIN": ["Myalgia", "Arthralgia", "Alanine aminotransferase increased", "Diarrhoea", "Nausea", "Rhabdomyolysis"],
    "LISINOPRIL": ["Hypotension", "Dizziness", "Headache", "Acute kidney injury", "Hyperglycaemia"],
    "PEMBROLIZUMAB": ["Fatigue", "Rash", "Pruritus", "Diarrhoea", "Nausea", "Hepatotoxicity", "Pneumonia", "Hypersensitivity"],
    "SEMAGLUTIDE": ["Nausea", "Vomiting", "Diarrhoea", "Abdominal pain", "Pancreatitis", "Hypoglycaemia"],
    "AMOXICILLIN": ["Diarrhoea", "Nausea", "Rash", "Hypersensitivity", "Anaphylactic reaction"],
    "ACETAMINOPHEN": ["Hepatotoxicity", "Alanine aminotransferase increased", "Nausea", "Vomiting", "Rash"],
    "PARACETAMOL": ["Hepatotoxicity", "Alanine aminotransferase increased", "Nausea", "Vomiting", "Rash"],
}


def map_to_meddra(verbatim: str) -> Tuple[str, str, str, float, str]:
    """Map a verbatim adverse event string to (PT, LLT, SOC, confidence, rationale)."""
    clean = verbatim.strip()
    lower = clean.lower()

    # 1. Exact PT match (case-insensitive)
    for pt, info in MEDDRA_TERMS.items():
        if lower == pt.lower():
            return pt, info["llts"][0], str(info["soc"]), 0.99, f"Exact MedDRA PT match for '{clean}'."

    # 2. Exact LLT or synonym substring match
    for pt, info in MEDDRA_TERMS.items():
        for syn in info["synonyms"]:  # type: ignore[attr-defined]
            if syn == lower or re.search(rf"\b{re.escape(syn)}\b", lower):
                conf = 0.94 if syn == lower else 0.88
                return pt, info["llts"][0], str(info["soc"]), conf, f"Matched clinical synonym '{syn}' to MedDRA PT '{pt}'."

    # 3. Fuzzy match against PT names
    all_pts = list(MEDDRA_TERMS.keys())
    matches = difflib.get_close_matches(clean, all_pts, n=1, cutoff=0.65)
    if matches:
        pt = matches[0]
        info = MEDDRA_TERMS[pt]
        ratio = difflib.SequenceMatcher(None, lower, pt.lower()).ratio()
        return pt, info["llts"][0], str(info["soc"]), round(max(0.70, ratio), 2), f"Fuzzy lexical match ({ratio:.2f}) to MedDRA PT '{pt}'."

    # 4. Fallback: preserve verbatim as PT under General disorders / Uncoded
    return clean.title(), clean, "General disorders and administration site conditions", 0.60, "No exact dictionary match; retained normalized verbatim term for human MedDRA coder review."


def check_listedness(suspect_drugs: List[str], pts: List[str]) -> Tuple[str, List[str], str]:
    """Check whether MedDRA PTs are listed/expected in reference product labels."""
    matched_terms: List[str] = []
    unlisted_terms: List[str] = []
    known_drug_found = False

    for drug in suspect_drugs:
        upper_drug = drug.upper()
        for ref_key, expected_pts in REFERENCE_PRODUCT_LABELS.items():
            if ref_key in upper_drug:
                known_drug_found = True
                for pt in pts:
                    if pt in expected_pts and pt not in matched_terms:
                        matched_terms.append(pt)

    for pt in pts:
        if pt not in matched_terms:
            unlisted_terms.append(pt)

    if not known_drug_found:
        return (
            "Indeterminate",
            [],
            f"Reference Safety Information (CCDS/USPI) not indexed locally for {', '.join(suspect_drugs) or 'suspect drug'}; manual label verification required.",
        )

    if unlisted_terms and matched_terms:
        return (
            "Unlisted / Unexpected",
            matched_terms,
            f"Mixed expectedness: {', '.join(matched_terms)} listed in RSI; {', '.join(unlisted_terms)} unlisted (case treated as Unlisted/Unexpected per ICH E2D).",
        )
    if unlisted_terms:
        return (
            "Unlisted / Unexpected",
            [],
            f"None of [{', '.join(unlisted_terms)}] are listed in the reference USPI/CCDS for {', '.join(suspect_drugs)}.",
        )
    return (
        "Listed / Expected",
        matched_terms,
        f"All coded events [{', '.join(matched_terms)}] are listed in the reference USPI/CCDS for {', '.join(suspect_drugs)}.",
    )
