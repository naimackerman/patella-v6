"""Template-based clinical narrative report generator."""

from typing import Dict


def _grade_descriptor(grade: int) -> str:
    """Convert OARSI osteophyte grade to text description."""
    return {0: "Absent", 1: "Small/doubtful", 2: "Definite/moderate", 3: "Large/severe"}.get(grade, "Unknown")


def _sclerosis_descriptor(grade: int) -> str:
    """Convert sclerosis grade to text description."""
    return {0: "None/minimal", 1: "Mild", 2: "Significant"}.get(grade, "Unknown")


def _jsn_descriptor(jsn_rate: float) -> str:
    """Convert JSN rate to text description."""
    if jsn_rate < 20:
        return "Normal"
    elif jsn_rate < 50:
        return "Narrowed"
    return "Severely narrowed"


def _asymmetry_descriptor(ratio: float) -> str:
    """Describe medial-lateral asymmetry."""
    if 0.8 < ratio < 1.2:
        return "Symmetric"
    elif ratio < 0.8:
        return "Asymmetric - predominantly medial"
    return "Asymmetric - predominantly lateral"


def _format_distribution(probs: list) -> str:
    """Format probability distribution as string."""
    labels = ["KL0", "KL1", "KL2", "KL3", "KL4"]
    parts = [f"{l}:{p:.1%}" for l, p in zip(labels, probs)]
    return " | ".join(parts)


def _generate_impression(jsn: Dict, osp: Dict, scl: Dict, kl_pred: Dict) -> str:
    """Generate clinical impression summary."""
    grade = kl_pred.get("grade", 0)
    impressions = []

    if grade == 0:
        impressions.append("No radiographic evidence of osteoarthritis.")
    elif grade == 1:
        impressions.append("Doubtful/minimal osteoarthritis. Possible early changes.")
    elif grade == 2:
        impressions.append("Mild osteoarthritis with definite osteophyte formation.")
    elif grade == 3:
        impressions.append("Moderate osteoarthritis with definite joint space narrowing.")
    elif grade == 4:
        impressions.append("Severe osteoarthritis with marked joint space loss.")

    # Note predominant compartment
    jsn_med = jsn.get("jsn_rate_medial", 0)
    jsn_lat = jsn.get("jsn_rate_lateral", 0)
    if jsn_med > jsn_lat + 15:
        impressions.append("Predominantly medial compartment involvement.")
    elif jsn_lat > jsn_med + 15:
        impressions.append("Predominantly lateral compartment involvement.")

    return " ".join(impressions)


def generate_clinical_report(jsn: Dict, osp: Dict, scl: Dict, kl_pred: Dict) -> str:
    """Generate structured clinical narrative from quantified features.

    Args:
        jsn: JSN feature dict (mJSW, JSN rates, ratio).
        osp: Osteophyte feature dict (per-site grades, sum).
        scl: Sclerosis feature dict (per-compartment grades, intensity).
        kl_pred: KL prediction dict (grade, confidence, probabilities).

    Returns:
        Formatted clinical report string.
    """
    report = f"""
AUTOMATED KOA ASSESSMENT REPORT (xrAI-OA)
============================================
KL Grade: {kl_pred.get('grade', '?')} (Confidence: {kl_pred.get('confidence', 0):.1%})
Grade distribution: {_format_distribution(kl_pred.get('probabilities', [0]*5))}

JOINT SPACE NARROWING:
- Medial compartment: mJSW = {jsn.get('mJSW_medial', 0):.1f} px
  (JSN rate: {jsn.get('jsn_rate_medial', 0):.0f}%)
  -> {_jsn_descriptor(jsn.get('jsn_rate_medial', 0))}
- Lateral compartment: mJSW = {jsn.get('mJSW_lateral', 0):.1f} px
  (JSN rate: {jsn.get('jsn_rate_lateral', 0):.0f}%)
  -> {_jsn_descriptor(jsn.get('jsn_rate_lateral', 0))}
- Medial-lateral ratio: {jsn.get('jsw_ratio', 0):.2f}
  ({_asymmetry_descriptor(jsn.get('jsw_ratio', 1.0))})

OSTEOPHYTES:
- Medial femur: OARSI Grade {osp.get('osp_grade_mf', 0)} ({_grade_descriptor(osp.get('osp_grade_mf', 0))})
- Lateral femur: OARSI Grade {osp.get('osp_grade_lf', 0)} ({_grade_descriptor(osp.get('osp_grade_lf', 0))})
- Medial tibia: OARSI Grade {osp.get('osp_grade_mt', 0)} ({_grade_descriptor(osp.get('osp_grade_mt', 0))})
- Lateral tibia: OARSI Grade {osp.get('osp_grade_lt', 0)} ({_grade_descriptor(osp.get('osp_grade_lt', 0))})
- Total osteophyte burden: {osp.get('osp_sum', 0)}/12

SUBCHONDRAL SCLEROSIS:
- Medial tibial plateau: {_sclerosis_descriptor(scl.get('scl_grade_medial', 0))}
  (Bone density index: {scl.get('scl_intensity_medial', 0):.0f},
   Trabecular complexity: FD={scl.get('scl_fractal_dim_med', 0):.3f})
- Lateral tibial plateau: {_sclerosis_descriptor(scl.get('scl_grade_lateral', 0))}
  (Bone density index: {scl.get('scl_intensity_lateral', 0):.0f},
   Trabecular complexity: FD={scl.get('scl_fractal_dim_lat', 0):.3f})

CLINICAL IMPRESSION:
{_generate_impression(jsn, osp, scl, kl_pred)}

NOTE: This is an AI-assisted assessment. All findings should be
confirmed by a qualified radiologist before clinical decision-making.
"""
    return report.strip()
