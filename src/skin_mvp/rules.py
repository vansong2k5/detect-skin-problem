"""Rule based triage heuristics for dermatology symptoms."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Tuple


class TriageLevel(str, Enum):
    A_SELFCARE = "A-SELFCARE"
    B_REMOTE = "B-PHARM/REMOTE"
    C_SOON = "C-SOON"
    D_EMERGENCY = "D-EMERGENCY"


@dataclass
class Symptoms:
    """Structured symptom intake used by the rules and prompts."""

    age: int
    days: int
    fever_c: float | None = None
    pain: bool = False
    itch: bool = False
    rapid_spread: bool = False
    pus: bool = False
    blisters: bool = False
    around_eye: bool = False
    lip_swelling: bool = False
    non_blanching: bool = False
    breathing_diff: bool = False
    mucosal: bool = False
    immunosuppressed: bool = False
    pregnant: bool = False
    new_mole_changing: bool = False
    notes: str | None = None

    def flags(self) -> Dict[str, bool]:
        return {
            "rapid_spread": self.rapid_spread,
            "around_eye": self.around_eye,
            "lip_swelling": self.lip_swelling,
            "non_blanching": self.non_blanching,
            "breathing_diff": self.breathing_diff,
            "mucosal": self.mucosal,
            "immunosuppressed": self.immunosuppressed,
            "pregnant": self.pregnant,
            "high_fever": self.fever_c is not None and self.fever_c >= 38.5,
            "severe_pain": self.pain and self.rapid_spread,
        }


RED_FLAG_FIELDS: Tuple[str, ...] = (
    "around_eye",
    "lip_swelling",
    "non_blanching",
    "breathing_diff",
    "mucosal",
    "immunosuppressed",
    "pregnant",
    "high_fever",
)


def has_red_flags(symptoms: Symptoms) -> List[str]:
    flags = symptoms.flags()
    return [name for name in RED_FLAG_FIELDS if flags.get(name, False)]


def level_from_rules(symptoms: Symptoms, img_signals: Dict[str, float] | None = None) -> Tuple[TriageLevel, List[str]]:
    """Basic severity stratification from symptoms and optional image signals."""

    reasons: List[str] = []
    red_flags = has_red_flags(symptoms)
    if red_flags:
        reasons.append("Detected critical red flags: " + ", ".join(red_flags))
        return TriageLevel.D_EMERGENCY, reasons

    if symptoms.rapid_spread or symptoms.pus:
        reasons.append("Rapidly spreading or purulent lesion")
        if symptoms.age < 12 or symptoms.age > 70:
            reasons.append("Extremes of age require urgent review")
        return TriageLevel.C_SOON, reasons

    if symptoms.blisters and (img_signals or {}).get("blister_count", 0) >= 3:
        reasons.append("Multiple blisters observed")
        return TriageLevel.C_SOON, reasons

    if symptoms.new_mole_changing:
        reasons.append("Evolving pigmented lesion")
        return TriageLevel.C_SOON, reasons

    if symptoms.pain or symptoms.itch:
        reasons.append("Symptomatic lesion but no high-risk features")
        return TriageLevel.B_REMOTE, reasons

    reasons.append("No concerning systemic or local features")
    return TriageLevel.A_SELFCARE, reasons
