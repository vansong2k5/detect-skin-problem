"""Prompt construction utilities for LLM reasoning."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict

from .rules import Symptoms, has_red_flags, level_from_rules


def build_prompt(symptoms: Symptoms, vision_summary: Dict[str, Any]) -> str:
    """Construct a safety-first prompt for downstream LLMs."""

    red_flags = has_red_flags(symptoms)
    rule_level, rule_reasons = level_from_rules(symptoms, vision_summary.get("Image_features", {}))

    payload = {
        "patient_symptoms": asdict(symptoms),
        "vision_summary": vision_summary,
        "rule_based_triage": {
            "level": rule_level.value,
            "reasons": rule_reasons,
            "red_flags": red_flags,
        },
    }

    instructions = (
        "You are an AI dermatology assistant."
        " Carefully review the data and provide a cautious differential diagnosis and triage."
        " Honour any D-EMERGENCY rules in the payload and never downgrade the severity."
        " Return a JSON object with keys: diagnosis, triage, confidence, red_flags, reasoning, care_advice."
        " Use short bullet strings (max 3) for reasoning and care advice."
    )

    prompt = f"{instructions}\n\n[DATA]\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    return prompt
