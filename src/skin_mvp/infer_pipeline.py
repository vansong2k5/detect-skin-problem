"""Inference pipeline combining vision embeddings and lightweight classifier."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import joblib
import numpy as np

from .features_vision import compute_image_signals, get_extractor
from .rules import Symptoms, TriageLevel, has_red_flags, level_from_rules


@dataclass
class VisionPrediction:
    predicted_condition: str
    confidence: float
    image_features: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary compatible with the prompt builder schema."""

        return {
            "Predicted_condition": self.predicted_condition,
            "Confidence": self.confidence,
            "Image_features": self.image_features,
        }


class VisionInferencePipeline:
    """Run inference using cached sklearn classifier and EfficientNet embeddings."""

    def __init__(self, artifact_path: str | Path, device: str = "cpu") -> None:
        self.artifact_path = Path(artifact_path)
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found at {artifact_path}")
        self.bundle = joblib.load(self.artifact_path)
        self.extractor = get_extractor(device=device)

    def _prepare_features(self, img: np.ndarray) -> tuple[np.ndarray, Dict[str, float]]:
        embedding = self.extractor.embed_array(img)
        signals = compute_image_signals(img)
        signal_cols = self.bundle.get("signal_columns", [])
        if signal_cols:
            extras = [float(signals.get(col[4:], 0.0)) for col in signal_cols]
            feature_vector = np.hstack([embedding, np.array(extras, dtype=float)])
        else:
            feature_vector = embedding
        return feature_vector, signals

    def predict_from_array(self, img: np.ndarray) -> VisionPrediction:
        if img is None:
            raise ValueError("Image array cannot be None")
        feature_vector, signals = self._prepare_features(img)
        scaler = self.bundle["scaler"]
        clf = self.bundle["clf"]
        label_map = self.bundle["label_map"]
        feature_scaled = scaler.transform([feature_vector])
        proba = clf.predict_proba(feature_scaled)[0]
        idx = int(np.argmax(proba))
        label = label_map[idx]
        confidence = float(proba[idx])
        return VisionPrediction(label, round(confidence, 4), signals)

    def predict_from_path(self, path: str) -> VisionPrediction:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Unable to read image: {path}")
        return self.predict_from_array(image)


def integrate_assessment(
    symptoms: Symptoms,
    vision: VisionPrediction,
    llm_response: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Combine rule-based triage with vision and optional LLM output."""

    red_flags = has_red_flags(symptoms)
    rule_level, rule_reasons = level_from_rules(symptoms, vision.image_features)

    final_level = rule_level
    final_reasons = list(rule_reasons)

    if red_flags:
        final_level = TriageLevel.D_EMERGENCY
        final_reasons.append("Red flag symptoms mandate emergency care")

    if llm_response:
        llm_triage = llm_response.get("triage")
        if llm_triage:
            try:
                llm_level = TriageLevel(llm_triage)
            except ValueError:
                llm_level = None
            if llm_level and llm_level != final_level:
                levels = list(TriageLevel)
                llm_idx = levels.index(llm_level)
                final_idx = levels.index(final_level)
                if llm_idx > final_idx:
                    final_level = levels[max(llm_idx, final_idx)]
                    final_reasons.append("LLM recommended higher urgency triage")
                elif vision.confidence < 0.45:
                    final_reasons.append("Low confidence vision prediction; prioritised rule-based triage")

    return {
        "triage": final_level.value,
        "vision": vision.to_dict(),
        "rule_based": {
            "level": rule_level.value,
            "reasons": rule_reasons,
        },
        "red_flags": red_flags,
        "final_reasons": final_reasons,
        "llm_response": llm_response,
    }


def predict_image_tags(image: Any, artifact_path: str | Path, device: str = "cpu") -> VisionPrediction:
    """Convenience wrapper mirroring the high-level API from the design doc."""

    pipeline = VisionInferencePipeline(artifact_path, device=device)
    if isinstance(image, str):
        return pipeline.predict_from_path(image)
    if isinstance(image, np.ndarray):
        return pipeline.predict_from_array(image)
    raise TypeError("image must be a file path or numpy array")

