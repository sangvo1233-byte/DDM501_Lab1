"""
ML model wrapper for credit default risk scoring.

The wrapper is the only place that knows how the model was trained: which
columns it expects, in which order, and what its output means. The API layer
above it deals in business objects, never in feature vectors.

"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from app.config import DECLINE_THRESHOLD, MODEL_PATH, MODEL_VERSION, REVIEW_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Column order the model was fitted on. Changing this list without retraining
# is the classic training-serving skew bug (Lesson 04).
FEATURE_COLUMNS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)


class CreditRiskModel:
    """Loads the trained pipeline and turns applications into decisions."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self._load_model()

    def _load_model(self) -> None:
        """Load the trained pipeline and its metadata from disk."""
        try:
            bundle = joblib.load(self.model_path)
            self.model = bundle["pipeline"]
            self.metadata = bundle.get("metadata", {})
            logger.info(
                "Model loaded from %s (type=%s, trained_at=%s)",
                self.model_path,
                self.metadata.get("model_type"),
                self.metadata.get("trained_at"),
            )
        except FileNotFoundError:
            logger.error(
                "Model file not found at %s. Run 'python scripts/train_model.py' "
                "to create it, or point MODEL_PATH at an existing artifact.",
                self.model_path,
            )
            raise

    @staticmethod
    def to_frame(applications: List[Dict[str, Any]]) -> pd.DataFrame:
        """Flatten API payloads into the wide frame the model was trained on.

        pay_status[0..5] maps to PAY_0, PAY_2..PAY_6: PAY_1 does not exist in
        the original dataset, and reproducing that faithfully avoids skew.
        """
        pay_names = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
        rows = []
        for a in applications:
            row = {
                "LIMIT_BAL": a["limit_bal"],
                "SEX": a["sex"],
                "EDUCATION": a["education"],
                "MARRIAGE": a["marriage"],
                "AGE": a["age"],
            }
            for i, name in enumerate(pay_names):
                row[name] = a["pay_status"][i]
            for i in range(6):
                row[f"BILL_AMT{i + 1}"] = a["bill_amt"][i]
                row[f"PAY_AMT{i + 1}"] = a["pay_amt"][i]
            rows.append(row)
        return pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    # -------------------------------------------------------------------------
    def predict_proba(self, applications: List[Dict[str, Any]]) -> np.ndarray:
        """Return the probability of default for each application. (PROVIDED)"""
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        frame = self.to_frame(applications)
        return self.model.predict_proba(frame)[:, 1]

    @staticmethod
    def decide(probability: float) -> Dict[str, str]:
        """Turn a probability into a risk band and an underwriting decision.

        The higher threshold is checked first; checking REVIEW first would
        send every HIGH applicant to manual review instead of declining it.
        """
        if probability >= DECLINE_THRESHOLD:
            return {"risk_band": "HIGH", "decision": "DECLINE"}
        if probability >= REVIEW_THRESHOLD:
            return {"risk_band": "MEDIUM", "decision": "REVIEW"}
        return {"risk_band": "LOW", "decision": "APPROVE"}

    def score(self, application: Dict[str, Any]) -> Dict[str, Any]:
        """Score one application and return the full response payload."""
        probability = float(self.predict_proba([application])[0])
        result = {
            "default_probability": round(probability, 4),
            "review_threshold": REVIEW_THRESHOLD,
            "decline_threshold": DECLINE_THRESHOLD,
            "model_version": MODEL_VERSION,
        }
        result.update(self.decide(probability))
        return result

    # -------------------------------------------------------------------------
    def score_batch(self, applications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score many applications in a single vectorised pass. (PROVIDED)

        Note this calls predict_proba ONCE for the whole batch rather than
        looping over self.score. One call to the model for 500 applicants is
        far cheaper than 500 calls — the same reasoning behind batch serving
        in Lesson 08.
        """
        probabilities = self.predict_proba(applications)
        results = []
        for probability in probabilities:
            probability = float(probability)
            result = {
                "default_probability": round(probability, 4),
                "review_threshold": REVIEW_THRESHOLD,
                "decline_threshold": DECLINE_THRESHOLD,
                "model_version": MODEL_VERSION,
            }
            result.update(self.decide(probability))
            results.append(result)
        return results

    # -------------------------------------------------------------------------
    def is_loaded(self) -> bool:
        """Check whether the model is ready to serve. (PROVIDED)"""
        return self.model is not None


# =============================================================================
# Singleton accessor 
# =============================================================================
_model_instance: Optional[CreditRiskModel] = None


def get_model() -> CreditRiskModel:
    """Get or create the model singleton."""
    global _model_instance
    if _model_instance is None:
        _model_instance = CreditRiskModel()
    return _model_instance
