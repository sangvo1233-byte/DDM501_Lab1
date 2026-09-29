"""
Pydantic schemas for request/response validation.

The schema is the API's contract. Everything the model assumes about its input
is stated here, so a malformed request fails at the edge with a clear 422
instead of producing a confident-looking wrong score.
"""

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator

PAY_STATUS_MIN = -2
PAY_STATUS_MAX = 8


# =============================================================================
# CreditApplication request schema
# =============================================================================
class CreditApplication(BaseModel):
    """One applicant, as the core banking system sends it."""

    limit_bal: float = Field(
        ..., gt=0, le=2_000_000, description="Credit limit in NT dollars", examples=[120000]
    )
    sex: Literal[1, 2] = Field(..., description="1 = male, 2 = female", examples=[2])
    education: Literal[1, 2, 3, 4] = Field(
        ..., description="1 = graduate school, 2 = university, 3 = high school, 4 = others",
        examples=[2],
    )
    marriage: Literal[1, 2, 3] = Field(
        ..., description="1 = married, 2 = single, 3 = others", examples=[2]
    )
    age: int = Field(..., ge=18, le=100, description="Age in years", examples=[34])
    pay_status: List[int] = Field(
        ...,
        min_length=6,
        max_length=6,
        description=(
            "Repayment status for months t-1 .. t-6. "
            "-2 = no consumption, -1 = paid in full, 0 = revolving credit, "
            "1..8 = months of payment delay."
        ),
        examples=[[0, 0, 0, 0, 0, 0]],
    )
    bill_amt: List[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Statement amount for months t-1 .. t-6 (negative = credit balance)",
        examples=[[50000, 48000, 47000, 45000, 44000, 43000]],
    )
    pay_amt: List[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Amount actually paid for months t-1 .. t-6",
        examples=[[2000, 2000, 1500, 1500, 1500, 1000]],
    )

    # Custom validators: rules a Field bound cannot express on list items
    @field_validator("pay_status")
    @classmethod
    def pay_status_in_domain(cls, values: List[int]) -> List[int]:
        """Every repayment status must be a code the dataset defines (-2..8)."""
        bad = [v for v in values if not PAY_STATUS_MIN <= v <= PAY_STATUS_MAX]
        if bad:
            raise ValueError(
                f"pay_status values must be between {PAY_STATUS_MIN} and "
                f"{PAY_STATUS_MAX}, got {bad}"
            )
        return values

    @field_validator("pay_amt")
    @classmethod
    def pay_amt_non_negative(cls, values: List[float]) -> List[float]:
        """A payment is money received; a negative one is a data error."""
        if any(v < 0 for v in values):
            raise ValueError(f"pay_amt values must not be negative, got {values}")
        return values


# =============================================================================
# PredictionResponse schema
# =============================================================================
class PredictionResponse(BaseModel):
    """Scoring result plus the decision derived from it."""

    default_probability: float = Field(..., ge=0.0, le=1.0)
    risk_band: Literal["LOW", "MEDIUM", "HIGH"]
    decision: Literal["APPROVE", "REVIEW", "DECLINE"]
    review_threshold: float
    decline_threshold: float
    model_version: str


# =============================================================================
# HealthResponse schema
# =============================================================================
class HealthResponse(BaseModel):
    """Liveness and readiness of the service."""

    status: Literal["healthy", "unhealthy"]
    model_loaded: bool
    model_version: str


# =============================================================================
# Batch schemas
# =============================================================================
class BatchPredictionRequest(BaseModel):
    """Up to 500 applications scored in one call."""

    applications: List[CreditApplication] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    """Results in the same order as the request."""

    predictions: List[PredictionResponse]
    total_count: int
