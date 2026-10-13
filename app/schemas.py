"""Request/response models. Literal types reject typos like "month to month"
with a clear 422 error instead of a silently wrong prediction."""
from typing import Literal

from pydantic import BaseModel, Field

YesNo = Literal["Yes", "No"]
NoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    customerID: str | None = None
    gender: Literal["Male", "Female"]
    SeniorCitizen: YesNo
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: NoInternet
    OnlineBackup: NoInternet
    DeviceProtection: NoInternet
    TechSupport: NoInternet
    StreamingTV: NoInternet
    StreamingMovies: NoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {"json_schema_extra": {"example": {
        "customerID": "demo-001", "gender": "Female", "SeniorCitizen": "No",
        "Partner": "No", "Dependents": "No", "tenure": 1, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 74.4, "TotalCharges": 74.4}}}


class Reason(BaseModel):
    feature: str
    value: str | float
    impact: float


class Prediction(BaseModel):
    customerID: str | None
    churn_probability: float
    risk_tier: Literal["high", "medium", "low"]
    top_reasons: list[Reason]
    model: str
