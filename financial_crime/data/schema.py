from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    beneficiary_id: str = Field(min_length=1)
    device_id: str = Field(min_length=1)
    merchant_id: str = "merchant-demo"
    timestamp: datetime
    amount: float = Field(ge=0, le=100_000_000)
    account_age_days: int = Field(ge=0)
    beneficiary_age_days: int = Field(ge=0)
    device_accounts_30d: int = Field(ge=1)
    transactions_1h: int = Field(ge=1)
    distance_from_usual_km: float = Field(ge=0)
    is_suspicious: int | None = Field(default=None, ge=0, le=1)
    source: str = "synthetic"


FEATURES = [
    "amount",
    "account_age_days",
    "beneficiary_age_days",
    "device_accounts_30d",
    "transactions_1h",
    "distance_from_usual_km",
]
