import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from financial_crime.data.schema import Transaction


class TransactionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: Literal["transaction.created"] = "transaction.created"
    schema_version: str = "1"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    payload: Transaction


class StreamOutput(BaseModel):
    topic: str
    key: str
    payload: dict[str, Any]
