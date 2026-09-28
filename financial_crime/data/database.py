from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from financial_crime.config import settings


class Base(DeclarativeBase):
    pass


class TransactionRow(Base):
    __tablename__ = "transactions"

    transaction_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(80), index=True)
    beneficiary_id: Mapped[str] = mapped_column(String(80), index=True)
    device_id: Mapped[str] = mapped_column(String(80), index=True)
    merchant_id: Mapped[str] = mapped_column(String(80), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    amount: Mapped[float] = mapped_column(Float)
    account_age_days: Mapped[int] = mapped_column(Integer)
    beneficiary_age_days: Mapped[int] = mapped_column(Integer)
    device_accounts_30d: Mapped[int] = mapped_column(Integer)
    transactions_1h: Mapped[int] = mapped_column(Integer)
    distance_from_usual_km: Mapped[float] = mapped_column(Float)
    is_suspicious: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    source: Mapped[str] = mapped_column(String(80))


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    source: Mapped[str] = mapped_column(String(255))
    row_count: Mapped[int] = mapped_column(Integer)
    schema_version: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class InvestigationRow(Base):
    __tablename__ = "investigations"

    investigation_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    transaction_id: Mapped[str] = mapped_column(String(80), index=True)
    status: Mapped[str] = mapped_column(String(40), default="requires-review")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class EvidenceRow(Base):
    __tablename__ = "evidence"

    evidence_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    investigation_id: Mapped[str] = mapped_column(String(100), index=True)
    category: Mapped[str] = mapped_column(String(40))
    source: Mapped[str] = mapped_column(String(100))
    claim: Mapped[str] = mapped_column(Text)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ToolCallRow(Base):
    __tablename__ = "tool_calls"

    call_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    tool_name: Mapped[str] = mapped_column(String(100), index=True)
    input_sha256: Mapped[str] = mapped_column(String(64))
    output_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(30))
    duration_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class AgentRunRow(Base):
    __tablename__ = "agent_runs"

    run_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    investigation_id: Mapped[str] = mapped_column(String(100), index=True)
    status: Mapped[str] = mapped_column(String(30))
    report_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


def make_session_factory():
    if settings.database_url.startswith("sqlite:///./"):
        Path(settings.database_url.removeprefix("sqlite:///./")).parent.mkdir(
            parents=True, exist_ok=True
        )
    connect_args = (
        {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
    )
    engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


SessionFactory = make_session_factory()
