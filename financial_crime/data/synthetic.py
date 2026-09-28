import random
from datetime import UTC, datetime, timedelta

import pandas as pd


def generate_transactions(rows: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a reproducible, explicitly synthetic transaction population."""
    if rows < 100:
        raise ValueError("rows must be at least 100 to preserve useful class splits")
    # Seeded pseudorandomness is used only to reproduce synthetic fixtures.
    rng = random.Random(seed)  # nosec B311
    base = datetime(2025, 1, 1, tzinfo=UTC)
    items: list[dict[str, object]] = []
    for index in range(rows):
        suspicious = int(rng.random() < 0.08)
        amount = rng.lognormvariate(7.0 if suspicious else 5.8, 0.75)
        if suspicious and rng.random() < 0.55:
            amount *= 3.2
        item = {
            "transaction_id": f"TX-{index + 1:07d}",
            "account_id": f"AC-{index // 4 + 1:06d}",
            "beneficiary_id": f"BN-{rng.randrange(max(1, rows // 8)):06d}",
            "device_id": f"DV-{rng.randrange(max(1, rows // 10)):06d}",
            "merchant_id": f"ME-{rng.randrange(80):04d}",
            "timestamp": base + timedelta(minutes=index * 11),
            "amount": round(amount, 2),
            "account_age_days": rng.randint(1, 120) if suspicious else rng.randint(90, 5000),
            "beneficiary_age_days": rng.randint(0, 5) if suspicious else rng.randint(30, 2500),
            "device_accounts_30d": rng.randint(3, 9) if suspicious else rng.randint(1, 2),
            "transactions_1h": rng.randint(3, 8) if suspicious else rng.randint(1, 2),
            "distance_from_usual_km": round(
                rng.uniform(300, 4000) if suspicious else rng.uniform(0, 80), 2
            ),
            "is_suspicious": suspicious,
            "source": "synthetic",
        }
        items.append(item)
    return pd.DataFrame(items)
