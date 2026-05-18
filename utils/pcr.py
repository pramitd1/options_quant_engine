"""Put/call ratio helpers shared by live capture and research reports."""

from __future__ import annotations

from typing import Any

from utils.numerics import safe_float


PCR_HIGH_CONTEXT_THRESHOLD = 1.288
PCR_LOW_CONTEXT_THRESHOLD = 0.835

PCR_BASIS_OPEN_INTEREST = "OPEN_INTEREST"
PCR_BASIS_VOLUME_ATM = "VOLUME_ATM"
PCR_BASIS_VOLUME = "VOLUME"
PCR_BASIS_UNAVAILABLE = "UNAVAILABLE"

PCR_BUCKET_HIGH = "HIGH_PCR"
PCR_BUCKET_MID = "MID_PCR"
PCR_BUCKET_LOW = "LOW_PCR"
PCR_BUCKET_UNAVAILABLE = "UNAVAILABLE"


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def normalize_pcr_basis(value: Any) -> str | None:
    """Return the canonical source basis for a PCR value."""
    text = str(value or "").upper().strip()
    aliases = {
        "OI": PCR_BASIS_OPEN_INTEREST,
        "PCR_OI": PCR_BASIS_OPEN_INTEREST,
        "OPEN_INTEREST": PCR_BASIS_OPEN_INTEREST,
        "OPENINTEREST": PCR_BASIS_OPEN_INTEREST,
        "VOLUME_ATM": PCR_BASIS_VOLUME_ATM,
        "ATM_VOLUME": PCR_BASIS_VOLUME_ATM,
        "NEAR_ATM_VOLUME": PCR_BASIS_VOLUME_ATM,
        "VOLUME": PCR_BASIS_VOLUME,
        "VOL": PCR_BASIS_VOLUME,
        "UNAVAILABLE": PCR_BASIS_UNAVAILABLE,
    }
    return aliases.get(text)


def classify_pcr_bucket(value: Any) -> str:
    """Classify a numeric PCR value into the engine's coarse context bucket."""
    parsed = safe_float(value, None)
    if parsed is None:
        return PCR_BUCKET_UNAVAILABLE
    if parsed >= PCR_HIGH_CONTEXT_THRESHOLD:
        return PCR_BUCKET_HIGH
    if parsed <= PCR_LOW_CONTEXT_THRESHOLD:
        return PCR_BUCKET_LOW
    return PCR_BUCKET_MID


def normalize_pcr_bucket_for_reporting(value: Any, *, default: str = "UNKNOWN") -> str:
    """Normalize PCR bucket variants into compact report labels."""
    text = str(value or "").upper().strip()
    if text in {"", "NA", "N/A", "NAN", "NONE", "NULL", "<NA>", "UNAVAILABLE", "UNKNOWN"}:
        return default
    if text in {"HIGH", "HIGH_PCR", "PUT_DOMINANT"}:
        return "HIGH"
    if text in {"MID", "MID_PCR", "NEUTRAL"}:
        return "MID"
    if text in {"LOW", "LOW_PCR", "CALL_DOMINANT"}:
        return "LOW"
    if text.endswith("_PCR"):
        return text.removesuffix("_PCR")
    return text


def select_canonical_pcr(
    *,
    open_interest_pcr: Any = None,
    volume_pcr_atm: Any = None,
    volume_pcr: Any = None,
    pcr_value: Any = None,
    pcr_basis: Any = None,
) -> dict[str, float | str | None]:
    """Select one canonical PCR value while retaining raw source readings.

    The hierarchy intentionally favors OI PCR because the mined historical
    priors were trained on OI-based PCR. Near-ATM volume PCR is the next best
    fallback because it is more responsive than full-chain volume PCR.
    """
    oi_value = safe_float(open_interest_pcr, None)
    atm_volume_value = safe_float(volume_pcr_atm, None)
    full_volume_value = safe_float(volume_pcr, None)
    explicit_value = safe_float(pcr_value, None)
    explicit_basis = normalize_pcr_basis(pcr_basis)

    if explicit_value is not None and explicit_basis == PCR_BASIS_OPEN_INTEREST and oi_value is None:
        oi_value = explicit_value
    elif explicit_value is not None and explicit_basis == PCR_BASIS_VOLUME_ATM and atm_volume_value is None:
        atm_volume_value = explicit_value
    elif explicit_value is not None and explicit_basis == PCR_BASIS_VOLUME and full_volume_value is None:
        full_volume_value = explicit_value

    if oi_value is not None:
        selected_value = oi_value
        selected_basis = PCR_BASIS_OPEN_INTEREST
    elif atm_volume_value is not None:
        selected_value = atm_volume_value
        selected_basis = PCR_BASIS_VOLUME_ATM
    elif full_volume_value is not None:
        selected_value = full_volume_value
        selected_basis = PCR_BASIS_VOLUME
    elif explicit_value is not None:
        selected_value = explicit_value
        selected_basis = explicit_basis or PCR_BASIS_UNAVAILABLE
    else:
        selected_value = None
        selected_basis = PCR_BASIS_UNAVAILABLE

    return {
        "open_interest_pcr": _round(oi_value),
        "volume_pcr_atm": _round(atm_volume_value),
        "volume_pcr": _round(full_volume_value),
        "pcr_value": _round(selected_value),
        "pcr_basis": selected_basis,
        "pcr_bucket": classify_pcr_bucket(selected_value),
    }
