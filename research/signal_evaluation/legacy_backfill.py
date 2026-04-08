"""
Module: legacy_backfill.py

Purpose:
    Reconstruct newly introduced signal-evaluation columns for legacy rows.

Role in the System:
    Part of the research signal-evaluation layer used for schema evolution backfills.

Key Outputs:
    Updated signal-evaluation dataframes with selected option diagnostics populated
    from saved chain snapshots where available.

Downstream Usage:
    Consumed by maintenance scripts prior to reporting and calibration runs.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import pandas as pd

from analytics.greeks_engine import _parse_expiry_years, compute_option_greeks
from config.settings import BASE_DIR, LOT_SIZE
from data.replay_loader import load_option_chain_snapshot


PROJECT_ROOT = Path(BASE_DIR)
PROPOSAL_CONFIDENCE_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _resolve_chain_path(raw_path: Any, project_root: Path) -> Path | None:
    if raw_path is None:
        return None
    token = str(raw_path).strip()
    if not token:
        return None

    candidate = Path(token)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    direct = (project_root / candidate).resolve()
    if direct.exists():
        return direct

    name_fallback = (project_root / "debug_samples" / "option_chain_snapshots" / candidate.name).resolve()
    if name_fallback.exists():
        return name_fallback

    return None


def _normalize_snapshot_match_key(raw_path: Any, *, project_root: Path | None = None) -> str | None:
    if raw_path is None or pd.isna(raw_path):
        return None

    token = str(raw_path).strip().replace("\\", "/")
    if not token:
        return None

    candidate = Path(token)
    normalized_root = project_root.resolve() if project_root is not None else None
    if candidate.is_absolute() and normalized_root is not None:
        try:
            return candidate.resolve().relative_to(normalized_root).as_posix()
        except Exception:
            pass

    parts = candidate.parts
    if "debug_samples" in parts:
        index = parts.index("debug_samples")
        return Path(*parts[index:]).as_posix()

    return candidate.as_posix()


def _first_existing_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _normalize_option_type(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    token = str(value).upper().strip()
    if not token:
        return None
    aliases = {
        "CE": "CE",
        "CALL": "CE",
        "C": "CE",
        "PE": "PE",
        "PUT": "PE",
        "P": "PE",
    }
    return aliases.get(token, token)


def _normalize_expiry_date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        token = str(value).strip()
        return token[:10] if token else None
    return parsed.strftime("%Y-%m-%d")


def _summarize_candidate_contracts(
    chain: pd.DataFrame,
    *,
    selected_strike: float | None,
    selected_option_type: Any,
    selected_expiry: Any,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    if chain is None or chain.empty:
        return []

    strike_col = _first_existing_column(chain, ("strikePrice", "STRIKE_PR"))
    if strike_col is None:
        return []

    type_col = _first_existing_column(chain, ("OPTION_TYP", "option_type", "optionType"))
    expiry_col = _first_existing_column(chain, ("EXPIRY_DT", "expiry", "expiry_date"))
    vol_col = _first_existing_column(chain, ("totalTradedVolume", "VOLUME"))
    oi_col = _first_existing_column(chain, ("openInterest", "OPEN_INT"))

    working = chain.copy()
    working[strike_col] = pd.to_numeric(working[strike_col], errors="coerce")
    working = working.dropna(subset=[strike_col])
    if working.empty:
        return []

    target_option_type = _normalize_option_type(selected_option_type)
    target_expiry = _normalize_expiry_date(selected_expiry)

    if type_col is not None:
        working["_normalized_option_type"] = working[type_col].map(_normalize_option_type)
    else:
        working["_normalized_option_type"] = None

    if expiry_col is not None:
        working["_normalized_expiry"] = working[expiry_col].map(_normalize_expiry_date)
    else:
        working["_normalized_expiry"] = None

    if selected_strike is None:
        working["_strike_distance"] = float("inf")
    else:
        working["_strike_distance"] = (working[strike_col] - float(selected_strike)).abs()

    working["_option_type_match"] = working["_normalized_option_type"] == target_option_type
    working["_expiry_exact_match"] = working["_normalized_expiry"] == target_expiry
    working["_expiry_month_match"] = False
    if target_expiry:
        target_month = target_expiry[:7]
        working["_expiry_month_match"] = working["_normalized_expiry"].astype(str).str.startswith(target_month)

    if vol_col is not None:
        working["_volume_sort"] = pd.to_numeric(working[vol_col], errors="coerce").fillna(0.0)
    else:
        working["_volume_sort"] = 0.0
    if oi_col is not None:
        working["_oi_sort"] = pd.to_numeric(working[oi_col], errors="coerce").fillna(0.0)
    else:
        working["_oi_sort"] = 0.0

    ranked = working.sort_values(
        ["_option_type_match", "_expiry_exact_match", "_expiry_month_match", "_strike_distance", "_volume_sort", "_oi_sort"],
        ascending=[False, False, False, True, False, False],
        kind="stable",
    ).head(max(int(top_n), 1))

    rows: list[dict[str, Any]] = []
    for _, candidate in ranked.iterrows():
        rows.append(
            {
                "strike": _safe_float(candidate.get(strike_col)),
                "strike_distance": _safe_float(candidate.get("_strike_distance")),
                "option_type_raw": candidate.get(type_col) if type_col is not None else None,
                "option_type_normalized": candidate.get("_normalized_option_type"),
                "option_type_match": bool(candidate.get("_option_type_match")),
                "expiry_raw": candidate.get(expiry_col) if expiry_col is not None else None,
                "expiry_normalized": candidate.get("_normalized_expiry"),
                "expiry_exact_match": bool(candidate.get("_expiry_exact_match")),
                "expiry_month_match": bool(candidate.get("_expiry_month_match")),
                "last_price": _safe_float(candidate.get("lastPrice")) or _safe_float(candidate.get("LAST_PRICE")),
                "open_interest": _safe_float(candidate.get(oi_col)) if oi_col is not None else None,
                "volume": _safe_float(candidate.get(vol_col)) if vol_col is not None else None,
            }
        )
    return rows


def audit_unresolved_signal_contract_matches(
    frame: pd.DataFrame,
    *,
    project_root: Path | None = None,
    top_n: int = 5,
) -> pd.DataFrame:
    root = project_root or PROJECT_ROOT
    if frame is None or frame.empty:
        return pd.DataFrame()

    chain_cache: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []
    required_missing_mask = (
        frame.get("selected_option_last_price", pd.Series(index=frame.index)).isna()
        | frame.get("selected_option_iv", pd.Series(index=frame.index)).isna()
        | frame.get("selected_option_delta", pd.Series(index=frame.index)).isna()
    )

    for _, row in frame[required_missing_mask].iterrows():
        snapshot_path = _resolve_chain_path(row.get("saved_chain_snapshot_path"), root)
        if snapshot_path is None:
            continue

        key = str(snapshot_path)
        if key not in chain_cache:
            try:
                chain_cache[key] = load_option_chain_snapshot(key)
            except Exception:
                chain_cache[key] = pd.DataFrame()

        chain = chain_cache[key]
        matched = _match_selected_contract(
            chain,
            strike=_safe_float(row.get("strike")),
            option_type=row.get("option_type"),
            selected_expiry=row.get("selected_expiry"),
        )
        if matched is not None:
            continue

        top_candidates = _summarize_candidate_contracts(
            chain,
            selected_strike=_safe_float(row.get("strike")),
            selected_option_type=row.get("option_type"),
            selected_expiry=row.get("selected_expiry"),
            top_n=top_n,
        )
        available_option_types = sorted(
            {
                value
                for value in chain.get(_first_existing_column(chain, ("OPTION_TYP", "option_type", "optionType")), pd.Series(dtype=object))
                .map(_normalize_option_type)
                .dropna()
                .astype(str)
                .tolist()
            }
        )
        available_expiries = sorted(
            {
                value
                for value in chain.get(_first_existing_column(chain, ("EXPIRY_DT", "expiry", "expiry_date")), pd.Series(dtype=object))
                .map(_normalize_expiry_date)
                .dropna()
                .astype(str)
                .tolist()
            }
        )
        audit_rows.append(
            {
                "signal_id": row.get("signal_id"),
                "signal_timestamp": row.get("signal_timestamp"),
                "symbol": row.get("symbol"),
                "direction": row.get("direction"),
                "saved_chain_snapshot_path": row.get("saved_chain_snapshot_path"),
                "resolved_chain_snapshot_path": str(snapshot_path),
                "selected_strike": _safe_float(row.get("strike")),
                "selected_option_type_raw": row.get("option_type"),
                "selected_option_type_normalized": _normalize_option_type(row.get("option_type")),
                "selected_expiry_raw": row.get("selected_expiry"),
                "selected_expiry_normalized": _normalize_expiry_date(row.get("selected_expiry")),
                "spot_at_signal": _safe_float(row.get("spot_at_signal")),
                "chain_rows": int(len(chain)),
                "available_option_types": json.dumps(available_option_types),
                "available_expiries": json.dumps(available_expiries[:20]),
                "top_candidate_count": int(len(top_candidates)),
                "top_candidates_json": json.dumps(top_candidates),
            }
        )

    return pd.DataFrame(audit_rows)


def _normalize_direction_to_option_type(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    token = str(value).upper().strip()
    mapping = {
        "CALL": "CE",
        "CE": "CE",
        "PUT": "PE",
        "PE": "PE",
    }
    return mapping.get(token)


def _load_confirmation_mode_replay_evidence(project_root: Path) -> pd.DataFrame:
    evidence_path = project_root / "research" / "ml_evaluation" / "confirmation_mode_replay" / "confirmation_mode_replay_delta.csv"
    if not evidence_path.exists():
        return pd.DataFrame()

    try:
        replay = pd.read_csv(evidence_path)
    except Exception:
        return pd.DataFrame()

    if replay.empty:
        return replay

    replay = replay.copy()
    replay["chain_snapshot_key"] = replay.get("chain_snapshot", pd.Series(index=replay.index, dtype=object)).map(
        lambda value: _normalize_snapshot_match_key(value, project_root=project_root)
    )
    replay["chain_snapshot_name"] = replay.get("chain_snapshot_key", pd.Series(index=replay.index, dtype=object)).map(
        lambda value: Path(str(value)).name if value else None
    )
    replay["spot_snapshot_key"] = replay.get("spot_snapshot", pd.Series(index=replay.index, dtype=object)).map(
        lambda value: _normalize_snapshot_match_key(value, project_root=project_root)
    )
    return replay


def _lookup_confirmation_mode_replay_evidence(
    audit_row: pd.Series,
    *,
    replay_evidence: pd.DataFrame,
    project_root: Path,
) -> dict[str, Any]:
    if replay_evidence is None or replay_evidence.empty:
        return {}

    chain_key = _normalize_snapshot_match_key(audit_row.get("resolved_chain_snapshot_path"), project_root=project_root)
    if not chain_key:
        return {}

    exact = replay_evidence.loc[replay_evidence.get("chain_snapshot_key", pd.Series(dtype=object)) == chain_key]
    if exact.empty:
        chain_name = Path(chain_key).name
        exact = replay_evidence.loc[replay_evidence.get("chain_snapshot_name", pd.Series(dtype=object)) == chain_name]
    if exact.empty:
        return {}

    replay_row = exact.iloc[0]
    trade_status = str(replay_row.get("trade_status_disc") or "").upper().strip() or None
    direction = str(replay_row.get("direction_disc") or "").upper().strip() or None
    confirmation_status = str(replay_row.get("confirmation_status_disc") or "").upper().strip() or None
    direction_source = str(replay_row.get("direction_source_x") or "").upper().strip() or None
    reason_code = str(replay_row.get("no_trade_reason_code_disc") or "").upper().strip() or None

    return {
        "archived_replay_match_type": "EXACT_CHAIN_SNAPSHOT",
        "archived_replay_timestamp": replay_row.get("timestamp_x"),
        "archived_trade_status": trade_status,
        "archived_direction": direction,
        "archived_direction_source": direction_source,
        "archived_confirmation_status": confirmation_status,
        "archived_no_trade_reason_code": reason_code,
        "archived_chain_snapshot": replay_row.get("chain_snapshot"),
        "archived_spot_snapshot": replay_row.get("spot_snapshot"),
    }


def _proposal_confidence_at_least(confidence: Any, minimum: str) -> bool:
    actual = PROPOSAL_CONFIDENCE_ORDER.get(str(confidence or "").upper().strip(), -1)
    required = PROPOSAL_CONFIDENCE_ORDER.get(str(minimum or "MEDIUM").upper().strip(), -1)
    return actual >= required


def _is_missing_scalar(value: Any) -> bool:
    if value is None or pd.isna(value):
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def propose_repairs_for_unresolved_signal_contract_matches(
    audit_frame: pd.DataFrame,
    *,
    project_root: Path | None = None,
) -> pd.DataFrame:
    if audit_frame is None or audit_frame.empty:
        return pd.DataFrame()

    root = project_root or PROJECT_ROOT
    replay_evidence = _load_confirmation_mode_replay_evidence(root)
    proposals: list[dict[str, Any]] = []
    for _, row in audit_frame.iterrows():
        direction_option_type = _normalize_direction_to_option_type(row.get("direction"))
        selected_option_type = _normalize_option_type(row.get("selected_option_type_raw"))
        target_option_type = direction_option_type or selected_option_type
        replay_match = _lookup_confirmation_mode_replay_evidence(row, replay_evidence=replay_evidence, project_root=root)
        archived_direction_option_type = _normalize_direction_to_option_type(replay_match.get("archived_direction"))

        try:
            candidates = json.loads(str(row.get("top_candidates_json") or "[]"))
        except Exception:
            candidates = []
        if not isinstance(candidates, list) or not candidates:
            continue

        filtered = candidates
        if target_option_type:
            preferred = [candidate for candidate in candidates if candidate.get("option_type_normalized") == target_option_type]
            if preferred:
                filtered = preferred

        chosen = filtered[0]
        reasons: list[str] = []
        confidence = "LOW"
        if direction_option_type:
            reasons.append(f"direction_implies_{direction_option_type}")
        if replay_match.get("archived_replay_match_type"):
            reasons.append(f"archived_replay_match_{str(replay_match['archived_replay_match_type']).lower()}")
        if archived_direction_option_type:
            reasons.append(f"archived_direction_implies_{archived_direction_option_type}")
        elif replay_match.get("archived_trade_status") == "NO_SIGNAL":
            reasons.append("archived_snapshot_was_no_signal")
        if chosen.get("expiry_exact_match"):
            reasons.append("expiry_exact_match")
        elif chosen.get("expiry_month_match"):
            reasons.append("expiry_month_match")
        if chosen.get("strike_distance") not in (None, "", float("inf")):
            reasons.append(f"nearest_strike_distance_{chosen.get('strike_distance')}")
        if chosen.get("option_type_match"):
            reasons.append("option_type_matches_existing_selection")

        if archived_direction_option_type and chosen.get("option_type_normalized") == archived_direction_option_type and chosen.get("expiry_exact_match"):
            confidence = "HIGH"
        elif direction_option_type and chosen.get("option_type_normalized") == direction_option_type and chosen.get("expiry_exact_match"):
            confidence = "HIGH"
        elif direction_option_type and chosen.get("option_type_normalized") == direction_option_type:
            confidence = "MEDIUM"
        elif chosen.get("expiry_exact_match"):
            confidence = "MEDIUM"

        proposals.append(
            {
                "signal_id": row.get("signal_id"),
                "signal_timestamp": row.get("signal_timestamp"),
                "symbol": row.get("symbol"),
                "resolved_chain_snapshot_path": row.get("resolved_chain_snapshot_path"),
                "selected_option_type_raw": row.get("selected_option_type_raw"),
                "selected_option_type_normalized": selected_option_type,
                "direction": row.get("direction"),
                "direction_implied_option_type": direction_option_type,
                "archived_direction": replay_match.get("archived_direction"),
                "archived_direction_implied_option_type": archived_direction_option_type,
                "archived_trade_status": replay_match.get("archived_trade_status"),
                "archived_direction_source": replay_match.get("archived_direction_source"),
                "archived_confirmation_status": replay_match.get("archived_confirmation_status"),
                "archived_no_trade_reason_code": replay_match.get("archived_no_trade_reason_code"),
                "archived_replay_match_type": replay_match.get("archived_replay_match_type"),
                "archived_replay_timestamp": replay_match.get("archived_replay_timestamp"),
                "selected_strike": row.get("selected_strike"),
                "selected_expiry_normalized": row.get("selected_expiry_normalized"),
                "proposed_option_type": chosen.get("option_type_normalized") or target_option_type,
                "proposed_strike": chosen.get("strike"),
                "proposed_expiry": chosen.get("expiry_normalized") or row.get("selected_expiry_normalized"),
                "proposed_direction": replay_match.get("archived_direction") if archived_direction_option_type else None,
                "proposal_confidence": confidence,
                "proposal_reasons": json.dumps(reasons),
                "chosen_candidate_json": json.dumps(chosen),
            }
        )

    return pd.DataFrame(proposals)


def apply_repair_proposals_to_dataset(
    frame: pd.DataFrame,
    proposals: pd.DataFrame,
    *,
    min_confidence: str = "MEDIUM",
) -> tuple[pd.DataFrame, dict[str, int]]:
    updated = pd.DataFrame(frame).copy()
    if updated.empty or proposals is None or proposals.empty:
        return updated, {
            "rows_considered": 0,
            "rows_accepted": 0,
            "rows_updated": 0,
            "direction_fields_applied": 0,
            "option_type_fields_applied": 0,
            "strike_fields_applied": 0,
            "expiry_fields_applied": 0,
        }

    accepted = proposals.loc[
        proposals.get("proposal_confidence", pd.Series(index=proposals.index, dtype=object)).map(
            lambda value: _proposal_confidence_at_least(value, min_confidence)
        )
    ].copy()

    direction_fields_applied = 0
    option_type_fields_applied = 0
    strike_fields_applied = 0
    expiry_fields_applied = 0
    rows_updated = 0

    signal_ids = updated.get("signal_id", pd.Series(index=updated.index, dtype=object)).astype(str)
    for _, proposal in accepted.iterrows():
        signal_id = str(proposal.get("signal_id") or "").strip()
        if not signal_id:
            continue

        matching_indices = updated.index[signal_ids == signal_id].tolist()
        if not matching_indices:
            continue

        row_changed = False
        for idx in matching_indices:
            proposed_direction = proposal.get("proposed_direction")
            if "direction" in updated.columns and _is_missing_scalar(updated.at[idx, "direction"]) and not _is_missing_scalar(proposed_direction):
                updated.at[idx, "direction"] = proposed_direction
                direction_fields_applied += 1
                row_changed = True

            proposed_option_type = proposal.get("proposed_option_type")
            if "option_type" in updated.columns and _is_missing_scalar(updated.at[idx, "option_type"]) and not _is_missing_scalar(proposed_option_type):
                updated.at[idx, "option_type"] = proposed_option_type
                option_type_fields_applied += 1
                row_changed = True

            proposed_strike = proposal.get("proposed_strike")
            if "strike" in updated.columns and _is_missing_scalar(updated.at[idx, "strike"]) and not _is_missing_scalar(proposed_strike):
                updated.at[idx, "strike"] = proposed_strike
                strike_fields_applied += 1
                row_changed = True

            proposed_expiry = proposal.get("proposed_expiry")
            if "selected_expiry" in updated.columns and _is_missing_scalar(updated.at[idx, "selected_expiry"]) and not _is_missing_scalar(proposed_expiry):
                updated.at[idx, "selected_expiry"] = proposed_expiry
                expiry_fields_applied += 1
                row_changed = True

            if row_changed and "direction_source" in updated.columns and _is_missing_scalar(updated.at[idx, "direction_source"]):
                archived_source = proposal.get("archived_direction_source")
                if not _is_missing_scalar(archived_source):
                    updated.at[idx, "direction_source"] = archived_source

        if row_changed:
            rows_updated += 1

    summary = {
        "rows_considered": int(len(proposals)),
        "rows_accepted": int(len(accepted)),
        "rows_updated": rows_updated,
        "direction_fields_applied": direction_fields_applied,
        "option_type_fields_applied": option_type_fields_applied,
        "strike_fields_applied": strike_fields_applied,
        "expiry_fields_applied": expiry_fields_applied,
    }
    return updated, summary


def partition_repair_proposals(
    proposals: pd.DataFrame,
    *,
    min_confidence: str = "MEDIUM",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if proposals is None or proposals.empty:
        empty = pd.DataFrame(proposals)
        return empty, empty

    accepted_mask = proposals.get("proposal_confidence", pd.Series(index=proposals.index, dtype=object)).map(
        lambda value: _proposal_confidence_at_least(value, min_confidence)
    )
    accepted = proposals.loc[accepted_mask].copy()
    review_queue = proposals.loc[~accepted_mask].copy()
    return accepted, review_queue


def _match_selected_contract(
    chain: pd.DataFrame,
    *,
    strike: float | None,
    option_type: str | None,
    selected_expiry: Any,
) -> pd.Series | None:
    if chain is None or chain.empty or strike is None:
        return None

    strike_col = _first_existing_column(chain, ("strikePrice", "STRIKE_PR"))
    type_col = _first_existing_column(chain, ("OPTION_TYP", "option_type", "optionType"))
    expiry_col = _first_existing_column(chain, ("EXPIRY_DT", "expiry", "expiry_date"))
    vol_col = _first_existing_column(chain, ("totalTradedVolume", "VOLUME"))
    oi_col = _first_existing_column(chain, ("openInterest", "OPEN_INT"))

    if strike_col is None:
        return None

    working_all = chain.copy()
    working_all[strike_col] = pd.to_numeric(working_all[strike_col], errors="coerce")
    working_all = working_all.dropna(subset=[strike_col])
    if working_all.empty:
        return None

    target_option_type = _normalize_option_type(option_type)
    target_expiry = _normalize_expiry_date(selected_expiry)

    def _apply_optional_filters(base: pd.DataFrame, *, strict_expiry: bool) -> pd.DataFrame:
        filtered = base
        if type_col is not None and target_option_type:
            type_series = filtered[type_col].map(_normalize_option_type)
            filtered = filtered.loc[type_series == target_option_type]
            if filtered.empty:
                return filtered

        if expiry_col is not None and target_expiry:
            normalized_expiry = filtered[expiry_col].map(_normalize_expiry_date)
            exact_mask = normalized_expiry == target_expiry
            if bool(exact_mask.any()):
                return filtered.loc[exact_mask]
            if strict_expiry:
                return filtered.iloc[0:0]

            target_month = target_expiry[:7]
            month_mask = normalized_expiry.astype(str).str.startswith(target_month)
            if bool(month_mask.any()):
                return filtered.loc[month_mask]
        return filtered

    working = _apply_optional_filters(working_all, strict_expiry=True)
    if working.empty:
        working = _apply_optional_filters(working_all, strict_expiry=False)
    if working.empty:
        return None

    exact_strike = working[working[strike_col].round(2) == round(float(strike), 2)]
    if not exact_strike.empty:
        working = exact_strike
    else:
        distance = (working[strike_col] - float(strike)).abs()
        nearest = float(distance.min()) if not distance.empty else None
        if nearest is None:
            return None
        working = working.loc[distance == nearest]

    sort_cols: list[str] = []
    ascending: list[bool] = []
    if vol_col is not None:
        working[vol_col] = pd.to_numeric(working[vol_col], errors="coerce").fillna(0.0)
        sort_cols.append(vol_col)
        ascending.append(False)
    if oi_col is not None:
        working[oi_col] = pd.to_numeric(working[oi_col], errors="coerce").fillna(0.0)
        sort_cols.append(oi_col)
        ascending.append(False)

    if sort_cols:
        working = working.sort_values(sort_cols, ascending=ascending, kind="stable")

    if working.empty:
        return None
    return working.iloc[0]


def _extract_contract_fields(row: pd.Series, source_path: Path, signal_row: pd.Series) -> dict[str, Any]:
    premium = _safe_float(row.get("lastPrice"))
    if premium is None:
        premium = _safe_float(row.get("LAST_PRICE"))

    volume = _safe_float(row.get("totalTradedVolume"))
    if volume is None:
        volume = _safe_float(row.get("VOLUME"))

    open_interest = _safe_float(row.get("openInterest"))
    if open_interest is None:
        open_interest = _safe_float(row.get("OPEN_INT"))

    iv = _safe_float(row.get("impliedVolatility"))
    if iv is None:
        iv = _safe_float(row.get("IV"))

    signal_timestamp = signal_row.get("signal_timestamp")
    expiry_value = signal_row.get("selected_expiry") or row.get("EXPIRY_DT")
    spot = _safe_float(signal_row.get("spot_at_signal"))
    strike = _safe_float(signal_row.get("strike"))
    option_type = str(signal_row.get("option_type") or "").upper().strip()

    greeks = None
    if iv not in (None, 0.0) and spot not in (None, 0.0) and strike not in (None, 0.0) and option_type in {"CE", "PE"}:
        tte = _parse_expiry_years(expiry_value, valuation_time=signal_timestamp)
        if tte is not None:
            greeks = compute_option_greeks(
                spot=spot,
                strike=strike,
                time_to_expiry_years=tte,
                volatility_pct=iv,
                option_type=option_type,
            )

    capital_per_lot = premium * float(LOT_SIZE) if premium not in (None, 0.0) else None

    fields = {
        "selected_option_last_price": premium,
        "selected_option_volume": int(volume) if volume is not None else None,
        "selected_option_open_interest": int(open_interest) if open_interest is not None else None,
        "selected_option_iv": iv,
        "selected_option_iv_is_proxy": False if iv is not None else None,
        "selected_option_iv_proxy_source": "CHAIN_SNAPSHOT" if iv is not None else None,
        "selected_option_delta": _safe_float((greeks or {}).get("DELTA")),
        "selected_option_delta_is_proxy": True if greeks else None,
        "selected_option_delta_proxy_source": "BACKFILLED_FROM_CHAIN_IV" if greeks else None,
        "selected_option_gamma": _safe_float((greeks or {}).get("GAMMA")),
        "selected_option_theta": _safe_float((greeks or {}).get("THETA")),
        "selected_option_vega": _safe_float((greeks or {}).get("VEGA")),
        "selected_option_vanna": _safe_float((greeks or {}).get("VANNA")),
        "selected_option_charm": _safe_float((greeks or {}).get("CHARM")),
        "selected_option_capital_per_lot": capital_per_lot,
        "selected_option_score": _safe_float(signal_row.get("selected_option_score")),
        "selected_option_source_snapshot": str(source_path),
    }
    return fields


def backfill_signal_contract_fields(
    frame: pd.DataFrame,
    *,
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    root = project_root or PROJECT_ROOT
    if frame is None or frame.empty:
        return pd.DataFrame(frame), {
            "rows_seen": 0,
            "rows_with_chain_path": 0,
            "rows_backfilled": 0,
            "rows_skipped_missing_snapshot": 0,
            "rows_skipped_no_contract": 0,
        }

    updated = frame.copy()
    chain_cache: dict[str, pd.DataFrame] = {}

    required_missing_mask = (
        updated.get("selected_option_last_price", pd.Series(index=updated.index)).isna()
        | updated.get("selected_option_iv", pd.Series(index=updated.index)).isna()
        | updated.get("selected_option_delta", pd.Series(index=updated.index)).isna()
    )

    rows_seen = int(len(updated))
    rows_with_chain_path = 0
    rows_backfilled = 0
    rows_skipped_missing_snapshot = 0
    rows_skipped_no_contract = 0

    for idx, row in updated[required_missing_mask].iterrows():
        snapshot_path = _resolve_chain_path(row.get("saved_chain_snapshot_path"), root)
        if snapshot_path is None:
            rows_skipped_missing_snapshot += 1
            continue
        rows_with_chain_path += 1

        key = str(snapshot_path)
        if key not in chain_cache:
            try:
                chain_cache[key] = load_option_chain_snapshot(key)
            except Exception:
                chain_cache[key] = pd.DataFrame()

        chain = chain_cache[key]
        contract_row = _match_selected_contract(
            chain,
            strike=_safe_float(row.get("strike")),
            option_type=row.get("option_type"),
            selected_expiry=row.get("selected_expiry"),
        )
        if contract_row is None:
            rows_skipped_no_contract += 1
            continue

        contract_fields = _extract_contract_fields(contract_row, snapshot_path, row)
        for field_name, field_value in contract_fields.items():
            if field_name not in updated.columns:
                continue
            if pd.isna(updated.at[idx, field_name]) and field_value is not None:
                updated.at[idx, field_name] = field_value

        entry_price = _safe_float(updated.at[idx, "entry_price"]) if "entry_price" in updated.columns else None
        target = _safe_float(updated.at[idx, "target"]) if "target" in updated.columns else None
        stop_loss = _safe_float(updated.at[idx, "stop_loss"]) if "stop_loss" in updated.columns else None
        if entry_price not in (None, 0.0):
            if "target_premium_return_pct" in updated.columns and pd.isna(updated.at[idx, "target_premium_return_pct"]) and target is not None:
                updated.at[idx, "target_premium_return_pct"] = round(((target - entry_price) / entry_price) * 100.0, 4)
            if "stop_loss_premium_return_pct" in updated.columns and pd.isna(updated.at[idx, "stop_loss_premium_return_pct"]) and stop_loss is not None:
                updated.at[idx, "stop_loss_premium_return_pct"] = round(((stop_loss - entry_price) / entry_price) * 100.0, 4)

        rows_backfilled += 1

    stats = {
        "rows_seen": rows_seen,
        "rows_with_chain_path": rows_with_chain_path,
        "rows_backfilled": rows_backfilled,
        "rows_skipped_missing_snapshot": rows_skipped_missing_snapshot,
        "rows_skipped_no_contract": rows_skipped_no_contract,
    }
    return updated, stats
