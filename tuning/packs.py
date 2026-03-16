"""
Module: packs.py

Purpose:
    Implement packs utilities for parameter search, validation, governance, or promotion workflows.

Role in the System:
    Part of the tuning layer that searches, validates, and governs candidate parameter packs.

Key Outputs:
    Experiment records, parameter candidates, validation summaries, and promotion decisions.

Downstream Usage:
    Consumed by shadow mode, promotion workflow, and parameter-pack governance.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from tuning.models import ParameterPack


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARAMETER_PACKS_DIR = PROJECT_ROOT / "config" / "parameter_packs"
RESEARCH_PARAMETER_PACKS_DIR = PROJECT_ROOT / "research" / "parameter_tuning" / "candidate_packs"
DEFAULT_PARAMETER_PACK_DIRS = (PARAMETER_PACKS_DIR, RESEARCH_PARAMETER_PACKS_DIR)


def _iter_pack_dirs(packs_dir: str | Path | Iterable[str | Path] | None = None) -> list[Path]:
    """
    Purpose:
        Process iter pack dirs for downstream use.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        packs_dir (str | Path | Iterable[str | Path] | None): Input associated with packs dir.
    
    Returns:
        list[Path]: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    if packs_dir is None:
        return [Path(path) for path in DEFAULT_PARAMETER_PACK_DIRS]
    if isinstance(packs_dir, (str, Path)):
        return [Path(packs_dir)]
    return [Path(path) for path in packs_dir]


def _resolve_pack_path(name: str, packs_dir: str | Path | Iterable[str | Path] | None = None) -> Path:
    """
    Purpose:
        Resolve pack path needed by downstream logic.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        name (str): Input associated with name.
        packs_dir (str | Path | Iterable[str | Path] | None): Input associated with packs dir.
    
    Returns:
        Path: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    for directory in _iter_pack_dirs(packs_dir):
        path = directory / f"{name}.json"
        if path.exists():
            return path
    search_roots = ", ".join(str(path) for path in _iter_pack_dirs(packs_dir))
    raise FileNotFoundError(f"Unknown parameter pack: {name} (searched: {search_roots})")


def _coerce_pack(payload: dict) -> ParameterPack:
    """
    Purpose:
        Coerce pack into a consistent representation.
    
    Context:
        Internal helper within the tuning layer. It isolates a reusable transformation so the surrounding code remains easy to follow.
    
    Inputs:
        payload (dict): Input associated with payload.
    
    Returns:
        ParameterPack: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    return ParameterPack(
        name=str(payload.get("name") or "").strip(),
        version=str(payload.get("version") or "1.0.0").strip(),
        description=str(payload.get("description") or "").strip(),
        parent=(str(payload.get("parent")).strip() or None) if payload.get("parent") is not None else None,
        notes=payload.get("notes"),
        tags=tuple(payload.get("tags") or ()),
        metadata=dict(payload.get("metadata") or {}),
        overrides=dict(payload.get("overrides") or {}),
    )


def list_parameter_packs(packs_dir: str | Path | Iterable[str | Path] | None = None) -> list[str]:
    """
    Purpose:
        List parameter packs available to the current workflow.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        packs_dir (str | Path | Iterable[str | Path] | None): Input associated with packs dir.
    
    Returns:
        list[str]: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    pack_names = set()
    for path in _iter_pack_dirs(packs_dir):
        if not path.exists():
            continue
        pack_names.update(file.stem for file in path.glob("*.json"))
    return sorted(pack_names)


def load_parameter_pack(name: str, packs_dir: str | Path | Iterable[str | Path] | None = None) -> ParameterPack:
    """
    Purpose:
        Process load parameter pack for downstream use.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        name (str): Input associated with name.
        packs_dir (str | Path | Iterable[str | Path] | None): Input associated with packs dir.
    
    Returns:
        ParameterPack: Result returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    path = _resolve_pack_path(name, packs_dir=packs_dir)
    payload = json.loads(path.read_text())
    pack = _coerce_pack(payload)
    if not pack.name:
        raise ValueError(f"Parameter pack {name} is missing a stable name")
    return pack


def resolve_parameter_pack(name: str, packs_dir: str | Path | Iterable[str | Path] | None = None) -> ParameterPack:
    """
    Purpose:
        Resolve parameter pack needed by downstream logic.
    
    Context:
        Public function within the tuning layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        name (str): Input associated with name.
        packs_dir (str | Path | Iterable[str | Path] | None): Input associated with packs dir.
    
    Returns:
        ParameterPack: Computed value returned by the helper.
    
    Notes:
        The output is designed to remain serializable so experiments, reports, and governance decisions can be reproduced later.
    """
    pack = load_parameter_pack(name, packs_dir=packs_dir)
    if not pack.parent:
        return pack

    parent = resolve_parameter_pack(pack.parent, packs_dir=packs_dir)
    overrides = dict(parent.overrides)
    overrides.update(pack.overrides)

    metadata = dict(parent.metadata)
    metadata.update(pack.metadata)

    tags = tuple(dict.fromkeys([*parent.tags, *pack.tags]).keys())
    return ParameterPack(
        name=pack.name,
        version=pack.version,
        description=pack.description or parent.description,
        parent=pack.parent,
        notes=pack.notes or parent.notes,
        tags=tags,
        metadata=metadata,
        overrides=overrides,
    )
