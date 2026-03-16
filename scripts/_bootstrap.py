"""
Module: _bootstrap.py

Purpose:
    Implement the bootstrap script used for repeatable operational or research tasks.

Role in the System:
    Part of the operational scripting layer that supports repeatable maintenance and research tasks.

Key Outputs:
    CLI side effects, maintenance artifacts, and repeatable batch jobs.

Downstream Usage:
    Consumed by operators and by repeatable development or research workflows.
"""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root_on_path(script_path: str | Path) -> Path:
    """
    Purpose:
        Ensure project root on path exists and is ready for use.
    
    Context:
        Public function within the operational scripting layer. It exposes a reusable step in this module's workflow.
    
    Inputs:
        script_path (str | Path): Input associated with script path.
    
    Returns:
        Path: Result returned by the helper.
    
    Notes:
        The helper keeps the surrounding module readable without changing runtime behavior.
    """
    project_root = Path(script_path).resolve().parent.parent
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    return project_root
