#!/usr/bin/env python3
"""Validate unused-candidate archival evidence and cooldown deprecations.

CI intent:
- If files are moved into archive/unused_candidates/<date>/..., require matching
  scan and manifest artifacts in research/reviews/.
- Regenerate current reference counts for candidate basenames and write a CI
  artifact for review.
- Enforce post-cooldown cleanup for deprecated compatibility wrappers.
"""

from __future__ import annotations

import csv
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = REPO_ROOT / "archive" / "unused_candidates"
REVIEWS_ROOT = REPO_ROOT / "research" / "reviews"


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {proc.stderr.strip()}")
    return proc.stdout


def _changed_files(base_ref: str) -> list[str]:
    try:
        out = _run(["git", "diff", "--name-only", f"origin/{base_ref}...HEAD"])
        files = [line.strip() for line in out.splitlines() if line.strip()]
        if files:
            return files
    except Exception:
        pass
    out = _run(["git", "diff", "--name-only", "HEAD~1...HEAD"])
    return [line.strip() for line in out.splitlines() if line.strip()]


def _count_refs(pattern: str, include_glob: str | None = None) -> int:
    grep_cmd = ["grep", "-RInE", pattern]
    if include_glob:
        grep_cmd.append(f"--include={include_glob}")
    grep_cmd.append(".")
    proc = subprocess.run(grep_cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    if proc.returncode not in (0, 1):
        raise RuntimeError(proc.stderr.strip())
    return len([line for line in proc.stdout.splitlines() if line.strip()])


def _validate_archival_dates(dates: set[str]) -> None:
    failures: list[str] = []
    for date in sorted(dates):
        manifest = REVIEWS_ROOT / f"unused_candidates_manifest_{date}.md"
        scan = REVIEWS_ROOT / f"unused_candidate_scan_{date}.csv"
        archive_date_root = ARCHIVE_ROOT / date

        if not manifest.exists():
            failures.append(f"Missing manifest for archival date {date}: {manifest}")
        if not scan.exists():
            failures.append(f"Missing scan CSV for archival date {date}: {scan}")
        if not archive_date_root.exists():
            failures.append(f"Missing archive date directory: {archive_date_root}")
            continue

        archived_py = sorted([p for p in archive_date_root.rglob("*.py")])
        if not archived_py:
            failures.append(f"No archived python files found under {archive_date_root}")
            continue

        if manifest.exists():
            text = manifest.read_text(encoding="utf-8")
            for path in archived_py:
                if path.name not in text:
                    failures.append(
                        f"Archived file {path.name} not referenced in manifest {manifest.name}"
                    )

        if scan.exists():
            with scan.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            required_cols = {"file", "import_refs", "grep_refs", "execution_evidence"}
            cols = set(rows[0].keys()) if rows else set()
            if not required_cols.issubset(cols):
                failures.append(
                    f"Scan CSV {scan.name} missing required columns: {sorted(required_cols - cols)}"
                )
            scanned_names = {Path((row.get("file") or "").strip()).name for row in rows}
            for path in archived_py:
                if path.name not in scanned_names:
                    failures.append(
                        f"Archived file {path.name} missing from scan CSV {scan.name}"
                    )

            regen_path = REVIEWS_ROOT / f"unused_candidate_scan_{date}.ci_regen.csv"
            with regen_path.open("w", encoding="utf-8", newline="") as out_csv:
                writer = csv.DictWriter(
                    out_csv,
                    fieldnames=["file", "import_refs_current", "grep_refs_current"],
                )
                writer.writeheader()
                for row in rows:
                    original = (row.get("file") or "").strip()
                    stem = Path(original).stem
                    base = Path(original).name
                    import_refs = _count_refs(
                        rf"import .*{re.escape(stem)}|from .*{re.escape(stem)} import",
                        include_glob="*.py",
                    )
                    grep_refs = _count_refs(rf"{re.escape(base)}|{re.escape(stem)}")
                    writer.writerow(
                        {
                            "file": original,
                            "import_refs_current": import_refs,
                            "grep_refs_current": grep_refs,
                        }
                    )

    if failures:
        raise RuntimeError("\n".join(failures))


def _enforce_wrapper_cooldown() -> None:
    wrapper = REPO_ROOT / "scripts" / "watchlist_realized_evaluation_20260407.py"
    cooldown_end = dt.date(2026, 5, 7)
    today = dt.date.today()
    if today < cooldown_end or not wrapper.exists():
        return

    proc = subprocess.run(
        ["grep", "-RIn", "watchlist_realized_evaluation_20260407.py", "."],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    refs = [line for line in proc.stdout.splitlines() if line.strip() and "watchlist_realized_evaluation_20260407.py" in line]
    # Self reference count at wrapper file itself is acceptable; external refs imply active dependency.
    external_refs = [line for line in refs if "scripts/watchlist_realized_evaluation_20260407.py" not in line]

    if not external_refs:
        raise RuntimeError(
            "Compatibility wrapper scripts/watchlist_realized_evaluation_20260407.py exceeded cooldown and has no external dependency. Remove it."
        )


def main() -> int:
    base_ref = sys.argv[1] if len(sys.argv) > 1 else "main"
    changed = _changed_files(base_ref)

    archive_re = re.compile(r"^archive/unused_candidates/(\d{4}-\d{2}-\d{2})/")
    archive_dates = {
        m.group(1)
        for path in changed
        for m in [archive_re.match(path)]
        if m
    }

    if archive_dates:
        _validate_archival_dates(archive_dates)

    _enforce_wrapper_cooldown()

    print("Unused-candidate evidence validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
