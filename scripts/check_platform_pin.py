#!/usr/bin/env python3
"""Verify the finplan-platform pin of FinanceModel (mirrors ``scripts/check_contracts_pin.py``).

The forward runner Lambda writes the beta experiment ledger through FinancialPlanning's
``finplan_platform.core.modeling_evidence``. FinanceModel never imports that source across
repositories at run time: it consumes the platform's reproducible wheel, vendored byte for byte
under ``vendor/finplan-platform/``. Fails unless ALL of these agree:

1. ``platform-pin.json``: package ``finplan-platform``, exact version, SHA-256 of the wheel and
   the 40-hex FinancialPlanning ``source_commit`` the wheel was built from;
2. the vendored wheel hashes to that SHA-256 and carries the pinned version;
3. ``pyproject.toml`` depends on ``finplan-platform==<version>`` exactly and ``[tool.uv.sources]``
   points at the pinned artifact;
4. ``uv.lock`` locks that version with the same wheel hash;
5. the installed distribution has that version (when installed).

The checks 2-5 are the contract pin checker's own (:func:`scripts.check_contracts_pin.check`, run
on ``platform-pin.json``). The contracts "0.x is beta-only" rule is a rule about *contract*
releases and is not applied: ``finplan-platform`` keeps version 0.1.0 and is identified by
``source_commit`` + ``sha256``; its only consumer (the forward runner) is deployed in beta only.

Re-pin: build the wheel from a FinancialPlanning commit (``git archive <commit> | tar -x``;
``SOURCE_DATE_EPOCH=315532800 uv build --wheel``), copy it into ``vendor/finplan-platform/``,
update ``platform-pin.json`` (sha256, source_commit) and run ``uv lock``.

Exit codes: 0 ok, 1 mismatch, 2 usage/config error. Runs offline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

__all__ = ["PACKAGE", "PIN_FILE", "check", "main"]

PACKAGE = "finplan-platform"
PIN_FILE = "platform-pin.json"


def check(root: Path = ROOT, *, pin_path: Path | None = None, check_installed: bool = True) -> list[str]:
    from scripts.check_contracts_pin import check as check_pin

    path = pin_path or root / PIN_FILE
    if not path.is_file():
        return [f"{PIN_FILE} is missing"]
    pin = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if pin.get("package") != PACKAGE:
        problems.append(f"{PIN_FILE}: package must be {PACKAGE!r}, got {pin.get('package')!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(pin.get("source_commit", ""))):
        problems.append(f"{PIN_FILE}: source_commit must be the 40-hex FinancialPlanning commit the wheel was built from")
    if pin.get("source_date_epoch") != 315532800:
        problems.append(f"{PIN_FILE}: source_date_epoch must be 315532800 (the reproducible platform build)")
    # env=None: the contracts 0.x rule does not apply to the platform package (see module doc).
    problems += check_pin(root, env=None, pin_path=path, check_installed=check_installed)
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--pin", type=Path, help=f"pin file (default: <root>/{PIN_FILE})")
    args = ap.parse_args(argv)
    try:
        problems = check(args.root, pin_path=args.pin)
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if problems:
        for p in problems:
            print(f"FAIL: {p}")
        return 1
    print("PASS: finplan-platform pin verified (version, wheel digest, source commit, pyproject, uv.lock)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
