"""Synthesized assemblies shared by the infrastructure tests (offline, synthesized once per session).

* ``assembly`` - the app as the pipeline builds it (``scripts/synth.py``: deployment synthesizer).
  Since contracts 1.0.0 (D16) every resource is synthesized: the job API, the job-execution role,
  the registry bucket policy and the CodeBuild log groups no longer wait for matrix rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest


@dataclass
class Assembly:
    directory: Path
    templates: dict[str, dict[str, Any]] = field(default_factory=dict)  # stack name -> template

    def stack(self, name: str) -> dict[str, Any]:
        return self.templates[name]

    def resources(self, name: str, rtype: str | None = None) -> dict[str, dict[str, Any]]:
        res = self.templates[name]["Resources"]
        return {k: v for k, v in res.items() if rtype is None or v.get("Type") == rtype}


def _load(directory: Path) -> Assembly:
    asm = Assembly(directory)
    for manifest in sorted(directory.rglob("manifest.json")):
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        for art in (doc.get("artifacts") or {}).values():
            if art.get("type") == "aws:cloudformation:stack":
                props = art.get("properties") or {}
                asm.templates[props["stackName"]] = json.loads((manifest.parent / props["templateFile"]).read_text(encoding="utf-8"))
    return asm


def _synth(tmp: Path, context: dict[str, Any]) -> Assembly:
    import os

    from scripts.synth import synth

    saved = os.environ.get("FINPLAN_RELEASE_BUILD")
    os.environ["FINPLAN_RELEASE_BUILD"] = "0"  # offline synth of the source tree, also inside a release build
    try:
        return _load(synth(tmp, context=context))
    finally:
        if saved is None:
            os.environ.pop("FINPLAN_RELEASE_BUILD", None)
        else:
            os.environ["FINPLAN_RELEASE_BUILD"] = saved


@pytest.fixture(scope="session")
def assembly(tmp_path_factory: pytest.TempPathFactory) -> Assembly:
    return _synth(tmp_path_factory.mktemp("asm-auto"), {})
