"""Task 10.6 and 10.5 documentation checks: no account identifiers (leak scan), every topic and every
operator parameter documented."""

from __future__ import annotations

from pathlib import Path

from finplan_contracts import leak_scan

DOCS = Path(__file__).resolve().parents[3] / "docs"


def test_operations_doc_covers_its_topics_and_leaks_nothing():
    text = (DOCS / "operations.md").read_text(encoding="utf-8")
    for topic in ("instance-prices", "cpu_research", "gpu", "approve_run.py", "cancel", "rollback_to_release_id", "in-flight", "auto-approve-usd"):
        assert topic in text, topic
    for name in ("operations.md", "pipeline.md", "bootstrap.md", "staging-and-registry.md"):
        assert leak_scan.scan_text((DOCS / name).read_text(encoding="utf-8"), name) == [], name


def test_operator_parameters_are_documented():
    """Every operator parameter the deployed control plane or suites read is documented (task 10.6)."""
    text = (DOCS / "operations.md").read_text(encoding="utf-8")
    for name in ("instance-prices", "integration-snapshot-id", "auto-approve-usd", "lease-limits", "production-candidate-principals"):
        assert f"`{name}`" in text, name
    assert "Price List API" in text and "scripts/instance_prices.py" in text
    boot = (DOCS / "bootstrap.md").read_text(encoding="utf-8")
    assert "/finplan/shared/financemodel/config/budget-enforced-role-names" in boot and "instance-prices" in boot
