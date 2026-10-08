#!/usr/bin/env python3
"""Approve (or inspect) a FinanceModel run as the human approver (design D4; FM-OQ-6 interim: a CLI
with the approver role).

Run it with credentials of the approver role ``finplan-<env>-financemodel-approver-role`` (for
example ``AWS_PROFILE=<approver-profile>``). It resolves the job endpoint from
``/finplan/<env>/financemodel/api/job-endpoint``, shows the run's cost estimate and asks for
confirmation before it sends ``POST /v1/jobs/{run_id}/approve`` signed with SigV4. The approval
records the approver role name and the time; the approved amount is the displayed estimate.

    uv run python scripts/approve_run.py --env beta run_01...            # show, confirm, approve
    uv run python scripts/approve_run.py --env beta run_01... --show     # show only

Tool-wrapper, agent and pipeline roles are refused by the API (``FORBIDDEN``) whatever this script
does. Nothing here is account-specific; no identifier is written to disk.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

__all__ = ["main", "signed_request"]


def signed_request(method: str, url: str, *, region: str, credentials: Any, body: dict[str, Any] | None = None, opener: Callable[..., Any] | None = None) -> tuple[int, dict[str, Any]]:
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    data = json.dumps(body).encode() if body is not None else None
    req = AWSRequest(method=method, url=url, data=data, headers={"Content-Type": "application/json"} if data else {})
    SigV4Auth(credentials, "execute-api", region).add_auth(req)
    prepared = urllib.request.Request(url, data=data, method=method, headers=dict(req.headers.items()))
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(prepared, timeout=20) as resp:
            return int(resp.status), json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as err:
        return int(err.code), json.loads(err.read() or b"{}")


def main(argv: list[str] | None = None, *, session: Any = None, opener: Callable[..., Any] | None = None, confirm: Callable[[str], str] = input) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_id")
    ap.add_argument("--env", required=True, choices=["beta", "gamma", "prod"])
    ap.add_argument("--region", default="us-east-2")
    ap.add_argument("--show", action="store_true", help="show the run and exit")
    ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = ap.parse_args(argv)
    if session is None:  # pragma: no cover - real credentials
        import boto3

        session = boto3.session.Session(region_name=args.region)
    endpoint = session.client("ssm").get_parameter(Name=f"/finplan/{args.env}/financemodel/api/job-endpoint")["Parameter"]["Value"].rstrip("/")
    creds = session.get_credentials()
    status, run = signed_request("GET", f"{endpoint}/v1/jobs/{args.run_id}", region=args.region, credentials=creds, opener=opener)
    if status != 200:
        print(json.dumps(run, indent=2), file=sys.stderr)
        return 1
    est = run.get("cost_estimate", {})
    print(f"run {run['run_id']}  state={run['state']}  job_type={run.get('job_type')}  purpose={run['purpose']}")
    print(f"estimate USD {est.get('estimated_usd_upper_bound')} ({est.get('budget_category')}; remaining USD {est.get('remaining_allocation_usd')}; price from {est.get('price_retrieved_at')})")
    if args.show:
        return 0
    if run["state"] != "awaiting_approval":
        print(f"not awaiting approval (state {run['state']}); nothing to do", file=sys.stderr)
        return 1
    if not args.yes and confirm("approve this run? [y/N] ").strip().lower() != "y":
        print("not approved")
        return 1
    body = {"approved_estimate_usd": est.get("estimated_usd_upper_bound"), "idempotency_key": f"approve-{args.run_id}"}
    status, out = signed_request("POST", f"{endpoint}/v1/jobs/{args.run_id}/approve", region=args.region, credentials=creds, body=body, opener=opener)
    print(json.dumps(out, indent=2))
    return 0 if status == 200 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
