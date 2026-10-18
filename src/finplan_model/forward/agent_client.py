"""Consumer-side HTTP client of the deployed beta FinanceAgent runtime (AgentCore).

FinanceModel's forward runner asks the deployed agent to run one named model tool
(``invoke_tool``) or the LangGraph recommendation workflow (``invoke_graph``) for an
isolated synthetic scenario. This module is the consumer side of the agent's public
invocation contract, copied from FinanceAgent ``finplan_agent.modeling.hosted`` /
``machine`` / ``capture.check_public`` (FinanceAgent c781fad) so that no FinanceAgent
source is imported across repositories at run time; FinanceAgent deletes its copy later.

Authentication: OAuth client credentials of the dedicated Cognito app client
``modeling-runner-client``. The secret NAME is published by FinanceAgent at
``/finplan/beta/financeagent/secret-ref/modeling-runner-client`` (the Secrets Manager
secret holds JSON ``{"client_id", "client_secret"}``); the token endpoint comes from
``/finplan/beta/financeagent/agent/authorizer-metadata-ref`` and the runtime ARN from
``/finplan/beta/financeagent/agent/runtime-ref``.

The capture callback freezes the exact request before the HTTP invocation. Credentials
stay in memory and are never returned, logged, or included in evidence.
"""
from __future__ import annotations

import base64
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable

REGION = "us-east-2"
#: Per-call HTTP timeout (seconds). The Lambda runner (900 s) runs up to three agent
#: approaches per slot; it stops invoking more when fewer than 240 s remain.
DEFAULT_TIMEOUT = 200
REFS = {"meta": "/finplan/{}/financeagent/agent/authorizer-metadata-ref",
        "secret": "/finplan/{}/financeagent/secret-ref/modeling-runner-client",
        "runtime": "/finplan/{}/financeagent/agent/runtime-ref"}

_KEYS = re.compile(r"^(authorization|access_token|refresh_token|id_token|client_secret|api_key|secret_access_key|session_token|password)$", re.I)
_VALUE = re.compile(r"Bearer\s+[A-Za-z0-9._~-]{16,}|AKIA[A-Z0-9]{16}|ASIA[A-Z0-9]{16}|X-Amz-(?:Credential|Signature)=", re.I)


class CredentialEvidenceError(ValueError):
    """Credential-shaped content in a payload that would become evidence."""


def check_public(value: Any) -> None:
    """Fail closed when a request/response contains credential-shaped keys or values."""
    if isinstance(value, dict):
        for key, val in value.items():
            if _KEYS.fullmatch(str(key)):
                raise CredentialEvidenceError("Credential-shaped evidence is prohibited")
            check_public(val)
    elif isinstance(value, (list, tuple)):
        for val in value:
            check_public(val)
    elif isinstance(value, str) and _VALUE.search(value):
        raise CredentialEvidenceError("Credential-shaped evidence is prohibited")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def machine_token(ssm: Any, secrets: Any, env: str, opener: Any) -> tuple[str, list[str]]:
    """Client-credentials access token; returns (token, sensitive strings to scrub)."""
    if env != "beta":
        raise ValueError("automatic comparison machine access is beta only")
    meta = json.loads(ssm.get_parameter(Name=REFS["meta"].format(env))["Parameter"]["Value"])
    endpoint = urllib.parse.urlsplit(meta["token_endpoint"])
    if endpoint.scheme != "https" or not (endpoint.hostname or "").endswith(".amazoncognito.com") or endpoint.path != "/oauth2/token" or endpoint.query:
        raise ValueError("invalid token endpoint")
    secret_id = ssm.get_parameter(Name=REFS["secret"].format(env))["Parameter"]["Value"]
    secret = json.loads(secrets.get_secret_value(SecretId=secret_id)["SecretString"])
    basic = base64.b64encode((secret["client_id"] + ":" + secret["client_secret"]).encode()).decode()
    request = urllib.request.Request(meta["token_endpoint"], data=b"grant_type=client_credentials", headers={
        "Authorization": "Basic " + basic, "Content-Type": "application/x-www-form-urlencoded"})
    with opener.open(request, timeout=30) as response:
        token = json.load(response)
    if str(token.get("token_type", "")).lower() != "bearer" or not token.get("access_token"):
        raise ValueError("invalid machine token response")
    return token["access_token"], [secret["client_secret"], basic]


def _session_and_timeout(session: Any, timeout: int) -> Any:
    if session is None:
        import boto3
        session = boto3.Session(region_name=REGION)
    if session.region_name != REGION:
        raise ValueError("beta comparison is restricted to us-east-2")
    if not 1 <= timeout <= 420:
        raise ValueError("bounded invocation timeout required")
    return session


def _require_isolated_scenario(arguments: dict[str, Any]) -> None:
    # An explicit scenario prevents falling back to the user's saved portfolio.
    if not all(k in arguments for k in ("input_snapshot_id", "as_of", "holdings")) or "portfolio_id" in arguments:
        raise ValueError("explicit shared snapshot and isolated holdings required")


def invoke_tool(tool: str, arguments: dict[str, Any], prompt: str, capture_before: Callable[[dict[str, Any]], None], *,
                session: Any = None, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Ask the deployed agent to run ONE named model tool for an isolated scenario.

    The prompt states the request in words; ``tool_request`` names the exact MCP tool and
    arguments, so the agent's result can be checked against what was asked.
    """
    session = _session_and_timeout(session, timeout)
    _require_isolated_scenario(arguments)
    if not tool or not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("tool name and prompt required")
    payload = {"prompt": prompt, "tool_request": {"name": tool, "arguments": arguments}, "stream": False, "capture_inputs": True}
    check_public(payload)
    final = _post(session, payload, capture_before, timeout,
                  {"numerical_policy": f"FinanceModel model run by the agent through MCP tool {tool}", "news": "Not consumed by this model tool"})
    if final.get("status") == "failed" and "error_code" in final and "answer" not in final:
        return final
    answer = final.get("answer") or {}
    final["requested_tool"] = tool
    final["tools_called"] = [e.get("tool") for e in answer.get("evidence") or []]
    recommendation = answer.get("recommendation")
    if isinstance(recommendation, dict) and isinstance(recommendation.get("target_weights"), list) and recommendation.get("cash_weight") is not None:
        final["target_weights"] = recommendation["target_weights"]
        final["cash_weight"] = recommendation["cash_weight"]
        final["served_strategy"] = recommendation.get("strategy")
    final["forecast"] = {"available": False, "status": "not_available", "reason": "A target allocation is not a portfolio value forecast"}
    return final


def invoke_graph(arguments: dict[str, Any], capture_before: Callable[[dict[str, Any]], None], *, session: Any = None,
                 timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    session = _session_and_timeout(session, timeout)
    _require_isolated_scenario(arguments)
    payload = {"recommendation": arguments, "stream": False, "capture_inputs": True}
    check_public(payload)
    final = _post(session, payload, capture_before, timeout,
                  {"numerical_policy": "FinanceModel served policy via recommend_portfolio; identity verified against the returned artifact", "news": "News not consumed by the graph request unless the served policy explicitly reports news features"})
    if final.get("status") == "failed" and "error_code" in final and "answer" not in final:
        return final
    recommendation = (final.get("answer") or {}).get("recommendation")
    if isinstance(recommendation, dict) and isinstance(recommendation.get("target_weights"), list) and recommendation.get("cash_weight") is not None:
        final["target_weights"] = recommendation["target_weights"]
        final["cash_weight"] = recommendation["cash_weight"]
        final["allocation_origin"] = "FinanceModel served policy via LangGraph recommend_portfolio tool"
        final["served_strategy"] = recommendation.get("strategy")
        final["forecast"] = {"available": False, "status": "not_available", "reason": "A target allocation is not a portfolio value forecast"}
    else:
        final["policy_status"] = "unavailable_no_validated_numerical_policy_returned"
    if "captured_inputs" not in final:
        final["input_capture_status"] = "unavailable_deployed_release_requires_capture_support"
    return final


def _post(session: Any, payload: dict[str, Any], capture_before: Callable[[dict[str, Any]], None], timeout: int,
          notes: dict[str, Any]) -> dict[str, Any]:
    """Capture the exact payload, then send it to the deployed beta AgentCore Runtime."""
    ssm = session.client("ssm")
    opener = urllib.request.build_opener(_NoRedirect)
    token, sensitive = machine_token(ssm, session.client("secretsmanager"), "beta", opener)
    sensitive.append(token)
    runtime = ssm.get_parameter(Name=REFS["runtime"].format("beta"))["Parameter"]["Value"]
    sid = "modeling_comparison_" + uuid.uuid4().hex
    url = "https://bedrock-agentcore." + REGION + ".amazonaws.com/runtimes/" + urllib.parse.quote(runtime, safe="") + "/invocations?qualifier=DEFAULT"
    capture_before({"framework": "langgraph", "runtime_session_id": sid, "payload": payload, **notes})
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={
        "Authorization": "Bearer " + token, "Content-Type": "application/json",
        "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": sid})
    start = time.monotonic()
    try:
        with opener.open(req, timeout=timeout) as response:
            final = json.load(response)
    except urllib.error.HTTPError as exc:
        return {"status": "failed", "error_code": "HOSTED_HTTP_" + str(exc.code), "session_id": sid}
    except (OSError, ValueError):
        return {"status": "failed", "error_code": "HOSTED_TRANSPORT_ERROR", "session_id": sid}
    if not isinstance(final, dict) or any(v and v in json.dumps(final) for v in sensitive):
        return {"status": "failed", "error_code": "HOSTED_RESPONSE_REJECTED", "session_id": sid}
    check_public(final)
    final["latency_seconds"] = time.monotonic() - start
    final["framework"] = "langgraph"
    usage = final.get("usage") or {}
    final["usage"] = {**usage, "cost_usd": usage.get("estimated_cost_usd"),
                      "cost_basis": "estimated_from_reported_provider_tokens_and_configured_rates" if usage.get("estimated_cost_usd") is not None else "not_available"}
    return final
