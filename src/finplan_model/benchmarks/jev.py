"""TypeSafe System One client and point-in-time buy/hold/sell portfolio adapter.

The request/response shapes were checked against the vendor's OpenAPI. Vendor probability
calibration is not assumed: labeled probabilities are evaluated separately from portfolio PnL.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Mapping

import numpy as np

from finplan_model.core.artifacts import canonical_json_bytes, sha256_checksum
from finplan_model.core.errors import FinplanError
from finplan_model.datasets.outbound import check_outbound_payload
from finplan_model.sim.strategy import TargetWeights

API = "https://api.typesafe.ai/v1/systemone"
SECRET = "finplan/shared/financemodel/jev-api-key"
LABELS = ("buy", "hold", "sell")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise FinplanError.dependency_unavailable("Jev redirects are refused", retryable=False, reason="jev_redirect_rejected")


def probabilities(answer):
    if not isinstance(answer, Mapping) or answer.get("type") != "choice":
        raise FinplanError.precondition("Jev returned a non-choice answer", reason="jev_response_invalid")
    p = answer.get("probabilities")
    if not isinstance(p, Mapping) or set(p) != set(LABELS):
        raise FinplanError.precondition("Jev did not return exactly buy/hold/sell probabilities", reason="jev_response_invalid")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in p.values()) or abs(sum(p.values()) - 1.) > .01:
        raise FinplanError.precondition("Jev probabilities are invalid", reason="jev_response_invalid")
    if answer.get("choice") not in p or p[answer["choice"]] + 1e-9 < max(p.values()):
        raise FinplanError.precondition("Jev choice disagrees with its probability vector", reason="jev_response_invalid")
    confidence = answer.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise FinplanError.precondition("Jev confidence is invalid", reason="jev_response_invalid")
    total = sum(p.values())
    return {k: float(p[k]) / total for k in LABELS}


class JevClient:
    def __init__(self, key, *, model="jev-latest", max_calls=64, max_input_tokens=200_000, transport=None, sleep=time.sleep, clock=time.monotonic):
        if not isinstance(key, str) or not key.strip():
            raise FinplanError.precondition("Jev secret is empty", reason="jev_secret_missing")
        self._key, self.model = key, model
        self.max_calls, self.max_input_tokens = max_calls, max_input_tokens
        self.transport, self.sleep, self.clock = transport or self._send, sleep, clock
        self.cache, self.calls, self.input_tokens, self.output_tokens = {}, 0, 0, 0
        self.returned_models, self.last_call = set(), None

    def _send(self, payload):
        request = urllib.request.Request(API, data=canonical_json_bytes(payload), headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"}, method="POST")
        with urllib.request.build_opener(NoRedirect).open(request, timeout=20) as response:
            data = response.read(512_001)
            if len(data) > 512_000:
                raise FinplanError.precondition("Jev response exceeded its bound", reason="jev_response_invalid")
            return json.loads(data)

    def ask(self, payload, *, ctx, raw_values=()):
        # A single central guard executes before cache lookup and before every outbound retry.
        check_outbound_payload(payload, ctx=ctx, raw_values=raw_values, destination="typesafe_systemone")
        key = sha256_checksum(canonical_json_bytes(payload))
        if key in self.cache:
            return self.cache[key]
        # Conservative per-request input bound is UTF-8 byte count; reserve before sending.
        reserve = len(canonical_json_bytes(payload))
        if self.calls >= self.max_calls or self.input_tokens + reserve > self.max_input_tokens:
            raise FinplanError.precondition("Jev call/token cap reached", reason="jev_usage_cap")
        for attempt in range(3):
            if self.calls >= self.max_calls or self.input_tokens + reserve > self.max_input_tokens:
                raise FinplanError.precondition("Jev retry would exceed the usage cap", reason="jev_usage_cap")
            if self.last_call is not None:
                self.sleep(max(0., .1 - (self.clock() - self.last_call)))
            self.calls += 1
            self.input_tokens += reserve  # charge the conservative reservation, including failed attempts
            self.last_call = self.clock()
            try:
                check_outbound_payload(payload, ctx=ctx, raw_values=raw_values, destination="typesafe_systemone")
                data = self.transport(payload)
                if not isinstance(data, dict) or not isinstance(data.get("model"), str) or not isinstance(data.get("answers"), dict) or set(data["answers"]) != set(payload["questions"]):
                    raise FinplanError.precondition("Jev response shape is invalid", reason="jev_response_invalid")
                for answer in data["answers"].values():
                    probabilities(answer)
                usage = data.get("usage") or {}
                actual = usage.get("input_tokens")
                output = usage.get("output_tokens")
                if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (actual, output)):
                    raise FinplanError.precondition("Jev usage is invalid", reason="jev_response_invalid")
                # Never subtract a conservative reservation; actual usage may exceed it.
                self.input_tokens += max(0, actual - reserve)
                self.output_tokens += output
                self.returned_models.add(data["model"])
                self.cache[key] = data
                return data
            except urllib.error.HTTPError as exc:
                # Never retain provider error bodies: these may reflect confidential payloads.
                retryable = exc.code in (429, 529, 500, 502, 503, 504)
                if not retryable or attempt == 2:
                    raise FinplanError.dependency_unavailable("Jev request failed", retryable=retryable, reason="jev_http_" + str(exc.code)) from None
                try:
                    delay = min(2., max(0., float(exc.headers.get("Retry-After", 2 ** attempt))))
                except (TypeError, ValueError):
                    delay = min(2., 2 ** attempt)
                self.sleep(delay)
            except (OSError, TimeoutError, ValueError):
                if attempt == 2:
                    raise FinplanError.dependency_unavailable("Jev request failed after bounded retries", retryable=True, reason="jev_network_unavailable") from None
                self.sleep(min(2., 2 ** attempt))

    @property
    def usage(self):
        return {"calls": self.calls, "input_token_upper_bound": self.input_tokens, "output_tokens": self.output_tokens,
            "requested_model": self.model, "returned_models": sorted(self.returned_models), "model_drift": len(self.returned_models) > 1,
            "cache_entries": len(self.cache), "external_vendor_estimated_usd": self.input_tokens * .042 / 1_000_000,
            "vendor_price_basis": "configured research planning rate; verify prepaid vendor pricing before approval", "billed_by": "typesafe_not_aws"}


def _bucket(value, edges, labels):
    return labels[int(np.searchsorted(edges, float(value), side="right"))]


def encode_state(view, holdings):
    """Only descriptors leave the account. No date, prices, quantities or raw returns."""
    descriptors = {}
    _, returns = view.returns(60)
    for instrument in view.instruments:
        j = view.instruments.index(instrument)
        if len(returns) < 20:
            raise FinplanError.precondition("Jev benchmark requires completed point-in-time history", reason="insufficient_history")
        values = returns[:, j]
        momentum = np.prod(1. + values[-20:]) - 1.
        vol = np.std(values[-20:]) * np.sqrt(252.)
        descriptors[instrument] = {
            "recent_trend": _bucket(momentum, [-.05, -.01, .01, .05], ["strongly_negative", "negative", "flat", "positive", "strongly_positive"]),
            "volatility": _bucket(vol, [.15, .30, .60], ["low", "moderate", "high", "very_high"]),
            "current_exposure": _bucket(holdings.weights.get(instrument, 0.), [.01, .10, .30, .60], ["none", "small", "moderate", "large", "concentrated"])}
    return {"scope": "retrospective research with unknown model pretraining cutoff", "horizon": "one trading month", "instruments": descriptors}


class JevStrategy:
    name = "jev"

    def __init__(self, client, *, ctx, raw_values=(), threshold=.20, max_delta=.10):
        self.client, self.ctx, self.raw_values = client, ctx, tuple(raw_values)
        self.threshold, self.max_delta = threshold, max_delta
        self.records = []

    def decide(self, view, holdings):
        if len(view.price_matrix("close")[0]) < 21:
            return TargetWeights(dict(holdings.weights), holdings.cash_weight, "no_effect", {"reason": "insufficient_history"})
        state = encode_state(view, holdings)
        questions = {i: {"type": "choice", "instructions": "Classify the next trading month's net return opportunity using only supplied descriptors.", "criteria": {
            "buy": "Expected return is sufficiently positive to justify increasing exposure after costs.",
            "hold": "Evidence does not justify changing exposure after costs.",
            "sell": "Expected return is sufficiently negative to justify reducing exposure after costs."}} for i in view.instruments}
        result = self.client.ask({"model": self.client.model, "state": state, "questions": questions}, ctx=self.ctx, raw_values=self.raw_values)
        current = {i: float(holdings.weights.get(i, 0.)) for i in view.instruments}
        target, rows, purchases = {}, [], {}
        for i in view.instruments:
            answer = result["answers"][i]
            p = probabilities(answer)
            signal = p["buy"] - p["sell"]
            delta = self.max_delta * signal if abs(signal) >= self.threshold and answer["confidence"] >= .5 else 0.
            # Reserve deliberate sales first. Purchases may spend only cash plus
            # those sales; normalization must never sell an instrument labeled hold.
            target[i] = max(0., current[i] + min(0., delta))
            purchases[i] = max(0., min(1. - current[i], delta))
            rows.append({"instrument_id": i, "probabilities": p, "choice": answer["choice"], "confidence": answer["confidence"], "sizing_signal": signal})
        available = max(0., 1. - sum(target.values()))
        requested = sum(purchases.values())
        scale = min(1., available / requested) if requested else 0.
        target = {i: target[i] + scale * purchases[i] for i in target}
        self.records.append({"decision_date": view.decision_session.isoformat(), "state": state, "answers": rows, "returned_model": result["model"], "request_hash": sha256_checksum(canonical_json_bytes({"model": self.client.model, "state": state, "questions": questions}))})
        return TargetWeights(target, max(0., 1. - sum(target.values())), "feasible", {"benchmark": "typesafe_jev", "classification_not_profitability": True, "raw_series_egress": False})


def calibration_report(records, market, *, horizon_sessions=21, cost_buffer=.002, observed_end=None):
    """Labels computed after predictions; never exposed as decision-time features."""
    samples = []
    for record in records:
        from datetime import date
        d = date.fromisoformat(record["decision_date"])
        k = market.session_index(d)
        if k + horizon_sessions >= len(market.sessions):
            continue
        end = market.sessions[k + horizon_sessions]
        if observed_end is not None and end > observed_end:
            continue
        for answer in record["answers"]:
            i = answer["instrument_id"]
            a, b = market.bar(i, d), market.bar(i, end)
            if a is None or b is None:
                continue
            ret = b.close / a.close - 1.
            label = "buy" if ret > cost_buffer else "sell" if ret < -cost_buffer else "hold"
            samples.append((answer["probabilities"], label))
    if not samples:
        return {"status": "not_available", "reason": "insufficient_forward_label_horizon", "horizon_sessions": horizon_sessions}
    brier = sum(sum((p[k] - float(k == label)) ** 2 for k in LABELS) for p, label in samples) / len(samples)
    nll = -sum(math.log(max(1e-12, p[label])) for p, label in samples) / len(samples)
    accuracy = sum(max(p, key=p.get) == label for p, label in samples) / len(samples)
    return {"status": "available", "samples": len(samples), "multiclass_brier": brier, "negative_log_likelihood": nll, "classification_accuracy": accuracy, "horizon_sessions": horizon_sessions,
        "label_cost_buffer": cost_buffer, "threshold_selection": "fixed released policy, no test tuning", "vendor_calibration_claim_verified": False, "classification_accuracy_is_not_portfolio_profitability": True,
        "dependence_warning": "Overlapping labels and shared market path reduce effective sample size"}
