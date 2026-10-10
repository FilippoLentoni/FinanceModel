"""Write-once classical analyses with verified references and a bounded newest-first index."""

from __future__ import annotations

import copy
import hashlib
import json
import re

from finplan_model.core.artifacts import (
    canonical_json_bytes,
    sha256_checksum,
    verify_checksum,
)
from finplan_model.core.errors import ErrorCode, FinplanError

ID = re.compile(r"^ca_[0-9a-f]{32}$")


def analysis_id(identity):
    return "ca_" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()[:32]


def require_analysis(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise FinplanError(
            ErrorCode.INVALID_IDENTIFIER,
            "invalid classical analysis id",
            field="analysis_id",
        )
    return value


def reference(doc):
    return {
        "artifact_id": doc["analysis_id"],
        "owner": "financemodel",
        "kind": "classical_analysis",
        "checksum": sha256_checksum(canonical_json_bytes(doc)),
        "content_type": "application/json",
    }


def public(doc):
    out = {
        k: copy.deepcopy(v)
        for k, v in doc.items()
        if k not in ("solve_inputs", "request_fingerprint", "internal")
    }
    out.update(
        {"analysis_ref": reference(doc), "contract_version": "1.5.0", "synthetic": True}
    )
    return out


class MemoryStore:
    def __init__(self):
        self.docs, self.claims = {}, {}

    def get(self, aid):
        require_analysis(aid)
        if aid not in self.docs:
            raise FinplanError(ErrorCode.NOT_FOUND, "classical analysis not found")
        return copy.deepcopy(self.docs[aid])

    def put(self, doc):
        aid = require_analysis(doc["analysis_id"])
        if aid in self.docs:
            old = self.docs[aid]
            if old["request_fingerprint"] != doc["request_fingerprint"]:
                raise FinplanError(
                    ErrorCode.IMMUTABLE_RECORD, "analysis identity conflicts"
                )
            return copy.deepcopy(old)
        self.docs[aid] = copy.deepcopy(doc)
        return copy.deepcopy(doc)

    def list(self, *, portfolio_id=None, kind=None, limit=20):
        rows = sorted(
            self.docs.values(),
            key=lambda d: (d["created_at"], d["analysis_id"]),
            reverse=True,
        )
        return [
            copy.deepcopy(d)
            for d in rows
            if (not portfolio_id or d.get("portfolio_id") == portfolio_id)
            and (not kind or d["analysis_kind"] == kind)
        ][:limit]

    def get_claim(self, key):
        return copy.deepcopy(self.claims.get(key))

    def claim(self, key, value):
        if key in self.claims and self.claims[key] != value:
            return False
        self.claims[key] = copy.deepcopy(value)
        return True


class S3Store:
    def __init__(self, client, bucket):
        self.client, self.bucket = client, bucket

    def _get(self, key):
        try:
            obj = self.client.get_object(Bucket=self.bucket, Key="classical/" + key)
            data = obj["Body"].read()
            verify_checksum(
                data,
                obj.get("Metadata", {}).get("checksum", ""),
                what="classical record",
            )
            return json.loads(data)
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code")
            if code in ("AccessDenied", "AccessDeniedException", "403"):
                # S3 also returns 403 for a missing key when ListBucket is
                # prefix-constrained. Probe only this exact key under the
                # bounded listing grant; an existing key or denied probe must
                # preserve the real authorization failure.
                object_key = "classical/" + key
                page = self.client.list_objects_v2(
                    Bucket=self.bucket, Prefix=object_key, MaxKeys=1
                )
                if not any(
                    item["Key"] == object_key for item in page.get("Contents", [])
                ):
                    raise FinplanError(
                        ErrorCode.NOT_FOUND, "classical record not found"
                    ) from None
            if code in (
                "NoSuchKey",
                "404",
                "NotFound",
            ):
                raise FinplanError(
                    ErrorCode.NOT_FOUND, "classical record not found"
                ) from None
            raise

    def _put(self, key, value):
        data = canonical_json_bytes(value)
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key="classical/" + key,
                Body=data,
                ContentType="application/json",
                Metadata={"checksum": sha256_checksum(data)},
                IfNoneMatch="*",
            )
            return True
        except Exception as exc:
            if getattr(exc, "response", {}).get("Error", {}).get("Code") in (
                "PreconditionFailed",
                "412",
                "ConditionalRequestConflict",
                "409",
            ):
                return False
            raise

    def get(self, aid):
        return self._get("records/" + require_analysis(aid) + ".json")

    def put(self, doc):
        aid = require_analysis(doc["analysis_id"])
        if not self._put("records/" + aid + ".json", doc):
            previous = self.get(aid)
            if previous["request_fingerprint"] != doc["request_fingerprint"]:
                raise FinplanError(
                    ErrorCode.IMMUTABLE_RECORD, "analysis identity conflicts"
                )
            doc = previous
        # Reverse timestamp provides newest-first listing without scanning all historical payloads.
        from datetime import datetime

        epoch = int(datetime.fromisoformat(doc["created_at"]).timestamp() * 1e6)
        key = f"index/{9999999999999999 - epoch:016d}-{aid}.json"
        self._put(
            key,
            {
                "analysis_id": aid,
                "analysis_kind": doc["analysis_kind"],
                "portfolio_id": doc.get("portfolio_id"),
            },
        )
        return doc

    def list(self, *, portfolio_id=None, kind=None, limit=20):
        args = {"Bucket": self.bucket, "Prefix": "classical/index/", "MaxKeys": 100}
        out = []
        for _ in range(10):
            page = self.client.list_objects_v2(**args)
            for item in page.get("Contents", []):
                entry = self._get(item["Key"].removeprefix("classical/"))
                if (not portfolio_id or entry.get("portfolio_id") == portfolio_id) and (
                    not kind or entry["analysis_kind"] == kind
                ):
                    out.append(self.get(entry["analysis_id"]))
                    if len(out) >= limit:
                        return out
            if not page.get("IsTruncated"):
                return out
            args["ContinuationToken"] = page["NextContinuationToken"]
        raise FinplanError.precondition(
            "analysis index exceeds the bounded search; narrow filters",
            reason="classical_index_search_bound",
        )

    def get_claim(self, key):
        if not re.fullmatch(r"[a-z0-9_/-]{1,180}", key):
            raise ValueError("invalid claim key")
        try:
            return self._get("claims/" + key + ".json")
        except FinplanError as exc:
            if exc.code == ErrorCode.NOT_FOUND:
                return None
            raise

    def claim(self, key, value):
        if not re.fullmatch(r"[a-z0-9_/-]{1,180}", key):
            raise ValueError("invalid claim key")
        if self._put("claims/" + key + ".json", value):
            return True
        return self._get("claims/" + key + ".json") == value
