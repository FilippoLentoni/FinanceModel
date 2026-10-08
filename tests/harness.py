"""Offline test harness (task 1.3; spec job-deployment-pipeline "No model fitting in CodeBuild or
Lambda", DEP-02). Loaded by ``tests/conftest.py`` for every test, and usable as a pytest plugin
(``-p tests.harness``).

While installed:

* **No network.** ``socket.connect``/``connect_ex`` to anything but loopback or a UNIX socket, and
  DNS resolution of non-local names, raise :class:`NetworkBlocked`.
* **No real AWS call.** botocore's HTTP sender raises :class:`AwsCallBlocked` before any request
  leaves the process. moto (``mock_aws``) still works: it answers inside botocore before the
  sender is reached.
* **No SageMaker call at all**, not even through moto: every ``sagemaker`` API operation raises
  :class:`AwsCallBlocked` (for example ``CreateProcessingJob``), so a build-stage test that tries to
  start a job fails the build. Control-plane tests use an injected fake SageMaker client instead.
  A test may opt into moto's in-process SageMaker mock with ``@pytest.mark.allow_sagemaker_moto``;
  real network stays blocked regardless.
* **Fake credentials only**: ``AWS_*`` variables point at dummy values and empty config files, so
  no test can pick up the developer's or the build role's credentials.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from typing import Any

import pytest

__all__ = ["OFFLINE_MARKER", "AwsCallBlocked", "NetworkBlocked", "install", "uninstall"]

#: Set while the offline harness is installed; the deployed suites assert it is absent (they must run
#: with the stage role's real credentials, never the fakes: platform lesson L5).
OFFLINE_MARKER = "FINPLAN_OFFLINE_TESTS"
_LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "0.0.0.0", ""}
_STATE: dict[str, Any] = {"installed": False, "allow_sagemaker": False}
_ORIG: dict[str, Any] = {}


class NetworkBlocked(RuntimeError):
    """A test tried to open a network connection."""


class AwsCallBlocked(RuntimeError):
    """A test tried to call AWS for real, or to call SageMaker at all."""


def _host_of(address: Any) -> str:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return str(address)


def _guarded_connect(orig: Any) -> Any:
    def connect(self: socket.socket, address: Any) -> Any:
        if getattr(socket, "AF_UNIX", None) is not None and self.family == socket.AF_UNIX:
            return orig(self, address)
        host = _host_of(address)
        if host in _LOCAL_HOSTS:
            return orig(self, address)
        raise NetworkBlocked(f"network access is blocked in tests (attempted connection to {host})")

    return connect


def _guarded_getaddrinfo(orig: Any) -> Any:
    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host is None or str(host) in _LOCAL_HOSTS:
            return orig(host, *args, **kwargs)
        raise NetworkBlocked(f"DNS resolution is blocked in tests ({host})")

    return getaddrinfo


_FAKE_ENV = {
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_SESSION_TOKEN": "testing",
    "AWS_DEFAULT_REGION": "us-east-2",
    "AWS_REGION": "us-east-2",
    "AWS_CONFIG_FILE": os.devnull,
    "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
    "AWS_EC2_METADATA_DISABLED": "true",
    # offline synth packages the source tree: never release mode (CodeBuild sets CODEBUILD_BUILD_ID
    # for the unit gate too), as the platform's tests/offline_env.py does
    "FINPLAN_RELEASE_BUILD": "0",
    OFFLINE_MARKER: "1",
}


def install() -> None:
    if _STATE["installed"]:
        return
    for k in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "AWS_CONTAINER_CREDENTIALS_FULL_URI", "AWS_CONTAINER_AUTHORIZATION_TOKEN", "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE", "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN", "AWS_ROLE_SESSION_NAME", "AWS_SECURITY_TOKEN", "FINPLAN_LAMBDA_BUNDLE_DIR"):
        _ORIG.setdefault(("env", k), os.environ.pop(k, None))
    for k, v in _FAKE_ENV.items():
        _ORIG.setdefault(("env", k), os.environ.get(k))
        os.environ[k] = v

    _ORIG["connect"] = socket.socket.connect
    _ORIG["connect_ex"] = socket.socket.connect_ex
    _ORIG["getaddrinfo"] = socket.getaddrinfo
    socket.socket.connect = _guarded_connect(_ORIG["connect"])  # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect(_ORIG["connect_ex"])  # type: ignore[method-assign]
    socket.getaddrinfo = _guarded_getaddrinfo(_ORIG["getaddrinfo"])  # type: ignore[assignment]

    try:
        import botocore.client
        import botocore.httpsession
    except ImportError:  # pragma: no cover - botocore is a dependency
        _STATE["installed"] = True
        return

    _ORIG["send"] = botocore.httpsession.URLLib3Session.send

    def blocked_send(self: Any, request: Any) -> Any:
        raise AwsCallBlocked(f"real AWS HTTP request blocked in tests ({getattr(request, 'method', '?')} to an AWS endpoint)")

    botocore.httpsession.URLLib3Session.send = blocked_send  # type: ignore[method-assign]

    _ORIG["make_api_call"] = botocore.client.BaseClient._make_api_call

    def guarded_make_api_call(self: Any, operation_name: str, api_params: Any) -> Any:
        service = self.meta.service_model.service_name
        if service in ("sagemaker", "sagemaker-runtime") and not _STATE["allow_sagemaker"]:
            raise AwsCallBlocked(f"SageMaker call blocked by the test harness: {operation_name} (build-stage tests must not call SageMaker; inject a fake client)")
        return _ORIG["make_api_call"](self, operation_name, api_params)

    botocore.client.BaseClient._make_api_call = guarded_make_api_call  # type: ignore[method-assign]
    _STATE["installed"] = True


def uninstall() -> None:
    if not _STATE["installed"]:
        return
    socket.socket.connect = _ORIG.pop("connect")  # type: ignore[method-assign]
    socket.socket.connect_ex = _ORIG.pop("connect_ex")  # type: ignore[method-assign]
    socket.getaddrinfo = _ORIG.pop("getaddrinfo")  # type: ignore[assignment]
    if "send" in _ORIG:
        import botocore.client
        import botocore.httpsession

        botocore.httpsession.URLLib3Session.send = _ORIG.pop("send")  # type: ignore[method-assign]
        botocore.client.BaseClient._make_api_call = _ORIG.pop("make_api_call")  # type: ignore[method-assign]
    for key in [k for k in _ORIG if isinstance(k, tuple) and k[0] == "env"]:
        val = _ORIG.pop(key)
        if val is None:
            os.environ.pop(key[1], None)
        else:
            os.environ[key[1]] = val
    _STATE["installed"] = False


# --------------------------------------------------------------------- pytest hooks
def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "allow_sagemaker_moto: the test may call SageMaker through moto's in-process mock (real network stays blocked)")
    install()


def pytest_unconfigure(config: pytest.Config) -> None:
    uninstall()


@pytest.fixture(autouse=True)
def _offline_harness(request: pytest.FixtureRequest) -> Iterator[None]:
    install()
    _STATE["allow_sagemaker"] = request.node.get_closest_marker("allow_sagemaker_moto") is not None
    try:
        yield
    finally:
        _STATE["allow_sagemaker"] = False
