# No `# syntax=` directive: the built-in BuildKit frontend suffices, and the directive would pull
# docker/dockerfile from Docker Hub in every build (anonymous pull limits on shared CodeBuild IPs).
#
# financemodel-cpu: the one CPU image of FinanceModel phase 1 (design D5; task 6.1).
# Entry points (ContainerArguments): prepare_dataset | run_backtest | run_benchmark | report
#
#   docker build -f container/Dockerfile -t financemodel-cpu .          # context = repository root
#   docker run --rm -v "$PWD/.local-run:/work" financemodel-cpu run_backtest \
#       --local /work --run-id <run_id> --fixture-snapshot /work/payload.json   # offline, no AWS
#
# The pipeline build stage builds this image ONCE, pushes it and promotes it by digest; job
# definitions in every environment reference <repository>@sha256:<digest> (DEP-03). Pin the base
# images by digest in the build stage with --build-arg PYTHON_IMAGE=...@sha256:<digest>.
#
# Dependencies come only from uv.lock (exact versions and hashes) with the vendored, digest-pinned
# finplan-contracts wheel. No dev or test dependencies, no tests, no fixtures, no market data, no
# credentials and no account identifiers are copied into the image. Strategy code runs only here,
# never in CodeBuild or Lambda (spec job-deployment-pipeline).
ARG PYTHON_IMAGE=public.ecr.aws/docker/library/python:3.12-slim-bookworm
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.23

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/financemodel/venv
WORKDIR /src
COPY pyproject.toml uv.lock README.md ./
COPY vendor/finplan-contracts/ ./vendor/finplan-contracts/
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable

FROM ${PYTHON_IMAGE} AS runtime
RUN groupadd --system --gid 10001 financemodel \
    && useradd --system --uid 10001 --gid financemodel --home-dir /opt/financemodel --shell /usr/sbin/nologin financemodel
COPY --from=build /opt/financemodel/venv /opt/financemodel/venv
COPY config/ /opt/financemodel/config/
ENV PATH=/opt/financemodel/venv/bin:${PATH} \
    FINPLAN_CONFIG_DIR=/opt/financemodel/config \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /opt/financemodel
USER 10001:10001
ENTRYPOINT ["python", "-m", "finplan_model.jobs"]
CMD ["--help"]
