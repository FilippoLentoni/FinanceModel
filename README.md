# FinanceModel

Research compute for the finplan platform: the common evaluator and paper execution simulator,
research datasets built from approved platform snapshots, baseline strategies, and the
experiment job interface with its control plane (OpenSpec change `add-research-job-foundation`).
Paper only: nothing here can place a live order.

- Layout, contract pin, build gates and shared interfaces: [docs/foundation.md](docs/foundation.md)
- Simulator conventions and configuration: [docs/simulator.md](docs/simulator.md)
- Offline model selection (controls vs. traditional optimizers vs. PPO/SAC, decision 27):
  [docs/model-selection.md](docs/model-selection.md); RL environment: [docs/rl-environment.md](docs/rl-environment.md)

```sh
uv sync --locked                                  # Python 3.12, pinned finplan-contracts 1.0.0
uv run pytest                                     # offline: no network, no AWS, no SageMaker
uv run python scripts/build_gates.py --stage pre  # contract pin, leak scan, copied-$id, conformance ...
npx aws-cdk@2 synth                               # offline synthesis of infra/app.py
```

This repository is public: it never contains account IDs, ARNs with accounts, bucket or endpoint
names, prices, secrets or retrieved market data. Fixtures are synthetic (`synthetic: true`).
