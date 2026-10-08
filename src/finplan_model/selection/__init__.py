"""Offline model selection (decision 27): one evaluator, fixed chronological splits, validation-only
selection and a single untouched test evaluation. :mod:`.protocol` is pure Python (the control plane
validates and freezes it); :mod:`.job` runs in the job image (it imports the RL learners lazily)."""
