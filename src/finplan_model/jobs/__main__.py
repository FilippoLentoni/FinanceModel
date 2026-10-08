"""``python -m finplan_model.jobs <job_type>``: the CPU image entry point."""

import sys

from .entrypoint import main

if __name__ == "__main__":
    sys.exit(main())
