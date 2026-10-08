"""Job interface and control plane (task groups 7 and 8).

* :mod:`.service` - :class:`JobService`: ``submit_job``, ``get_job_status``, ``get_job_result``,
  ``cancel_job``, ``list_jobs``, ``approve_run``, the dispatcher tick and the SageMaker
  state-change handler;
* :mod:`.api` - HTTP routing for API Gateway (IAM auth); :mod:`.handlers` - Lambda entry points;
* :mod:`.validation`, :mod:`.costs`, :mod:`.leases`, :mod:`.sagemaker`, :mod:`.states`,
  :mod:`.store`, :mod:`.settings`, :mod:`.auth`, :mod:`.policies` - the building blocks.

See ``docs/job-interface.md`` (consumers) and ``docs/job-execution.md`` (operations and wiring).
"""

from .auth import Principal
from .service import JobService, ServiceDeps, status_document
from .settings import SsmSettings, StaticSettings
from .store import TABLE_SPEC, DynamoRunStore, InMemoryRunStore

__all__ = ["DynamoRunStore", "InMemoryRunStore", "JobService", "Principal", "ServiceDeps", "SsmSettings", "StaticSettings", "TABLE_SPEC", "status_document"]
