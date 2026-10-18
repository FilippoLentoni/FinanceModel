"""Private durable EC2 report-refresh hints, containing no financial ledger values."""
from __future__ import annotations
import os
import uuid
from pathlib import Path
from finplan_model.core.artifacts import canonical_json_bytes,sha256_checksum


def enqueue(directory, task, *, max_pending=10000):
    root=Path(directory);root.mkdir(mode=0o700,parents=True,exist_ok=True)
    import fcntl
    # All producers serialize the capacity check with publication. Readers may
    # remove completed tasks concurrently; that can only free capacity.
    descriptor=os.open(root/'.producer.lock',os.O_WRONLY|os.O_CREAT,0o600)
    with os.fdopen(descriptor,'wb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _enqueue_locked(root,task,max_pending)


def _enqueue_locked(root,task,max_pending):
    filename=sha256_checksum(canonical_json_bytes([task['experiment_id'],task['event_id'],task['reporting_day']])).split(':')[1]+'.json'
    target=root/filename;raw=canonical_json_bytes(task)
    if target.exists():
        if target.read_bytes()!=raw:raise ValueError('permanent report-refresh task conflicts')
        return str(target)
    with os.scandir(root) as entries:
        if sum(1 for entry in entries if entry.is_file() and entry.name.endswith('.json'))>=max_pending:
            raise ValueError('durable report-refresh queue full; no task discarded')
    temporary=root/('.pending-'+uuid.uuid4().hex)
    fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
        try:os.link(temporary,target)
        except FileExistsError:
            if target.read_bytes()!=raw:raise ValueError('concurrent report-refresh task conflicts')
        directory_fd=os.open(root,os.O_RDONLY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
    finally:temporary.unlink(missing_ok=True)
    return str(target)
