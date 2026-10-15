"""Windows workaround for screamingface 0.1.3 (Windows only).

`screamingface._results.cases._publish_index` calls `os.fsync` on a file opened read-only ("rb").
POSIX allows that; Windows raises `OSError: [Errno 9] Bad file descriptor`, which the SDK reports
as `result_storage_failed` AFTER the paid model calls have finished. This shim makes `os.fsync`
ignore EBADF on Windows only. It changes durability only (an fsync of an already-written file is
skipped), never the data. Import this module before running any evaluation.
"""

from __future__ import annotations

import errno
import os

_real_fsync = os.fsync


def _tolerant_fsync(fd: int) -> None:
    try:
        _real_fsync(fd)
    except OSError as exc:
        if exc.errno != errno.EBADF:
            raise


def apply() -> None:
    if os.name == "nt" and os.fsync is _real_fsync:
        os.fsync = _tolerant_fsync


apply()
