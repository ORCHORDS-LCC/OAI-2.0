"""Status markers for the OAI-2.0 package.

Every module exposes a ``STATUS`` constant so callers can introspect what
is wired vs. what is only design. See :mod:`oai2.core` for the
:class:`Status` enum.
"""

from __future__ import annotations

from . import Status

# Top-level package status — updated when subsystems move from PROPOSED to
# EXPERIMENTAL or IMPLEMENTED.
PACKAGE_STATUS = Status.PROPOSED

__all__ = ["PACKAGE_STATUS"]
