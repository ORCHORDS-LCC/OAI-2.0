"""Deployment artefacts that are not part of the importable runtime.

Present so ``deploy.cloudflare.knowledge_worker.entry`` is a real package and
the Worker entrypoint can be exercised by tests offline. The marker is empty on
purpose: nothing here is imported at runtime, and the modules under it bind no
identifiers of their own.
"""
