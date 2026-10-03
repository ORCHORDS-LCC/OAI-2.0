"""The Cloudflare Python Worker entrypoint for the knowledge transport.

Package marker so the entrypoint is importable in tests with a stubbed
``workers`` module. ``entry.py`` itself imports ``workers``, which exists only
inside the Worker runtime, so tests inject a stub before importing it.
"""
