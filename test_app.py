"""Worker-distribution acceptance checks for the fleet worker image.

This module is the executable acceptance fixture for the OAI-2.0 fleet worker
distribution (master #241, child #243). It is intentionally stdlib-only and
side-effect free so it can be executed inside the worker container on a
qualified node as an assigned verification job, and independently re-run on
the supervising host to confirm the result.

It asserts the *distribution contract* only:

* the command profile file is well formed and bounded;
* the image is non-root and contains no baked credential or host-private path;
* the entrypoint consumes node identity at run time and never starts a
  coordinator, gateway or replacement control plane.

It does not assert anything about a particular node's hardware, so a pass here
means "this distribution is correctly scoped", not "this node is qualified".
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
WORKER_DIR = REPO_ROOT / "deploy" / "worker"

PROFILES_PATH = WORKER_DIR / "profiles.json"
DOCKERFILE_PATH = WORKER_DIR / "Dockerfile"
ENTRYPOINT_PATH = WORKER_DIR / "entrypoint.sh"

# Profiles the worker image must always be able to advertise, because the
# cluster grants a job only when every required capability is present.
REQUIRED_PROFILES = ("smoke", "pass-probe", "fail-probe", "python-unittest")

# Secrets, credentials and machine-private locations must never be baked into
# an image. These are the classes of value that would make the image unsafe to
# distribute, plus host paths that would tie the image to one machine.
FORBIDDEN_DOCKERFILE_PATTERNS = (
    r"QPIPE_CLUSTER_TOKEN\s*=\s*\S",
    r"QPIPE_ORIGIN_TOKEN\s*=\s*\S",
    r"BEGIN [A-Z ]*PRIVATE KEY",
    r"/Users/",
    r"/home/[a-z0-9_]+/",
    r"[A-Za-z]:\\\\?Users\\\\?",
    r"\.gguf\b",
    r"--model\b",
)


class ProfileFileTests(unittest.TestCase):
    """The profile allow-list is the worker's only tool surface."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))

    def test_profiles_document_is_an_object(self) -> None:
        self.assertIsInstance(self.raw, dict, "profiles.json must be an object")

    def test_profiles_container_present(self) -> None:
        self.assertIn("profiles", self.raw, "profiles.json must define 'profiles'")
        self.assertIsInstance(self.raw["profiles"], dict)

    def test_required_profiles_are_advertised(self) -> None:
        profiles = self.raw["profiles"]
        missing = [name for name in REQUIRED_PROFILES if name not in profiles]
        self.assertEqual([], missing, f"worker is missing approved profiles: {missing}")

    def test_every_profile_is_bounded_argv(self) -> None:
        profiles = self.raw["profiles"]
        self.assertTrue(profiles, "at least one profile is required")
        for name, profile in profiles.items():
            with self.subTest(profile=name):
                self.assertIsInstance(profile, dict)
                argv = profile.get("argv")
                self.assertIsInstance(argv, list, f"{name}.argv must be a list")
                self.assertTrue(argv, f"{name}.argv must not be empty")
                for argument in argv:
                    self.assertIsInstance(argument, str, f"{name}.argv must be strings")
                self.assertNotIn(
                    "sh",
                    argv[:1],
                    f"{name} must use fixed argv, not a shell string",
                )
                timeout = profile.get("timeout_sec")
                self.assertIsInstance(timeout, int, f"{name}.timeout_sec must be an int")
                self.assertGreater(timeout, 0, f"{name}.timeout_sec must be positive")
                cwd = profile.get("cwd")
                self.assertIsInstance(cwd, str, f"{name}.cwd must be a string")
                self.assertFalse(
                    Path(cwd).is_absolute() or ".." in Path(cwd).parts,
                    f"{name}.cwd must stay inside the job workspace",
                )


class ImageBoundaryTests(unittest.TestCase):
    """The image ships the worker only, and never a credential."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")
        cls.entrypoint = ENTRYPOINT_PATH.read_text(encoding="utf-8")

    def test_runs_as_non_root(self) -> None:
        self.assertRegex(
            self.dockerfile,
            r"(?m)^USER\s+10001:10001\s*$",
            "worker must run as the unprivileged uid/gid 10001",
        )

    def test_no_baked_secret_or_host_private_path(self) -> None:
        for pattern in FORBIDDEN_DOCKERFILE_PATTERNS:
            with self.subTest(pattern=pattern):
                self.assertIsNone(
                    re.search(pattern, self.dockerfile, re.IGNORECASE),
                    f"Dockerfile must not contain {pattern!r}",
                )

    def test_entrypoint_requires_runtime_node_identity(self) -> None:
        for variable in (
            "QPIPE_CLUSTER_URL",
            "QPIPE_CLUSTER_TOKEN",
            "QPIPE_NODE_ID",
        ):
            with self.subTest(variable=variable):
                # Fail closed: `${VAR:?message}` aborts when the variable is
                # unset or empty, so the worker never starts without identity.
                self.assertRegex(
                    self.entrypoint,
                    re.compile(rf"\$\{{{variable}:\?"),
                    f"entrypoint must fail closed when {variable} is unset",
                )

    def test_entrypoint_never_starts_a_coordinator(self) -> None:
        # An unavailable pipeline must not cause the worker to stand up a
        # replacement control plane; that authority stays in the existing
        # pipeline (master #241 non-negotiable boundaries).
        for forbidden in ("cluster serve", "qpipe serve", "qpipe start"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(
                    forbidden,
                    self.entrypoint,
                    f"entrypoint must not launch {forbidden!r}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
