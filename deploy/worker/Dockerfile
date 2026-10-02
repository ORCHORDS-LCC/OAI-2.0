# OAI-2.0 fleet worker — minimal, non-root, worker-only.
#
# Scope (master #241 / #243): this image carries ONLY the worker client and the
# selected execution backend. It must NOT carry the pipeline coordinator, the
# public gateway, the routing/lease authority, teacher/grader services or any
# knowledge corpus. Those stay in the existing architecture.
#
# Build (explicit target platform; never accept an Apple-Silicon image for the
# P50 just because it built on the Mac):
#
# Prepare a staging context containing q-pipe's ./qpipe plus this repository's
# deploy/worker directory, then build explicitly for the P50 architecture:
#
#   mkdir -p /tmp/oai2-worker-build
#   cp -a /path/to/q-pipe/qpipe /tmp/oai2-worker-build/qpipe
#   cp -a deploy /tmp/oai2-worker-build/deploy
#   docker buildx build \
#     --platform linux/amd64 \
#     -f /tmp/oai2-worker-build/deploy/worker/Dockerfile \
#     -t oai2-worker:1.0.0 \
#     --load \
#     /tmp/oai2-worker-build
#
# The worker code path is stdlib-only, so no pip install and no dependency wheel
# are needed.
#
# No credential, token, model weight or machine-private path may ever be baked
# into this image. Node identity and secrets are supplied at runtime.

FROM python:3.12-slim-bookworm

ARG WORKER_VERSION="1.0.0"
ARG GIT_VERSION="2.47.2"

LABEL org.opencontainers.image.title="oai2-fleet-worker" \
      org.opencontainers.image.description="OAI-2.0 worker-only fleet node (verification role)" \
      org.opencontainers.image.version="${WORKER_VERSION}" \
      org.opencontainers.image.base.name="docker.io/library/python:3.12-slim-bookworm"

# ca-certificates: HTTPS clone of the assigned repository.
# git: the worker clones/checks out a job-scoped workspace.
# No MLX, no CUDA, no Metal, no model weights, no knowledge corpus.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    ; \
    rm -rf /var/lib/apt/lists/*; \
    git --version; \
    python3 --version

# Run as an unprivileged, non-root uid. The worker never needs to write outside
# its work root and never needs the Docker socket or privileged mode.
RUN set -eux; \
    groupadd --gid 10001 worker; \
    useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin worker

# Worker source only (no tests, no docs, no .env, no venv, no macOS cruft).
WORKDIR /opt/worker
COPY qpipe/ /opt/worker/qpipe/

# Bounded writable state: job-scoped workspaces, command profiles, worker log.
RUN set -eux; \
    mkdir -p /var/lib/worker/root /var/lib/worker/profiles; \
    chown -R worker:worker /var/lib/worker /opt/worker; \
    chmod -R a-w /opt/worker

COPY --chmod=0644 deploy/worker/profiles.json /var/lib/worker/profiles/profiles.json
COPY --chmod=0755 deploy/worker/entrypoint.sh /usr/local/bin/worker-entrypoint

USER 10001:10001

# Bounded, non-privileged volumes (bind or named) are supplied at run time for
# /var/lib/worker/root. Never mount the whole home directory or the host docker.sock.
VOLUME ["/var/lib/worker/root"]

ENV PYTHONPATH=/opt/worker \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/var/lib/worker

ENTRYPOINT ["/usr/local/bin/worker-entrypoint"]
