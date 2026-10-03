# Numerical promotion evidence

Machine-checkable evidence that a real optimized path was run through the
canonical promotion gate in `oai2/model/numerical_promotion.py`, tied to an
exact source SHA and a proven serving identity.

## Why this directory exists

Issue #217's closure rule requires the comparison matrix to be "consumed by
backend/export/kernel promotion gates and locally reproduced without
runners". The gate existed and was tested; nothing had ever been run *through*
it on a real backend, so REQ-NUM-021 (a reference implementation or declared
oracle) and REQ-NUM-026 (artifacts identifying model/runtime/backend/config
versions) had no artifact behind them.

`llamacpp_runtime.py` (added in `a133744`) is the first real BACKEND candidate
in this repository, so it is the first thing that can close that gap.

## What an artifact here does and does not claim

Each `backend_*.json` records **one thing**: that a named backend was run
through the canonical gate on a named operation, and what the gate decided.

It does **not** claim:

- that the backend is or is not fit for production;
- an accuracy figure for any suite (that is `evidence/accuracy/`);
- that a `passed: true` result means the model is correct — it means the
  backend matched the *declared oracle* on the probed samples.

The `scope` field inside each artifact repeats this, so the claim travels with
the file rather than living only in a commit message.

## Reproducing

```
uv run python scripts/numerical_backend_evidence.py
```

Requires a running `llama-server` on `127.0.0.1:8851` (the NORMAL lane).
Serving identity is read from `/props` and the weights are hashed from the file
the server reports; the harness refuses to emit an artifact if it cannot prove
which model is resident, because a numerical claim about an unidentified model
is not a claim about anything.

The artifact filename carries the source SHA that produced it, and the
`harness.dirty` flag must be `false` — a dirty tree means the SHA does not
describe the code that produced the number. `tests/test_numerical_backend_evidence.py`
enforces that.

## Current result

`backend_noran_*.json`: the llama.cpp NORMAL lane
(SmolLM2-1.7B-Instruct, Q4_K_M, `n_ctx=8192`) was probed on exact integer
arithmetic at `temperature=0.0`, fixed seed.

The gate **refused promotion and fell back to the reference**: `17 + 25` was
answered `32`, not `42`. The other two probes matched. This is a
capability observation about that quantized 1.7B model, recorded as what the
gate decided — not a defect in the gate, and not a claim about the NORMAL
lane's fitness for production.
