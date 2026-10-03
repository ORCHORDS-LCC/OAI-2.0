# Fleet Worker Evidence — node `p50`

**Status:** living document. Verification-role worker accepted; inference qualification **BLOCKED**.
**Document date:** 2026-10-03
**Repository revision at time of writing:** `e8af56de8d42dd0d1446ab8e7d3c541b0d6e9939`
**Baseline measurement revision (the pinned ref all baseline jobs ran at):** `2983c5ccbfeb4364c41d73ff2aecb19d438939da`
**Measurement harness:** `scripts/measure_worker_contribution.py`
**Scope of this document:** what the fleet worker demonstrably does, what it does not do, and what remains unproven.

---

## 0. Read this before quoting any number from this repository

1. **`p50` is a VERIFICATION-role worker, not an inference node.** It is a worker-only Docker
   container running a bounded, allow-listed command surface (`deploy/worker/profiles.json`, #255).
   It advertises `verification`, `verification:smoke`, `verification:pass-probe`,
   `verification:fail-probe`, `verification:python-unittest` and `max_leases: 1`. It has no
   model server, no accelerator runtime and no serving role. **No output produced against `p50`
   may be presented as inference performance, generative quality, or token/throughput capacity.**

2. **Inference qualification is BLOCKED, not PASSED.** No approved CPU-inference artifact exists
   under **#246**. Until such an artifact is approved and actually run on a node, the inference
   lane has **no passing evidence and must not be reported as qualified**. Absence of an artifact
   is a *block*, not a soft pass and not a silent omission.

3. **End-to-end latency for the measured workload is WORSE on the P50.** For this workload the
   local Mac is faster (Cell A median 1.178s vs Cell B mean 4.549s enqueue-to-finish). The
   demonstrated benefit of the worker is **offload — work leaving the Mac and completing on the
   node (5/5 useful jobs)** — not latency. **A speedup must never be claimed from this
   measurement.**

4. **PAUSE / STOP / DUPLICATE-IDENTITY acceptance cases are BLOCKED.** They were not run. They
   must not be reported as passing, and they must not be quietly omitted from a summary that
   otherwise reads as a clean qualification.

Every acceptance case in this document carries an explicit label:
**PASS** (run, and the evidence supports it) · **FAIL** (run, and it did not hold) ·
**BLOCKED** (cannot be run, or must not be run, until a stated precondition is met) ·
**NOT RUN** (not attempted; no evidence either way).

---

## 1. Established baseline — ALREADY MEASURED, do not re-measure

This section is a **recorded result**, not a live re-run. It is the comparison point the harness
in §4 is designed to reproduce.

- **Baseline recorded:** 2026-10-03 (this document). The measurement itself was performed earlier
  in the 2026-10-02 → 2026-10-03 fleet window; the baseline record does not carry a precise
  measurement timestamp, so none is asserted here.
- **Reps:** 5 per cell, all at pinned revision `2983c5ccbfeb`.
- **Useful work (identical in both cells):** `python3 -m unittest -v test_app` on this repository,
  executed against a checkout of the pinned revision.
- **Cell A (Mac-local):** shallow clone, then local execution.
- **Cell B (Mac + P50):** `qpipe cluster enqueue-verification` → the worker node executes the same
  command in its own workspace.

| Metric | Value | n |
| --- | --- | --- |
| Cell A — Mac-local, total | mean **1.178 s** | 5 |
| Cell B — enqueue-to-finish, total | mean **4.549 s** | 5 |
| Cell B — of which queue / admission | mean **0.714 s** | 5 |
| Cell B — of which execution on node | mean **2.512 s** | 5 |
| Fleet jobs executed on node `p50` | **5 / 5** | 5 |
| Fleet jobs `ok` | **5 / 5** | 5 |
| Fleet jobs at revision `2983c5ccbfeb` | **5 / 5** | 5 |

**Interpretation (already established, and carried forward unchanged).** For THIS workload the P50
is **slower end to end**: 4.549 s against 1.178 s. The value demonstrated is **offload** — zero
useful work performed on the Mac, five useful jobs completed on the node, every one at the pinned
revision. A latency or speedup claim from this lane is not supportable and must not be made.

**Where the baseline is thin.** The baseline reports means only. It does not separate the first
(cold, repository-materialising) execution from later (warm) executions, it gives no min/median/
p95/max distribution, and it does not sample host resource use. §4 exists to close those gaps.

---

## 1a. Re-measured with the harness — LIVE RUN, cold/warm separated

Run 2026-10-03, harness `scripts/measure_worker_contribution.py`, same work, same pinned revision,
`--reps 5`, `--node p50`, `--cluster-url http://127.0.0.1:10534`. **All 12 correctness gates PASS**;
the harness printed the measured results only because it did.

| Metric | Cold (n=1) | Warm (n=4) |
| --- | --- | --- |
| Cell A — Mac-local, total | 1.169 s (preparation/clone 1.133 s) | median **0.0339 s**, p95 0.0340 s |
| Cell B — enqueue-to-finish, total | **3.2158 s** | median **4.6929 s**, p95 4.9085 s |
| Cell B — queue / admission | 0.6824 s | 1.9415 s |
| Cell B — execution on node | 2.4115 s | 2.5550 s |
| Cell A useful jobs per minute | — | 229.956 |
| Cell B useful jobs per minute | — | 13.648 |

Cell B fleet jobs **77, 78, 79, 80, 81** — all `succeeded`, all executed on `p50`, all
`resolved_revision` exactly `2983c5ccbfeb4364c41d73ff2aecb19d438939da`. Both cells emitted the
same `work_argv_sha256` `66c8faead5ce10ae`, which is what makes the two cells comparable at all.

**Cold vs warm, honestly.** The Mac is fast warm (0.034 s) and slow cold (1.169 s) because only
the first rep pays the clone. The node shows the opposite shape: its cold rep is the *fastest* of
its five (3.216 s) and its warm reps are *slower* (median 4.693 s). The node clones inside every
job — the worker makes a disposable clone per job by design — so it never amortises preparation,
while the Mac prepares once and reuses it. That is a real property of the current worker
architecture, not a measurement artefact.

**Two earlier runs are void and are not quoted.** The first run reported a queue time of
~1.79 × 10⁹ s: the harness had subtracted a local `time.monotonic()` instant (seconds since boot)
from a cluster wall-clock epoch. That is a mixed-clock defect, fixed in `plausible_duration()`
and locked by three new self-test checks. The second run lost the control plane mid-run (another
agent restarted it) and the harness correctly refused to publish its statistics. Neither run is
cited as a result; only the clean run above is.

**Benefit, stated without inflation.** 5 of 5 useful jobs executed on `p50`, 0 on the Mac. The
demonstrated value is **Mac offload**. The P50 is ~140× slower end to end on this workload. No
speedup, no inference throughput and no GPU capacity is claimed or measurable here.

---

## 2. Acceptance case register

| # | Acceptance case | Label | Evidence / reason |
| --- | --- | --- | --- |
| A1 | Worker distribution is correctly scoped and carries no baked credential | **PASS** | `test_app.py` — 8 tests, OK, at HEAD; asserts non-root uid 10001, no secret/host-path in `Dockerfile`, fixed-argv profiles only, fail-closed entrypoint. |
| A2 | Node `p50` executes an assigned verification job successfully | **PASS** | Baseline §1: 5/5 jobs `ok` on `p50` at the pinned revision. |
| A3 | Job executes at the revision that was requested (no silent drift) | **PASS** | Baseline §1: 5/5 at `2983c5ccbfeb`; also gated on every future run by the harness (§4.3). |
| A4 | Node stays enrolled and accepts work | **PASS** | Baseline §1 required 5 sequential successful admissions on one node. |
| A5 | Work is offloaded off the Mac | **PASS** | 5/5 useful jobs completed on `p50`; Mac performed the work in Cell B zero times. |
| A6 | **PAUSE** acceptance case | **BLOCKED** | Not run. Requires a deliberate pause/resume drill on the live host with a running job. Blocked until a scheduled window with host access is approved. |
| A7 | **STOP** acceptance case | **BLOCKED** | Not run. Requires a deliberate stop/drain drill on the live host. Blocked for the same reason as A6. |
| A8 | **DUPLICATE-IDENTITY** acceptance case | **BLOCKED** | Not run. Requires two nodes enrolling under one identity to be demonstrated rejected or de-duplicated. Blocked until a second node is available. |
| A9 | **CPU-inference qualification** (artifact under #246) | **BLOCKED** | No approved CPU-inference artifact exists under #246. Nothing was run; nothing may be reported as qualified. |
| A10 | Cold vs warm preparation/execution separation for both cells | **NOT RUN** | Harness built and self-tested (§4). The real A/B run against the live fleet has not been executed. |
| A11 | Latency distribution (min/median/p95/max) for both cells | **NOT RUN** | Same as A10. |
| A12 | Host resource use sampled during execution | **NOT RUN** | Same as A10. |

**Reading rule:** A1–A5 passing does **not** qualify the node for anything beyond verification
work. A6–A9 are open and must appear as open in any roll-up. A10–A12 are pending measurement, not
pending judgement.

---

## 3. Defect → failing test → fix → passing evidence → remaining limitations

### 3.1 D1 — WSL2 tears down the VM, taking `dockerd` and the node with it

- **Symptom.** The node dropped out of the fleet within about a minute of every launch.
- **Why the obvious fix fails.** A container `--restart` policy cannot help, because the policy
  lives *inside* `dockerd`, and `dockerd` is what died. A detached launch therefore has nothing
  left to enforce the policy.
- **Failing test / reproduction.** Live observation on the P50 host: launch the runtime owner, let
  the launching process exit, wait ~60 s, and the node is gone from `GET /v1/cluster/nodes`.
  *(Live host behaviour; NOT re-run by the author of this document — see §3.3.)*
- **Fix.** A single **attached, never-returning** `wsl.exe` invocation
  (`deploy/worker/start-worker.ps1`) whose process holds the WSL2 session open, driving a resident
  supervisor (`deploy/worker/worker-supervisor.sh`) that ensures `docker` is active and re-asserts
  `docker start oai2-worker` every 30 s.
- **Passing evidence.** §1 baseline: 5/5 jobs admitted to and completed on `p50`, which requires the
  node to have stayed enrolled across the whole run. `test_app.py` asserts the entrypoint never
  launches a replacement control plane (A1).
- **Remaining limitation.** Uptime is now coupled to one long-lived attached process. If that
  process is killed, or the Windows host reboots and nothing re-invokes the launcher, the node goes
  offline again. There is **no** host-level autostart (no scheduled task / service registered in
  this repository), so recovery is currently manual. Not measured: time-to-recovery.

### 3.2 D2 — Multi-line script passed through PowerShell `bash -c` executed only its first line

- **Symptom.** Same failure signature as D1: node offline within about a minute of every launch.
- **Cause.** The launcher passed the supervision loop to `bash -c` as a multi-line PowerShell
  here-string. PowerShell split it on newlines, so bash received only the first line and exited 0
  immediately — the loop that was supposed to hold the session alive never ran.
- **Failing test / reproduction.** Launch, then `bash -c` receives a single line and the process
  exits 0 with no supervision loop entered. *(Same provenance as D1.)*
- **Fix.** Pass a **real file as a single argument** to `bash` (`-e bash /path/worker-supervisor.sh`)
  instead of a here-string. One argument cannot be split on newlines, so the whole script runs.
- **Passing evidence.** §1 baseline 5/5 (as D1); the supervisor is a single non-returning process
  by construction, which is what §3.1 depends on.
- **Remaining limitation.** The fix is specific to the invocation shape. Any future launcher that
  re-introduces an inline multi-line script reintroduces D2. There is no automated regression test
  for the PowerShell invocation itself — `test_app.py` covers the *container* contract, not the
  *Windows-host launcher*.

### 3.3 D3 — Harness defect found and fixed by the harness's own self-test

Recorded because it is a worked example of the same discipline, on the measurement tool itself.

- **Defect (two, both caught before any live run).**
  1. `revisions_match()` accepted a bare prefix equality, so an abbreviated requested ref that
     diverged from the resolved sha could not be distinguished from a genuine match. Worse, a
     one-character ref would have satisfied the correctness gate, defeating it.
  2. The report's own `scope_note` disclaimer contained a hardware-performance word, so the
     claim guard rejected the harness's own honest disclaimer. The guard had real teeth.
- **Failing test.** The `--dry-run` self-test: `gate_accepts_good_record` FAIL, and
  `synthetic_report_claim_free` FAIL.
- **Fix.** (1) Require the shorter revision to be ≥ 7 hex characters and require the longer to be
  a valid hex revision token before accepting a prefix match; a new self-test case
  `gate_refuses_trivial_prefix` locks this in. (2) Reworded the disclaimer to state the same
  boundary without using forbidden vocabulary.
- **Passing evidence.** `--dry-run` now reports 20/20 self-test checks passed, exit 0.
- **Remaining limitation.** The two node-identity gates (D1, D2) are **live-host observations from
  the fleet brief and the repository's own commit history**, not observations made by the author of
  this document. They are labelled accordingly and were not re-run here.

### 3.4 D4 — Harness mixed two clocks and published an epoch as a duration (found on the FIRST live run)

- **Defect.** The enqueue instant was captured with `time.monotonic()` (seconds since boot, ≈40487
  on this host) while `grant` and `finish` come from the cluster's wall-clock epoch. The subtraction
  yielded ~1.79 × 10⁹, and the harness printed it as `queue_s`. `grant_to_finish_s` was unaffected,
  which is exactly why it survived the self-test: the self-test never exercised mixed-clock
  arithmetic.
- **Failing test.** `duration_rejects_mixed_clock_arithmetic` — added as a self-test check; it
  fails against the pre-fix code and passes after.
- **Fix.** The enqueue instant is now a wall clock captured separately from the monotonic rep timer,
  the cluster-reported enqueue stamp is preferred, and every phase duration passes through
  `plausible_duration()`, which returns `None` for negative, non-finite or > 1 day values instead of
  publishing them. An unrecoverable phase is now reported unavailable, not invented.
- **Passing evidence.** Self-test 23/23 PASS. The clean live run in §1a reports real queue values
  (0.6824 s cold, 1.9415 s warm) instead of an epoch.
- **Remaining limitation.** `plausible_duration()` bounds a phase at 86 400 s. A job that genuinely
  queued for more than a day would be reported as unavailable rather than measured. That is
  deliberate: this harness enqueues one bounded verification job per rep and is not a queue-depth
  study.

---

## 4. Measurement harness — `scripts/measure_worker_contribution.py`

Stdlib only, no third-party dependency. Built and self-tested; **the real A/B measurement has not
been run** (see A10–A12). It exists so the §1 baseline can be reproduced with the cold/warm
distinction, a latency distribution, and host sampling that the baseline lacks.

### 4.1 CLI surface

| Flag | Default | Purpose |
| --- | --- | --- |
| `--reps N` | `5` | Repetitions per cell. Rep 0 is cold; reps 1..N-1 are warm. |
| `--ref SHA` | `HEAD` | The pinned revision both cells run at. |
| `--repo-url URL` | *(required)* | Repository the worker clones and the Mac checks out. |
| `--check PROFILE` | `python-unittest` | Worker check profile; **also defines cell A's argv**, so both cells provably run identical work. |
| `--node ID` | `p50` | The node every fleet job is required to execute on. |
| `--json-out PATH` | *(none)* | Machine-readable report for CI/archive capture. |
| `--cluster-url URL` | `$QPIPE_CLUSTER_URL` or `http://127.0.0.1:10534` | Cluster control plane. |
| `--qpipe PATH` | q-pipe venv CLI | Enqueue path. |
| `--poll-interval`, `--timeout`, `--sample-interval` | `0.2`, `300.0`, `1.0` | Poll, per-job timeout, host sampling rate. |
| `--auth-header`, `--auth-scheme` | `Authorization`, `Bearer` | Cluster token header (token read from `$QPIPE_CLUSTER_TOKEN`; never printed, never a CLI arg). |
| `--local-argv` | *(none)* | Escape hatch to override cell A's argv. |
| `--dry-run` / `--self-test` | off | Validate config, run self-tests, print the plan. **Zero cluster calls.** |
| `--quiet` | off | Suppress the human-readable report. |

**Exit codes:** `0` all gates passed · `1` correctness gate failure · `2` configuration error.

### 4.2 How cold is separated from warm — field names

Both cells are labelled per rep, and the latency blocks are reported **separately** for each phase
and never blended into a single headline mean.

- Per-rep discriminator: **`phase`** ∈ {`"cold"`, `"warm"`}. `classify_phase(0) == "cold"`;
  every later index is `"warm"`.
- Cell A cold is *forced cold*: the checkout is materialised from scratch (`git init` +
  `fetch --depth 1` for a sha ref, `git clone --depth 1 --branch` for a branch), so preparation
  cost is real rather than a warm cache hit.
- Per-rep timing fields: **`total_s`** (= preparation + execution), **`preparation_s`**,
  **`execution_s`**.
- Cell A block: `preparation_s.cold` / `preparation_s.warm` — warm is `0` by construction, since
  warm reps reuse the prepared tree.
- Cell B per-rep fields: **`job_id`**, **`node_id`**, **`state`**, **`ok`**, **`requested_ref`**,
  **`resolved_revision`**, **`enqueue_to_grant_s`** (queue/admission), **`grant_to_finish_s`**
  (execution on node), **`total_s`**.
- Cell B block: **`phase_split_s.cold.{queue_s, execution_s, n}`** and
  `phase_split_s.warm.{queue_s, execution_s, n}`.
- Latency distributions, each reporting `n`, `min_s`, `median_s`, `p95_s`, `max_s`, `mean_s`,
  `p95_method`: **`latency_s.cold`**, **`latency_s.warm`**, **`latency_s.all`** (per cell).
  `p95` is **nearest-rank**, so at n=5 the p95 *is* the max; the report prints `p95_method` so a
  reader is never misled by that.
- **Honest limitation of the cell B split.** Node-side repository materialisation is *not*
  separately observable in the job record. The harness therefore isolates it by the **cold/warm rep
  contrast** (rep 0 includes it, later reps do not) rather than reporting a fabricated preparation
  number. If the scheduler events do not expose grant/finish instants, the affected rep's split is
  reported as `null` with a note — never guessed.

### 4.3 Correctness gates (these decide the exit code)

Gates run **before** statistics are considered meaningful. `judge_record()` is the single source of
truth; on failure the process exits non-zero and the report prints
*"CORRECTNESS GATES FAILED … statistics above are VOID."*

| Gate | Condition | Failure mode covered |
| --- | --- | --- |
| `cell_a_rep<i>_cold/warm` | local command exit `0` **and** the prepared checkout's `rev-parse HEAD` matches `--ref` | silent checkout drift |
| `cell_b_rep<i>_cold/warm` | `ok` is true | job did not succeed |
| " | `resolved_revision` present and equals `--ref` (prefix-tolerant, ≥ 7 hex chars) | wrong code executed |
| " | `node_id` present **and** equals `--node` | work did not run on the expected node |
| " | enqueue + poll completed within `--timeout` | enqueue/poll error, recorded per rep |
| `expected_node_advertises_verification` | `GET /v1/cluster/nodes` shows the node with `verification` in its capabilities | wrong role, wrong node, cluster unreachable |
| `cell_a_work_is_matched` | cell A argv == the `--check` profile argv | the two cells silently doing different work |

Matched-work proof: both cells emit **`work_argv_sha256`** (SHA-256 over the joined argv). Identical
digests are the evidence that Cell A and Cell B ran *the same* useful work.

### 4.4 How the harness structurally cannot print an inference/throughput claim

Four independent mechanisms, all exercised by the self-test:

1. **Fail-closed vocabulary guard.** `assert_verification_scoped()` scans the **fully rendered
   report and the serialised JSON** for 45 generative-performance vocabulary classes (hardware
   names, token-rate units, latency-to-first-token style names, serving-stack names, output-quality
   terms, and rate units). A match raises `InferenceClaimError` → **exit 1**, and nothing is printed
   or written. The guard has no allow-list escape hatch and cannot be satisfied by reformatting.
2. **The guard is proven to have teeth.** `--dry-run` plants one probe per vocabulary class and
   requires all 45 to be rejected (`claim_guard_rejects_all_vocabulary`), *and* requires a full
   synthetic report + JSON payload to pass the same guard (`synthetic_report_claim_free`). It has
   already caught a real violation in this harness (§3.3 D3).
3. **Measurement is structurally incapable of it.** The harness runs one fixed, allow-listed
   verification argv from `deploy/worker/profiles.json` and reads a job record. There is no code
   path that produces a token count, a hardware utilisation figure, or an output-quality score.
   The report's own capacity label is fixed: `capacity_class: "verification-execution-only"`,
   `worker_role: "verification"`.
4. **Traffic is constrained, not just intended.** `validate_cluster_url()` refuses any port that
   looks like a model server, coordinator or gateway, and `assert_local_argv_offline()` rejects any
   cell A argument containing a URL or a service/model switch. So the harness cannot obtain such a
   measurement in the first place, and the self-test verifies 3/3 service probes are blocked.

### 4.5 Boundaries the harness respects

Contacts **only** the cluster control plane (`enqueue-verification`, `GET /v1/cluster/jobs/<id>`,
`/v1/cluster/events`, `/v1/cluster/nodes`). It does not touch the Mac model server, the
coordinator, or the gateway, and it sends no foreground Mac inference traffic. Host sampling is
best-effort (`os.getloadavg()`; memory via `sysctl`/`vm_stat` on macOS) at a low rate; an
unavailable metric is reported as `null` and **never fails the run**.

### 4.6 Validated state of the harness (actually executed)

| Check | Result |
| --- | --- |
| `python3 -m py_compile scripts/measure_worker_contribution.py` | **PASS** |
| `python3 scripts/measure_worker_contribution.py --help` | **PASS** (exit 0) |
| `--dry-run` self-test (no cluster calls) | **PASS** — 20/20 checks, exit 0 |
| Dry run under a network/subprocess trap (`urlopen` and `git`/`qpipe`/`ssh`/`curl`/`docker` spawns made to raise) | **PASS** — exit 0, 0 network calls, 0 spawns |
| `python3 -m unittest test_app` | **PASS** — 8 tests, OK |

**Not run:** any real measurement. No cluster job was enqueued, no host was contacted, no `ssh` was
used, no cluster port was dialed. A6–A12 remain open exactly as labelled in §2.

---

## 5. Host / runtime configuration — what changed, and where it lives

This table exists so a reader can tell **what was changed on the P50 host** (outside the
repository, not reconstructible from it) from **what is only in the repository**. Items marked
*host* are properties of the Windows machine's live state; a fresh clone of this repository does
**not** reproduce them.

| Item | Value | Lives in | Changed on the P50 host? | Notes |
| --- | --- | --- | --- | --- |
| Node id | `p50` | host runtime env | **Yes** | Advertised identity used by the baseline. |
| Node role / capabilities | `verification`, `verification:smoke`, `verification:pass-probe`, `verification:fail-probe`, `verification:python-unittest` | host runtime | **Yes** | Verification capacity only. No serving role. |
| Node metadata | `max_leases: 1` | host runtime | **Yes** | Single-lease: one job at a time. Bounds achievable throughput. |
| Container name | `oai2-worker` | `worker-supervisor.sh` | **Yes** (container created/registered on host) | The supervisor re-asserts this exact name. |
| WSL2 distro | `Ubuntu-24.04` | `start-worker.ps1` | **Yes** | The launcher targets this distro by name. |
| WSL launcher | `C:\Windows\System32\wsl.exe` | `start-worker.ps1` | **Yes** | Invoked attached and blocking (§3.1). |
| WSL user | `root` (`-u root`) | `start-worker.ps1` | **Yes** | Needed to manage the `docker` service. |
| **Executed copy of the supervisor** | `/mnt/c/Users/P50/q-pipe/worker-supervisor.sh` | **host filesystem only** | **Yes** | ⚠️ The path the launcher executes is **not** the repository path. The repo copy is `deploy/worker/worker-supervisor.sh`; it must be **copied to the host** and kept in sync manually. A stale host copy is a live drift risk. |
| `docker` service management | `systemctl start docker`, retried up to 60× at 2 s | `worker-supervisor.sh` | **Yes** | Non-fatal if unavailable; supervisor continues retrying. |
| Container re-assert loop | `docker start oai2-worker` every 30 s | `worker-supervisor.sh` | **Yes** | Keeps the restart policy meaningful. |
| Host-level autostart on Windows boot | **absent** | — | **No** | ⚠️ Known gap (§3.1). Nothing in this repository registers a scheduled task or service. Recovery after reboot is manual. |
| Worker image contents | worker only, uid 10001, no credentials | `deploy/worker/Dockerfile` | via image build | Enforced by A1. |
| Command allow-list | fixed argv per profile, no shell interpolation | `deploy/worker/profiles.json` | via image build | The scoped tool surface (#255). |
| Container entrypoint contract | requires `QPIPE_CLUSTER_URL` / `QPIPE_CLUSTER_TOKEN` / `QPIPE_NODE_ID` at run time; never starts a coordinator | `deploy/worker/entrypoint.sh` | via image build | Enforced by A1. |
| Measurement harness | `scripts/measure_worker_contribution.py` | **repository only** | **No** | Runs on the Mac; not deployed to the node. |
| This evidence document | `docs/WORKER_EVIDENCE.md` | **repository only** | **No** | — |

**Node credential.** `QPIPE_CLUSTER_TOKEN` is supplied to the worker process at run time by the
host environment. It is deliberately **not** recorded here, not baked into the image, and not
accepted as a CLI argument. A new value must be issued on the host; nothing in this repository
needs to change to rotate it.

---

## 6. What this document does not claim

- It does **not** claim the measurement in §4 was run. Only the baseline in §1 was measured.
- It does **not** claim `p50` is qualified for inference. That is **BLOCKED** (#246, A9).
- It does **not** claim any latency improvement, speedup or throughput gain. The measured
  end-to-end latency is **worse** on the P50; the benefit is offload.
- It does **not** claim PAUSE / STOP / DUPLICATE-IDENTITY acceptance. Those are **BLOCKED** (A6–A8).
- It does **not** assert live host behaviour observed by the author of this document. D1/D2 are
  recorded from the fleet brief and the repository's own commit history; §3.3 says so explicitly.
- It does **not** treat a BLOCKED case as a PASS, anywhere, in any roll-up derived from it.
