# Fleet Worker Evidence — node `p50`

**Status:** living document. Verification-role worker accepted; inference qualification **BLOCKED**; the hardened supervisor is **staged but not active**.
**Document date:** 2026-10-03
**Scope:** what the fleet worker demonstrably does, what it does not do, what is only staged, and what remains unproven.

Every acceptance case carries an explicit label:

- **PASS** — run, and the evidence supports it.
- **FAIL** — run, and it did not hold.
- **BLOCKED** — cannot be run, or must not be run, until a stated precondition is met.
- **NOT RUN** — not attempted; no evidence either way.

**Staged ≠ active.** "Staged" means the bytes are correct on the host. "Active" means the currently running process loaded them. These are three different states and §1 separates them. No claim in this document is made about code that is staged but not yet loaded.

---

## 0. Read this before quoting any number from this repository

1. **`p50` is a VERIFICATION-role worker, not an inference node.** It is a worker-only Docker container
   running a bounded, allow-listed command surface (`deploy/worker/profiles.json`, #255). It advertises
   `verification`, `verification:smoke`, `verification:pass-probe`, `verification:fail-probe`,
   `verification:python-unittest`, `max_leases: 1`. It has no model server, no accelerator runtime and
   no serving role. **No output produced against `p50` may be presented as inference performance,
   generative quality, or token/throughput capacity.**

2. **Inference qualification is BLOCKED, not PASSED.** No approved CPU-inference artifact exists under
   **#246** (and none under **#250**). The inference lane has **no passing evidence and must not be
   reported as qualified**. Absence of an artifact is a *block*, not a soft pass and not a silent
   omission.

3. **End-to-end latency for the measured workload is WORSE on the P50** (§4). The demonstrated benefit
   is **offload** — work leaving the Mac and completing on the node — not latency. **A speedup must
   never be claimed from this measurement.** The measurement harness enforces this structurally, not
   by convention (§6.4).

4. **The hardened supervisor is not running.** The owner-state fixes in §5.2 are **PASS as code and
   regression tests**, and are **NOT ACTIVE on the host**. The live process is still executing the
   previous revision. See §1. Do not describe this lane as "hardened in production".

5. **The cluster-side lifecycle controls were never exercised.** Pause, revoke, expired-lease handling
   and duplicate-result rejection are **NOT RUN** — they belong to the control plane, not to this lane
   (A6–A8). Supervisor exit is **not** a cluster pause, drain, revoke or removal (§3).

6. **Host-failure recovery is NOT RUN.** Reboot, sleep/wake, address change, DNS/link failure,
   controller restart, disk exhaustion and memory pressure have not been exercised (A13).

---

## 1. The five identities — what is active, what is only staged

These are five distinct things. Collapsing any two of them is how a staged change gets reported as a
deployed one. All values below were read from the live host on 2026-10-03.

### 1.1 Repository revision (source of truth in git)

| Item | Value |
| --- | --- |
| Branch | `main` (direct; no branch, no PR, no force push) |
| Revision at the time of writing | see §1.6 — verify with `git rev-parse HEAD` |
| Working tree | clean at commit time; `git status --short` empty |

### 1.2 Staged on the P50 host (correct bytes, not yet loaded)

| File | sha256 | Size | Matches repository? |
| --- | --- | --- | --- |
| `/mnt/c/Users/P50/q-pipe/worker-supervisor.sh` | `62973c1c13b51dab5b65d13cdf5d7147fc72e5d0f1c593d2a433948176c9b771` | 31 040 B | **NO — stale.** This is the *previous* round's revision. |
| `/mnt/c/Users/P50/q-pipe/worker-supervisor.sh.prev` | `48fce7161ed67b7896eb394697e048c4ffe9702f5e85409b9fe5bc3d138c3335` | 2 187 B | The original minimal supervisor. |

The repository copy of `worker-supervisor.sh` is now `8c9c0e2fbc5ac5a6…`, 48 688 B. **The host copy is
two revisions behind and must be re-staged before any activation.**

### 1.3 Active runtime (what is executing right now)

| Item | Observed value |
| --- | --- |
| Scheduled task | `OAI2WorkerRuntime`, State `Running` |
| Principal / logon | UserId `P50`, LogonType `S4U`, RunLevel `Highest` |
| Last run / result | 2026-10-03 11:29:32 · `0x41301` = `SCHED_S_TASK_RUNNING` |
| Supervisor process | PID 579, started **2026-10-03 11:29:37** |
| WSL boot | 2026-10-03 11:28:45 (process start = boot + 51.8 s, from `/proc/579/stat` field 22) |
| **Revision the live process is running** | **the OLD one** — see below |

**Proof that the live process predates the hardened script**, by two independent observations:

1. The staged file's mtime is **12:27:56**, i.e. 58 minutes *after* the process started at 11:29:37. A
   process that began at 11:29:37 read the file as it was then — the 2 187 B revision, which is exactly
   what is preserved as `worker-supervisor.sh.prev`.
2. `/var/lib/worker` **does not exist**. Only the newer revision creates that state directory (and exits
   2 if it cannot). Its absence is positive evidence that the running code has no owner-state handling
   at all. (It is also a *deliberate* absence: no production state directory has been created by any
   test — see A14.)

> A `/proc/<pid>` directory mtime is **not** a reliable process start time and was initially misread as
> one. The authoritative sources are `/proc/<pid>/stat` field 22 plus `/proc/stat` `btime`.

### 1.4 Image and runtime identity (what the work actually runs inside)

| Item | Value |
| --- | --- |
| Image | `oai2-worker:1.0.1`, id `sha256:387cb9449683b6080bff440f3f2a2433ff0483b16e94f56e8eeed3163aedea2b` |
| Image created | 2026-10-02T22:25:43Z |
| Platform / user | linux/amd64 · `10001:10001` (non-root) |
| Limits | 2 CPU · 2 GiB · 256 PIDs |
| Toolchain | Python 3.12.15 · Docker CLI 29.8.2 · kernel 6.18.40.1 |
| Container | `oai2-worker`, started 2026-10-03T04:30:26Z, `RestartCount` **0** |
| Image-baked `profiles.json` | sha256 `175907483ccde08204d4e09770499d32ffca778d33a899e8e37c07aac26bb080` — **matches** the repository file |

### 1.5 Verified workload revision (what the jobs in §4 actually executed)

| Item | Value |
| --- | --- |
| Pinned revision | `2983c5ccbfeb4364c41d73ff2aecb19d438939da` |
| Work identity | `work_argv_sha256` `66c8faead5ce10ae…` — identical in both cells |
| Jobs | 77, 78, 79, 80, 81 — all `succeeded`, all executed on `p50` |

Note this is *not* the repository revision in §1.1. The measured jobs ran the pinned workload revision;
the supervisor and harness code in this document are later. They are different things and are not
conflated.

### 1.6 What activation requires — and why it was not done here

Bringing the hardened supervisor into effect means restarting the scheduled task. On this host that
releases the WSL2 session, which is what holds `dockerd` up; if `dockerd` dies first, the node leaves
the fleet. Restarting the task, `docker`, WSL or the host independently is **not authorised to this
lane**. Activation must be performed by, or with, the runtime/controller owner. A controlled handover
design is in §3.2. **It is a design; it has not been executed.**

---

## 2. Acceptance case register

Source of truth for "source": the test id, job id, or host observation. "Active runtime" says whether
the evidence came from code that is running on the host today.

| # | Acceptance case | Label | Source | Active runtime? | Limits |
| --- | --- | --- | --- | --- | --- |
| A1 | Worker distribution correctly scoped, no baked credential | **PASS** | `test_app.py`, 8 tests OK | image rebuilt from repo | asserts uid 10001, no secret/host path, fixed-argv profiles, fail-closed entrypoint |
| A2 | Node `p50` executes an assigned verification job | **PASS** | jobs 77–81, all `succeeded` | yes | 5 jobs, one workload, one profile |
| A3 | Job executes at the requested revision (no silent drift) | **PASS** | jobs 77–81 `resolved_revision` = pinned sha; harness gate `gate_accepts_exact_pinned_sha` | yes | prefix-only refs are now rejected by the gate |
| A4 | Node stays enrolled and accepts work | **PASS** | 5 sequential admissions; hb_age 0–1 s over ~9.5 min | yes | no reboot / sleep / network-loss window (see A13) |
| A5 | Work is offloaded off the Mac | **PASS** | 5/5 jobs on `p50`, 0 on the Mac in Cell B | yes | `max_leases: 1`; not a capacity study |
| A6 | **PAUSE** (cluster-side admission pause) | **NOT RUN** | — | n/a | owned by the control plane (`oai2/runtime/admission.py`); not this lane's control |
| A7 | **STOP** (cluster-side drain/revoke) | **NOT RUN** | — | n/a | as A6. Supervisor exit is **not** a drain (§3) |
| A8 | **DUPLICATE-IDENTITY** rejection / de-duplication | **NOT RUN** | — | n/a | needs two nodes under one identity; second node is owned by another agent |
| A9 | **CPU-inference qualification** (#246 / #250) | **BLOCKED** | — | n/a | no approved artifact exists. Nothing may be reported as qualified |
| A10 | Cold vs warm preparation/execution separation, both cells | **PASS** (code) / **PARTIAL** (data) | harness §4.1; live run 2026-10-03 | n/a (Mac-side tool) | labels now come from observed preparation, not rep index. Node-side prep is summed from worker evidence; **no remote rep is ever warm** by architecture |
| A11 | Latency distribution (min/median/p95/max) for both cells | **PASS** | harness, nearest-rank p95, n reported with every distribution | n/a | n is small; the report states that a p95 over 4 observations is simply the max |
| A12 | Host resource use sampled during execution | **PARTIAL** | `sampler_state` + per-metric status | n/a | sampler now actually starts (`sampler.start()` was previously never called). **Local Mac only** — every metric is labelled `host: mac-local`; **no `p50` metric is sampled at all** |
| A13 | Host-failure recovery: reboot, sleep/wake, address change, DNS/link loss, controller restart, disk full, memory pressure, GPU loss, image upgrade/rollback | **NOT RUN** | — | n/a | needs an explicitly authorised window; not run unilaterally |
| A14 | Owner-intent state handling: absent vs damaged, atomic update, interruption, parser agreement | **PASS** (code + tests) | `SupervisorOwnerStateTests`, 15 tests; 5/5 injected defects caught | **NO — staged only** | exercised hermetically (stub `docker`/`systemctl`/`wsl`, scratch state dir). **Not run on the host** |
| A15 | Default regression tests cannot reach live Docker / WSL / systemctl / controller / production container | **PASS** | `TestIsolationEnforcement`, 11 tests | n/a | see §5.3, including the incident that motivated it |
| A16 | Supervisor `dockerd`-down, container-present and S4U-registration paths on the real host | **NOT RUN** | — | partially: the S4U registration *is* live and working | the other two are unexercised |
| A17 | PowerShell syntax of `start-worker.ps1` / `install-worker-task.ps1` | **NOT RUN** | — | the task is registered and Running | `pwsh` is unavailable on the authoring Mac, so the scripts were never parsed by PowerShell |

**Reading rule.** A1–A5 passing qualifies the node for **verification work only**. A6–A8 and A13 are
open and must appear as open in any roll-up. A9 stays BLOCKED. A14–A15 are real passes **on code that
is not yet active** — say so every time.

---

## 3. Lifecycle — the actual flow

### 3.1 The control flow, end to end

```
   OWNER INTENT                    VALIDATION                    LIFECYCLE DECISION
  ───────────────                 ───────────                    ───────────────────
  /var/lib/worker/          ┌──────────────────────┐      ┌───────────────────────────────┐
  supervisor-intent         │ read_intent()        │      │ THIS PROCESS OWNS ONE THING: │
  (durable sentinel)        │                      │      │ suspend its own management    │
                            │ absent?  → run       │      │ (requirement A)                │
  written atomically:       │ regular file?        │      │                               │
   temp in same dir         │ readable?            │      │ it does NOT own:              │
   + rename (commit point)  │ first NON-BLANK line │      │  · worker-process restart  →   │
                            │ exact match to       │      │    the container entrypoint    │
  values: run|stop|          │ run|stop|pause|      │      │  · application reconnect  →    │
  pause|remove|clear         │ remove                │      │    the worker client           │
                            │                      │      │  · cluster pause/drain/revoke │
                            │ any other state →    │      │    → the CONTROL PLANE         │
                            │ FAULT, exit 12       │      │    (oai2/runtime/admission.py) │
                            └──────────┬───────────┘      └───────────────┬───────────────┘
                                       │ FAIL                          │ run
                                       ▼                               ▼
                              ┌─────────────────┐            ┌──────────────────────────┐
                              │ nothing started │            │ WARMER STATE             │
                              │ exit 12         │            │  host/container runtime  │
                              │ "FAULT, not a   │            └────────────┬─────────────┘
                              │  shutdown"      │                         │
                              └─────────────────┘                         ▼
                                                        ┌──────────────────────────┐
                                          WORKER STATE  │ container oai2-worker up│
                                          (host)        │ RestartCount observed,  │
                                                        │ StartedAt compared      │
                                                        └────────────┬─────────────┘
                                                                     │ enrolled + heartbeat
                                                                     ▼
                                          ASSIGNED JOB    ┌──────────────────────────┐
                                          (cluster)       │ admission: max_leases 1  │
                                                        │ queue → grant           │
                                                        └────────────┬─────────────┘
                                                                     │ enqueue-verification
                                                                     ▼
                                          WORKER EXECUTES  ┌──────────────────────────┐
                                          (in container)   │ clone at pinned sha     │
                                                        │ run allow-listed argv   │
                                                        └────────────┬─────────────┘
                                                                     │
                                                                     ▼
                                          INDEPENDENTLY     ┌──────────────────────────┐
                                          CHECKED RESULT   │ checks[]: per-check     │
                                          (cluster side)    │  exit code + duration   │
                                                        │ evidence[]: git-clone / │
                                                        │  fetch / checkout steps │
                                                        │ resolved_revision = sha │
                                                        └────────────┬─────────────┘
                                                                     │
                                                                     ▼
                                                        job_succeeded / job_failed
                                                        + job_enqueued/job_leased
                                                          events (one clock domain)
```

**The two ownership boundaries that this diagram exists to make unmissable:**

- **Requirement A** — the supervisor owns *host/container runtime start only*. The entrypoint loop owns
  worker-process restart; the worker client owns reconnect.
- **"Pause worker admissions" is a different control.** The worker stays enrolled and heartbeating but
  must not be handed new jobs. The supervisor **cannot** implement this and does not pretend to. It
  suspends its own management; the control plane owns admission.

### 3.2 Controlled handover design (NOT EXECUTED)

Activating a new supervisor revision without losing the node. **Not run.** It exists so the runtime
owner can execute it deliberately rather than improvise.

| # | Step | Why | Verification | Rollback |
| --- | --- | --- | --- | --- |
| 1 | Confirm a quiet window: no campaign measurement in flight, controller healthy, node heartbeat fresh | an aborted handover during another agent's window is the actual risk | `/healthz` 200, `p50` in `/v1/cluster/nodes`, hb_age < 5 s | stop — nothing changed |
| 2 | Drain the node: stop handing `p50` new work (control-plane control, **not** the supervisor) | avoids a job dying mid-handover | no in-flight lease | un-drain |
| 3 | Record the five identities of §1 **before** touching anything | this is the only pre-change reference point | saved | — |
| 4 | Re-stage `worker-supervisor.sh` from the repository; verify sha256 equals the repo copy | the host is currently two revisions stale | digest match | — |
| 5 | Write owner intent `stop` **while the old supervisor is still running**, via `--set-intent stop` | the old revision has no `--set-intent`; until the new code is loaded the sentinel is ignored, so this step is a no-op today and becomes meaningful after step 6 | sentinel present | `--set-intent run` |
| 6 | Restart the scheduled task so the new revision is loaded | the WSL2 session is released and re-established by the task itself — **do not** stop WSL, Docker or the host by hand | task State `Running`, new `LastRunTime` | re-stage the previous file and restart again |
| 7 | Verify exactly one active supervisor: a single `worker-supervisor.sh` process, and `/var/lib/worker` now exists | two supervisors would fight over the container; this is the specific failure this step catches | `pgrep -fc` on the argv signature = 1, and the state dir exists | stop the duplicate |
| 8 | Verify the container survived: same `StartedAt` if it was never meant to change, `RestartCount` unchanged | a restart here means `dockerd` went away and came back | `docker inspect` before/after | n/a |
| 9 | Re-enable admission | a node that is up but not admitted looks healthy and does nothing | node accepts work | re-drain |
| 10 | Run one real verification job and check `resolved_revision` | proves the new code path executes real work | job `succeeded` at the pinned sha | `--set-intent stop` |

**Ordering constraint that matters most:** steps 4–7 must happen in one uninterrupted stretch, because
between "WSL session released" and "supervisor resident again" the node is genuinely down. This is why
the handover is a single coordinated action and not something a background retry should be left to
finish.

---

## 4. Dated observations — the measurement record

These are **historical observations, recorded on the dates shown**. They are not re-run by this
document and are not claims about the current code.

### 4.1 Baseline run — 2026-10-02 → 2026-10-03, pinned `2983c5ccbfeb`

5 reps per cell, `python3 -m unittest -v test_app` as the workload in both.

| Metric | Value | n |
| --- | --- | --- |
| Cell A — Mac-local total | mean **1.178 s** | 5 |
| Cell B — enqueue-to-finish total | mean **4.549 s** | 5 |
| Cell B — queue / admission | mean **0.714 s** | 5 |
| Cell B — execution on node | mean **2.512 s** | 5 |
| Jobs executed on `p50` | **5 / 5** | 5 |
| Jobs at revision `2983c5ccbfeb` | **5 / 5** | 5 |

**What this baseline could not show:** it reported means only. It did not separate cold from warm, gave
no distribution, and sampled no host resources. §6 (the harness) exists to close those gaps.

### 4.2 Clean live A/B run — 2026-10-03, jobs 77–81, pinned `2983c5ccbfeb`

`--reps 5 --node p50`, cluster `http://127.0.0.1:10534`, same `work_argv_sha256` in both cells.

| Metric | Cold (n=1) | Warm (n=4) |
| --- | --- | --- |
| Cell A — Mac-local total | 1.169 s (preparation 1.133 s) | median **0.0339 s**, p95 0.0340 s |
| Cell B — enqueue-to-finish total | **3.2158 s** | median **4.6929 s**, p95 4.9085 s |
| Cell B — queue / admission | 0.6824 s | 1.9415 s |
| Cell B — execution on node | 2.4115 s | 2.5550 s |
| Cell A useful jobs/min | — | 229.956 |
| Cell B useful jobs/min | — | 13.648 |

**The shape is real, and the earlier document's explanation of it was wrong.** This document previously
carried the claim that the node "never amortises preparation" because later reps exclude the clone, and
that cell B's preparation was "not separately observable". Both were artefacts of labelling reps by
index. Corrected reading: the worker makes a **disposable clone per job by design**, so *every* remote
rep pays preparation. There is no warm remote rep. The harness now derives this from the worker's own
`evidence[]` steps (`git-clone` / `git-fetch` / `git-checkout`) rather than from the rep number, and
reports node-side preparation seconds summed from those steps.

**Benefit, stated without inflation.** 5 of 5 useful jobs executed on `p50`, 0 on the Mac. The
demonstrated value is **Mac offload**. The P50 is roughly 140× slower end to end on this workload. No
speedup, no inference throughput and no GPU capacity is claimed or measurable here.

**Two earlier runs are void and are not quoted.** The first reported a queue time of ~1.79 × 10⁹ s
(mixed-clock subtraction, §5.4 D4). The second lost the control plane mid-run and the harness
correctly refused to publish. Neither is cited as a result.

**Open measurement limits, stated rather than hidden.**

- The workload is one allow-listed profile (`python-unittest`) at one revision. It is not a
  representative sample of useful verification work.
- 4 warm observations is not a distribution. The report now says so in words, and prints n beside every
  statistic.
- **Host sampling in that run reported `sample_count: 0`** because `sampler.start()` was never called.
  That is a defect in the harness, not a property of the host, and it is now fixed (§6.5). The numbers
  above therefore carry **no** host-resource evidence.
- All host metrics describe the **Mac only**. The harness explicitly does not sample `p50` and labels
  every metric with the host it describes.

---

## 5. Defects → failing test → fix → evidence → remaining limits

### 5.1 D1/D2 — the node could not stay online (fixed 2026-10-03, live)

- **Symptom.** The node dropped out of the fleet within about a minute of every launch.
- **Two independent causes.** (a) The scheduled task ran as `SYSTEM`, and WSL refuses a local system
  account (`Wsl/WSL_E_LOCAL_SYSTEM_NOT_SUPPORTED`) — so nothing ran at all. (b) The launcher *exited*
  after handing off, which released the WSL2 session; WSL2 then tore down the VM, taking `dockerd` and
  the container with it. A container `--restart` policy cannot help, because the policy lives *inside*
  `dockerd` and `dockerd` is what died.
- **Fix.** Register `OAI2WorkerRuntime` for account `P50` with `LogonType S4U` (no stored password), and
  make the launcher a single attached, never-returning `wsl.exe` invocation driving a resident
  supervisor that holds the session open. The supervision script is passed as a **real file argument**
  (`-e bash /path/worker-supervisor.sh`), never as an inline multi-line `bash -c` here-string, which
  PowerShell split on newlines (that was D2: only the first line ever ran).
- **Evidence.** Task `Running` as `P50`/S4U; node heartbeat 0–1 s over ~9.5 min; jobs 77–81 admitted
  and completed.
- **Remaining limit.** Uptime still depends on one long-lived attached process, now anchored by a
  scheduled task. Time-to-recovery after a fault is **NOT RUN** (A13).

### 5.2 Owner-state handling (fixed in code this round; **NOT ACTIVE**)

Four real defects, all reproduced against the pre-fix script before being fixed:

- **Absent was conflated with damaged.** An empty, whitespace-only or unreadable sentinel each resolved
  to `run`, so a *damaged* file silently granted permission to supervise and then failed later for a
  completely misleading reason. Now **only absence** means `run`; empty, unreadable, non-regular and
  dangling-symlink all fail closed with exit 12 and distinct health states.
- **The parser disagreed with its own header.** The header documented "first non-blank line"; the code
  used `head -n 1`, so a sentinel written with a leading blank line (`\nstop\n`) resolved to `run` —
  the exact inverse of the owner's instruction. Fixed, and the surrounding-whitespace-only
  normalisation is now edge-only (the old `tr -d ' \t'` also deleted *interior* spaces, so `s top` was
  accepted as `stop`).
- **No atomic update path.** Added `write_intent_atomically()` — temp file in the same directory, then
  `rename`, with the rename as the commit point — exposed to the owner as `--set-intent`.
- **Interruption was untestable.** "Fails closed on a partial value" was prose. Added the
  `WORKER_SIMULATE_TORN_INTENT=1` test seam so the case is executable; the exact-match rule is also what
  makes a genuinely truncated value such as `sto` unrecognised.

**Owner interface** (`bash worker-supervisor.sh …`, no docker contact in these two modes):

```
--set-intent run|stop|pause|remove   write the sentinel atomically, exit
--set-intent clear                   remove it; absence means run
--resolve-intent                     print the intent that would be honoured, exit
```

**Evidence.** `SupervisorOwnerStateTests`, 15 tests, all passing. To show the suite is not vacuous, five
defects were re-injected one at a time and every one was caught: empty→run (2 tests), `head -n 1`
parser (1), `tr -d` interior spaces (2), truncated-redirect write (1), removal of the `-r` readability
check (1). The file was restored byte-for-byte after each run (sha256 verified).

**Remaining limits.** All of this was exercised hermetically with stub `docker`/`systemctl`/`wsl` and a
scratch state dir. **None of it has run on the host.** There is no `fsync`, so durability across a
machine-level power loss is explicitly out of scope and the script says so. The two owner-side modes are
placed before any docker call so they cannot start, stop or inspect anything.

### 5.3 Test isolation — including the incident that caused it

**The incident, recorded not softened.** A validation pass that was described as isolated ran the
supervisor with `--container oai2-worker` and intent `stop`. The supervisor honoured it and **stopped
the live production worker container.** The node's return to service was the running supervisor's luck,
not a control. Three concrete holes made that possible: the test PATH *prepended* its stub dir to the
inherited PATH (so the real `docker` and real `qpipe` stayed reachable); the suite's own default target
**was** the production container name; and there was no isolation layer at all.

**What replaced it** — enforced, not documented:

- `assert_isolated_env()` raises `IsolationViolation` (a subclass of `AssertionError`, so it is always a
  *failing test*, never a skip) when any PATH entry lies outside the test's own temp stub dir, when any
  infrastructure binary resolves outside it (checked by `realpath`, so symlinking the real `docker` in
  under another name is caught), when a required stub is missing, or when a live cluster endpoint or
  docker context was inherited.
- It runs in three places: at module setup (before any test body), at cluster construction, and again on
  **every** env handed to a subprocess — after `extra_env` is merged, so a test cannot edit its way past.
- `isolated_target()` refuses the production container name and any state dir outside the test's temp
  root (and the known production dirs by name).
- The single way past either refusal is `OAI2_LIVE_TESTS=1`, which is **off by default** and is not used
  by anything in the module — `FakeCluster` hardcodes `live=False` with no override.
- `rm` is supplied as a suite-written stub bounded to the cluster's own temp root, because on a managed
  machine the ambient `rm` may be a wrapper that refuses to delete outside the workspace; the
  supervisor's `clear` path has to be tested for real.

**Evidence.** 11 isolation tests, all passing. Negative controls: the guard rejects the *old* PATH
construction with 26 distinct violations, rejects a symlink to the real `docker` placed under another
name, and rejects this host's real PATH. Running with `OAI2_LIVE_TESTS=1` fails exactly one test — the
one that asserts the default configuration — and the other 67 still pass and stay isolated, which is
the evidence that the opt-in cannot reach production. *(Verified directly: `OAI2_LIVE_TESTS=1 python3 -m unittest tests.test_worker_lifecycle` → 68 run, 1 failure, and that failure is `test_iso_r8`, the test that asserts the default configuration.)*

### 5.4 Measurement-harness defects

- **Prefix-equality revisions.** `revisions_match()` accepted a bare prefix match, so a diverging
  abbreviated ref could pass. Now a prefix must be ≥ 7 hex characters and both sides must be valid hex
  revision tokens; `gate_rejects_prefix_only_match` and `gate_rejects_wrong_resolved_revision` lock it.
- **The guard rejected the harness's own honest disclaimer.** Recorded because it shows the guard has
  teeth rather than being decorative.
- **Mixed clocks (D4).** The enqueue instant was captured with `time.monotonic()` (seconds since boot)
  while grant/finish came from the cluster's wall clock, and the subtraction was published as a
  duration (~1.79 × 10⁹ s). Now every instant is a `Stamp` tagged with its clock domain;
  `verify_same_clock()` raises `ClockDomainError`, `phase_duration()` reports the phase unavailable with
  `same_clock=False`, and a missing instant is reported unavailable rather than invented. The 86 400 s
  plausibility ceiling is kept and explicitly labelled a **filter, not validation**
  (`plausibility_filter_is_not_validation`).
- **Vacuous matched-work gate.** `cell_a_work_is_matched` compared a local variable with itself. Now the
  remote identity is recovered from the worker's own result evidence — the `checks[]` entry whose
  `profile` matches, its `argv` digested, compared against the pinned local digest, with the node
  required to advertise `verification:<profile>`. Missing `checks[]`, missing `argv`, a profile that ran
  twice, an argv that changed between reps, or an unadvertised capability all **fail closed** as
  `remote identity unproven`.
- **Phase labelled by index.** `classify_phase` took a rep number. It now takes observed state:
  cell A labels cold only if it actually built a checkout; cell B labels from the worker's `evidence[]`
  git steps.
- **Sampler never started.** `sampler.start()` was defined but never called, which is why the live run
  reported `sample_count: 0`. Now started at the top of `measure()` and stopped in a `finally`.
  `never-ran` is distinguished from `unsupported` and from a real count of 0.
- **`ZeroDivisionError` on an all-zero-rate cell** (found by driving the real `measure()` with a stubbed
  cluster) — now a clean exit code.

### 5.5 Probe defects (L1–L5, R1–R7, fixed)

`/proc`-based process discovery (the image has no `ps`/`pkill`), an anchored argv signature with
self-exclusions via a unique marker, pid+starttime identity so a recycled pid cannot pass, mandatory
replacement, a bounded deadline, and acceptance that never depends on `docker logs`. Live
`lifecycle-probe.sh` run: CASE A PASS, CASE B PASS, `RestartCount` stayed 0.

---

## 6. Measurement harness — `scripts/measure_worker_contribution.py`

Stdlib only, no third-party dependency. Corrected in place; it is still one harness.

### 6.1 What it can and cannot answer

| Comparison | Label | Why |
| --- | --- | --- |
| (a) as-is: Mac-local vs Mac+worker, each as it naturally is | **SUPPORTED** | both cells run the identical allow-listed argv, pinned to one sha |
| (b) matched fresh preparation in both cells | **SUPPORTED** | cell A materialises from scratch; cell B's clone is read from the worker's own evidence |
| (c) matched prepared execution (prepare once, execute many) | **NOT SUPPORTED** | the worker makes a disposable clone **per job**; the report states this and leaves it unmeasured rather than approximating it |
| (d) useful verification capacity | **SUPPORTED, with a caveat** | one profile at one revision with `max_leases: 1` is not a representative workload, and the report says so |

### 6.2 Boundaries it respects

It contacts only the cluster control plane (`enqueue-verification`, `/v1/cluster/jobs/<id>`,
`/v1/cluster/events`, `/v1/cluster/nodes`). It does not touch the Mac model server, the coordinator or
the gateway, and sends no foreground Mac inference traffic. `validate_cluster_url()` refuses any port
that looks like a model server, coordinator or gateway; `assert_local_argv_offline()` rejects any cell A
argument containing a URL or a service/model switch.

### 6.3 Gates run before statistics

`judge_record()` is the single source of truth. On failure the process exits non-zero and prints
*"CORRECTNESS GATES FAILED … statistics above are VOID."* A failed gate **voids** the statistics rather
than publishing partial results.

Covered: local exit 0 **and** the prepared checkout's `rev-parse HEAD` matching `--ref`; job success;
`resolved_revision` equal to the pinned sha; `node_id` equal to `--node`; enqueue/poll inside `--timeout`;
the node advertising the pinned profile; failed check exit code / timeout / cancellation; remote argv,
profile, capability and revision identity; and clock-domain agreement.

### 6.4 How it structurally cannot print a generative-performance claim

1. **Fail-closed vocabulary guard.** `assert_verification_scoped()` scans the fully rendered report and
   the serialised JSON for **50** vocabulary classes — hardware names, token-rate units, serving-stack
   names, output-quality terms, rate units, and comparative-performance terms (`throughput`, `speedup`,
   `faster than`, …). A match raises and **exits 1** with nothing printed or written. No allow-list
   escape hatch.
2. **The guard is proven to have teeth.** The self-test plants one probe per class and requires all to be
   rejected, *and* requires a full synthetic report and JSON payload to pass the same guard. It has
   already caught a real violation in this harness (§5.4).
3. **Measurement is structurally incapable of it.** The harness runs one fixed allow-listed argv and
   reads a job record. No code path produces a token count, a hardware utilisation figure or an
   output-quality score. The report's capacity label is fixed at `verification-execution-only` and the
   worker role at `verification`.
4. **Traffic is constrained, not merely intended** (see §6.2).

### 6.5 Sampler honesty

`state()`/`metric_status()` distinguish **`never-ran`**, **`unsupported`** (the platform cannot provide
it) and a real sample count. Every metric carries its host: local figures are labelled
`host: mac-local (this Mac, Darwin arm64)`, and the report prints an explicit line that **no metric for
the p50 host was sampled** and that local figures must never be read as the node's. A short rep with a
1 s interval can legitimately contain 0 samples; that is reported as a real count beside the sampler
state, so it is never confused with the sampler not running.

### 6.6 Validated state (actually executed)

| Check | Result |
| --- | --- |
| `python3 -m py_compile scripts/measure_worker_contribution.py` | **PASS** |
| `--self-test` (with `--repo-url`) | **PASS — 46/46 checks, exit 0** |
| `--help` | **PASS** (exit 0) |
| Self-test under a network + subprocess trap | **PASS** — 0 network calls, 0 spawns |
| End-to-end `measure()` with a stubbed cluster | good path exit 0; mismatched argv, mismatched profile, missing `checks[]` → gate FAIL exit 1; failed checks → gate FAIL exit 1; wrong revision → gate FAIL exit 1 |
| Static lint | **NOT RUN** — no linter is installed on the authoring machine. `py_compile` and the self-test are the only static checks claimed here |

**Not run by the harness's own tests:** any live measurement. §4.2 is a dated observation from a
separately executed run.

---

## 7. Host / runtime configuration

This table separates **what lives in the repository** from **what was changed on the P50 host**. Items
marked *host* are properties of the live machine; a fresh clone does **not** reproduce them.

| Item | Value | Lives in | Changed on host? | Notes |
| --- | --- | --- | --- | --- |
| Node id / role | `p50` · `verification` | host runtime | Yes | verification capacity only |
| Node capabilities | `verification{,:smoke,:pass-probe,:fail-probe,:python-unittest}` | host runtime | Yes | no serving role |
| Node metadata | `max_leases: 1` | host runtime | Yes | one job at a time |
| Container name | `oai2-worker` | `worker-supervisor.sh` | Yes | the supervisor re-asserts this name |
| WSL distro / launcher / user | `Ubuntu-24.04` · `C:\Windows\System32\wsl.exe` · `root` | `start-worker.ps1` | Yes | attached and blocking |
| **Executed supervisor copy** | `/mnt/c/Users/P50/q-pipe/worker-supervisor.sh` | **host filesystem only** | Yes | ⚠️ **not** the repository path. Must be copied to the host and kept in sync. **Currently two revisions stale** (§1.2) |
| Host autostart | `OAI2WorkerRuntime`, account `P50`, `S4U` | `install-worker-task.ps1` | Yes | ✅ **now present** — an earlier version of this document wrongly said autostart was absent. `install-worker-task.ps1` is idempotent, supports `-ShowCurrent` / `-DryRun`, and **refuses to fall back to SYSTEM** |
| `docker` service management | start attempt, then escalation | `worker-supervisor.sh` | Yes | the old "continuing to retry" message was false and was fixed |
| Container re-assert loop | periodic | `worker-supervisor.sh` | Yes | keeps the restart policy meaningful |
| Owner intent sentinel | `/var/lib/worker/supervisor-intent` | `worker-supervisor.sh` | **No** | only the staged revision reads it; it does not exist on the host |
| Worker image | worker only, uid 10001, no credentials | `deploy/worker/Dockerfile` | via build | enforced by A1 |
| Command allow-list | fixed argv per profile | `deploy/worker/profiles.json` | via build | image-baked digest matches the repo |
| Entrypoint contract | requires the three `QPIPE_*` vars at run time; never starts a coordinator | `deploy/worker/entrypoint.sh` | via build | enforced by A1 |
| Acceptance fixture | `test_app.py` | repo | No | the workload the measured jobs ran |
| Lifecycle tests | `tests/test_worker_lifecycle.py` | repo | No | 68 tests, hermetic |
| Measurement harness | `scripts/measure_worker_contribution.py` | repo | No | runs on the Mac |
| This document | `docs/WORKER_EVIDENCE.md` | repo | No | — |

**Node credential.** `QPIPE_CLUSTER_TOKEN` is supplied to the worker at run time by the host environment.
It is deliberately not recorded here, not baked into the image, and not accepted as a CLI argument.
Rotation requires no repository change.

**Local environment discrepancy, unresolved and unowned.** `~/.config/q-pipe/cluster.env` on the Mac
sets `QPIPE_CLUSTER_URL=http://192.168.100.37:10534`, which is **unreachable**; the live control plane
is `http://127.0.0.1:10534` (HTTP 401 without a token, i.e. alive). Every run in this lane therefore
overrode the variable explicitly. The file was **not** edited, because it is shared configuration and
another agent may depend on it. **This needs an owner's decision** — it silently voids any run that
relies on the environment default.

---

## 8. What this document does not claim

- It does **not** claim the hardened supervisor is active. It is staged, and the host copy is stale.
- It does **not** claim `p50` is qualified for inference. **BLOCKED** (#246 / #250, A9).
- It does **not** claim any latency improvement, speedup or throughput gain. The measured end-to-end
  latency is **worse** on the P50; the benefit is offload.
- It does **not** claim cluster-side pause, stop, drain, revoke or duplicate-identity handling. **NOT
  RUN** (A6–A8), and the supervisor does not implement them.
- It does **not** claim host-failure recovery. **NOT RUN** (A13).
- It does **not** claim PowerShell syntax was verified. `pwsh` was unavailable (A17).
- It does **not** treat BLOCKED or NOT RUN as PASS, anywhere, in any roll-up derived from it.
- It does **not** claim the A14/A15 results describe the running system. They describe **staged code**.
