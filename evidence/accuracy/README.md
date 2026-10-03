# Accuracy evidence

Machine-checkable accuracy baselines, tied to an exact source SHA and a
proven serving identity.

## Why this directory exists

Issue #240 carried an accuracy figure of **`0/12`** attributed to
`evidence/latency-20261003/summary.json`. That path has never existed in
this repository — there was no `evidence/` directory at all. The figure was
therefore not reproducible and not refutable by anyone, which is exactly the
defect the #240 audit raised. A correctness number that cannot be re-derived
is not evidence.

**That `0/12` is superseded.** The re-measured, traceable baseline is
**4/12** at `a1337442df2f60294cfa31a8d5ef569488c8b752`.

Do not compare the two numbers as if they were the same measurement. They
are not: the old one has no task set, no oracle and no artifact behind it,
so there is no way to know which tasks it contained or how they were scored.
The new one is reproducible by anyone on this repository.

## What makes a record here admissible

| Property | How it is enforced |
| --- | --- |
| Tied to source | `harness.commit` + `harness.dirty` in every record |
| Tied to actual weights | Serving identity read from `/props`, never from a CLI flag |
| Decidable | Regex oracle in `oai2/evals/deterministic_hard.py`; no model or human grades the output |
| Oracle is correct | `test_deterministic_hard_oracles_are_correct` re-derives all 12 answers |
| Reproducible | `temperature=0.0` + fixed seed; raw output stored verbatim |
| Auditable | Raw model text and computed failure class stored per case |

A record missing any of these is not admissible as a correctness claim,
regardless of the score it reports.

## `deterministic_hard_12_a133744.json`

Suite `deterministic_hard_12`, 12 cases, scorer `regex_or`.

| | |
| --- | --- |
| Harness | `a1337442df2f60294cfa31a8d5ef569488c8b752` (`dirty: false`) |
| Serving | `SmolLM2-1.7B-Instruct-Q4_K_M.gguf`, 4 slots × 8192 ctx |
| Sampling | `temperature=0.0`, `seed=20261004`, `max_tokens=64` |
| Result | **4 / 12 verified-correct** (0.3333) |

Reproducibility was checked directly: two consecutive runs produced
**12/12 byte-identical raw outputs** and the same 4/12.

### What passed

`days_in_a_week`, `count_letter_in_word`, `geometric_sequence`,
`leap_year_length` — short factual or single-step items.

### What failed, and how

The taxonomy is computed from the response shape, not hand-written, so it
cannot be adjusted to suit a result.

| Class | Count | Cases |
| --- | ---: | --- |
| `arithmetic_error` | 5 | `arithmetic_multistep`, `modular_arithmetic`, `unit_conversion`, `rate_word_problem`, `distinct_letter_count` |
| `fact_error` | 1 | `capital_city` — answered *Sydney* (largest city, not the capital) |
| `logic_error` | 1 | `boolean_logic` — answered *False*, i.e. stopped after the `AND` and ignored the `OR` |
| `symbolic_corruption` | 1 | `string_reversal` — answered `drocorb`, which contains a `b` that does not appear in `orchords` at all |

Two of these are worth separating from ordinary wrongness:

- **`rate_word_problem` answered 120**, which is `60 × 2` — the model
  inverted the operation instead of dividing by 1.5.
- **`string_reversal` is not a permutation of its input.** A near-miss
  reversal would be a formatting or attention failure; emitting a character
  that was never in the source is generative. The classifier separates
  `wrong_permutation` from `symbolic_corruption` for exactly this reason.

### How to read this number

4/12 is a **correctness** result, not a throughput result, and the two must
not be traded against each other. It says the NORMAL lane's resident model
handles single-step recall reliably and multi-step execution unreliably at
this size. Any future throughput claim should be read against this baseline:
speed on a task the model cannot do is not an improvement.
