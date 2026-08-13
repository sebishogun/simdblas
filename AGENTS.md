# AGENTS.md — working on simdblas

Instructions for coding agents. Read the required documents before changing
anything; this file is the summary, the documents are the detail.

## Product and boundary

simdblas is a BLAS implementation for gonum, backed by
github.com/sebishogun/simd. **No cgo** — the kernels are compiled ahead of
time and committed as assembly.

- The whole exported surface is one type: `Implementation struct{
  gonum.Implementation }`. The four BLAS interfaces are satisfied by
  construction (tested by `TestSatisfiesBLAS`).
- Callers install it at startup with `blas64.Use(simdblas.Implementation{})`
  and `blas32.Use(...)`. Registration is **caller-owned and process-global**:
  a library must never change a caller's BLAS, and both precisions are
  separate backends.
- **Delegation defines compatibility.** Every override guards its fast path
  and hands everything else — wrong shape, too small, invalid arguments — to
  the embedded gonum method. Delegated calls are gonum's own code, panics
  included. Invalid arguments must never be rejected here.
- Non-goals: accelerating everything; bit-exact agreement with gonum on
  accelerated paths; last-bit reproducibility across versions; performance
  claims off amd64; changing a caller's global BLAS from a library.

## Current status

- Latest published release: **v1.0.1**. `main` carries the `simd v1.20.0`
  dependency bump, not yet tagged as a simdblas release.
- Which routines are accelerated and where the thresholds sit are
  **measurements, not promises**, and may move in a minor release. Within a
  version results are deterministic and identical on every architecture.
- The roadmap is **not shipped**. Nothing in docs/roadmap.md or
  docs/plans/ exists in the code until it has passed the production bar
  (below). Do not document planned routines as if they shipped.

## Required reading, in order

1. `README.md` — the shipped front page, numbers, status.
2. `docs/guide.md` — usage, what gets faster, measuring.
3. `docs/architecture.md` — package boundary, data flow, promises.
4. `docs/lld/delegation-and-guards.md` — every accelerated routine, guard and
   threshold.
5. `docs/lld/level-2-and-level-3.md` — how the matrix paths are built.
6. `docs/verification.md` — the gates and what counts as a number.
7. `docs/roadmap.md` — evaluated future work, not promises.
8. `docs/wrong.md` — measurements that argued against changes.
9. `CHANGELOG.md` — release history and the v1.0 promise.
10. `CONTRIBUTING.md` — the bar for a change.
11. `docs/plans/*` — before doing any planned work.

## Rules

- **Disassemble first.** Before proposing a cause for anything slow, before
  writing a variant, before reading a profile delta:

  ```
  go test -c -o /tmp/x.test .
  go tool objdump -s 'simdblas\.FunctionName' /tmp/x.test
  ```

  Register spills, missed bounds-check elimination, index multiplies,
  inlining and `memmove` calls only show up there.
- **The 8.3% floor.** Code-layout noise here is 8.3%; anything smaller cannot
  be told from nothing by wall clock, and more samples do not help. Below the
  floor, judge with `perf stat -e instructions:u,cycles:u` (layout-independent)
  and the disassembly.
- **Quiet machine, interleaved A/B, minima.** Load average under 1.
  `-count 6`, minimum of runs, comparison run twice, worse of two runs
  reported, both implementations in one process on the same inputs — never
  across sessions.
- **Bare gates.** Never pipe a gate through `tail` (or anything else) without
  `set -o pipefail` — the pipe launders failures into green exits. Run gates
  bare: `go test ./... && go test -race ./... && go vet ./... && git diff --check`.
- **Numerical.** Accelerated answers may differ from gonum's in the last bits
  (fixed accumulation order); delegated answers must be gonum's bits exactly.
  `nrm2` is the one numerical fallback (overflow/subnormal → gonum's scaled
  algorithm). The untargeted triangle of symmetric/rank-k routines must come
  out exactly as it went in.
- **Concurrency.** gemm and gemv use simd's parallel kernels; scratch comes
  from `sync.Pool` and never escapes the call. The race suite is a CI gate.
- **New accelerated routine:** guard + delegate (invalid arguments included),
  differential tests at both bars, a `TestFastPathActuallyRuns`-style
  engagement test (or direct guard tests where results cannot distinguish —
  see `TestGuards`), minimum-work/length set at the first clearly-ahead size,
  disassembly, then measurement.
- **Production bar:** gonum differential correctness, fast-path engagement,
  no regression for guarded shapes, disassembly, performance outside the 8.3%
  floor.
- **wrong.md policy.** Measurements that argue against a change belong in
  `docs/wrong.md`, append-only, whether or not any code changed; never invent
  a measurement; never edit or delete an old entry — add a new one.
- **Docs-only changes** touch only `.md` files: never Go source, tests,
  `go.mod`/`go.sum`, workflows, or benchmark artifacts in the same change.
- **Verification before claiming done.** Run the gates, confirm output, then
  state the outcome. Never claim green without running it.
