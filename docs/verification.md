# Verification

How every claim in this repository is checked: the exact commands, what each
one is for, what counts as a number, and what happens when a number is smaller
than the machine can say. This is the checklist a change must pass before it is
claimed done, and the checklist CI runs.

## The gates

```
go test ./...
go test -race ./...
go vet ./...
```

`go test ./...` is the differential suite: every override is checked against
the gonum routine it replaces, at two bars. `go test -race ./...` covers the
concurrent kernels — gemm and gemv run concurrently above a threshold.
`go vet ./...` is the static pass.

CI (`.github/workflows/ci.yml`) runs, in order, on every push to main and every
pull request:

1. `go vet ./...`
2. `go test ./...`
3. `go test -race ./...`
4. `go run github.com/sebishogun/simd/cmd/simdinfo`

The last step reports the simd tier selected on the runner; it is how a CI
result can be tied to an instruction set.

## The differential oracle

Two bars, on purpose (diff_test.go:12-21, CONTRIBUTING):

- **Where the fast path does not apply, this package delegates** — so the
  answer must be gonum's *exactly, to the bit*. Anything else means a guard is
  wrong and a fast path ran when it should not have. `closeTo`/`closeSlice`
  compare raw bits here.
- **Where it does apply**, the bar is a relative tolerance (`tol = 1e-12` in
  the main suite), because the accumulation orders differ legitimately — a
  fixed sixteen-accumulator tree here, gonum's four-way unrolled loop.
  NaN-vs-NaN agreement is part of the check.

Invalid arguments are delegated, and `TestInvalidArgumentsStillPanic` asserts
the panic is gonum's own, verbatim. The symmetric suites add their own bars
(1e-10 to 1e-12, and 1e-4 for float32) and `TestSymmetricRoutinesLeaveOtherTriangleAlone`
asserts the BLAS contract that the untargeted triangle is never written.

## Fast-path engagement

A suite that passes by delegating everything looks identical to one that
works, so:

- `TestFastPathActuallyRuns` requires `Ddot` and `Dnrm2` **not** to return
  gonum's exact bits on the fast path — the two accumulate differently, so a
  match means the accelerated path stopped running and the differential tests
  went vacuous (fastpath_test.go:10-41).
- `Dgemm` cannot be checked that way: the blocked kernel and gonum's both
  accumulate sequentially over `k` and agree bit-for-bit at every size tried,
  so its guards are tested directly — `TestGuards` (gemm, gemv, Level 1) and
  `TestGerGuard` assert the contiguous case accelerates and every deviation
  delegates (fastpath_test.go:66-138, diff_test.go:266-283).

## Docs checks (part of `go test ./...`)

- `TestReadmeThresholdsMatchConstants` parses the README's bold threshold list
  and compares it against the constants in level1.go — the facts written in
  prose are machine-gated against the code so they cannot drift.
- `TestDocLinksResolve` checks every local link in its explicit document list
  — README.md, CONTRIBUTING.md, docs/guide.md and CHANGELOG.md — points at a
  file that exists. Links inside docs/architecture.md, the LLDs, the roadmap
  and the plans are not covered by the test and must be checked by hand; a
  Markdown-only change that adds or moves a document should verify them
  manually before committing.
- `TestPackageDocNamesTheEntryPoint` keeps the package doc naming
  `blas64.Use` and `Implementation`.
- `TestSatisfiesBLAS` fails the build if the embedding ever stops satisfying
  all four BLAS interfaces.

## The benchmark harness and its limits

The committed harness (`bench_test.go`) covers exactly five routines — `Ddot`,
`Daxpy`, `Dnrm2`, `Dgemm`, `Dgemv` — both implementations in the same process
on the same inputs, at fixed sizes:

- vector routines: 1,000 / 100,000 / 1,000,000 elements;
- `Dgemm`: 64 / 128 / 256 / 512;
- `Dgemv`: 256 / 1024 / 2048.

The README's symmetric, decomposition and whole-workload rows are
**release-time measurements, not continuous gates**: no committed benchmark
covers `symm`, `syr2k`, `syr`, `trsm`, `trmm`, `syrk`, `syr2`, `ger`, or the
decompositions. A number for one of those is a measurement made under the
rules below, not a `go test` assertion.

Command:

```
go test -run '^$' -bench . -count 6 .
```

## What counts as a number

- **Quiet machine.** Load average under 1 before each run; a busy machine's
  numbers are worse than none (the README's own table was once measured
  against a 32-thread docker lane — see [wrong.md](wrong.md)).
- **Minimum of runs, not the mean.** Benchmark noise is one-sided —
  interference only makes a run slower — so the fastest run is the least
  interfered with. `-count 6` at least, and the comparison run twice, reporting
  the worse of the two runs.
- **Interleaved A/B.** Both implementations in the same process on the same
  inputs; comparing two builds compares two builds.
- **The 8.3% floor.** Code-layout noise here is 8.3%; anything smaller cannot
  be told from nothing by wall clock, and more samples do not help — layout
  noise is per-build, not per-run. For changes expected to be worth less than
  that, compare instructions and cycles instead:

  ```
  perf stat -e instructions:u,cycles:u ./bench
  ```

  which are layout-independent — and read the disassembly, which is the only
  thing that explains why.
- **Disassemble first.** Before proposing a cause for anything slow, before
  writing a variant, before reading a profile delta, build and read the
  instructions:

  ```
  go test -c -o /tmp/x.test .
  go tool objdump -s 'pkg\.functionName' /tmp/x.test
  ```

  Register pressure (a spilled loop counter costs 18% and no counter reports
  it), bounds-check elimination, index multiplies, inlining and
  `append`-becomes-`memmove` only show up there. A/B builds must be
  interleaved in one session and compared on the minimum, never across
  sessions.
- **Bare gates.** Never pipe a gate through `tail` or anything else without
  `set -o pipefail` — the pipe reports the last command's status and a failure
  vanishes. Run gates bare.

## Cross-architecture and release matrix

- Performance is claimed for **amd64 only**; the README says so, and reports
  from other architectures are the contribution the project most needs
  (CONTRIBUTING).
- Correctness: simd v1.20.0 has real-hardware coverage on amd64 and arm64 NEON
  and emulated coverage for its other targets. `TestAnswerDoesNotDependOnTier`
  asserts repeatability here; cross-tier agreement is proven in the simd
  repository.
- A release requires: the full gate set green at the tag commit, the CHANGELOG
  updated with what changed, the README's release rows re-measured (they are
  the shipped record), and the threshold/prose facts machine-checked by the
  docs tests.

## Release checks

Before tagging:

1. `git diff v<last>..HEAD --stat` — confirm what actually changed.
2. Gates green at the tag commit (the list at the top of this file).
3. CHANGELOG entry describes the diff; README updated if any number or
   threshold moved, with the measurement beside it.
4. `git diff --check` clean.
5. Tag pushed; the version in the README Status section matches the tag.
