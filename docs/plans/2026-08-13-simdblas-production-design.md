# simdblas production design record

2026-08-13. This file records the **approved production direction** — the one
the project has been shipping since v1.0.0 and through v1.0.1 — and the
governance around it, so that future work has a single place to check what was
decided and why. It is a record, not a proposal: the shipped behaviour is
current v1 behaviour, and where a future evaluation could change something
that is stated explicitly as an evaluation, never as a plan to ship.

Companion document: [2026-08-13-simdblas-production.md](2026-08-13-simdblas-production.md)
(the implementation plan for future work) and [../roadmap.md](../roadmap.md)
(the evaluated, not-promised list of future work).

## Current shipped boundary

- One exported type: `Implementation struct{ gonum.Implementation }`
  (simdblas.go:74). The API is stable; v1.0.0 committed that it stays an
  embedding.
- Latest published release: **v1.0.1**. `main` carries the `simd v1.20.0`
  dependency bump, not yet tagged as a simdblas release (README Status).
- Accelerated in both precisions: Level 1 `dot`, `nrm2`, `asum`, `axpy`,
  `scal`, `swap`, `rot`; Level 2 `gemv`, `ger`, `syr`, `syr2`; Level 3 `gemm`,
  `symm`, `trsm`, `trmm`, `syrk`, `syr2k`. Everything else is gonum's own
  implementation, reached by delegation (CHANGELOG v1.0.0, the LLDs).
- No cgo. Kernels are compiled ahead of time and committed as assembly.

## Goals

- Make everything above BLAS — gonum's `mat`, `stat`, `optimize` — faster by
  swapping one process-global backend, with no change to callers.
- Keep correctness absolute: a call this package does not accelerate *is* a
  gonum call, panics included.
- Keep results deterministic within a version and identical across
  architectures, by fixing the accumulation order instead of following the
  vector width.
- Keep the win measurable: nothing ships without a measurement behind it.

## Non-goals

- Accelerating every BLAS routine. The unaccelerated set is a decision with a
  measurement or an argument behind it, not an oversight (README, CHANGELOG
  v1.0.0).
- Bit-exact agreement with gonum on accelerated paths, or across versions —
  BLAS does not specify bit-exact results, and a call may move between the
  accelerated and delegated paths across versions, which accumulate
  differently (CHANGELOG v1.0.0).
- Performance claims off amd64. Everything in the README was measured on one
  Zen 5 machine; the thresholds are measured constants.
- Changing the caller's BLAS from a library. Registration is the
  application's decision (`blas64.Use`/`blas32.Use` at startup).

## Architecture decisions

1. **Embed, don't reimplement.** `Implementation` embeds `gonum.Implementation`
   so every method exists and is correct from the first line; each accelerated
   routine is an override of a working method, and the 60-odd routines nobody
   has measured a win for stay gonum's (simdblas.go:62-74).
2. **Guard, then delegate.** Every override checks a pure guard predicate and
   hands anything outside the fast path — including invalid arguments — to the
   embedded method. Panics a caller relies on come from gonum and read exactly
   as they always have (level1.go:13-17).
3. **Pack rather than special-case.** The general gemm path copies operands
   into pooled contiguous scratch, applying transposes on the way in, and
   scales and accumulates back row-wise; operands already in the kernel's
   layout are used where they lie (gemm.go:119-193).
4. **Block and densify at Level 3.** `trsm`/`trmm` walk diagonal blocks of
   size 64, delegating the small triangular pieces to gonum and doing the
   rectangular updates with accelerated gemm; `trmm` below the block size
   densifies the triangle instead; `symm`/`syrk`/`syr2k` densify or compute a
   full product and fold the requested triangle back. Densifying is the same
   trade as packing and is paid for at the same sizes (the LLDs).
5. **Parallel kernels where gonum is parallel.** `gemm` and `gemv` use simd's
   parallel kernels, which divide by output row and stay bit-identical to the
   serial kernels (level23.go:9-16).
6. **Pooled scratch.** The general paths allocate nothing per call; buffers
   come from `sync.Pool` and belong to the call (gemm.go:31-54).

## Compatibility decisions

- Delegation defines compatibility: whatever gonum does, simdblas does,
  because the call is gonum's. Tested at two bars — bit-exact where delegated,
  relative tolerance (1e-12) where accelerated — plus
  `TestInvalidArgumentsStillPanic` for the panic text and `TestSatisfiesBLAS`
  for the interfaces (diff_test.go, iface_test.go).
- Fast-path engagement is tested (`TestFastPathActuallyRuns`), so a suite
  that passes by delegating everything cannot look like one that works; where
  engagement cannot be distinguished by results (`gemm`), the guards are
  tested directly (fastpath_test.go).

## Numerical decisions

- Fixed accumulation order (sixteen-accumulator tree) for determinism within
  a version and across architectures; cross-tier agreement is supplied by simd
  v1.20.0.
- `nrm2` is the one numerical fallback: BLAS defines it to avoid spurious
  overflow and underflow, so an overflowing or subnormal sum of squares
  delegates to gonum's scaled algorithm; the common path pays one comparison
  (level1.go:75-133).
- `beta == 0` never reads C, per BLAS (gemm.go:171-192).
- Thresholds and the accelerated set are measurements, not promises; they may
  move in a minor release (CHANGELOG v1.0.0).

## Ownership and concurrency decisions

- Registration is caller-owned and process-global: `blas32` and `blas64` are
  separate backends and both must be registered for both precisions.
- Accelerated routines must not mutate anything beyond the BLAS contract: the
  untargeted triangle of symmetric/rank-k routines comes out exactly as it
  went in, and the race suite is a CI gate because gemm and gemv run
  concurrently above a threshold.

## Performance decisions

- A number is only a number on a quiet machine (load average under 1), taken
  as the minimum of several runs (`-count 6`), with the comparison run twice
  and the worse of the two runs reported, interleaved in one session — never
  across sessions.
- The code-layout noise floor is 8.3%. Below it, wall clock cannot distinguish
  a change; the verdict comes from `perf stat instructions:u,cycles:u` and the
  disassembly. Disassembly comes before any performance hypothesis
  (verification.md).
- A regression is defined as any shipped shape being slower than gonum; the
  guards exist so that a call that would lose delegates instead.

## Release decisions

- v1.0.0 committed the API (one exported type, an embedding) and the
  delegation contract; v1.0.1 re-measured every number on an idle machine and
  corrected the record both ways (CHANGELOG v1.0.1).
- A minor release may gain a fast path or move a threshold; the README and
  CHANGELOG are part of the release, and the threshold prose is
  machine-checked against the constants (`TestReadmeThresholdsMatchConstants`).
- Release gates: `go test ./...`, `go test -race ./...`, `go vet ./...`,
  `git diff --check`, docs tests green at the tag commit, CHANGELOG updated
  (verification.md).

## Current v1 behaviour versus future evaluations

Everything above is what v1 ships today. The items in [../roadmap.md](../roadmap.md)
— a blocked `syrk`, wider direct `gemm`/`gemv` paths, complex and banded/packed
support, `trsv`, a cross-architecture measurement programme, and releasing the
untagged `simd v1.20.0` state — are evaluations with exit criteria, not
commitments, and a routine enters production only through the five-gate bar
stated there. An evaluation that loses is recorded in [../wrong.md](../wrong.md)
and does not ship.
