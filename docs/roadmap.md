# Roadmap

Future work only, staged, with measurable exit criteria. **Nothing on this page
is shipped.** The shipped surface is the README and the two LLDs
([delegation-and-guards](lld/delegation-and-guards.md),
[level-2-and-level-3](lld/level-2-and-level-3.md)); the current release state
is in the README's Status section and CHANGELOG.md.

Every item here is an *evaluation* with a stated source, not a promise. The
project's own record says why that is: which routines are accelerated and where
the thresholds sit are measurements, and both should move when a better
measurement exists (CHANGELOG v1.0.0). An evaluation that loses is not a
failure; it is recorded in [wrong.md](wrong.md) and the code is not shipped.

## How a routine enters production

A candidate becomes shipped only after all of these pass, in order:

1. **Gonum differential correctness** — the two-bar suite (bit-exact where the
   guard delegates, relative tolerance where it accelerates) covers the
   routine's full shape space, including the shapes that must delegate.
2. **Fast-path engagement** — a `TestFastPathActuallyRuns`-style test proves
   the accelerated path runs and is not a pass-by-delegation suite.
3. **No regression for guarded shapes** — the minimum-work and minimum-length
   guards keep the routine from being slower than gonum anywhere a caller can
   reach; small sizes keep delegating.
4. **Disassembly** — the built instructions are read before any performance
   conclusion; register pressure, bounds checks and inlining are checked, not
   guessed (see [verification.md](verification.md)).
5. **Performance outside the 8.3% layout floor** — measured on a quiet
   machine, interleaved A/B, minima of several runs; below the floor, judged
   on `perf stat instructions:u,cycles:u` rather than wall clock.

## Stage 0 — release the current state

**Status:** the latest published release is v1.0.1; `main` carries the
`simd v1.20.0` dependency bump, which is not yet tagged as a simdblas release
(README Status).

**Work:** decide the version for the diff between v1.0.1 and main (the
dependency bump; no routine changes), write the CHANGELOG entry, run the full
verification set, tag and publish.

**Exit criteria:** the tag exists and points at a commit where `go test ./...`,
`go test -race ./...` and `go vet ./...` pass; the CHANGELOG entry describes
exactly what changed since v1.0.1.

## Stage 1 — routine evaluations

Each item is an experiment on a branch: prototype, measure, then either keep
with tests or delete and record the measurement in [wrong.md](wrong.md). The
implementation plan is [docs/plans/2026-08-13-simdblas-production.md](plans/2026-08-13-simdblas-production.md).

### E1. Blocked `syrk` that only touches the triangle

**Source:** trmm.go:188-196 — the shipped syrk computes the whole product and
keeps half, and the code says "a blocked form that only touches the triangle
would be better and is not written; this is the version that could be measured
today."

**Evaluation:** prototype a blocked syrk that skips the implied half, against
the full-product path, at sizes straddling the current `2*(1<<15)` guard and
inside decompositions (`mat.Cholesky`). The full product does 2× the arithmetic
a triangle-only routine needs; the question is whether the block bookkeeping
repays skipping the other half.

**Loses if:** no win outside the 8.3% floor anywhere in the size sweep, or
Cholesky does not move; then the prototype is deleted and the measurement
recorded.

### E2. Widening the direct `gemm`/`gemv` fast paths

**Source:** level23.go:18-33 — the direct gemm path is deliberately narrow
("no transpose, alpha of 1, beta of 0 and natural leading dimensions"), and
the comment says widening it — a transposed B, a non-zero beta — "is worth
doing against measurements rather than in advance, since each variant needs
either another kernel or a scratch buffer".

**Evaluation:** measure the direct path's gap to the packed general path for
transposed, scaled and windowed shapes at the sizes `mat.Mul` and LAPACK
produce. If packing already covers the shapes without losing, the evaluation
is a measurement that records why the narrow guard is right.

**Loses if:** the packed path is within noise for every shape tried; the
finding is recorded and the guard stays.

### E3. Complex support

**Source:** README — "everything complex" is unaccelerated and "a larger job
than it looks"; the banded and packed storage variants "have not been
attempted; gonum's `mat` rarely produces the first".

**Evaluation:** first establish whether simd v1.20.0 has complex kernels at
all. If not, the evaluation ends with that finding recorded — adding kernels
upstream is a different project. If kernels exist, prototype the complex
Level 1 reductions against gonum's, with the same two-bar and engagement
tests.

**Loses if:** no kernels exist upstream, or the differential bars cannot be
met; recorded, not shipped.

### E4. Banded and packed storage variants

**Source:** README — not attempted; gonum's `mat` rarely produces them.

**Evaluation:** measure how often gonum's `mat` actually routes banded or
packed calls into BLAS, and whether the shapes (non-unit strides, asymmetric
lengths) match what the Level 1 guards can take. The likely outcome is a
recorded measurement that they stay delegated.

**Loses if:** the call frequency or the win does not justify the paths;
recorded.

### E5. `trsv`

**Source:** README — "a triangular solve is sequential — `x[i]` depends on
`x[0..i-1]` — so it is a scan rather than a map, and Level 2 has no work to
amortise a blocked algorithm against."

**Evaluation:** confirm the sequential argument on hardware — a strided solve
against gonum's at the sizes LAPACK uses. Expected to lose; the deliverable is
the measurement behind the README claim, or a surprise if there is one.

**Loses if:** expected; the measurement is recorded in [wrong.md](wrong.md)
and the README claim stands.

### E6. Blocked transpose packing

**Source:** gemm.go:72-75 — packing a transposed operand walks down a column,
the cache-hostile direction; the code notes "blocking it would help, and it
has not been measured as mattering against the multiply that follows — worth
revisiting if a profile ever says otherwise."

**Evaluation:** a profile of a transposed-heavy workload (a decomposition that
reads Aᵀ), then a blocked transpose pack if the profile supports it.

**Loses if:** the profile shows the packing is still noise against the
multiply; the finding is recorded.

## Stage 2 — cross-architecture measurement

**Source:** README Status and CONTRIBUTING — everything was measured on one
Zen 5 amd64 machine; the thresholds are measured constants; simd v1.20.0 has
real-hardware correctness coverage on amd64 and arm64 NEON and emulated
coverage elsewhere, which "does not establish simdblas performance there";
reports from other architectures are "the contribution it most needs".

**Work:** arm64 NEON first (real-hardware correctness already exists there),
then the emulated targets if a runner can be found. Deliverable is a
measurement report against the README table, with threshold adjustments where
the constants are wrong for the machine.

**Exit criteria:** a committed report; each threshold that moved carries its
measurement next to it; every result is either in the README or in
[wrong.md](wrong.md) with a reason.

## Closed doors

`symv` and `trmv` were written, measured and deleted — 0.29× at n=192 and
0.11× at n=1024 against gonum, because the densify trick that pays for Level 3
(n³ of work on n² of data) is the same order as the work it prepares for
Level 2 (n² on n²). They are not re-opened without a new argument, and the
measurement is in [wrong.md](wrong.md). `trsv` is an open evaluation (E5) but
expected to land next to them.
