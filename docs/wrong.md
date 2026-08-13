# wrong.md — decisions this repository did not take

Measurements that argued against a change, and corrections to numbers that
were published wrong. This file is **append-only decision evidence**: a
finding that cost a measurement belongs here whether or not any code changed,
and an entry here is what stops the next person from re-running the experiment
or re-believing the wrong number.

Policy: append new entries at the bottom; never edit or delete old ones. If a
re-measurement overturns an entry, add a new entry that says so and links to
the old one. Never invent a measurement — an entry without a source (README,
CHANGELOG, a commit, or a code comment) is worse than no entry, because it is
evidence with no provenance.

## 2026-08-01 — `symv` and `trmv` were written, measured and deleted

The densify trick that makes `symm` 18× was applied to the symmetric and
triangular Level 2 matrix-vector routines. Level 3 does n³ of arithmetic on n²
of data, so filling in the implied half of a matrix is free next to the
multiply; Level 2 does n² on n², so the densify is the *same order* as the
work it is preparing — and it replaces gonum reading n²/2 stored elements with
n² of branchy copying plus a full matrix-vector multiply.

Measured against gonum: **0.29× at n=192 and 0.11× at n=1024** — the
accelerated version was slower than delegating. Both variants were deleted and
never shipped. The README and CHANGELOG v1.0.0 record the numbers; the commit
(d42fb41) records the deletion. `trsv` was not attempted for the same reason —
a triangular solve is sequential, a scan rather than a map.

## 2026-08-01 — a busy machine published 2.27× where the quiet one says 4.81×

An early draft of the README numbers was measured while a docker
cross-compilation lane was using all 32 threads. It reported `Dgemm` at 512 as
**2.27×**; the same measurement on an idle machine is **4.81×**. Benchmark
noise is one-sided — interference only makes a run slower — which is why the
project takes the minimum of several runs on a machine with load average under
1 (README, "Measuring").

## 2026-08-01 — v1.0.1 re-measurement corrected both ways

The first genuinely idle full run (load average under 1 at the start of each
of two runs, against 29 for the worst of the earlier ones) corrected the
release table; the two runs behind each figure now agree to within 2%, which
they did not before (commit a4346ea, CHANGELOG v1.0.1):

| row | published | corrected |
|---|---|---|
| least squares 5000×200 | 1.42× | 1.52× (understated) |
| covariance + Cholesky 5000×300 | 4.43× | 4.36× (overstated) |
| `Dgemm` 512 | 4.8× | 4.2× |
| `Dgemv` 1024 | 2.3× | 2.5× |

## 2026-08-01 — serial `gemv` kernel measured 0.90× and was replaced

`Dgemv` with simd.go's serial kernel measured **0.90×** against gonum at
n=1024 — ten percent slower — on two independent runs. A single earlier run
had said 1.35× the other way, and was not believed: two agreeing runs are the
only reason the 0.90× was believed. The call now goes to
`simd.GemvParallelInto`. The measurement and the decision are next to the
guard in level23.go:75-81.

## 2026-08-01 — serial `gemm` kernel was three times slower at n=512

gonum's `Dgemm` fans out over `GOMAXPROCS` above a size threshold. Against a
single-threaded kernel this made gemm **three times slower at n=512** — a
faster inner loop losing to more cores. `Dgemm` uses `MatMulParallelInto`,
which divides by output row and stays bit-identical to the serial kernel
(level23.go:9-16).

## 2026-08-01 — Level 1 lost to gonum's SSE2 below ~64 elements

The kernels behind Level 1 fall back to a plain Go loop under their own
threshold — right for a general-purpose library, wrong here, because gonum's
Level 1 is hand-written SSE2 rather than a Go loop. Falling back to portable
code handed gonum the win: `dot` **0.46×** at n=8, `axpy` **0.62×**, `asum`
**0.52×**, `scal` **0.55×** (the full table is in the README). Each routine now
has a minimum length set at the first size that is clearly ahead rather than
at break-even — dot 64, axpy 48, scal 48, swap 48, rot 48, asum 16, nrm2 8 —
and below it the call is delegated, which costs one extra method call (0.4 to
0.7 ns, measured) and is the floor on how much slower this package can be for
any call (CHANGELOG v0.4.0, level1.go:19-49).

## 2026-08-01 — the nrm2 magnitude pre-check was measured and rejected

Checking whether a vector needs the scaled algorithm *up front* costs a whole
extra pass: **483 µs against 54 µs** for a million elements — most of the win,
spent to buy an answer that is almost always already correct. The shipped
design instead runs the fast path and detects its own failure (a sum of
squares of +Inf, or below the smallest normal), then delegates to gonum's
scaled algorithm; the common path pays one comparison (level1.go:84-110).

## 2026-08-01 — routing every LU `ger` call through the per-row path lost 12%

Before `gerRowMinLen` existed, every `ger` call an LU makes — one per column,
with a row length that shrinks by one each time — went through the per-row
path. That made a 256-square factorisation **12% slower** than leaving it
alone, because the per-row path wins only the width of the inner loop and has
to cover one dispatch per row first. The row-length threshold (256) delegates
the short calls (level1_blas.go:90-110).

## 2026-08-01 — `trmm` Right/Lower was wrong while every left-hand case passed

The blocked trmm used the left-hand convention for which side of the diagonal
the off-diagonal span lies on. That made Right/Lower produce wrong answers
while every left-hand combination — and Right/Upper — passed. The fix inverts
the convention for the right side, and the code says so next to it
(trmm.go:75-87). The lesson, recorded there: a case that only a subset of the
eight side/triangle/transpose combinations exercises will not be caught by the
rest.

## 2026-08-01 — a whole workload was 0.73× and no routine was slow

An ordinary least-squares fit on a 2000×50 matrix came out at **0.73×** — no
single routine was slow, and the combination was: applying a QR factor to one
right-hand side multiplies 2000×50 by 50×1, 100,000 multiply-accumulates
against 102,050 elements to copy. The packing was more expensive than the
arithmetic it was preparing, and it passed a threshold that only counted mnk.
The fix was a guard on arithmetic per element copied (intensity ≥ 8), plus
skipping the redundant copies of operands already in the kernel's layout; the
workload went to 1.00× (CHANGELOG v0.5.0, gemm.go:87-118).

## 2026-08-01 — an apparent 1.00× that is not a finding

`syr` and `syr2` measured 1.00× at n=64. That is not a measurement of the
routines: 64 is below `syrMinLen` (128), so those calls were delegating, which
is the intended behaviour. Recorded here so nobody re-runs the experiment
expecting a finding (commit d42fb41).
