# LLD: delegation and guards

The exact shipped surface: every accelerated routine, its guard predicate, its
thresholds, and what happens when the guard fails. Sourced from the code; the
README and [guide](../guide.md) carry the crossover *measurements*, which this
file does not repeat. The constants here are the current facts and the source
of truth is the code (level1.go, level1_blas.go, level23.go, gemm.go,
symmetric.go, trsm.go, trmm.go, level3_f32.go).

## The delegation contract

A guard failure always delegates to the embedded `gonum.Implementation` method
with the original arguments. Three consequences, all tested:

- **Invalid arguments are delegated, never rejected here.** Gonum validates and
  panics with the message callers already rely on. Reproducing someone else's
  panic text is a fast path that will get it wrong eventually (level1.go:13-17,
  `TestInvalidArgumentsStillPanic`).
- **Where delegation applies, the answer must be gonum's exactly, to the bit.**
  Anything else means a guard is wrong (diff_test.go).
- **Delegation costs one extra method call** — measured at +0.4 to +0.7 ns on a
  3-6 ns operation. That is the floor on how much slower this package can be
  for any call, and it only applies below the thresholds (README).

## Level 1: unit strides and minimum lengths

The guard is `unit` (one vector: `incX == 1 && n >= 0 && len(x) >= n`) or
`unit2` (both vectors: `incX == 1 && incY == 1 && n >= 0` and both lengths),
plus a minimum length per routine (level1.go:41-49, 51-59):

| routine | minimum length | constant | kernel |
|---|---|---|---|
| `dot` | 64 | `minLenDot` | `simd.Dot` |
| `nrm2` | 8 | `minLenNrm2` | `simd.SumSquares` + `math.Sqrt` (see below) |
| `asum` | 16 | `minLenAsum` | `simd.L1Norm` |
| `axpy` | 48 | `minLenAxpy` | `simd.AddScaled` |
| `scal` | 48 | `minLenScal` | `simd.Scale` |
| `swap` | 48 | `minLenSwap` | `simd.Swap` |
| `rot` | 48 | `minLenRot` | `simd.Rotate` |

Every minimum is set at the first size that is clearly ahead of gonum rather
than at break-even, because gonum's Level 1 is hand-written SSE2 and beats a
general-purpose kernel's own scalar fallback at small sizes. Below the minimum
the call is delegated, which is strictly better than running a scalar loop:
same answer, gonum's assembly instead (level1.go:19-40, README).

Strided calls delegate. So do calls where `n` exceeds the slice length or `n`
is negative — the guard doubles as the validity check.

### `nrm2` is the one numerical fallback

BLAS defines `nrm2` to avoid spurious overflow and underflow; the obvious
implementation does not. The fast path runs and its own failure is detected:
if the sum of squares is `+Inf` (overflow) or below the smallest normal value
(`minNormalF64` = 0x1p-1022 for float64, `minNormalF32` = 0x1p-126 for
float32, level1.go:131-132 — the point where the mantissa is already being
eaten), the call goes to gonum's scaled algorithm
(level1.go:93-133). The subnormal case is the easy one to get wrong: a vector
of 1e-160s squares to 1e-320, which is not zero, so a zero test passes it
through with six digits gone. Checking magnitudes up front instead was measured
at 483 µs against 54 µs for a million elements and rejected — see
[wrong.md](../wrong.md). The common path pays one comparison.

## Level 2

### `ger`: two accelerated paths, one guard on top

`Dger`/`Sger` first validate shape (`incY == 1`, `m >= 0`, `n >= 0`,
`lda >= n`, and the slice lengths), then choose (level1_blas.go:56-88):

- `gerFast` — `incX == 1 && incY == 1 && lda == n` — the single-call kernel
  `simd.RankOneInto`.
- otherwise, if `n >= gerRowMinLen` (256, level1_blas.go:110), the per-row
  path `gerRows`: a rank-1
  update is `m` independent axpys, and `simd.AddScaled` takes each row as its
  own contiguous slice, so any `lda` and any `incX` work. This is the shape an
  LU actually passes: a strided column as `x` and a trailing submatrix as the
  window.
- otherwise delegate.

`gerRowMinLen` exists because one accelerated call per row is one dispatch per
row, and gonum's own `ger` is already a per-row axpy against SSE2 — routing
every LU `ger` call through the per-row path made a 256-square factorisation
12% slower (level1_blas.go:90-110, [wrong.md](../wrong.md)).

### `gemv`: the narrow direct guard

`gemvFast` requires `NoTrans`, `alpha == 1`, `beta == 0`, `incX == 1`,
`incY == 1`, `m >= 0`, `n >= 0` and `lda == n`; with the slice lengths checked
in the caller, the call runs `simd.GemvParallelInto` (level23.go:82-103).
Everything else — transposed, scaled, strided, windowed — delegates. There is
no packed general path for `gemv`; see the
[roadmap](../roadmap.md) for the open question about widening it.

### `syr` and `syr2`: per-row, no scratch

`syrOK` requires `n >= syrMinLen` (128), `incX == 1`, a valid `Uplo`,
`lda >= n` and the slice lengths; `syr2` additionally requires `incY == 1` and
`len(y) >= n` (symmetric.go:176-254). The part of each row inside the requested
triangle is a contiguous span, so each row is one `simd.AddScaled`; nothing is
densified and no scratch is allocated. Only the requested triangle is touched
— the other must come out exactly as it went in
(`TestSymmetricRoutinesLeaveOtherTriangleAlone`).

## Level 3: work guards instead of length guards

Level 3 guards are on arithmetic and intensity, because the packed and blocked
paths win by the ratio of arithmetic to data movement, not by total work
(README, CHANGELOG v0.5.0):

| routine | guard | shipped path |
|---|---|---|
| `gemm` direct | `gemmFast`: `NoTrans`, `NoTrans`, `alpha == 1`, `beta == 0`, `lda == k`, `ldb == n`, `ldc == n`, non-negative dims, slice lengths | `simd.MatMulParallelInto` in place |
| `gemm` general | `gemmWorthPacking` (work `>= gemmGeneralMinWork` = 1<<15, gemm.go:85, and intensity `>= gemmMinIntensity` = 8, gemm.go:100) **and** `gemmValid` | `gemmGeneral`: pack transposed/windowed/scaled operands into pooled scratch, multiply, scale and accumulate |
| `trsm` | valid side/uplo/transpose/diag, `lda >= nt`, `ldb >= n`, lengths, work `>= trsmMinWork` (1<<15, trsm.go:31) **and** `nt > trsmBlock` (64, trsm.go:27) | blocked: gonum solves the diagonal blocks, accelerated gemm does the updates |
| `trmm` | same validity set, work `>= trsmMinWork` (1<<15 — the trsm constant, reused at trmm.go:45) | blocked; dense path (`trmmDense`) when `nt <= 64` |
| `symm` | `symmOK`: valid side/uplo, `lda >= na`, lengths, work `>= gemmGeneralMinWork` (1<<15, symmetric.go:116) | densify the stored triangle, `gemmGeneral` |
| `syrk` | valid uplo/transpose, lengths, work `>= 2*gemmGeneralMinWork` (computed inline, trmm.go:209) | full product via `gemmGeneral`, fold the requested triangle back |
| `syr2k` | same, work `>= 2*gemmGeneralMinWork` (computed inline, symmetric.go:173) | two `gemmGeneral` products into scratch, `foldTriangle` |

`ConjTrans` is accepted wherever `Trans` is; for the real types it is treated
as a transpose (`tA != blas.NoTrans`). The validity predicates are shared with
the delegation decision, so an invalid call — including a `transpose` value
that is not one of the three — delegates and gonum panics as usual.

The general gemm path counts only the operands that would actually be copied
(`gemmIntensity`): an operand already in the kernel's layout is used where it
lies rather than copied, and the copy count is what the intensity ratio divides
by (gemm.go:102-109, CHANGELOG v0.5.0).

## Scratch and allocation ownership

The general paths allocate nothing per call: `gemmScratch` and `gemmScratch32`
are `sync.Pool`s of grow-on-demand buffers, acquired with `getScratch` and
returned with `put` (gemm.go:31-54). The pooled buffer belongs to the call and
is never visible to the caller; a decomposition calls gemm once per block and
an allocation each time would show up as GC pressure in a routine that had
none. `syrk`, `syr2k`, `symm` and `trmmDense` draw from the same pools.

## Fast-path engagement

The differential suite would pass if every guard rejected everything — the
answers would simply be gonum's. Two things prevent that:

- `TestFastPathActuallyRuns` requires `Ddot` and `Dnrm2` **not** to return
  gonum's exact bits on the fast path (fastpath_test.go).
- `Dgemm` cannot be checked that way — the blocked kernel and gonum's both
  accumulate sequentially over `k` and agree bit-for-bit at every size tried —
  so its guards (and `gemv`'s, `ger`'s and Level 1's) are tested directly by
  `TestGuards` and `TestGerGuard`, which assert the contiguous case is
  accelerated and every deviation delegates (fastpath_test.go:71-138,
  diff_test.go:266-283).

## What delegates unconditionally

Banded and packed storage variants, everything complex, `symv`, `trmv` and
`trsv` are not overridden at all: the embedded gonum method is the entire
implementation, exactly as it was before this package existed (README,
CHANGELOG v1.0.0).
