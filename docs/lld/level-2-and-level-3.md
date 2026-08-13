# LLD: level 2 and level 3

How the matrix routines are built: the exact shipped paths for `gemv`, `ger`,
`syr`, `syr2`, `gemm`, `trsm`, `trmm`, `symm`, `syrk` and `syr2k`, in both
precisions. Guard predicates, thresholds and validity rules are in
[delegation-and-guards.md](delegation-and-guards.md); this file is about what
the accelerated paths do with a call that passes the guard.

## Row-major semantics

BLAS leading-dimension semantics apply throughout: a matrix is a row-major
slice with an explicit leading dimension, and a leading dimension larger than
the logical width makes the operand a window onto a bigger array. The simd
kernels underneath take contiguous, row-major, *naturally strided* slices and
no leading dimension. Every general path exists to reconcile the two: it packs
what the kernel cannot see directly, applies any transpose on the way in, and
respects the leading dimension when writing results back.

## Level 2

### `gemv`

`y = A*x` with `NoTrans`, `alpha == 1`, `beta == 0`, unit strides and
`lda == n` runs `simd.GemvParallelInto(y, a, x, m, n)` (level23.go:87-103).
It is the parallel variant because the serial kernel measured **0.90×** against
gonum at n=1024 — ten percent slower — on two independent runs, which is the
only reason it was believed: a single earlier run had said 1.35× the other way
(level23.go:75-81, [wrong.md](../wrong.md)). Everything else delegates; there
is no packed `gemv`.

### `ger`

`A += alpha*x*yᵀ` has two shipped paths (level1_blas.go:56-88):

- `gerFast` (unit strides, `lda == n`): `simd.RankOneInto` in one call.
- otherwise, for row length `>= 256`: `gerRows`, which is the same rank-1
  update as `m` independent axpys, one `simd.AddScaled` per row. This covers
  the shapes LAPACK actually passes — a strided column as `x` and a trailing
  submatrix window — so an LU is not stuck on the delegated path. The row scale
  is computed once and the product rounds before the add, matching the kernel
  and gonum, so this path returns the same bits as the single-call path
  (level1_blas.go:112-131).

### `syr` and `syr2`

`A += alpha*x*xᵀ` (and the two-vector form) over one triangle. No scratch and
no densifying: the part of each row inside the triangle is a contiguous span,
so each row is one or two `simd.AddScaled` calls (symmetric.go:176-244). The
other triangle is never written — that is a BLAS contract, tested by
`TestSymmetricRoutinesLeaveOtherTriangleAlone`.

## Level 3

### `gemm`: direct and general

**Direct path** (`gemmFast`): `C = A*B` in place via
`simd.MatMulParallelInto` — no copies, no allocation. This is the shape
gonum's `mat.Mul` produces for dense matrices that are not views, the common
call, and the one where the register-blocked kernel reaches about 90% of a
machine's single-core peak (level23.go:18-33).

**General path** (`gemmGeneral`, gemm.go:119-193) handles everything else
that passes `gemmWorthPacking` and `gemmValid`: any transpose, any alpha and
beta, and windows via leading dimensions. It decides per operand whether a
copy is needed (`needA`, `needB`, `needC`):

- an operand already in the kernel's layout is used where it lies — LAPACK's
  windows have the wrong leading dimension but its right-hand sides usually do
  not, and skipping a copy of an `m×k` operand is worth more than tuning the
  copy;
- transposed operands are packed with the transpose applied on the way in
  (`packMatrix`, gemm.go:61-81);
- C is written straight through only when the kernel's output layout is
  already C's and nothing has to be scaled or accumulated afterwards;
  otherwise the product lands in pooled scratch and the scale-and-accumulate
  runs one row at a time so that `ldc` is respected without another copy.

`beta == 0` is not the same as multiplying by zero: BLAS says C is *not read*
when beta is zero, so a C full of NaN or uninitialised memory must not poison
the result (gemm.go:171-192). The scale branches (`beta == 0 && alpha == 1`,
`beta == 0`, `beta == 1`, general) exist to avoid touching C's rows twice.

### `trsm`: blocked solve

`op(A)*X = alpha*B` (or the right-hand side), solved in place in `b`, blocked
when the work is `>= 1<<15` and the triangular order `nt > 64` (trsm.go:153-174).
The algorithm is deliberately not clever (trsm.go:5-21):

1. Scale `B` by `alpha` once up front, then solve with an implicit alpha of
   one — identical bookkeeping for every block instead of special-casing the
   first. `alpha == 0` just zeroes `B`.
2. Walk the diagonal blocks in the direction `trsmForward` picks: a transpose
   swaps which triangle is read, so Upper-with-transpose behaves exactly as
   Lower-without does, and the four cases collapse to an exclusive-or on each
   side (trsm.go:39-46).
3. Hand each diagonal block's small triangular solve to gonum's own `Dtrsm`
   (the scalar triple loop is fine at 64² per block).
4. Do the rectangular update between blocks with the accelerated gemm:
   `B[r0:] -= op(A)[r0:, j:j+jb] * B[j:j+jb, :]` with `alpha = -1`, `beta = 1`,
   choosing the sub-block offset for the transposed case (trsm.go:111-145).

All of the arithmetic that scales as O(n³) ends up in gemm; what stays scalar
is O(nb²) per block for the fixed block size 64, which matches the block size
gonum's own LAPACK uses for these factorisations.

### `trmm`: blocked multiply, dense for small triangles

`B = alpha*op(A)*B` (or the right-hand side), work `>= 1<<15`
(trmm.go:31-123):

- **`nt <= 64`**: the dense path `trmmDense` fills the triangle out into a
  full matrix (unit diagonal written as 1), hands the whole thing to
  `gemmGeneral`, and copies the result back through `ldb` (trmm.go:125-186).
  Densifying zeroes nt² and doubles the arithmetic, but this is the shape
  `Dlarfb` arrives in — a 64-wide triangle against a B as tall as the matrix —
  and blocking a 64-wide triangle produces one block and no gemm at all. That
  is what took QR from 1.05× to 1.98× at n=256 (README, trmm.go:125-137).
- **`nt > 64`**: blocked, with the diagonal block solved by gonum's own
  `Dtrmm` and the off-diagonal span updated by accelerated gemm.
  `trmmForward` is the *opposite* of `trsmForward` (trmm.go:13-23): a solve
  consumes the rows it has produced, so it walks toward them; a multiply
  consumes the rows it has not yet overwritten, so it walks away. `effLower`
  computes which triangle is actually read after any transpose (trmm.go:25-28).

One past bug is worth remembering: using the left-hand convention for both
sides is what made Right/Lower wrong while every left-hand case passed — which
side of the diagonal the off-diagonal span lies on inverts between Left and
Right, and the code says so next to the fix (trmm.go:78-87,
[wrong.md](../wrong.md)).

### `symm`: densify and multiply

`C = alpha*A*B + beta*C` with `A` symmetric, work `>= 1<<15`
(symmetric.go:64-117). The stored triangle is densified into a full n×n matrix
(`densifySym` reads the implied half from the stored side), then the call is a
`gemmGeneral` with the densified matrix as an operand. Densifying costs n² of
copying against n³ of arithmetic — the same trade as packing a windowed
operand, paid for at the same sizes (symmetric.go:8-22).

### `syrk` and `syr2k`: full product, fold one triangle

Both compute the whole product and fold only the requested triangle back,
because the multiply they use is the accelerated one and the alternative is
gonum's scalar triple loop (trmm.go:188-196):

- `syrk` (`C = alpha*A*Aᵀ + beta*C` or `alpha*Aᵀ*A`, work `>= 2*(1<<15)`):
  one `gemmGeneral` product into pooled scratch, then a row-wise fold that
  respects `beta` (0/1/other) and writes only the requested triangle
  (trmm.go:197-247). A blocked form that only touches the triangle would be
  better and is not written — it is an open evaluation in the
  [roadmap](../roadmap.md).
- `syr2k` (`C = alpha*A*Bᵀ + alpha*B*Aᵀ + beta*C`, work `>= 2*(1<<15)`): two
  `gemmGeneral` products into the same scratch — the second with `beta = 1`,
  accumulating on top of the first — then `foldTriangle`
  (symmetric.go:119-174).

`foldTriangle` and the `syrk` fold both guarantee the other triangle comes out
exactly as it went in; that invariant is what makes these routines not just
gemm calls (symmetric.go:39-41, trmm.go:224-226).

## Concurrency and scratch rules

- `gemm` and `gemv` use simd.go's **parallel** kernels
  (`MatMulParallelInto`, `GemvParallelInto`). They divide by output row, which
  leaves each element's summation over `k` alone, so the result is
  bit-identical to the serial kernel (level23.go:9-16). Gonum's own `gemm` fans
  out over `GOMAXPROCS` above a size threshold, so a single-threaded kernel
  measured three times slower at n=512 and the parallel variant is required,
  not a bonus (level23.go:10-13).
- All scratch comes from the `sync.Pool` buffers (`gemmScratch`,
  `gemmScratch32`); a call acquires with `getScratch`, defers `put`, and never
  retains the buffer after returning (gemm.go:31-54).
- The race suite (`go test -race ./...`) is a CI gate because these routines
  run concurrently above a threshold (ci.yml, CONTRIBUTING).

## Overflow and numerical tolerances

- `nrm2` is the only numerical *fallback* in the package (overflow/subnormal
  sums of squares delegate to gonum's scaled algorithm) — see
  [delegation-and-guards.md](delegation-and-guards.md).
- `beta == 0` in `gemmGeneral` never reads C (gemm.go:171-192).
- `trsm` tests use well-conditioned triangles — a dominant diagonal keeps the
  solve from amplifying rounding into something a tolerance cannot cover
  (trsm_test.go).
- The differential bar: relative tolerance `1e-12` where accelerated, bit
  equality where delegated (diff_test.go:21); the symmetric suites use `1e-10`
  to `1e-12` and `1e-4` for float32 (symmetric_test.go).
