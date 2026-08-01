# Changelog

## v0.5.0

**A realistic workload was 27% slower; it is not any more.** Everything before
this was measured routine by routine, and an ordinary least-squares fit on a
2000x50 matrix came out at 0.73x — no single routine was slow, and the
combination was.

The cause is that a size threshold is the wrong guard for `gemm`. Packing costs
O(mk + kn + mn) and the multiply is O(mnk), so the packed path wins by the ratio
between them, and that ratio is not implied by the total work. Applying a QR
factor to one right-hand side multiplies 2000x50 by 50x1: 100,000
multiply-accumulates against 102,050 elements to copy. The packing was more
expensive than the arithmetic it was preparing, and it passed a threshold that
only counted mnk.

Two fixes. Operands already in the layout the kernel wants are no longer copied
at all — LAPACK's windows have the wrong leading dimension but its right-hand
sides usually do not — and the guard is now on arithmetic per element copied
rather than on total work.

Measured end to end, worse of two runs:

	least squares  2000x50    0.73x -> 1.00x
	least squares  5000x200   1.18x -> 1.43x
	covariance + Cholesky 5000x300   3.84x -> 4.16x
	inference, 128x512x1024            1.81x

Skipping the redundant copies helped the large cases as much as the guard
helped the small one.

## v0.4.0

**Nothing is slower than gonum any more.** Level 1 below roughly 64 elements
was, and the reason is worth stating: the kernels fall back to a plain Go loop
under their own threshold, which is right for a general-purpose library and
wrong here, because gonum's Level 1 is hand-written SSE2 rather than a Go loop.
Falling back to portable code handed gonum the win.

Measured, gonum against this, before the fix — `dot` 0.46× at n=4 and 0.57× at
16, `axpy` 0.68× at 16, `scal` 0.55× at 8. Each routine now has a minimum
length, set at the first size clearly ahead rather than at break-even: dot 64,
axpy 48, scal 48, swap 48, rot 48, asum 16, nrm2 8. Below it the call is
delegated, which costs one extra method call — 0.4 to 0.7 ns, measured — and
that is now the floor on how much slower this can be for any call.

Also in this release: runnable examples for every entry point, a guide, and
this file.

## v0.3.0

**Decompositions are accelerated.** `mat.QR` 1.98×, `mat.Inverse` 1.54×,
`mat.Cholesky` 1.51×, `mat.LU` 1.25×. They were flat at 1.00× before.

The reason they were flat is that LAPACK almost never calls BLAS in the shape
application code does: `Dgetrf` multiplies submatrix windows with an alpha of
−1, `Dgetf2` passes `ger` a strided column, and half the work is in `trsm` and
`trmm`, which had no fast path at all. Every guard rejected every call.

`gemm` gained a general path that packs windowed, transposed and scaled
operands into pooled scratch. `trsm`, `trmm` and `syrk` are blocked on top of
it, in both precisions, all eight combinations of side, triangle and transpose.
`trmm` also has a dense path for a triangle at or below the block size, which is
the shape `Dlarfb` uses and what took QR from 1.05× to 1.98× at n=256.

## v0.2.0

`Dswap`, `Drot` and `Dger`, on the kernels added in simd.go v1.2.0. `Dger` has a
per-row path for the strided column and submatrix window LAPACK passes, gated at
a row length of 256.

## v0.1.0

First release. `Implementation` embeds `gonum.Implementation`, so all four BLAS
interfaces are satisfied and every routine is correct before any of it is fast.
Level 1, `gemv` and `gemm` accelerated in both precisions.
