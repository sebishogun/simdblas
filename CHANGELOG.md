# Changelog

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
