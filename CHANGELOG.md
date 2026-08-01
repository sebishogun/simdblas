# Changelog

## v1.0.1

Documentation only. Every number re-measured at a load average under 1, which is
the first time the machine has been genuinely idle for a full run.

The corrections go both ways and none is large: least squares at 5000x200 was
understated at 1.42x and is 1.52x; covariance plus Cholesky at 5000x300 was
overstated at 4.43x and is 4.36x; Dgemm at 512 was 4.8x and is 4.2x. The two
runs behind each figure now agree to within 2%, which they did not before.

The narrowest case is stated properly: 1.13x for a dot product of 100,000
elements, where the memory bus is the limit rather than the arithmetic, and
1.00x for the smallest whole workload.

## v1.0.0

**Stable.** The API is one exported type and it is not going to change.

### What 1.0 commits to

`Implementation` stays, and it stays an embedding of `gonum.Implementation`.
That is the whole surface, and it is what makes the rest of the promise cheap:
every BLAS interface remains satisfied because gonum's implementation satisfies
them, and every routine that is not accelerated behaves exactly as gonum's does,
including the panics.

Anything outside a fast path is delegated — wrong shape, too small, or an
argument that would make the answer wrong. A call never silently returns
something a BLAS would not.

### What 1.0 does not commit to

**Which routines are accelerated, or where the thresholds are.** Both are
measurements, not decisions, and both should move when a better measurement
exists. A routine may gain a fast path in a minor release; a threshold may move
in either direction.

**The last unit in the last place, across versions.** Within a version the
result is deterministic and identical on every architecture — that is inherited
from simd.go and it is the reason to use this. Across versions, a call may move
between the accelerated and the delegated path, and those two accumulate
differently. Pin a version if you diff floating-point output between builds.

**Speed on anything but amd64.** Every number in the README was measured on one
Zen 5 machine. The thresholds are measured constants, so a different cache or a
narrower vector unit may well want different ones.

### Accelerated in 1.0

Level 1: `dot`, `nrm2`, `asum`, `axpy`, `scal`, `swap`, `rot`. Level 2: `gemv`,
`ger`, `syr`, `syr2`. Level 3: `gemm`, `symm`, `trsm`, `trmm`, `syrk`, `syr2k`.
Both precisions throughout.

Not accelerated, with the reason stated in the README rather than left as a gap:
the banded and packed storage variants, everything complex, `symv` and `trmv`
(measured at 0.11x and deleted), and `trsv` (sequential by nature).

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
