package simdblas

import "github.com/sebishogun/simd"

// The complex Level 1 routines gonum has and simd has a kernel for.
//
// simd v1.20.0 ships generated assembly for three complex shapes, across
// SSE2, AVX2 and AVX-512 on amd64 and the equivalent tiers elsewhere:
// dotComplex (the bilinear sum a[i]*b[i]), dotConjComplex (the Hermitian sum
// conj(a[i])*b[i]) and scaleComplex (a complex vector times a real scalar).
// Those map exactly onto six BLAS routines: Zdotu/Cdotu, Zdotc/Cdotc and
// Zdscal/Csscal. No mapping is approximate and none needs scratch.
//
// The rest of the complex surface stays gonum's:
//
//   - Zscal, Cscal, Zaxpy, Caxpy take a COMPLEX scalar. simd has no
//     vector-by-complex-scalar kernel, so the only route is a full-length
//     broadcast of alpha plus MulComplex -- an allocation and a second pass
//     over the data for work gonum does in one. Pinned in
//     TestComplexScalarCandidatesAreNotShipped.
//   - Dzasum is the sum of |Re| + |Im|, not the sum of moduli.
//     AbsComplexInto computes the modulus, which is a different number.
//   - Dznrm2 and Scnrm2 scale as they go to avoid overflow.
//     DotComplexConj(x, x) is the squared norm and overflows on inputs
//     gonum handles, so it is not a drop-in.
//   - Zswap, Zcopy, Izamax have no complex kernel.

// Minimum lengths, below which gonum's own routine gets the call.
//
// Provisional, not yet measured: the machine was not quiet when these were
// written, and the family rule is that a threshold moves only on an
// interleaved A/B minimum taken under load average 1. They start at the
// real-precision thresholds for the same shapes (minLenDot, minLenScal),
// which is a conservative place to start -- a complex element is two floats,
// so the same element count is twice the work and if anything the crossover
// comes sooner. Task 4 step 3 of the production plan replaces these with
// measured values and this comment with the table.
const (
	minLenDotComplex  = 64
	minLenScalComplex = 48
)

// unitC reports whether a one-vector complex Level 1 call is the contiguous,
// valid case. unit cannot be reused: its type parameter is simd.Number, which
// is the real element types only.
func unitC[C simd.Complex](n int, x []C, incX int) bool {
	return incX == 1 && n >= 0 && len(x) >= n
}

// unitC2 is the same for the two-vector routines.
func unitC2[C simd.Complex](n int, x []C, incX int, y []C, incY int) bool {
	return incX == 1 && incY == 1 && n >= 0 && len(x) >= n && len(y) >= n
}

// Zdotu returns the bilinear product sum(x[i]*y[i]).
//
// Not the inner product: no conjugation. Zdotc is the one linear algebra
// usually means.
func (impl Implementation) Zdotu(n int, x []complex128, incX int, y []complex128, incY int) complex128 {
	if !unitC2(n, x, incX, y, incY) || n < minLenDotComplex {
		return impl.Implementation.Zdotu(n, x, incX, y, incY)
	}
	return simd.DotComplex(x[:n], y[:n])
}

// Zdotc returns the Hermitian inner product sum(conj(x[i])*y[i]).
func (impl Implementation) Zdotc(n int, x []complex128, incX int, y []complex128, incY int) complex128 {
	if !unitC2(n, x, incX, y, incY) || n < minLenDotComplex {
		return impl.Implementation.Zdotc(n, x, incX, y, incY)
	}
	return simd.DotComplexConj(x[:n], y[:n])
}

// Cdotu is Zdotu in single precision.
func (impl Implementation) Cdotu(n int, x []complex64, incX int, y []complex64, incY int) complex64 {
	if !unitC2(n, x, incX, y, incY) || n < minLenDotComplex {
		return impl.Implementation.Cdotu(n, x, incX, y, incY)
	}
	return simd.DotComplex(x[:n], y[:n])
}

// Cdotc is Zdotc in single precision.
func (impl Implementation) Cdotc(n int, x []complex64, incX int, y []complex64, incY int) complex64 {
	if !unitC2(n, x, incX, y, incY) || n < minLenDotComplex {
		return impl.Implementation.Cdotc(n, x, incX, y, incY)
	}
	return simd.DotComplexConj(x[:n], y[:n])
}

// Zdscal scales x by the real alpha, in place.
func (impl Implementation) Zdscal(n int, alpha float64, x []complex128, incX int) {
	if !unitC(n, x, incX) || n < minLenScalComplex {
		impl.Implementation.Zdscal(n, alpha, x, incX)
		return
	}
	simd.ScaleComplex(x[:n], alpha)
}

// Csscal scales x by the real alpha, in place.
func (impl Implementation) Csscal(n int, alpha float32, x []complex64, incX int) {
	if !unitC(n, x, incX) || n < minLenScalComplex {
		impl.Implementation.Csscal(n, alpha, x, incX)
		return
	}
	simd.ScaleComplex(x[:n], alpha)
}
