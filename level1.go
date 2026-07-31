package simdblas

import (
	"math"

	"github.com/sebishogun/simd"
)

// The Level 1 routines are memory-bound: one pass over the data, no reuse. The
// win is smaller than in Level 3 and still real, because gonum's versions are
// SSE2 at best and a plain Go loop on anything that is not amd64.
//
// Each of these is the same shape. Check that the arguments are the contiguous
// case, and hand anything else — including anything invalid — back to gonum,
// which validates and panics with the message callers expect. A fast path that
// has to reproduce someone else's panic text is a fast path that will get it
// wrong eventually.

// unit reports whether a one-vector Level 1 call is the contiguous, valid case.
func unit[T simd.Number](n int, x []T, incX int) bool {
	return incX == 1 && n >= 0 && len(x) >= n
}

// unit2 is the same for the two-vector routines.
func unit2[T simd.Number](n int, x []T, incX int, y []T, incY int) bool {
	return incX == 1 && incY == 1 && n >= 0 && len(x) >= n && len(y) >= n
}

func (impl Implementation) Ddot(n int, x []float64, incX int, y []float64, incY int) float64 {
	if !unit2(n, x, incX, y, incY) {
		return impl.Implementation.Ddot(n, x, incX, y, incY)
	}
	return simd.Dot(x[:n], y[:n])
}

func (impl Implementation) Sdot(n int, x []float32, incX int, y []float32, incY int) float32 {
	if !unit2(n, x, incX, y, incY) {
		return impl.Implementation.Sdot(n, x, incX, y, incY)
	}
	return simd.Dot(x[:n], y[:n])
}

// Dnrm2 is the one routine here that cannot simply be handed to the vector
// unit, and the reason is in the BLAS specification rather than in the kernel.
//
// nrm2 is *defined* to avoid spurious overflow and underflow: gonum computes it
// by scaling to the largest element first, so a vector of 1e200s returns
// 8.48e200 rather than +Inf. The sum of squares overflows there, and flushes to
// zero for a vector of 1e-200s. Swapping the backend must not turn a finite
// result into an infinity.
//
// Checking the magnitudes first works and costs a whole extra pass — measured
// at 483us against 54us for a million elements, which spends most of the win to
// buy an answer that is almost always already correct. So instead the fast path
// runs and its own failure is detected: the sum of squares overflowing produces
// +Inf, and underflowing produces a zero that a non-empty vector can only
// otherwise reach by being all zeros. Both are exactly the cases gonum's scaled
// algorithm exists for, and both then go to it.
//
// The common path pays one comparison.
func (impl Implementation) Dnrm2(n int, x []float64, incX int) float64 {
	if !unit(n, x, incX) {
		return impl.Implementation.Dnrm2(n, x, incX)
	}
	if n == 0 {
		return 0
	}
	ss := simd.SumSquares(x[:n])
	if math.IsInf(ss, 1) || ss < minNormalF64 {
		// Overflowed, or sank into the subnormals. The second case is the one
		// that is easy to get wrong: a vector of 1e-160s squares to 1e-320,
		// which is not zero, so a test for zero passes it through with six
		// digits of the mantissa already gone. An all-zero vector lands here
		// too, and gonum returns zero for it, so the fallback is correct rather
		// than merely safe.
		return impl.Implementation.Dnrm2(n, x, incX)
	}
	return math.Sqrt(ss)
}

func (impl Implementation) Snrm2(n int, x []float32, incX int) float32 {
	if !unit(n, x, incX) {
		return impl.Implementation.Snrm2(n, x, incX)
	}
	if n == 0 {
		return 0
	}
	ss := simd.SumSquares(x[:n])
	if math.IsInf(float64(ss), 1) || ss < minNormalF32 {
		return impl.Implementation.Snrm2(n, x, incX)
	}
	return float32(math.Sqrt(float64(ss)))
}

// The smallest normal values of each type. Below these the mantissa is being
// eaten by the exponent and the sum of squares has already lost precision that
// no amount of care afterwards recovers.
const (
	minNormalF64 = 0x1p-1022
	minNormalF32 = 0x1p-126
)

func (impl Implementation) Dasum(n int, x []float64, incX int) float64 {
	if !unit(n, x, incX) {
		return impl.Implementation.Dasum(n, x, incX)
	}
	return simd.L1Norm(x[:n])
}

func (impl Implementation) Sasum(n int, x []float32, incX int) float32 {
	if !unit(n, x, incX) {
		return impl.Implementation.Sasum(n, x, incX)
	}
	return simd.L1Norm(x[:n])
}

// Daxpy is y += alpha*x, the operation BLAS calls axpy and this library calls
// AddScaled. One pass over both slices rather than a scale followed by an add.
func (impl Implementation) Daxpy(n int, alpha float64, x []float64, incX int, y []float64, incY int) {
	if !unit2(n, x, incX, y, incY) {
		impl.Implementation.Daxpy(n, alpha, x, incX, y, incY)
		return
	}
	simd.AddScaled(y[:n], x[:n], alpha)
}

func (impl Implementation) Saxpy(n int, alpha float32, x []float32, incX int, y []float32, incY int) {
	if !unit2(n, x, incX, y, incY) {
		impl.Implementation.Saxpy(n, alpha, x, incX, y, incY)
		return
	}
	simd.AddScaled(y[:n], x[:n], alpha)
}

func (impl Implementation) Dscal(n int, alpha float64, x []float64, incX int) {
	if !unit(n, x, incX) {
		impl.Implementation.Dscal(n, alpha, x, incX)
		return
	}
	simd.Scale(x[:n], alpha)
}

func (impl Implementation) Sscal(n int, alpha float32, x []float32, incX int) {
	if !unit(n, x, incX) {
		impl.Implementation.Sscal(n, alpha, x, incX)
		return
	}
	simd.Scale(x[:n], alpha)
}
