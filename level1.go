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

// Minimum lengths, below which gonum's own routine is faster and gets the call.
//
// This is the one place a fast path can lose without any guard noticing. The
// kernels behind these have their own threshold — under a few dozen elements
// they run a plain Go loop, because crossing into assembly costs more than it
// saves — but gonum's Level 1 is *hand-written SSE2*, not a Go loop, so falling
// back to portable code hands it the win. Delegating instead is strictly
// better: same answer, and gonum's assembly rather than a scalar loop.
//
// Measured on Zen 5, float64, gonum against this, ratios crossing 1.0:
//
//	n      asum   axpy    dot   nrm2   scal
//	8      0.52   0.62   0.46   1.09   0.55
//	16     1.04   0.72   0.57   2.15   0.59
//	32     1.75   0.97   0.77   3.70   0.98
//	48     2.57   1.10   0.94   5.16   1.06
//	64     3.62   1.29   1.15   6.47   1.07
//	256   14.03   2.42   2.26  18.75   1.79
//
// Each threshold is the first length that is clearly ahead, not the break-even
// point: being wrong here costs speed in one direction and a regression in the
// other, and a regression is what a caller notices.
const (
	minLenDot  = 64
	minLenAxpy = 48
	minLenScal = 48
	minLenAsum = 16
	minLenNrm2 = 8
	minLenSwap = 48
	minLenRot  = 48
)

// unit reports whether a one-vector Level 1 call is the contiguous, valid case.
func unit[T simd.Number](n int, x []T, incX int) bool {
	return incX == 1 && n >= 0 && len(x) >= n
}

// unit2 is the same for the two-vector routines.
func unit2[T simd.Number](n int, x []T, incX int, y []T, incY int) bool {
	return incX == 1 && incY == 1 && n >= 0 && len(x) >= n && len(y) >= n
}

func (impl Implementation) Ddot(n int, x []float64, incX int, y []float64, incY int) float64 {
	if !unit2(n, x, incX, y, incY) || n < minLenDot {
		return impl.Implementation.Ddot(n, x, incX, y, incY)
	}
	return simd.Dot(x[:n], y[:n])
}

func (impl Implementation) Sdot(n int, x []float32, incX int, y []float32, incY int) float32 {
	if !unit2(n, x, incX, y, incY) || n < minLenDot {
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
	if !unit(n, x, incX) || n < minLenNrm2 {
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
	if !unit(n, x, incX) || n < minLenNrm2 {
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
	if !unit(n, x, incX) || n < minLenAsum {
		return impl.Implementation.Dasum(n, x, incX)
	}
	return simd.L1Norm(x[:n])
}

func (impl Implementation) Sasum(n int, x []float32, incX int) float32 {
	if !unit(n, x, incX) || n < minLenAsum {
		return impl.Implementation.Sasum(n, x, incX)
	}
	return simd.L1Norm(x[:n])
}

// Daxpy is y += alpha*x, the operation BLAS calls axpy and this library calls
// AddScaled. One pass over both slices rather than a scale followed by an add.
func (impl Implementation) Daxpy(n int, alpha float64, x []float64, incX int, y []float64, incY int) {
	if !unit2(n, x, incX, y, incY) || n < minLenAxpy {
		impl.Implementation.Daxpy(n, alpha, x, incX, y, incY)
		return
	}
	simd.AddScaled(y[:n], x[:n], alpha)
}

func (impl Implementation) Saxpy(n int, alpha float32, x []float32, incX int, y []float32, incY int) {
	if !unit2(n, x, incX, y, incY) || n < minLenAxpy {
		impl.Implementation.Saxpy(n, alpha, x, incX, y, incY)
		return
	}
	simd.AddScaled(y[:n], x[:n], alpha)
}

func (impl Implementation) Dscal(n int, alpha float64, x []float64, incX int) {
	if !unit(n, x, incX) || n < minLenScal {
		impl.Implementation.Dscal(n, alpha, x, incX)
		return
	}
	simd.Scale(x[:n], alpha)
}

func (impl Implementation) Sscal(n int, alpha float32, x []float32, incX int) {
	if !unit(n, x, incX) || n < minLenScal {
		impl.Implementation.Sscal(n, alpha, x, incX)
		return
	}
	simd.Scale(x[:n], alpha)
}
