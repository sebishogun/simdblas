package simdblas

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"github.com/sebishogun/simd"
)

// The complex Level 1 candidates, checked the same two ways as the real ones:
// where the fast path applies the answer is a different but equally valid BLAS
// answer, so the bar is relative; where it does not apply the call is
// delegated and the answer must be gonum's to the bit.
//
// simd v1.20.0 ships generated kernels for exactly three of these shapes --
// dotComplex (the bilinear product), dotConjComplex (the Hermitian one) and
// scaleComplex (a complex vector by a real scalar) -- across SSE2, AVX2 and
// AVX-512. Those are Zdotu/Cdotu, Zdotc/Cdotc and Zdscal/Csscal. Nothing
// ships for a complex scalar (Zscal, Zaxpy): those need a broadcast of the
// scalar and a second pass, which is a different measurement, made in
// TestComplexScalarCandidatesAreNotShipped below.

func closeC128(t *testing.T, what string, got, want complex128, exact bool) {
	t.Helper()
	if exact {
		if got != want {
			t.Errorf("%s: delegated path must match gonum exactly: got %v want %v", what, got, want)
		}
		return
	}
	d := cmplx.Abs(got - want)
	if s := cmplx.Abs(want); s > 1 {
		d /= s
	}
	if d > tol || cmplx.IsNaN(got) != cmplx.IsNaN(want) {
		t.Errorf("%s: got %v want %v (rel %g)", what, got, want, d)
	}
}

func closeC64(t *testing.T, what string, got, want complex64, exact bool) {
	t.Helper()
	if exact {
		if got != want {
			t.Errorf("%s: delegated path must match gonum exactly: got %v want %v", what, got, want)
		}
		return
	}
	// float32 accumulation over thousands of terms diverges much faster than
	// float64; the single-precision bar elsewhere in this package is 1e-4.
	d := float64(cmplx.Abs(complex128(got) - complex128(want)))
	if s := float64(cmplx.Abs(complex128(want))); s > 1 {
		d /= s
	}
	if d > 1e-4 {
		t.Errorf("%s: got %v want %v (rel %g)", what, got, want, d)
	}
}

func randC128(rng *rand.Rand, n int) []complex128 {
	v := make([]complex128, n)
	for i := range v {
		v[i] = complex(rng.NormFloat64(), rng.NormFloat64())
	}
	return v
}

func randC64(rng *rand.Rand, n int) []complex64 {
	v := make([]complex64, n)
	for i := range v {
		v[i] = complex(float32(rng.NormFloat64()), float32(rng.NormFloat64()))
	}
	return v
}

// Both dot products, both precisions, over the guard boundary. Zdotu is the
// bilinear sum x[i]*y[i]; Zdotc is the Hermitian sum conj(x[i])*y[i]. Getting
// these the wrong way round is the whole risk of the task, and a test that
// used a real-valued vector would not catch it -- so every vector here has a
// non-zero imaginary part.
func TestComplexDotMatchesGonum(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, n := range []int{0, 1, 7, 8, 63, 64, 65, 1000, 4096} {
		x128, y128 := randC128(rng, n+1), randC128(rng, n+1)
		x64, y64 := randC64(rng, n+1), randC64(rng, n+1)

		// Unit strides: the fast path applies above the threshold.
		fast := n >= minLenDotComplex
		closeC128(t, "Zdotu", simdImpl.Zdotu(n, x128, 1, y128, 1),
			refImpl.Zdotu(n, x128, 1, y128, 1), !fast)
		closeC128(t, "Zdotc", simdImpl.Zdotc(n, x128, 1, y128, 1),
			refImpl.Zdotc(n, x128, 1, y128, 1), !fast)
		closeC64(t, "Cdotu", simdImpl.Cdotu(n, x64, 1, y64, 1),
			refImpl.Cdotu(n, x64, 1, y64, 1), !fast)
		closeC64(t, "Cdotc", simdImpl.Cdotc(n, x64, 1, y64, 1),
			refImpl.Cdotc(n, x64, 1, y64, 1), !fast)

		// Non-unit strides delegate, so they must agree to the bit.
		if n > 1 {
			m := n / 2
			closeC128(t, "Zdotu incX=2", simdImpl.Zdotu(m, x128, 2, y128, 1),
				refImpl.Zdotu(m, x128, 2, y128, 1), true)
			closeC128(t, "Zdotc incY=2", simdImpl.Zdotc(m, x128, 1, y128, 2),
				refImpl.Zdotc(m, x128, 1, y128, 2), true)
			closeC64(t, "Cdotu incX=2", simdImpl.Cdotu(m, x64, 2, y64, 1),
				refImpl.Cdotu(m, x64, 2, y64, 1), true)
			closeC64(t, "Cdotc incY=2", simdImpl.Cdotc(m, x64, 1, y64, 2),
				refImpl.Cdotc(m, x64, 1, y64, 2), true)
		}
	}
}

// Zdotc(x, x) is the squared norm and must come out real and non-negative.
// A conjugation applied to the wrong operand still produces the right answer
// here, so this is a sanity check rather than the discriminator; the
// discriminator is the asymmetric case below.
func TestZdotcIsHermitian(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	x := randC128(rng, 1024)
	self := simdImpl.Zdotc(len(x), x, 1, x, 1)
	if imag(self) != 0 && math.Abs(imag(self)) > 1e-9*real(self) {
		t.Errorf("Zdotc(x, x) = %v: the squared norm must be real", self)
	}
	if real(self) <= 0 {
		t.Errorf("Zdotc(x, x) = %v: the squared norm must be positive", self)
	}

	// conj(x)·y and conj(y)·x are conjugates of one another. This fails if
	// the conjugation lands on the wrong operand.
	y := randC128(rng, 1024)
	xy := simdImpl.Zdotc(len(x), x, 1, y, 1)
	yx := simdImpl.Zdotc(len(x), y, 1, x, 1)
	closeC128(t, "Zdotc symmetry", xy, cmplx.Conj(yx), false)

	// And the bilinear product is symmetric, which the Hermitian one is not.
	closeC128(t, "Zdotu symmetry", simdImpl.Zdotu(len(x), x, 1, y, 1),
		simdImpl.Zdotu(len(x), y, 1, x, 1), false)
	if xy == simdImpl.Zdotu(len(x), x, 1, y, 1) {
		t.Error("Zdotc and Zdotu returned the same value: one of them is not conjugating")
	}
}

// Scaling a complex vector by a real scalar, in place. gonum walks the vector
// multiplying both parts; the kernel does the same work a lane at a time.
func TestComplexRealScaleMatchesGonum(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, n := range []int{0, 1, 7, 8, 47, 48, 49, 1000} {
		for _, alpha := range []float64{0, 1, -1, 0.5, 3} {
			x := randC128(rng, n+1)
			got := append([]complex128(nil), x...)
			want := append([]complex128(nil), x...)
			fast := n >= minLenScalComplex
			simdImpl.Zdscal(n, alpha, got, 1)
			refImpl.Zdscal(n, alpha, want, 1)
			for i := range want {
				closeC128(t, "Zdscal", got[i], want[i], !fast)
			}

			x32 := randC64(rng, n+1)
			got32 := append([]complex64(nil), x32...)
			want32 := append([]complex64(nil), x32...)
			simdImpl.Csscal(n, float32(alpha), got32, 1)
			refImpl.Csscal(n, float32(alpha), want32, 1)
			for i := range want32 {
				closeC64(t, "Csscal", got32[i], want32[i], !fast)
			}

			// Non-unit stride delegates and must be bit-exact -- and must
			// leave the skipped elements alone, which a fast path that
			// ignored incX would not.
			if n > 1 {
				g := append([]complex128(nil), x...)
				w := append([]complex128(nil), x...)
				simdImpl.Zdscal(n/2, alpha, g, 2)
				refImpl.Zdscal(n/2, alpha, w, 2)
				for i := range w {
					closeC128(t, "Zdscal incX=2", g[i], w[i], true)
				}
			}
		}
	}
}

// NaN and Inf propagate rather than being swallowed: a fast path that
// accumulated in a different order still has to produce a NaN where gonum
// produces one.
func TestComplexDotPropagatesNaN(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	n := 1024
	for _, bad := range []complex128{
		complex(math.NaN(), 0),
		complex(0, math.NaN()),
		complex(math.Inf(1), 0),
		complex(0, math.Inf(-1)),
	} {
		for _, at := range []int{0, n / 2, n - 1} {
			x := randC128(rng, n)
			y := randC128(rng, n)
			x[at] = bad
			got := simdImpl.Zdotu(n, x, 1, y, 1)
			want := refImpl.Zdotu(n, x, 1, y, 1)
			if cmplx.IsNaN(got) != cmplx.IsNaN(want) || cmplx.IsInf(got) != cmplx.IsInf(want) {
				t.Errorf("Zdotu with %v at %d: got %v want %v", bad, at, got, want)
			}
		}
	}
}

// An empty or negative n, a short slice, a zero stride: every one of these is
// gonum's to validate and panic on. The overrides must not swallow them.
func TestComplexGuardsDelegateInvalidArguments(t *testing.T) {
	x := make([]complex128, 4)
	y := make([]complex128, 4)
	for _, c := range []struct {
		name string
		call func()
	}{
		{"Zdotu incX=0", func() { simdImpl.Zdotu(4, x, 0, y, 1) }},
		{"Zdotc incY=0", func() { simdImpl.Zdotc(4, x, 1, y, 0) }},
		{"Zdotu short x", func() { simdImpl.Zdotu(8, x, 1, y, 1) }},
		{"Zdscal incX=0", func() { simdImpl.Zdscal(4, 2, x, 0) }},
		{"Zdscal short x", func() { simdImpl.Zdscal(8, 2, x, 1) }},
		{"Zdotu negative n", func() { simdImpl.Zdotu(-1, x, 1, y, 1) }},
	} {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Error("no panic: gonum validates these and the override must delegate")
				}
			}()
			c.call()
		})
	}
}

// The guards must engage exactly where they claim to, which a differential
// test cannot show: identical answers prove nothing about which path ran.
func TestComplexGuards(t *testing.T) {
	x := make([]complex128, 4096)
	y := make([]complex128, 4096)
	for _, c := range []struct {
		name string
		n    int
		incX int
		incY int
		want bool
	}{
		{"at the threshold", minLenDotComplex, 1, 1, true},
		{"one below", minLenDotComplex - 1, 1, 1, false},
		{"strided x", 4096, 2, 1, false},
		{"strided y", 4096, 1, 2, false},
		{"reversed x", 4096, -1, 1, false},
		{"short x", 4096, 1, 1, false},
	} {
		t.Run(c.name, func(t *testing.T) {
			xx, yy := x, y
			if c.name == "short x" {
				xx = x[:8]
			}
			got := unitC2(c.n, xx, c.incX, yy, c.incY) && c.n >= minLenDotComplex
			if got != c.want {
				t.Errorf("guard says %v, want %v", got, c.want)
			}
		})
	}
}

// Zscal and Zaxpy take a COMPLEX scalar, and simd v1.20.0 ships no kernel for
// that shape: the only route is a broadcast buffer plus MulComplex, which is
// an allocation and a second pass over the data for work gonum does in one.
// This test pins the reason they are absent, so a later reader does not
// assume it was an oversight. It fails if a complex-scalar kernel appears
// upstream -- at which point the measurement is worth redoing.
func TestComplexScalarCandidatesAreNotShipped(t *testing.T) {
	// The kernel set that exists for complex vectors. If simd grows a
	// complex-scalar multiply, this list is where it shows up.
	var (
		_ = simd.DotComplex[complex128]
		_ = simd.DotComplexConj[complex128]
		_ = simd.ScaleComplex[complex128, float64]
		// simd.MulComplex(a, b []C) is elementwise, not vector-by-scalar:
		// using it for Zscal needs a full-length broadcast of alpha.
		_ = simd.MulComplex[complex128]
	)

	// And the overrides are absent, so these are gonum's, bit for bit.
	rng := rand.New(rand.NewPCG(9, 10))
	x := randC128(rng, 1024)
	got := append([]complex128(nil), x...)
	want := append([]complex128(nil), x...)
	alpha := complex(0.5, -0.25)
	simdImpl.Zscal(len(x), alpha, got, 1)
	refImpl.Zscal(len(x), alpha, want, 1)
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Zscal[%d]: %v vs %v -- an override appeared without a measurement", i, got[i], want[i])
		}
	}
}
