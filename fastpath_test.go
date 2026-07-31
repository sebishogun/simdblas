package simdblas

import (
	"gonum.org/v1/gonum/blas"
	"math"
	"math/rand/v2"
	"testing"
)

// Everything in diff_test.go would also pass if every guard were inverted and
// this package delegated all of it — the answers would simply be gonum's, which
// is what those tests compare against. So this asserts the opposite: on the
// fast path the answer must NOT be bit-identical to gonum's, because the two
// accumulate in different orders and a large enough reduction always disagrees
// in the last place.
//
// If one of these starts passing trivially, the fast path stopped running.
func TestFastPathActuallyRuns(t *testing.T) {
	r := rand.New(rand.NewPCG(29, 31))
	const n = 4096
	x, y := make([]float64, n), make([]float64, n)
	for i := range x {
		x[i], y[i] = r.NormFloat64(), r.NormFloat64()
	}

	t.Run("Ddot", func(t *testing.T) {
		got, want := simdImpl.Ddot(n, x, 1, y, 1), refImpl.Ddot(n, x, 1, y, 1)
		if got == want {
			t.Errorf("Ddot returned gonum's exact bits (%#x); the accelerated path "+
				"did not run and the differential tests are vacuous",
				math.Float64bits(got))
		}
	})

	t.Run("Dnrm2", func(t *testing.T) {
		if simdImpl.Dnrm2(n, x, 1) == refImpl.Dnrm2(n, x, 1) {
			t.Error("Dnrm2 returned gonum's exact bits; the accelerated path did not run")
		}
	})

}

// The property the package doc claims: the answer does not depend on which
// instruction set was selected. gonum's own does — its assembly and its
// pure-Go fallback disagree — which is why a gonum result is not reproducible
// between amd64 and arm64.
func TestAnswerDoesNotDependOnTier(t *testing.T) {
	r := rand.New(rand.NewPCG(37, 41))
	const n = 4096
	x, y := make([]float64, n), make([]float64, n)
	for i := range x {
		x[i], y[i] = r.NormFloat64(), r.NormFloat64()
	}
	// simd selects its tier at startup and GOSIMD pins it, so this asserts
	// repeatability here and the cross-tier suite in the simd repository proves
	// the rest. What matters is that the value is a fixed function of the input.
	first := simdImpl.Ddot(n, x, 1, y, 1)
	for i := 0; i < 100; i++ {
		if got := simdImpl.Ddot(n, x, 1, y, 1); got != first {
			t.Fatalf("Ddot is not deterministic: %#x then %#x",
				math.Float64bits(first), math.Float64bits(got))
		}
	}
}

// The guards decide everything: whether a call is accelerated or handed back to
// gonum. Dgemm cannot be checked the way Ddot is, because the blocked kernel
// and gonum's both accumulate sequentially over k and agree bit-for-bit at
// every size tried — a matching result there says nothing about which ran. So
// the guards are tested directly.
func TestGuards(t *testing.T) {
	t.Run("gemm", func(t *testing.T) {
		cases := []struct {
			name                   string
			tA, tB                 blas.Transpose
			alphaIs1, betaIs0      bool
			lda, ldb, ldc, m, n, k int
			want                   bool
		}{
			{"contiguous", blas.NoTrans, blas.NoTrans, true, true, 4, 5, 5, 3, 5, 4, true},
			{"A transposed", blas.Trans, blas.NoTrans, true, true, 4, 5, 5, 3, 5, 4, false},
			{"B transposed", blas.NoTrans, blas.Trans, true, true, 4, 5, 5, 3, 5, 4, false},
			{"alpha != 1", blas.NoTrans, blas.NoTrans, false, true, 4, 5, 5, 3, 5, 4, false},
			{"beta != 0", blas.NoTrans, blas.NoTrans, true, false, 4, 5, 5, 3, 5, 4, false},
			{"A is a window", blas.NoTrans, blas.NoTrans, true, true, 9, 5, 5, 3, 5, 4, false},
			{"B is a window", blas.NoTrans, blas.NoTrans, true, true, 4, 9, 5, 3, 5, 4, false},
			{"C is a window", blas.NoTrans, blas.NoTrans, true, true, 4, 5, 9, 3, 5, 4, false},
			{"negative m", blas.NoTrans, blas.NoTrans, true, true, 4, 5, 5, -1, 5, 4, false},
		}
		for _, c := range cases {
			got := gemmFast(c.tA, c.tB, c.m, c.n, c.k, c.alphaIs1, c.betaIs0, c.lda, c.ldb, c.ldc)
			if got != c.want {
				t.Errorf("%s: gemmFast = %v, want %v", c.name, got, c.want)
			}
		}
	})

	t.Run("gemv", func(t *testing.T) {
		if !gemvFast(blas.NoTrans, 3, 5, true, true, 5, 1, 1) {
			t.Error("the contiguous case must be accelerated")
		}
		for _, c := range []struct {
			name string
			ok   bool
		}{
			{"transposed", gemvFast(blas.Trans, 3, 5, true, true, 5, 1, 1)},
			{"alpha != 1", gemvFast(blas.NoTrans, 3, 5, false, true, 5, 1, 1)},
			{"beta != 0", gemvFast(blas.NoTrans, 3, 5, true, false, 5, 1, 1)},
			{"A is a window", gemvFast(blas.NoTrans, 3, 5, true, true, 9, 1, 1)},
			{"strided x", gemvFast(blas.NoTrans, 3, 5, true, true, 5, 2, 1)},
			{"strided y", gemvFast(blas.NoTrans, 3, 5, true, true, 5, 1, 2)},
		} {
			if c.ok {
				t.Errorf("%s must be delegated to gonum, not accelerated", c.name)
			}
		}
	})

	t.Run("level 1", func(t *testing.T) {
		x, y := make([]float64, 8), make([]float64, 8)
		if !unit(8, x, 1) || !unit2(8, x, 1, y, 1) {
			t.Error("the contiguous case must be accelerated")
		}
		for _, c := range []struct {
			name string
			ok   bool
		}{
			{"strided", unit(8, x, 2)},
			{"negative n", unit(-1, x, 1)},
			{"n past the end of x", unit(9, x, 1)},
			{"strided y", unit2(8, x, 1, y, 2)},
			{"n past the end of y", unit2(9, x, 1, y, 1)},
		} {
			if c.ok {
				t.Errorf("%s must be delegated to gonum, not accelerated", c.name)
			}
		}
	})
}
