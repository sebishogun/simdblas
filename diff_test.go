package simdblas

import (
	"math"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/gonum"
)

// Every override is checked against the routine it replaces.
//
// Two different bars, on purpose. Where the fast path does NOT apply this
// implementation delegates, so the answer must be gonum's exactly, to the bit —
// anything else means a guard is wrong and a fast path ran when it should not
// have. Where it does apply the answer is a different but equally valid BLAS
// answer, so the bar is a relative tolerance: reductions here use a
// sixteen-accumulator tree and gonum's use a four-way unrolled loop, and they
// disagree in the last place or two.
const tol = 1e-12

var (
	simdImpl Implementation
	refImpl  gonum.Implementation
)

func closeTo(t *testing.T, what string, got, want float64, exact bool) {
	t.Helper()
	if exact {
		if got != want {
			t.Errorf("%s: delegated path must match gonum exactly: got %#x want %#x",
				what, math.Float64bits(got), math.Float64bits(want))
		}
		return
	}
	d := math.Abs(got - want)
	if s := math.Abs(want); s > 1 {
		d /= s
	}
	if d > tol || math.IsNaN(got) != math.IsNaN(want) {
		t.Errorf("%s: got %v want %v (rel %g)", what, got, want, d)
	}
}

func closeSlice(t *testing.T, what string, got, want []float64, exact bool) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d vs %d", what, len(got), len(want))
	}
	for i := range got {
		if exact {
			if got[i] != want[i] {
				t.Errorf("%s[%d]: delegated path must match gonum exactly: %#x vs %#x",
					what, i, math.Float64bits(got[i]), math.Float64bits(want[i]))
				return
			}
			continue
		}
		d := math.Abs(got[i] - want[i])
		if s := math.Abs(want[i]); s > 1 {
			d /= s
		}
		if d > tol {
			t.Errorf("%s[%d]: got %v want %v", what, i, got[i], want[i])
			return
		}
	}
}

func randSlice(r *rand.Rand, n int) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = r.NormFloat64()
	}
	return s
}

func TestLevel1MatchesGonum(t *testing.T) {
	r := rand.New(rand.NewPCG(7, 11))
	// n=0 and n=1 straddle the empty case; 3 and 17 are below the library's
	// dispatch threshold, where it runs a plain loop; the rest are above it.
	for _, n := range []int{0, 1, 3, 17, 64, 100, 1000, 4096} {
		for _, inc := range []int{1, 2} {
			exact := inc != 1 // non-unit stride is delegated verbatim
			x := randSlice(r, n*inc+1)
			y := randSlice(r, n*inc+1)

			closeTo(t, "Ddot", simdImpl.Ddot(n, x, inc, y, inc), refImpl.Ddot(n, x, inc, y, inc), exact)
			closeTo(t, "Dnrm2", simdImpl.Dnrm2(n, x, inc), refImpl.Dnrm2(n, x, inc), exact)
			closeTo(t, "Dasum", simdImpl.Dasum(n, x, inc), refImpl.Dasum(n, x, inc), exact)

			const alpha = 1.7
			ya, yb := append([]float64(nil), y...), append([]float64(nil), y...)
			simdImpl.Daxpy(n, alpha, x, inc, ya, inc)
			refImpl.Daxpy(n, alpha, x, inc, yb, inc)
			closeSlice(t, "Daxpy", ya, yb, exact)

			xa, xb := append([]float64(nil), x...), append([]float64(nil), x...)
			simdImpl.Dscal(n, alpha, xa, inc)
			refImpl.Dscal(n, alpha, xb, inc)
			closeSlice(t, "Dscal", xa, xb, exact)
		}
	}
}

func TestGemvMatchesGonum(t *testing.T) {
	r := rand.New(rand.NewPCG(13, 17))
	for _, m := range []int{1, 5, 64, 200} {
		for _, n := range []int{1, 5, 64, 200} {
			for _, tA := range []blas.Transpose{blas.NoTrans, blas.Trans} {
				for _, alpha := range []float64{1, 2.5} {
					for _, beta := range []float64{0, 1.5} {
						a := randSlice(r, m*n)
						rows, cols := m, n
						if tA == blas.Trans {
							rows, cols = n, m
						}
						x := randSlice(r, cols)
						y1, y2 := randSlice(r, rows), []float64(nil)
						y2 = append(y2, y1...)

						simdImpl.Dgemv(tA, m, n, alpha, a, n, x, 1, beta, y1, 1)
						refImpl.Dgemv(tA, m, n, alpha, a, n, x, 1, beta, y2, 1)

						fast := tA == blas.NoTrans && alpha == 1 && beta == 0
						closeSlice(t, "Dgemv", y1, y2, !fast)
					}
				}
			}
		}
	}
}

func TestGemmMatchesGonum(t *testing.T) {
	r := rand.New(rand.NewPCG(19, 23))
	for _, m := range []int{1, 7, 64, 129} {
		for _, k := range []int{1, 7, 64, 129} {
			for _, n := range []int{1, 7, 64, 129} {
				for _, tA := range []blas.Transpose{blas.NoTrans, blas.Trans} {
					for _, alpha := range []float64{1, 0.5} {
						for _, beta := range []float64{0, 2} {
							a := randSlice(r, m*k)
							b := randSlice(r, k*n)
							c1 := randSlice(r, m*n)
							c2 := append([]float64(nil), c1...)
							lda := k
							if tA == blas.Trans {
								lda = m
							}
							simdImpl.Dgemm(tA, blas.NoTrans, m, n, k, alpha, a, lda, b, n, beta, c1, n)
							refImpl.Dgemm(tA, blas.NoTrans, m, n, k, alpha, a, lda, b, n, beta, c2, n)

							// Accelerated whenever the shape is valid and the
							// work is worth packing for — which since the
							// general path exists is nearly everything, so
							// exactness is only required below the
							// threshold, where this still delegates.
							accel := gemmWorthPacking(tA, blas.NoTrans, m, n, k, lda, n, n, alpha == 1, beta == 0) &&
								gemmValid(tA, blas.NoTrans, m, n, k, lda, n, n, len(a), len(b), len(c1))
							closeSlice(t, "Dgemm", c1, c2, !accel)
						}
					}
				}
			}
		}
	}
}

// A caller that relies on BLAS panicking for bad input must keep getting the
// panic, and the same one — which is why invalid arguments are delegated rather
// than rejected here.
func TestInvalidArgumentsStillPanic(t *testing.T) {
	cases := []struct {
		name string
		fn   func(blas.Float64)
	}{
		{"Ddot negative n", func(b blas.Float64) { b.Ddot(-1, make([]float64, 4), 1, make([]float64, 4), 1) }},
		{"Ddot short x", func(b blas.Float64) { b.Ddot(8, make([]float64, 2), 1, make([]float64, 8), 1) }},
		{"Dscal zero inc", func(b blas.Float64) { b.Dscal(4, 2, make([]float64, 4), 0) }},
		{"Dgemm short c", func(b blas.Float64) {
			b.Dgemm(blas.NoTrans, blas.NoTrans, 4, 4, 4, 1, make([]float64, 16), 4, make([]float64, 16), 4, 0, make([]float64, 2), 4)
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			want := recovered(func() { c.fn(refImpl) })
			got := recovered(func() { c.fn(simdImpl) })
			if want == nil {
				t.Skipf("gonum does not panic for this input")
			}
			if got == nil {
				t.Fatalf("gonum panics with %v; this implementation did not panic", want)
			}
			if got != want {
				t.Errorf("panic differs: got %v, gonum gives %v", got, want)
			}
		})
	}
}

func recovered(f func()) (v any) {
	defer func() { v = recover() }()
	f()
	return nil
}

// The decomposition routines, against the gonum ones they replace. Same two
// bars as above: exact where this delegates, tolerance where it does not.
func TestDecompositionRoutinesMatchGonum(t *testing.T) {
	r := rand.New(rand.NewPCG(73, 79))

	t.Run("Dswap", func(t *testing.T) {
		for _, n := range []int{0, 1, 17, 1000} {
			for _, inc := range []int{1, 2} {
				x1, y1 := randSlice(r, n*inc+1), randSlice(r, n*inc+1)
				x2 := append([]float64(nil), x1...)
				y2 := append([]float64(nil), y1...)
				simdImpl.Dswap(n, x1, inc, y1, inc)
				refImpl.Dswap(n, x2, inc, y2, inc)
				closeSlice(t, "Dswap x", x1, x2, true)
				closeSlice(t, "Dswap y", y1, y2, true)
			}
		}
	})

	t.Run("Drot", func(t *testing.T) {
		for _, n := range []int{0, 1, 17, 1000} {
			for _, inc := range []int{1, 2} {
				c, s := 0.6, 0.8
				x1, y1 := randSlice(r, n*inc+1), randSlice(r, n*inc+1)
				x2 := append([]float64(nil), x1...)
				y2 := append([]float64(nil), y1...)
				simdImpl.Drot(n, x1, inc, y1, inc, c, s)
				refImpl.Drot(n, x2, inc, y2, inc, c, s)
				// A rotation is elementwise, so this is exact on both paths.
				closeSlice(t, "Drot x", x1, x2, true)
				closeSlice(t, "Drot y", y1, y2, true)
			}
		}
	})

	t.Run("Dger", func(t *testing.T) {
		for _, m := range []int{0, 1, 7, 64} {
			for _, n := range []int{0, 1, 7, 64} {
				for _, lda := range []int{n, n + 3} {
					if lda == 0 {
						continue
					}
					x, y := randSlice(r, m), randSlice(r, n)
					a1 := randSlice(r, m*lda+1)
					a2 := append([]float64(nil), a1...)
					simdImpl.Dger(m, n, 1.5, x, 1, y, 1, a1, lda)
					refImpl.Dger(m, n, 1.5, x, 1, y, 1, a2, lda)
					// The kernel hoists the row scale exactly as gonum does, so
					// even the accelerated path is exact here.
					closeSlice(t, "Dger", a1, a2, true)
				}
			}
		}
	})
}

// As with the rest, the differential test would pass if every guard rejected
// everything. gerFast is what decides, so it is checked directly.
func TestGerGuard(t *testing.T) {
	if !gerFast(4, 5, 5, 1, 1) {
		t.Error("the contiguous case must be accelerated")
	}
	for _, c := range []struct {
		name string
		ok   bool
	}{
		{"A is a window", gerFast(4, 5, 9, 1, 1)},
		{"strided x", gerFast(4, 5, 5, 2, 1)},
		{"strided y", gerFast(4, 5, 5, 1, 2)},
		{"negative m", gerFast(-1, 5, 5, 1, 1)},
	} {
		if c.ok {
			t.Errorf("%s must be delegated to gonum", c.name)
		}
	}
}
