package simdblas

import (
	"math"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/blas"
)

func symMat(r *rand.Rand, n int) []float64 {
	a := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			v := r.NormFloat64()
			a[i*n+j], a[j*n+i] = v, v
		}
	}
	return a
}

func closeAll(t *testing.T, what string, got, want []float64, tol float64) {
	t.Helper()
	for i := range got {
		d := math.Abs(got[i] - want[i])
		s := math.Max(1, math.Abs(want[i]))
		if d/s > tol {
			t.Fatalf("%s: differs at %d: %v vs %v", what, i, got[i], want[i])
		}
	}
}

func TestDsymmMatchesGonum(t *testing.T) {
	r := rand.New(rand.NewPCG(149, 151))
	for _, side := range []blas.Side{blas.Left, blas.Right} {
		for _, ul := range []blas.Uplo{blas.Lower, blas.Upper} {
			for _, alpha := range []float64{1, -1, 0.5} {
				for _, beta := range []float64{0, 1, 2} {
					for _, sz := range [][2]int{{8, 8}, {64, 40}, {129, 70}, {150, 150}} {
						m, n := sz[0], sz[1]
						na := m
						if side == blas.Right {
							na = n
						}
						a := symMat(r, na)
						b := make([]float64, m*n)
						c := make([]float64, m*n)
						for i := range b {
							b[i], c[i] = r.NormFloat64(), r.NormFloat64()
						}
						c1 := append([]float64(nil), c...)
						c2 := append([]float64(nil), c...)
						simdImpl.Dsymm(side, ul, m, n, alpha, a, na, b, n, beta, c1, n)
						refImpl.Dsymm(side, ul, m, n, alpha, a, na, b, n, beta, c2, n)
						closeAll(t, "Dsymm", c1, c2, 1e-10)
					}
				}
			}
		}
	}
}

func TestDsyr2kMatchesGonum(t *testing.T) {
	r := rand.New(rand.NewPCG(157, 163))
	for _, ul := range []blas.Uplo{blas.Lower, blas.Upper} {
		for _, tr := range []blas.Transpose{blas.NoTrans, blas.Trans} {
			for _, alpha := range []float64{1, -0.5} {
				for _, beta := range []float64{0, 1, 2} {
					for _, sz := range [][2]int{{8, 8}, {64, 32}, {150, 100}} {
						n, k := sz[0], sz[1]
						rows, cols := n, k
						if tr != blas.NoTrans {
							rows, cols = k, n
						}
						a := make([]float64, rows*cols)
						b := make([]float64, rows*cols)
						for i := range a {
							a[i], b[i] = r.NormFloat64(), r.NormFloat64()
						}
						c := make([]float64, n*n)
						for i := range c {
							c[i] = r.NormFloat64()
						}
						c1 := append([]float64(nil), c...)
						c2 := append([]float64(nil), c...)
						simdImpl.Dsyr2k(ul, tr, n, k, alpha, a, cols, b, cols, beta, c1, n)
						refImpl.Dsyr2k(ul, tr, n, k, alpha, a, cols, b, cols, beta, c2, n)
						closeAll(t, "Dsyr2k", c1, c2, 1e-10)
					}
				}
			}
		}
	}
}

func TestDsyrAndDsyr2MatchGonum(t *testing.T) {
	r := rand.New(rand.NewPCG(167, 173))
	for _, ul := range []blas.Uplo{blas.Lower, blas.Upper} {
		for _, n := range []int{8, 64, 128, 200} {
			x := make([]float64, n)
			y := make([]float64, n)
			for i := range x {
				x[i], y[i] = r.NormFloat64(), r.NormFloat64()
			}
			a := make([]float64, n*n)
			for i := range a {
				a[i] = r.NormFloat64()
			}
			a1 := append([]float64(nil), a...)
			a2 := append([]float64(nil), a...)
			simdImpl.Dsyr(ul, n, 1.5, x, 1, a1, n)
			refImpl.Dsyr(ul, n, 1.5, x, 1, a2, n)
			closeAll(t, "Dsyr", a1, a2, 1e-12)

			a1 = append([]float64(nil), a...)
			a2 = append([]float64(nil), a...)
			simdImpl.Dsyr2(ul, n, 1.5, x, 1, y, 1, a1, n)
			refImpl.Dsyr2(ul, n, 1.5, x, 1, y, 1, a2, n)
			closeAll(t, "Dsyr2", a1, a2, 1e-12)
		}
	}
}

// syr, syr2, syrk and syr2k all write one triangle. The other must be
// untouched, which computing a full product makes easy to get wrong.
func TestSymmetricRoutinesLeaveOtherTriangleAlone(t *testing.T) {
	const n, k = 150, 100
	r := rand.New(rand.NewPCG(179, 181))
	x := make([]float64, n)
	for i := range x {
		x[i] = r.NormFloat64()
	}
	ab := make([]float64, n*k)
	for i := range ab {
		ab[i] = r.NormFloat64()
	}
	check := func(name string, ul blas.Uplo, a []float64) {
		t.Helper()
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				other := (ul == blas.Lower && j > i) || (ul == blas.Upper && j < i)
				if other && a[i*n+j] != -777 {
					t.Fatalf("%s uplo=%v: wrote into the other triangle at (%d,%d)", name, ul, i, j)
				}
			}
		}
	}
	for _, ul := range []blas.Uplo{blas.Lower, blas.Upper} {
		a := make([]float64, n*n)
		for i := range a {
			a[i] = -777
		}
		simdImpl.Dsyr(ul, n, 1, x, 1, a, n)
		check("Dsyr", ul, a)

		a = make([]float64, n*n)
		for i := range a {
			a[i] = -777
		}
		simdImpl.Dsyr2(ul, n, 1, x, 1, x, 1, a, n)
		check("Dsyr2", ul, a)

		a = make([]float64, n*n)
		for i := range a {
			a[i] = -777
		}
		simdImpl.Dsyr2k(ul, blas.NoTrans, n, k, 1, ab, k, ab, k, 0, a, n)
		check("Dsyr2k", ul, a)
	}
}

func TestFloat32SymmetricMatchGonum(t *testing.T) {
	r := rand.New(rand.NewPCG(191, 193))
	n, k := 150, 100
	a := make([]float32, n*k)
	b := make([]float32, n*k)
	for i := range a {
		a[i], b[i] = float32(r.NormFloat64()), float32(r.NormFloat64())
	}
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(r.NormFloat64())
	}
	for _, ul := range []blas.Uplo{blas.Lower, blas.Upper} {
		c := make([]float32, n*n)
		for i := range c {
			c[i] = float32(r.NormFloat64())
		}
		c1 := append([]float32(nil), c...)
		c2 := append([]float32(nil), c...)
		simdImpl.Ssyr2k(ul, blas.NoTrans, n, k, 0.5, a, k, b, k, 1, c1, n)
		refImpl.Ssyr2k(ul, blas.NoTrans, n, k, 0.5, a, k, b, k, 1, c2, n)
		for i := range c1 {
			if math.Abs(float64(c1[i]-c2[i]))/math.Max(1, math.Abs(float64(c2[i]))) > 1e-4 {
				t.Fatalf("Ssyr2k differs at %d: %v vs %v", i, c1[i], c2[i])
			}
		}
		a1 := make([]float32, n*n)
		a2 := make([]float32, n*n)
		simdImpl.Ssyr(ul, n, 1.5, x, 1, a1, n)
		refImpl.Ssyr(ul, n, 1.5, x, 1, a2, n)
		for i := range a1 {
			if math.Abs(float64(a1[i]-a2[i])) > 1e-4 {
				t.Fatalf("Ssyr differs at %d", i)
			}
		}
	}
}
