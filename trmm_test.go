package simdblas

import (
	"math"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/blas"
)

func triangular(r *rand.Rand, nt int) []float64 {
	a := make([]float64, nt*nt)
	for i := 0; i < nt; i++ {
		for j := 0; j < nt; j++ {
			a[i*nt+j] = r.NormFloat64() * 0.1
		}
		a[i*nt+i] = 2 + r.Float64()
	}
	return a
}

func TestDtrmmMatchesGonum(t *testing.T) {
	r := rand.New(rand.NewPCG(103, 107))
	for _, side := range []blas.Side{blas.Left, blas.Right} {
		for _, ul := range []blas.Uplo{blas.Lower, blas.Upper} {
			for _, tA := range []blas.Transpose{blas.NoTrans, blas.Trans} {
				for _, d := range []blas.Diag{blas.Unit, blas.NonUnit} {
					for _, alpha := range []float64{1, -1, 0.5, 0} {
						for _, sz := range [][2]int{{8, 8}, {64, 40}, {65, 70}, {129, 33}, {200, 150}} {
							m, n := sz[0], sz[1]
							nt := m
							if side == blas.Right {
								nt = n
							}
							a := triangular(r, nt)
							b := make([]float64, m*n)
							for i := range b {
								b[i] = r.NormFloat64()
							}
							b1 := append([]float64(nil), b...)
							b2 := append([]float64(nil), b...)
							simdImpl.Dtrmm(side, ul, tA, d, m, n, alpha, a, nt, b1, n)
							refImpl.Dtrmm(side, ul, tA, d, m, n, alpha, a, nt, b2, n)
							for i := range b1 {
								diff := math.Abs(b1[i] - b2[i])
								scale := math.Max(1, math.Abs(b2[i]))
								if diff/scale > 1e-10 {
									t.Fatalf("side=%v uplo=%v trans=%v diag=%v alpha=%v m=%d n=%d: "+
										"differs at %d: %v vs %v", side, ul, tA, d, alpha, m, n, i, b1[i], b2[i])
								}
							}
						}
					}
				}
			}
		}
	}
}

func TestDsyrkMatchesGonum(t *testing.T) {
	r := rand.New(rand.NewPCG(109, 113))
	for _, ul := range []blas.Uplo{blas.Lower, blas.Upper} {
		for _, tr := range []blas.Transpose{blas.NoTrans, blas.Trans} {
			for _, alpha := range []float64{1, -1, 0.5} {
				for _, beta := range []float64{0, 1, 2} {
					for _, sz := range [][2]int{{8, 8}, {64, 32}, {100, 70}, {150, 150}} {
						n, k := sz[0], sz[1]
						rows, cols := n, k
						if tr != blas.NoTrans {
							rows, cols = k, n
						}
						a := make([]float64, rows*cols)
						for i := range a {
							a[i] = r.NormFloat64()
						}
						c := make([]float64, n*n)
						for i := range c {
							c[i] = r.NormFloat64()
						}
						c1 := append([]float64(nil), c...)
						c2 := append([]float64(nil), c...)
						simdImpl.Dsyrk(ul, tr, n, k, alpha, a, cols, beta, c1, n)
						refImpl.Dsyrk(ul, tr, n, k, alpha, a, cols, beta, c2, n)
						for i := range c1 {
							diff := math.Abs(c1[i] - c2[i])
							scale := math.Max(1, math.Abs(c2[i]))
							if diff/scale > 1e-10 {
								t.Fatalf("uplo=%v trans=%v alpha=%v beta=%v n=%d k=%d: "+
									"differs at %d (row %d col %d): %v vs %v",
									ul, tr, alpha, beta, n, k, i, i/n, i%n, c1[i], c2[i])
							}
						}
					}
				}
			}
		}
	}
}

// The untouched triangle of C must be left exactly as it was — that is the only
// thing distinguishing syrk from a gemm, and computing the full product makes
// it easy to overwrite by accident.
func TestDsyrkLeavesOtherTriangleAlone(t *testing.T) {
	const n, k = 150, 100
	r := rand.New(rand.NewPCG(127, 131))
	a := make([]float64, n*k)
	for i := range a {
		a[i] = r.NormFloat64()
	}
	for _, ul := range []blas.Uplo{blas.Lower, blas.Upper} {
		c := make([]float64, n*n)
		for i := range c {
			c[i] = -12345
		}
		simdImpl.Dsyrk(ul, blas.NoTrans, n, k, 1, a, k, 0, c, n)
		for i := 0; i < n; i++ {
			for j := 0; j < n; j++ {
				untouched := (ul == blas.Lower && j > i) || (ul == blas.Upper && j < i)
				if untouched && c[i*n+j] != -12345 {
					t.Fatalf("uplo=%v: wrote into the other triangle at (%d,%d)", ul, i, j)
				}
			}
		}
	}
}

// The float32 level-3 routines are a transcription of the float64 ones, which
// is exactly the kind of thing that goes wrong in one branch and nowhere else.
// Same coverage: every combination, against gonum.
func TestFloat32Level3MatchesGonum(t *testing.T) {
	r := rand.New(rand.NewPCG(137, 139))
	tri32 := func(nt int) []float32 {
		a := make([]float32, nt*nt)
		for i := 0; i < nt; i++ {
			for j := 0; j < nt; j++ {
				a[i*nt+j] = float32(r.NormFloat64()) * 0.1
			}
			a[i*nt+i] = float32(3 + r.Float64())
		}
		return a
	}
	check := func(name string, got, want []float32) {
		t.Helper()
		for i := range got {
			d := math.Abs(float64(got[i] - want[i]))
			s := math.Max(1, math.Abs(float64(want[i])))
			if d/s > 1e-4 {
				t.Fatalf("%s: differs at %d: %v vs %v", name, i, got[i], want[i])
			}
		}
	}
	for _, side := range []blas.Side{blas.Left, blas.Right} {
		for _, ul := range []blas.Uplo{blas.Lower, blas.Upper} {
			for _, tA := range []blas.Transpose{blas.NoTrans, blas.Trans} {
				for _, d := range []blas.Diag{blas.Unit, blas.NonUnit} {
					for _, sz := range [][2]int{{64, 40}, {129, 33}, {200, 150}} {
						m, n := sz[0], sz[1]
						nt := m
						if side == blas.Right {
							nt = n
						}
						a := tri32(nt)
						b := make([]float32, m*n)
						for i := range b {
							b[i] = float32(r.NormFloat64())
						}
						b1 := append([]float32(nil), b...)
						b2 := append([]float32(nil), b...)
						simdImpl.Strsm(side, ul, tA, d, m, n, 0.5, a, nt, b1, n)
						refImpl.Strsm(side, ul, tA, d, m, n, 0.5, a, nt, b2, n)
						check("Strsm", b1, b2)

						b1 = append([]float32(nil), b...)
						b2 = append([]float32(nil), b...)
						simdImpl.Strmm(side, ul, tA, d, m, n, 0.5, a, nt, b1, n)
						refImpl.Strmm(side, ul, tA, d, m, n, 0.5, a, nt, b2, n)
						check("Strmm", b1, b2)
					}
				}
			}
		}
	}
	for _, ul := range []blas.Uplo{blas.Lower, blas.Upper} {
		for _, tr := range []blas.Transpose{blas.NoTrans, blas.Trans} {
			for _, beta := range []float32{0, 1, 2} {
				n, k := 150, 100
				rows, cols := n, k
				if tr != blas.NoTrans {
					rows, cols = k, n
				}
				a := make([]float32, rows*cols)
				for i := range a {
					a[i] = float32(r.NormFloat64())
				}
				c := make([]float32, n*n)
				for i := range c {
					c[i] = float32(r.NormFloat64())
				}
				c1 := append([]float32(nil), c...)
				c2 := append([]float32(nil), c...)
				simdImpl.Ssyrk(ul, tr, n, k, 0.5, a, cols, beta, c1, n)
				refImpl.Ssyrk(ul, tr, n, k, 0.5, a, cols, beta, c2, n)
				check("Ssyrk", c1, c2)
			}
		}
	}
}
