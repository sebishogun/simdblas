package simdblas

import (
	"math"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/blas"
)

// All eight combinations of side, triangle and transpose, both diagonal kinds,
// several alphas, at sizes that straddle the block size and the work threshold.
// LAPACK uses every one of them, so there is no useful subset to skip.
func TestDtrsmMatchesGonum(t *testing.T) {
	r := rand.New(rand.NewPCG(83, 89))
	for _, side := range []blas.Side{blas.Left, blas.Right} {
		for _, ul := range []blas.Uplo{blas.Lower, blas.Upper} {
			for _, tA := range []blas.Transpose{blas.NoTrans, blas.Trans} {
				for _, d := range []blas.Diag{blas.Unit, blas.NonUnit} {
					for _, alpha := range []float64{1, -1, 0.5, 0} {
						for _, sz := range [][2]int{{8, 8}, {64, 40}, {65, 70}, {129, 33}, {200, 200}} {
							m, n := sz[0], sz[1]
							nt := m
							if side == blas.Right {
								nt = n
							}
							// A well-conditioned triangle: a dominant diagonal
							// keeps the solve from amplifying rounding into
							// something a tolerance cannot cover.
							a := make([]float64, nt*nt)
							for i := 0; i < nt; i++ {
								for j := 0; j < nt; j++ {
									a[i*nt+j] = r.NormFloat64() * 0.1
								}
								a[i*nt+i] = 4 + r.Float64()
							}
							b := make([]float64, m*n)
							for i := range b {
								b[i] = r.NormFloat64()
							}
							b1 := append([]float64(nil), b...)
							b2 := append([]float64(nil), b...)

							simdImpl.Dtrsm(side, ul, tA, d, m, n, alpha, a, nt, b1, n)
							refImpl.Dtrsm(side, ul, tA, d, m, n, alpha, a, nt, b2, n)

							for i := range b1 {
								diff := math.Abs(b1[i] - b2[i])
								scale := math.Max(1, math.Abs(b2[i]))
								if diff/scale > 1e-9 || math.IsNaN(b1[i]) != math.IsNaN(b2[i]) {
									t.Fatalf("side=%v uplo=%v trans=%v diag=%v alpha=%v m=%d n=%d: "+
										"differs at %d: %v vs %v",
										side, ul, tA, d, alpha, m, n, i, b1[i], b2[i])
								}
							}
						}
					}
				}
			}
		}
	}
}

// The blocked path must actually run for the large cases, or the test above
// passes by delegating everything.
func TestDtrsmBlockedPathRuns(t *testing.T) {
	r := rand.New(rand.NewPCG(97, 101))
	const nt, n = 200, 200
	a := make([]float64, nt*nt)
	for i := 0; i < nt; i++ {
		for j := 0; j < nt; j++ {
			a[i*nt+j] = r.NormFloat64() * 0.1
		}
		a[i*nt+i] = 4 + r.Float64()
	}
	b := make([]float64, nt*n)
	for i := range b {
		b[i] = r.NormFloat64()
	}
	b1 := append([]float64(nil), b...)
	b2 := append([]float64(nil), b...)
	simdImpl.Dtrsm(blas.Left, blas.Lower, blas.NoTrans, blas.NonUnit, nt, n, 1, a, nt, b1, n)
	refImpl.Dtrsm(blas.Left, blas.Lower, blas.NoTrans, blas.NonUnit, nt, n, 1, a, nt, b2, n)
	same := true
	for i := range b1 {
		if b1[i] != b2[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("Dtrsm matched gonum bit-for-bit across the whole matrix; the " +
			"blocked path did not run and the differential test is vacuous")
	}
}

func TestTrsmForwardDirection(t *testing.T) {
	// A transpose swaps which triangle is read, so these must pair up.
	cases := []struct {
		side blas.Side
		ul   blas.Uplo
		tA   blas.Transpose
		want bool
	}{
		{blas.Left, blas.Lower, blas.NoTrans, true},
		{blas.Left, blas.Upper, blas.NoTrans, false},
		{blas.Left, blas.Upper, blas.Trans, true},
		{blas.Left, blas.Lower, blas.Trans, false},
		{blas.Right, blas.Upper, blas.NoTrans, true},
		{blas.Right, blas.Lower, blas.NoTrans, false},
		{blas.Right, blas.Lower, blas.Trans, true},
		{blas.Right, blas.Upper, blas.Trans, false},
	}
	for _, c := range cases {
		if got := trsmForward(c.side, c.ul, c.tA); got != c.want {
			t.Errorf("side=%v uplo=%v trans=%v: forward=%v, want %v", c.side, c.ul, c.tA, got, c.want)
		}
	}
}
