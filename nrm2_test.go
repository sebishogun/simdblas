package simdblas

import (
	"math"
	"testing"
)

// nrm2 is defined by BLAS to avoid spurious overflow and underflow, and the
// obvious implementation — sqrt of the sum of squares — does not. Before the
// range guard this returned +Inf for a vector of 1e200s and zero for a vector
// of 1e-200s, where gonum returns the correct finite value in both cases.
//
// That is the difference between a result in the last place and a result that
// is wrong, and it is the kind of thing a caller discovers in production rather
// than in a unit test, because normal data never reaches it.
func TestNrm2DoesNotOverflowOrUnderflow(t *testing.T) {
	for _, v := range []float64{
		1e-300, 1e-200, 1e-160, 1e-150, 1e-100, 1, 1e100, 1e150, 1e160, 1e200, 1e300,
		math.MaxFloat64, math.SmallestNonzeroFloat64,
	} {
		for _, n := range []int{1, 2, 72, 1000} {
			x := make([]float64, n)
			for i := range x {
				x[i] = v
			}
			got, want := simdImpl.Dnrm2(n, x, 1), refImpl.Dnrm2(n, x, 1)
			if math.IsInf(got, 0) && !math.IsInf(want, 0) {
				t.Errorf("v=%g n=%d: returned +Inf where gonum returns %g", v, n, want)
				continue
			}
			if got == 0 && want != 0 {
				t.Errorf("v=%g n=%d: flushed to zero where gonum returns %g", v, n, want)
				continue
			}
			if d := math.Abs(got - want); want != 0 && d/want > 1e-12 {
				t.Errorf("v=%g n=%d: got %g want %g", v, n, got, want)
			}
		}
	}
}

func TestSnrm2DoesNotOverflowOrUnderflow(t *testing.T) {
	for _, v := range []float32{1e-38, 1e-20, 1e-19, 1, 1e19, 1e20, 1e38, math.MaxFloat32} {
		for _, n := range []int{1, 2, 72, 1000} {
			x := make([]float32, n)
			for i := range x {
				x[i] = v
			}
			got, want := simdImpl.Snrm2(n, x, 1), refImpl.Snrm2(n, x, 1)
			if math.IsInf(float64(got), 0) && !math.IsInf(float64(want), 0) {
				t.Errorf("v=%g n=%d: returned +Inf where gonum returns %g", v, n, want)
				continue
			}
			if got == 0 && want != 0 {
				t.Errorf("v=%g n=%d: flushed to zero where gonum returns %g", v, n, want)
			}
		}
	}
}

// NaN and Inf must behave as gonum does rather than being caught by the range
// guard and silently changed.
func TestNrm2NonFinite(t *testing.T) {
	for _, name := range []string{"NaN", "+Inf", "-Inf"} {
		var v float64
		switch name {
		case "NaN":
			v = math.NaN()
		case "+Inf":
			v = math.Inf(1)
		case "-Inf":
			v = math.Inf(-1)
		}
		x := []float64{1, v, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
		got, want := simdImpl.Dnrm2(len(x), x, 1), refImpl.Dnrm2(len(x), x, 1)
		if math.IsNaN(got) != math.IsNaN(want) || (!math.IsNaN(got) && got != want) {
			t.Errorf("%s: got %g want %g", name, got, want)
		}
	}
}
