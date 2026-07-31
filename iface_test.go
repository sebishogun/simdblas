package simdblas

import (
	"testing"

	"gonum.org/v1/gonum/blas"
)

// The point of embedding is that this cannot regress silently.
func TestSatisfiesBLAS(t *testing.T) {
	var _ blas.Float64 = Implementation{}
	var _ blas.Float32 = Implementation{}
	var _ blas.Complex64 = Implementation{}
	var _ blas.Complex128 = Implementation{}
}
