package simdblas

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/blas"
)

// Both implementations, same inputs, same machine, in one process. Anything
// else compares two builds rather than two implementations.

func benchVec(b *testing.B, n int, f func(x, y []float64)) {
	r := rand.New(rand.NewPCG(1, 2))
	x, y := make([]float64, n), make([]float64, n)
	for i := range x {
		x[i], y[i] = r.NormFloat64(), r.NormFloat64()
	}
	b.SetBytes(int64(n * 8))
	b.ResetTimer()
	for b.Loop() {
		f(x, y)
	}
}

func BenchmarkDdot(b *testing.B) {
	for _, n := range []int{1000, 100000, 1000000} {
		b.Run(fmt.Sprintf("n=%d/simd", n), func(b *testing.B) {
			benchVec(b, n, func(x, y []float64) { sinkF = simdImpl.Ddot(n, x, 1, y, 1) })
		})
		b.Run(fmt.Sprintf("n=%d/gonum", n), func(b *testing.B) {
			benchVec(b, n, func(x, y []float64) { sinkF = refImpl.Ddot(n, x, 1, y, 1) })
		})
	}
}

func BenchmarkDaxpy(b *testing.B) {
	for _, n := range []int{1000, 100000, 1000000} {
		b.Run(fmt.Sprintf("n=%d/simd", n), func(b *testing.B) {
			benchVec(b, n, func(x, y []float64) { simdImpl.Daxpy(n, 1.5, x, 1, y, 1) })
		})
		b.Run(fmt.Sprintf("n=%d/gonum", n), func(b *testing.B) {
			benchVec(b, n, func(x, y []float64) { refImpl.Daxpy(n, 1.5, x, 1, y, 1) })
		})
	}
}

func BenchmarkDnrm2(b *testing.B) {
	for _, n := range []int{1000, 100000, 1000000} {
		b.Run(fmt.Sprintf("n=%d/simd", n), func(b *testing.B) {
			benchVec(b, n, func(x, y []float64) { sinkF = simdImpl.Dnrm2(n, x, 1) })
		})
		b.Run(fmt.Sprintf("n=%d/gonum", n), func(b *testing.B) {
			benchVec(b, n, func(x, y []float64) { sinkF = refImpl.Dnrm2(n, x, 1) })
		})
	}
}

func benchGemm(b *testing.B, impl blas.Float64, n int) {
	r := rand.New(rand.NewPCG(3, 5))
	a := make([]float64, n*n)
	m := make([]float64, n*n)
	for i := range a {
		a[i], m[i] = r.NormFloat64(), r.NormFloat64()
	}
	c := make([]float64, n*n)
	// 2*n^3 flops for a square multiply.
	b.SetBytes(int64(2 * n * n * n))
	b.ResetTimer()
	for b.Loop() {
		impl.Dgemm(blas.NoTrans, blas.NoTrans, n, n, n, 1, a, n, m, n, 0, c, n)
	}
}

func BenchmarkDgemm(b *testing.B) {
	for _, n := range []int{64, 128, 256, 512} {
		b.Run(fmt.Sprintf("n=%d/simd", n), func(b *testing.B) { benchGemm(b, simdImpl, n) })
		b.Run(fmt.Sprintf("n=%d/gonum", n), func(b *testing.B) { benchGemm(b, refImpl, n) })
	}
}

func BenchmarkDgemv(b *testing.B) {
	for _, n := range []int{256, 1024, 2048} {
		r := rand.New(rand.NewPCG(7, 9))
		a := make([]float64, n*n)
		for i := range a {
			a[i] = r.NormFloat64()
		}
		x, y := make([]float64, n), make([]float64, n)
		for _, c := range []struct {
			name string
			impl blas.Float64
		}{{"simd", simdImpl}, {"gonum", refImpl}} {
			b.Run(fmt.Sprintf("n=%d/%s", n, c.name), func(b *testing.B) {
				b.SetBytes(int64(n * n * 8))
				for b.Loop() {
					c.impl.Dgemv(blas.NoTrans, n, n, 1, a, n, x, 1, 0, y, 1)
				}
			})
		}
	}
}

var sinkF float64
