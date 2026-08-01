package simdblas

import (
	"github.com/sebishogun/simd"
	"gonum.org/v1/gonum/blas"
)

// Level 2 and Level 3.
//
// Dgemm goes to simd.MatMulParallelInto rather than the serial kernel. gonum's
// Dgemm fans out over GOMAXPROCS above a size threshold, and against a
// single-threaded kernel that made this three times slower at n=512 — a
// faster inner loop losing to more cores. The parallel variant removes the
// handoff that used to be here: it divides by output row, which leaves each
// element's summation over k alone, so the result is still bit-identical to the
// serial kernel.
//
// BLAS states these more generally than this library implements them. Dgemm is
//
//	C = alpha*op(A)*op(B) + beta*C
//
// where op is an optional transpose, and the leading dimensions let A, B and C
// be windows onto larger arrays. simd.MatMulInto computes dst = a*b over
// contiguous, untransposed, naturally strided slices.
//
// So the fast path is the case where those coincide, and everything else goes
// to gonum. That is deliberately narrow for a first release: it is the shape
// gonum's own mat.Mul produces for dense matrices that are not views, which is
// the common call, and it is the one where the register-blocked kernel reaches
// about 90% of a machine's single-core peak. Widening it — a transposed B, a
// non-zero beta — is worth doing against measurements rather than in advance,
// since each variant needs either another kernel or a scratch buffer, and a
// scratch buffer means an allocation in a routine that currently has none.

// gemmFast reports whether Dgemm/Sgemm arguments are the contiguous,
// untransposed, unscaled case.
//
// beta == 0 matters as much as alpha == 1: with beta non-zero the result
// accumulates into C, and MatMulInto overwrites it.
func gemmFast(tA, tB blas.Transpose, m, n, k int, alphaIs1, betaIs0 bool, lda, ldb, ldc int) bool {
	return tA == blas.NoTrans && tB == blas.NoTrans &&
		alphaIs1 && betaIs0 &&
		m >= 0 && n >= 0 && k >= 0 &&
		lda == k && ldb == n && ldc == n
}

func (impl Implementation) Dgemm(tA, tB blas.Transpose, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	if gemmFast(tA, tB, m, n, k, alpha == 1, beta == 0, lda, ldb, ldc) &&
		len(a) >= m*k && len(b) >= k*n && len(c) >= m*n {
		simd.MatMulParallelInto(c[:m*n], a[:m*k], b[:k*n], m, k, n)
		return
	}
	if int64(m)*int64(n)*int64(k) >= gemmGeneralMinWork &&
		gemmValid(tA, tB, m, n, k, lda, ldb, ldc, len(a), len(b), len(c)) {
		gemmGeneral(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
		return
	}
	impl.Implementation.Dgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

func (impl Implementation) Sgemm(tA, tB blas.Transpose, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	if gemmFast(tA, tB, m, n, k, alpha == 1, beta == 0, lda, ldb, ldc) &&
		len(a) >= m*k && len(b) >= k*n && len(c) >= m*n {
		simd.MatMulParallelInto(c[:m*n], a[:m*k], b[:k*n], m, k, n)
		return
	}
	if int64(m)*int64(n)*int64(k) >= gemmGeneralMinWork &&
		gemmValid(tA, tB, m, n, k, lda, ldb, ldc, len(a), len(b), len(c)) {
		gemmGeneral(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
		return
	}
	impl.Implementation.Sgemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// gemvFast is the same question for the matrix-vector routines. y = A*x with
// unit strides and a naturally strided A.
//
// Dgemv goes to the parallel variant for the same reason Dgemm does. Measured
// against gonum at n=1024 the serial kernel was 0.90x — ten percent slower —
// and two independent runs agreed on it, which is the only reason it was
// believed: a single earlier run had said 1.35x the other way.
func gemvFast(tA blas.Transpose, m, n int, alphaIs1, betaIs0 bool, lda, incX, incY int) bool {
	return tA == blas.NoTrans && alphaIs1 && betaIs0 &&
		incX == 1 && incY == 1 && m >= 0 && n >= 0 && lda == n
}

func (impl Implementation) Dgemv(tA blas.Transpose, m, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int) {
	if !gemvFast(tA, m, n, alpha == 1, beta == 0, lda, incX, incY) ||
		len(a) < m*n || len(x) < n || len(y) < m {
		impl.Implementation.Dgemv(tA, m, n, alpha, a, lda, x, incX, beta, y, incY)
		return
	}
	simd.GemvParallelInto(y[:m], a[:m*n], x[:n], m, n)
}

func (impl Implementation) Sgemv(tA blas.Transpose, m, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	if !gemvFast(tA, m, n, alpha == 1, beta == 0, lda, incX, incY) ||
		len(a) < m*n || len(x) < n || len(y) < m {
		impl.Implementation.Sgemv(tA, m, n, alpha, a, lda, x, incX, beta, y, incY)
		return
	}
	simd.GemvParallelInto(y[:m], a[:m*n], x[:n], m, n)
}
