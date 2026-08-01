package simdblas

import (
	"github.com/sebishogun/simd"
	"gonum.org/v1/gonum/blas"
)

// The symmetric routines.
//
// A symmetric matrix is stored as one triangle and the other half is implied,
// which is the whole reason these exist as separate routines instead of being
// gemm calls. The kernels underneath know nothing about triangles, so the
// approach throughout this file is the one already used for syrk: fill the
// implied half in, hand the result to the accelerated multiply, and fold only
// the requested triangle back.
//
// Densifying costs n² of copying against n²k or n³ of arithmetic. It is the
// same trade as packing a windowed operand for gemm and it is paid for at the
// same sizes, so these carry the same intensity guard.
//
// The rank-1 and rank-2 updates do not need any of that: they touch one
// triangle of the output and nothing else, so each row is a contiguous span
// that AddScaled can take directly.

// densifySym writes the full n×n matrix implied by one stored triangle.
func densifySym[T float32 | float64](dst, a []T, n, lda int, upper bool) {
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			// Read from whichever side of the diagonal is actually stored.
			ri, rj := i, j
			if (upper && j < i) || (!upper && j > i) {
				ri, rj = j, i
			}
			dst[i*n+j] = a[ri*lda+rj]
		}
	}
}

// foldTriangle accumulates src into the requested triangle of c, and only that
// triangle: the other half must come out exactly as it went in.
func foldTriangle[T float32 | float64](c []T, src []T, n, ldc int, upper bool, beta T) {
	for i := 0; i < n; i++ {
		lo, hi := 0, i+1
		if upper {
			lo, hi = i, n
		}
		row := c[i*ldc+lo : i*ldc+hi]
		s := src[i*n+lo : i*n+hi]
		switch beta {
		case 0:
			copy(row, s)
		case 1:
			for j := range row {
				row[j] += s[j]
			}
		default:
			for j := range row {
				row[j] = row[j]*beta + s[j]
			}
		}
	}
}

// Dsymm computes C = alpha*A*B + beta*C with A symmetric, or C = alpha*B*A when
// side is Right.
func (impl Implementation) Dsymm(s blas.Side, ul blas.Uplo, m, n int, alpha float64,
	a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {

	na := m
	if s == blas.Right {
		na = n
	}
	if !symmOK(s, ul, m, n, na, lda, ldb, ldc, len(a), len(b), len(c)) {
		impl.Implementation.Dsymm(s, ul, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
		return
	}
	full, put := getScratch[float64](na * na)
	defer put()
	densifySym(full, a, na, lda, ul == blas.Upper)

	if s == blas.Left {
		gemmGeneral(blas.NoTrans, blas.NoTrans, m, n, na, alpha, full, na, b, ldb, beta, c, ldc)
		return
	}
	gemmGeneral(blas.NoTrans, blas.NoTrans, m, n, na, alpha, b, ldb, full, na, beta, c, ldc)
}

func (impl Implementation) Ssymm(s blas.Side, ul blas.Uplo, m, n int, alpha float32,
	a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {

	na := m
	if s == blas.Right {
		na = n
	}
	if !symmOK(s, ul, m, n, na, lda, ldb, ldc, len(a), len(b), len(c)) {
		impl.Implementation.Ssymm(s, ul, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
		return
	}
	full, put := getScratch[float32](na * na)
	defer put()
	densifySym(full, a, na, lda, ul == blas.Upper)

	if s == blas.Left {
		gemmGeneral(blas.NoTrans, blas.NoTrans, m, n, na, alpha, full, na, b, ldb, beta, c, ldc)
		return
	}
	gemmGeneral(blas.NoTrans, blas.NoTrans, m, n, na, alpha, b, ldb, full, na, beta, c, ldc)
}

func symmOK(s blas.Side, ul blas.Uplo, m, n, na, lda, ldb, ldc, lenA, lenB, lenC int) bool {
	return m > 0 && n > 0 &&
		(s == blas.Left || s == blas.Right) &&
		(ul == blas.Upper || ul == blas.Lower) &&
		lda >= na && ldb >= n && ldc >= n &&
		lenA >= (na-1)*lda+na && lenB >= (m-1)*ldb+n && lenC >= (m-1)*ldc+n &&
		int64(m)*int64(n)*int64(na) >= gemmGeneralMinWork
}

// Dsyr2k computes C = alpha*A*Bᵀ + alpha*B*Aᵀ + beta*C, or the transposed form,
// writing one triangle.
func (impl Implementation) Dsyr2k(ul blas.Uplo, t blas.Transpose, n, k int, alpha float64,
	a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {

	rows, cols := n, k
	if t != blas.NoTrans {
		rows, cols = k, n
	}
	if !syr2kOK(ul, t, n, k, rows, cols, lda, ldb, ldc, len(a), len(b), len(c)) {
		impl.Implementation.Dsyr2k(ul, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
		return
	}
	tB := blas.Trans
	if t != blas.NoTrans {
		tB = blas.NoTrans
	}
	full, put := getScratch[float64](n * n)
	defer put()
	// alpha*A*Bᵀ, then accumulate alpha*B*Aᵀ on top of it.
	gemmGeneral(t, tB, n, n, k, alpha, a, lda, b, ldb, 0, full, n)
	gemmGeneral(t, tB, n, n, k, alpha, b, ldb, a, lda, 1, full, n)
	foldTriangle(c, full, n, ldc, ul == blas.Upper, beta)
}

func (impl Implementation) Ssyr2k(ul blas.Uplo, t blas.Transpose, n, k int, alpha float32,
	a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {

	rows, cols := n, k
	if t != blas.NoTrans {
		rows, cols = k, n
	}
	if !syr2kOK(ul, t, n, k, rows, cols, lda, ldb, ldc, len(a), len(b), len(c)) {
		impl.Implementation.Ssyr2k(ul, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
		return
	}
	tB := blas.Trans
	if t != blas.NoTrans {
		tB = blas.NoTrans
	}
	full, put := getScratch[float32](n * n)
	defer put()
	gemmGeneral(t, tB, n, n, k, alpha, a, lda, b, ldb, 0, full, n)
	gemmGeneral(t, tB, n, n, k, alpha, b, ldb, a, lda, 1, full, n)
	foldTriangle(c, full, n, ldc, ul == blas.Upper, beta)
}

func syr2kOK(ul blas.Uplo, t blas.Transpose, n, k, rows, cols, lda, ldb, ldc, lenA, lenB, lenC int) bool {
	return n > 0 && k > 0 &&
		(ul == blas.Upper || ul == blas.Lower) &&
		(t == blas.NoTrans || t == blas.Trans || t == blas.ConjTrans) &&
		lda >= cols && ldb >= cols && ldc >= n &&
		lenA >= (rows-1)*lda+cols && lenB >= (rows-1)*ldb+cols &&
		lenC >= (n-1)*ldc+n &&
		int64(n)*int64(n)*int64(k) >= 2*gemmGeneralMinWork
}

// Dsyr is A += alpha*x*xᵀ over one triangle.
//
// No scratch and no densifying: the part of each row that lies in the triangle
// is a contiguous span, and adding a scaled vector to a contiguous span is
// exactly AddScaled. This is the rank-1 update restricted to half the matrix.
func (impl Implementation) Dsyr(ul blas.Uplo, n int, alpha float64, x []float64, incX int, a []float64, lda int) {
	if !syrOK(ul, n, incX, lda, len(x), len(a)) {
		impl.Implementation.Dsyr(ul, n, alpha, x, incX, a, lda)
		return
	}
	for i := 0; i < n; i++ {
		lo, hi := 0, i+1
		if ul == blas.Upper {
			lo, hi = i, n
		}
		simd.AddScaled(a[i*lda+lo:i*lda+hi], x[lo:hi], alpha*x[i])
	}
}

func (impl Implementation) Ssyr(ul blas.Uplo, n int, alpha float32, x []float32, incX int, a []float32, lda int) {
	if !syrOK(ul, n, incX, lda, len(x), len(a)) {
		impl.Implementation.Ssyr(ul, n, alpha, x, incX, a, lda)
		return
	}
	for i := 0; i < n; i++ {
		lo, hi := 0, i+1
		if ul == blas.Upper {
			lo, hi = i, n
		}
		simd.AddScaled(a[i*lda+lo:i*lda+hi], x[lo:hi], alpha*x[i])
	}
}

// Dsyr2 is A += alpha*x*yᵀ + alpha*y*xᵀ over one triangle.
func (impl Implementation) Dsyr2(ul blas.Uplo, n int, alpha float64, x []float64, incX int,
	y []float64, incY int, a []float64, lda int) {

	if !syrOK(ul, n, incX, lda, len(x), len(a)) || incY != 1 || len(y) < n {
		impl.Implementation.Dsyr2(ul, n, alpha, x, incX, y, incY, a, lda)
		return
	}
	for i := 0; i < n; i++ {
		lo, hi := 0, i+1
		if ul == blas.Upper {
			lo, hi = i, n
		}
		row := a[i*lda+lo : i*lda+hi]
		simd.AddScaled(row, y[lo:hi], alpha*x[i])
		simd.AddScaled(row, x[lo:hi], alpha*y[i])
	}
}

func (impl Implementation) Ssyr2(ul blas.Uplo, n int, alpha float32, x []float32, incX int,
	y []float32, incY int, a []float32, lda int) {

	if !syrOK(ul, n, incX, lda, len(x), len(a)) || incY != 1 || len(y) < n {
		impl.Implementation.Ssyr2(ul, n, alpha, x, incX, y, incY, a, lda)
		return
	}
	for i := 0; i < n; i++ {
		lo, hi := 0, i+1
		if ul == blas.Upper {
			lo, hi = i, n
		}
		row := a[i*lda+lo : i*lda+hi]
		simd.AddScaled(row, y[lo:hi], alpha*x[i])
		simd.AddScaled(row, x[lo:hi], alpha*y[i])
	}
}

// syrMinLen is the order below which the per-row calls cost more than they
// save. A rank-1 update over a triangle averages n/2 elements per row, so the
// rows are short even when the matrix is not, and each one is a dispatch.
const syrMinLen = 128

func syrOK(ul blas.Uplo, n, incX, lda, lenX, lenA int) bool {
	return n >= syrMinLen && incX == 1 &&
		(ul == blas.Upper || ul == blas.Lower) &&
		lda >= n && lenX >= n && lenA >= (n-1)*lda+n
}
