package simdblas

import "github.com/sebishogun/simd"

// The routines gonum's decompositions are built from. LU, QR and Cholesky are
// rank-1 updates, Givens rotations and row swaps rather than matrix multiplies,
// so accelerating gemm alone left mat.Solve and mat.Inverse untouched.

func (impl Implementation) Dswap(n int, x []float64, incX int, y []float64, incY int) {
	if !unit2(n, x, incX, y, incY) || n < minLenSwap {
		impl.Implementation.Dswap(n, x, incX, y, incY)
		return
	}
	simd.Swap(x[:n], y[:n])
}

func (impl Implementation) Sswap(n int, x []float32, incX int, y []float32, incY int) {
	if !unit2(n, x, incX, y, incY) || n < minLenSwap {
		impl.Implementation.Sswap(n, x, incX, y, incY)
		return
	}
	simd.Swap(x[:n], y[:n])
}

func (impl Implementation) Drot(n int, x []float64, incX int, y []float64, incY int, c, s float64) {
	if !unit2(n, x, incX, y, incY) || n < minLenRot {
		impl.Implementation.Drot(n, x, incX, y, incY, c, s)
		return
	}
	simd.Rotate(x[:n], y[:n], c, s)
}

func (impl Implementation) Srot(n int, x []float32, incX int, y []float32, incY int, c, s float32) {
	if !unit2(n, x, incX, y, incY) || n < minLenRot {
		impl.Implementation.Srot(n, x, incX, y, incY, c, s)
		return
	}
	simd.Rotate(x[:n], y[:n], c, s)
}

// Dger is the rank-1 update, and the shape it arrives in decides how much of
// it can be accelerated.
//
// simd.RankOneInto takes neither a leading dimension nor a stride for x, so the
// single-call path needs a matrix whose rows are exactly n apart and an x that
// is contiguous. LAPACK's Dgetf2 — the inner loop of an LU — supplies neither:
// it passes a *column* of the working matrix as x, so incX is lda, and it
// updates a trailing submatrix, so lda is the width of the parent rather than
// n. Guarding on those two conditions alone meant Dger delegated on every call
// an LU ever made, which is exactly the case it was written for.
//
// So there is a second path. A rank-1 update is m independent axpys, one per
// row, and simd.AddScaled handles an arbitrary row offset because each row is
// its own contiguous slice. That costs one call per row instead of one per
// matrix and works for any lda and any incX.
func (impl Implementation) Dger(m, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int) {
	if incY != 1 || m < 0 || n < 0 || lda < n || len(y) < n ||
		len(x) < 1+(m-1)*abs(incX) || len(a) < (m-1)*lda+n {
		impl.Implementation.Dger(m, n, alpha, x, incX, y, incY, a, lda)
		return
	}
	if gerFast(m, n, lda, incX, incY) {
		simd.RankOneInto(a[:m*n], x[:m], y[:n], alpha, m, n)
		return
	}
	if n < gerRowMinLen {
		impl.Implementation.Dger(m, n, alpha, x, incX, y, incY, a, lda)
		return
	}
	gerRows(a, x, y[:n], alpha, m, n, lda, incX)
}

func (impl Implementation) Sger(m, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int) {
	if incY != 1 || m < 0 || n < 0 || lda < n || len(y) < n ||
		len(x) < 1+(m-1)*abs(incX) || len(a) < (m-1)*lda+n {
		impl.Implementation.Sger(m, n, alpha, x, incX, y, incY, a, lda)
		return
	}
	if gerFast(m, n, lda, incX, incY) {
		simd.RankOneInto(a[:m*n], x[:m], y[:n], alpha, m, n)
		return
	}
	if n < gerRowMinLen {
		impl.Implementation.Sger(m, n, alpha, x, incX, y, incY, a, lda)
		return
	}
	gerRows(a, x, y[:n], alpha, m, n, lda, incX)
}

// gerRowMinLen is the row length below which the per-row path costs more than
// it saves.
//
// One accelerated call per row means one dispatch per row, and gonum's own Dger
// is already a per-row axpy against its SSE2 assembly — so the only thing being
// won is the width of the inner loop, and it has to cover the call overhead
// first. Measured on Zen 5, float64, a windowed matrix with a strided x:
//
//	row length    16    0.20x
//	              64    0.72x
//	             128    0.91x
//	             256    1.07x
//	             512    1.34x
//	            1024    1.50x
//
// Below the threshold this hands the call to gonum. That matters more than it
// looks: an LU calls ger once per column with a row length that shrinks by one
// each time, so most of its calls are short ones, and routing all of them
// through here made a 256-square factorisation 12% slower than leaving it
// alone.
const gerRowMinLen = 256

// gerRows is the rank-1 update as m axpys, for the strided and windowed shapes
// the single-call kernel cannot take.
//
// The row scale is computed once and the product rounds before the add, both
// matching what the kernel and gonum do, so this path returns the same bits as
// the other two.
func gerRows[T float32 | float64](a, x, y []T, alpha T, m, n, lda, incX int) {
	if m == 0 || n == 0 {
		return
	}
	ix := 0
	if incX < 0 {
		ix = -(m - 1) * incX
	}
	for i := 0; i < m; i++ {
		s := alpha * x[ix]
		simd.AddScaled(a[i*lda:i*lda+n], y, s)
		ix += incX
	}
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// gerFast is separated out so the guard can be tested without building a
// matrix, the same way gemmFast is.
func gerFast(m, n, lda, incX, incY int) bool {
	return incX == 1 && incY == 1 && lda == n && m >= 0 && n >= 0
}
