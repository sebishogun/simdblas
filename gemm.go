package simdblas

import (
	"sync"

	"github.com/sebishogun/simd"
	"gonum.org/v1/gonum/blas"
)

// The general gemm path.
//
// BLAS states gemm as C = alpha*op(A)*op(B) + beta*C, where op is an optional
// transpose and the leading dimensions let each operand be a window onto a
// larger array. simd.MatMulInto computes dst = a*b over contiguous, row-major,
// naturally strided slices, and no kernel can take the general form directly:
// dst, a, b, m, k, n, lda, ldb and ldc is nine integer arguments against the
// six the SysV ABI passes in registers.
//
// The first release therefore accelerated only the case where the two
// coincide, and delegated everything else. That turned out to exclude nearly
// every call a decomposition makes — LAPACK's blocked Dgetrf multiplies
// submatrix windows with an alpha of -1 — so mat.LU, mat.Solve and mat.Inverse
// saw no change at all.
//
// This is the fix, and it is what a real BLAS does: copy the operands into
// contiguous scratch, applying the transpose on the way in, multiply, then
// scale and accumulate. The copies are O(mk + kn + mn) against O(mnk) of
// arithmetic, so they are noise once k is more than a few — for LAPACK's own
// blocking, 448x448x64, the packing is 2.5% of the multiply.

// gemmScratch hands out reusable buffers so the packed path does not allocate
// per call. A decomposition calls gemm once per block, and an allocation each
// time would show up as GC pressure in a routine that had none.
var gemmScratch = sync.Pool{New: func() any { return new([]float64) }}
var gemmScratch32 = sync.Pool{New: func() any { return new([]float32) }}

func getScratch[T float32 | float64](n int) (buf []T, put func()) {
	switch any(buf).(type) {
	case []float64:
		p := gemmScratch.Get().(*[]float64)
		if cap(*p) < n {
			*p = make([]float64, n)
		}
		*p = (*p)[:n]
		return any(*p).([]T), func() { gemmScratch.Put(p) }
	default:
		p := gemmScratch32.Get().(*[]float32)
		if cap(*p) < n {
			*p = make([]float32, n)
		}
		*p = (*p)[:n]
		return any(*p).([]T), func() { gemmScratch32.Put(p) }
	}
}

// packMatrix copies an operand into a contiguous rows×cols buffer, transposing
// if the caller asked for op(X) = Xᵀ.
//
// rows and cols describe the result — the shape after any transpose — so a
// caller states what it wants rather than what it has.
func packMatrix[T float32 | float64](dst, src []T, rows, cols, ld int, trans bool) {
	if !trans {
		if ld == cols {
			copy(dst, src[:rows*cols])
			return
		}
		for i := 0; i < rows; i++ {
			copy(dst[i*cols:(i+1)*cols], src[i*ld:i*ld+cols])
		}
		return
	}
	// src is cols×rows with stride ld; walking it by destination row reads down
	// a column, which is the cache-hostile direction. Blocking it would help,
	// and it has not been measured as mattering against the multiply that
	// follows — worth revisiting if a profile ever says otherwise.
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			dst[i*cols+j] = src[j*ld+i]
		}
	}
}

// gemmGeneralMinWork is the multiply-accumulate count below which packing costs
// more than the wider inner loop wins.
//
// Three copies plus a multiply against gonum doing the multiply in place: the
// copies are linear in the operands and the multiply is cubic, so the threshold
// is where the cubic term stops being small. Measured, see the README.
const gemmGeneralMinWork = 1 << 15

// gemmGeneral computes C = alpha*op(A)*op(B) + beta*C through contiguous
// scratch. It assumes its arguments have already been validated.
func gemmGeneral[T float32 | float64](tA, tB blas.Transpose, m, n, k int,
	alpha T, a []T, lda int, b []T, ldb int, beta T, c []T, ldc int) {

	transA := tA != blas.NoTrans
	transB := tB != blas.NoTrans

	buf, put := getScratch[T](m*k + k*n + m*n)
	defer put()
	pa := buf[:m*k]
	pb := buf[m*k : m*k+k*n]
	pc := buf[m*k+k*n:]

	packMatrix(pa, a, m, k, lda, transA)
	packMatrix(pb, b, k, n, ldb, transB)

	simd.MatMulParallelInto(pc, pa, pb, m, k, n)

	// Scale and accumulate back into C, one row at a time so that ldc is
	// respected without another copy.
	//
	// beta == 0 is not the same as multiplying by zero: BLAS says C is not read
	// when beta is zero, so a C full of NaN or uninitialised memory must not
	// poison the result.
	for i := 0; i < m; i++ {
		row := c[i*ldc : i*ldc+n]
		src := pc[i*n : (i+1)*n]
		switch {
		case beta == 0 && alpha == 1:
			copy(row, src)
		case beta == 0:
			// row = alpha*src
			copy(row, src)
			simd.Scale(row, alpha)
		case beta == 1:
			// row += alpha*src
			simd.AddScaled(row, src, alpha)
		default:
			simd.Scale(row, beta)
			simd.AddScaled(row, src, alpha)
		}
	}
}

// gemmValid reports whether the arguments describe a multiplication this
// package can carry out, leaving anything else — including anything invalid —
// to gonum, so that the panics a caller depends on come from there.
func gemmValid(tA, tB blas.Transpose, m, n, k, lda, ldb, ldc, lenA, lenB, lenC int) bool {
	if m < 0 || n < 0 || k < 0 {
		return false
	}
	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		return false
	}
	if tB != blas.NoTrans && tB != blas.Trans && tB != blas.ConjTrans {
		return false
	}
	if m == 0 || n == 0 || k == 0 {
		return false // nothing to do; let gonum handle the edge semantics
	}
	// Rows of op(A) are k wide and there are m of them, or the transpose.
	ra, ca := m, k
	if tA != blas.NoTrans {
		ra, ca = k, m
	}
	rb, cb := k, n
	if tB != blas.NoTrans {
		rb, cb = n, k
	}
	if lda < ca || ldb < cb || ldc < n {
		return false
	}
	return lenA >= (ra-1)*lda+ca && lenB >= (rb-1)*ldb+cb && lenC >= (m-1)*ldc+n
}
