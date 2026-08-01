package simdblas

import "gonum.org/v1/gonum/blas"

// Triangular solve with multiple right-hand sides, blocked.
//
// trsm is the other half of every blocked decomposition — Dgetrf, Dpotrf and
// Dgeqrf all alternate it with gemm — and gonum's is a scalar triple loop. It
// was the largest remaining reason a factorisation saw no benefit from this
// package after gemm was widened.
//
// The algorithm is the standard one and it is deliberately not clever: split
// the triangular operand into diagonal blocks, hand each small triangular solve
// to gonum, and do the rectangular update between them with the accelerated
// gemm. All of the arithmetic that scales as O(n^3) ends up in gemm, and what
// stays in the scalar solve is O(nb^2) per block for a fixed block size.
//
// LAPACK asks for all eight combinations of side, triangle and transpose, so
// there is no useful subset to special-case. What varies between them is only
// the direction the blocks are walked and which sub-block the update reads, and
// those are computed rather than written out eight times.

// trsmBlock is the diagonal block size. The scalar solve inside each block is
// O(nb^2) per right-hand side and the gemm between blocks is O(nb*n*m), so a
// larger block means less gemm and more scalar work. 64 matches the block size
// gonum's own LAPACK uses for these factorisations.
const trsmBlock = 64

// trsmMinWork is the size below which blocking is not worth the bookkeeping and
// the whole call goes to gonum.
const trsmMinWork = 1 << 15

// trsmForward reports whether the diagonal blocks are walked from the top-left
// downward.
//
// A transpose swaps which triangle is being read, so Upper-with-transpose
// behaves exactly as Lower-without does. The four cases collapse to one
// exclusive-or on each side.
func trsmForward(side blas.Side, ul blas.Uplo, tA blas.Transpose) bool {
	lower := ul == blas.Lower
	trans := tA != blas.NoTrans
	if side == blas.Left {
		return lower != trans
	}
	return lower == trans
}

// dtrsmBlocked solves op(A)*X = alpha*B or X*op(A) = alpha*B in place in b.
//
// It scales B by alpha once up front, then solves with an implicit alpha of
// one — which is what makes the per-block bookkeeping identical for every
// block instead of special-casing the first.
func (impl Implementation) dtrsmBlocked(side blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag,
	m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {

	if alpha != 1 {
		for i := 0; i < m; i++ {
			row := b[i*ldb : i*ldb+n]
			if alpha == 0 {
				for j := range row {
					row[j] = 0
				}
				continue
			}
			impl.Dscal(n, alpha, row, 1)
		}
	}
	if alpha == 0 {
		return
	}

	// nt is the order of the triangular operand.
	nt := m
	if side == blas.Right {
		nt = n
	}
	fwd := trsmForward(side, ul, tA)

	for s := 0; s < nt; s += trsmBlock {
		// j is the start of this diagonal block, walking from whichever end.
		j := s
		if !fwd {
			j = nt - s - trsmBlock
			if j < 0 {
				j = 0
			}
		}
		jb := trsmBlock
		if fwd {
			if j+jb > nt {
				jb = nt - j
			}
		} else {
			jb = nt - s
			if jb > trsmBlock {
				jb = trsmBlock
			}
		}
		if jb <= 0 {
			break
		}

		// The rest of the operand: what is still unsolved after this block.
		var r0, rn int
		if fwd {
			r0, rn = j+jb, nt-j-jb
		} else {
			r0, rn = 0, j
		}

		if side == blas.Left {
			// Solve the jb rows of B that this diagonal block owns.
			impl.Implementation.Dtrsm(side, ul, tA, d, jb, n, 1,
				a[j*lda+j:], lda, b[j*ldb:], ldb)
			if rn <= 0 {
				continue
			}
			// B[r0:] -= op(A)[r0:, j:j+jb] * B[j:j+jb, :]
			//
			// Without a transpose that sub-block starts at (r0, j); with one,
			// op(A)[r,c] is A[c,r], so the same sub-block is A[j:j+jb, r0:]
			// read transposed.
			aoff, at := r0*lda+j, tA
			if tA != blas.NoTrans {
				aoff = j*lda + r0
			}
			impl.Dgemm(at, blas.NoTrans, rn, n, jb, -1,
				a[aoff:], lda, b[j*ldb:], ldb, 1, b[r0*ldb:], ldb)
			continue
		}

		// Right side: the block owns jb columns of B.
		impl.Implementation.Dtrsm(side, ul, tA, d, m, jb, 1,
			a[j*lda+j:], lda, b[j:], ldb)
		if rn <= 0 {
			continue
		}
		// B[:, r0:] -= B[:, j:j+jb] * op(A)[j:j+jb, r0:]
		aoff, at := j*lda+r0, tA
		if tA != blas.NoTrans {
			aoff = r0*lda + j
		}
		impl.Dgemm(blas.NoTrans, at, m, rn, jb, -1,
			b[j:], ldb, a[aoff:], lda, 1, b[r0:], ldb)
	}
}

// Dtrsm solves a triangular system with multiple right-hand sides.
//
// Blocked through the accelerated gemm when the problem is big enough to pay
// for the bookkeeping; gonum's own otherwise, and for anything this cannot
// validate.
func (impl Implementation) Dtrsm(side blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag,
	m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {

	nt := m
	if side == blas.Right {
		nt = n
	}
	ok := m >= 0 && n >= 0 && m > 0 && n > 0 &&
		(side == blas.Left || side == blas.Right) &&
		(ul == blas.Upper || ul == blas.Lower) &&
		(tA == blas.NoTrans || tA == blas.Trans || tA == blas.ConjTrans) &&
		(d == blas.Unit || d == blas.NonUnit) &&
		lda >= nt && ldb >= n &&
		len(a) >= (nt-1)*lda+nt && len(b) >= (m-1)*ldb+n &&
		int64(m)*int64(n)*int64(nt) >= trsmMinWork && nt > trsmBlock

	if !ok {
		impl.Implementation.Dtrsm(side, ul, tA, d, m, n, alpha, a, lda, b, ldb)
		return
	}
	impl.dtrsmBlocked(side, ul, tA, d, m, n, alpha, a, lda, b, ldb)
}
