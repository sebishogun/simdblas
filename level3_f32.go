// Code in this file mirrors trsm.go and trmm.go for float32. The algorithms,
// the block size and the thresholds are the same and the reasoning is written
// out there rather than repeated here; what differs is the element type and the
// gonum method each small step delegates to.
//
// It is a transcription rather than a generic because the BLAS interface fixes
// the method names — Dtrsm and Strsm are different methods on the same type —
// and threading the four delegated operations through as closures cost more
// clarity than the duplication does.

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

// strsmBlocked solves op(A)*X = alpha*B or X*op(A) = alpha*B in place in b.
//
// It scales B by alpha once up front, then solves with an implicit alpha of
// one — which is what makes the per-block bookkeeping identical for every
// block instead of special-casing the first.
func (impl Implementation) strsmBlocked(side blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag,
	m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) {

	if alpha != 1 {
		for i := 0; i < m; i++ {
			row := b[i*ldb : i*ldb+n]
			if alpha == 0 {
				for j := range row {
					row[j] = 0
				}
				continue
			}
			impl.Sscal(n, alpha, row, 1)
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
			impl.Implementation.Strsm(side, ul, tA, d, jb, n, 1,
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
			impl.Sgemm(at, blas.NoTrans, rn, n, jb, -1,
				a[aoff:], lda, b[j*ldb:], ldb, 1, b[r0*ldb:], ldb)
			continue
		}

		// Right side: the block owns jb columns of B.
		impl.Implementation.Strsm(side, ul, tA, d, m, jb, 1,
			a[j*lda+j:], lda, b[j:], ldb)
		if rn <= 0 {
			continue
		}
		// B[:, r0:] -= B[:, j:j+jb] * op(A)[j:j+jb, r0:]
		aoff, at := j*lda+r0, tA
		if tA != blas.NoTrans {
			aoff = r0*lda + j
		}
		impl.Sgemm(blas.NoTrans, at, m, rn, jb, -1,
			b[j:], ldb, a[aoff:], lda, 1, b[r0:], ldb)
	}
}

// Strsm solves a triangular system with multiple right-hand sides.
//
// Blocked through the accelerated gemm when the problem is big enough to pay
// for the bookkeeping; gonum's own otherwise, and for anything this cannot
// validate.
func (impl Implementation) Strsm(side blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag,
	m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) {

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
		impl.Implementation.Strsm(side, ul, tA, d, m, n, alpha, a, lda, b, ldb)
		return
	}
	impl.strsmBlocked(side, ul, tA, d, m, n, alpha, a, lda, b, ldb)
}

// Triangular matrix multiply and symmetric rank-k update, both blocked through
// the accelerated gemm.
//
// These are the two that decide QR and Cholesky. gonum's blocked Dgeqrf spends
// itself in Strmm — twenty-four call sites against sixteen for Sgemm — and
// Dpotrf alternates Strsm with Ssyrk. Neither had a fast path, which is why QR
// moved 1.02x at n=512 while the multiply-heavy routines moved 1.2x.

// Strmm computes B = alpha*op(A)*B or B = alpha*B*op(A) with A triangular.
func (impl Implementation) Strmm(side blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag,
	m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) {

	nt := m
	if side == blas.Right {
		nt = n
	}
	ok := m > 0 && n > 0 &&
		(side == blas.Left || side == blas.Right) &&
		(ul == blas.Upper || ul == blas.Lower) &&
		(tA == blas.NoTrans || tA == blas.Trans || tA == blas.ConjTrans) &&
		(d == blas.Unit || d == blas.NonUnit) &&
		lda >= nt && ldb >= n &&
		len(a) >= (nt-1)*lda+nt && len(b) >= (m-1)*ldb+n &&
		int64(m)*int64(n)*int64(nt) >= trsmMinWork

	if !ok {
		impl.Implementation.Strmm(side, ul, tA, d, m, n, alpha, a, lda, b, ldb)
		return
	}
	if nt <= trsmBlock {
		impl.strmmDense(side, ul, tA, d, m, n, alpha, a, lda, b, ldb, nt)
		return
	}

	fwd := trmmForward(side, ul, tA)
	lower := effLower(ul, tA)

	for s := 0; s < nt; s += trsmBlock {
		j := s
		jb := trsmBlock
		if !fwd {
			j = nt - s - trsmBlock
			if j < 0 {
				jb = trsmBlock + j
				j = 0
			}
		} else if j+jb > nt {
			jb = nt - j
		}
		if jb <= 0 {
			break
		}

		// The off-diagonal span of this block row (or column): everything in
		// the triangle that is not the diagonal block itself.
		//
		// Which side of the diagonal that is inverts between Left and Right. On
		// the left a lower triangle contributes the columns *before* the block,
		// because result row i sums op(A)[i,l] for l <= i. On the right the same
		// triangle contributes the columns *after* it, because result column j
		// sums op(A)[l,j] for l >= j. Using the left-hand convention for both is
		// what made Right/Lower wrong while every left-hand case passed.
		lowerLike := lower
		if side == blas.Right {
			lowerLike = !lower
		}
		off, cnt := 0, j
		if !lowerLike {
			off, cnt = j+jb, nt-j-jb
		}

		if side == blas.Left {
			// The diagonal block first: it reads only the rows it writes.
			impl.Implementation.Strmm(side, ul, tA, d, jb, n, alpha,
				a[j*lda+j:], lda, b[j*ldb:], ldb)
			if cnt <= 0 {
				continue
			}
			// B[j:j+jb] += alpha * op(A)[j:j+jb, off:off+cnt] * B[off:off+cnt]
			aoff := j*lda + off
			if tA != blas.NoTrans {
				aoff = off*lda + j
			}
			impl.Sgemm(tA, blas.NoTrans, jb, n, cnt, alpha,
				a[aoff:], lda, b[off*ldb:], ldb, 1, b[j*ldb:], ldb)
			continue
		}

		impl.Implementation.Strmm(side, ul, tA, d, m, jb, alpha,
			a[j*lda+j:], lda, b[j:], ldb)
		if cnt <= 0 {
			continue
		}
		// B[:, j:j+jb] += alpha * B[:, off:off+cnt] * op(A)[off:off+cnt, j:j+jb]
		aoff := off*lda + j
		if tA != blas.NoTrans {
			aoff = j*lda + off
		}
		impl.Sgemm(blas.NoTrans, tA, m, jb, cnt, alpha,
			b[off:], ldb, a[aoff:], lda, 1, b[j:], ldb)
	}
}

// strmmDense handles a triangular operand small enough that blocking it buys
// nothing: it fills the triangle out into a full matrix and hands the whole
// thing to gemm.
//
// This is the shape QR arrives in. LAPACK's Dlarfb applies a block reflector by
// multiplying by a k×k triangular factor where k is the block size, against a B
// with as many rows as the matrix is tall — so the triangle is 64 across and the
// multiply is enormous. Blocking a 64-wide triangle produces one block and no
// gemm at all, which is why QR moved 1.05x while everything else moved.
//
// Densifying costs nt² of zeroing and doubles the arithmetic, since half the
// full matrix is the zeros just written. Both are paid against a multiply that
// is m*nt*nt and now runs on the vector unit.
func (impl Implementation) strmmDense(side blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag,
	m, n int, alpha float32, a []float32, lda int, b []float32, ldb int, nt int) {

	lower := effLower(ul, tA)
	unit := d == blas.Unit

	buf, put := getScratch[float32](nt*nt + m*n)
	defer put()
	full := buf[:nt*nt]
	out := buf[nt*nt:]
	for i := range full {
		full[i] = 0
	}
	for i := 0; i < nt; i++ {
		for j := 0; j < nt; j++ {
			inTri := j <= i
			if !lower {
				inTri = j >= i
			}
			if !inTri {
				continue
			}
			if i == j && unit {
				full[i*nt+j] = 1
				continue
			}
			// op(A)[i][j] is A[i][j] transposed or not.
			if tA == blas.NoTrans {
				full[i*nt+j] = a[i*lda+j]
			} else {
				full[i*nt+j] = a[j*lda+i]
			}
		}
	}

	if side == blas.Left {
		// B = alpha * full(nt×nt) * B(m×n), where nt == m.
		gemmGeneral(blas.NoTrans, blas.NoTrans, m, n, nt, alpha, full, nt, b, ldb, 0, out, n)
		for i := 0; i < m; i++ {
			copy(b[i*ldb:i*ldb+n], out[i*n:(i+1)*n])
		}
		return
	}
	// B = alpha * B(m×nt) * full(nt×nt), where nt == n.
	gemmGeneral(blas.NoTrans, blas.NoTrans, m, nt, nt, alpha, b, ldb, full, nt, 0, out, nt)
	for i := 0; i < m; i++ {
		copy(b[i*ldb:i*ldb+nt], out[i*nt:(i+1)*nt])
	}
}

// Ssyrk computes C = alpha*A*Aᵀ + beta*C, or alpha*Aᵀ*A, writing only one
// triangle of C.
//
// This one is done by computing the whole product and keeping half, which is
// twice the arithmetic a purpose-built syrk would do. It still wins, because
// the multiply it doubles is the accelerated one and the alternative is gonum's
// scalar triple loop — the factor of two is smaller than the factor the vector
// unit brings. A blocked form that only touches the triangle would be better
// and is not written; this is the version that could be measured today.
func (impl Implementation) Ssyrk(ul blas.Uplo, t blas.Transpose, n, k int, alpha float32,
	a []float32, lda int, beta float32, c []float32, ldc int) {

	rows, cols := n, k
	if t != blas.NoTrans {
		rows, cols = k, n
	}
	ok := n > 0 && k > 0 &&
		(ul == blas.Upper || ul == blas.Lower) &&
		(t == blas.NoTrans || t == blas.Trans || t == blas.ConjTrans) &&
		lda >= cols && ldc >= n &&
		len(a) >= (rows-1)*lda+cols && len(c) >= (n-1)*ldc+n &&
		int64(n)*int64(n)*int64(k) >= 2*gemmGeneralMinWork

	if !ok {
		impl.Implementation.Ssyrk(ul, t, n, k, alpha, a, lda, beta, c, ldc)
		return
	}

	tB := blas.Trans
	if t != blas.NoTrans {
		tB = blas.NoTrans
	}
	full, put := getScratch[float32](n * n)
	defer put()
	gemmGeneral(t, tB, n, n, k, alpha, a, lda, a, lda, 0, full, n)

	// Fold the computed half into C, respecting beta and touching only the
	// triangle the caller asked for — the other one must be left exactly as it
	// was, which is the whole reason syrk is not just a gemm.
	for i := 0; i < n; i++ {
		lo, hi := i, n
		if ul == blas.Lower {
			lo, hi = 0, i+1
		}
		row := c[i*ldc+lo : i*ldc+hi]
		src := full[i*n+lo : i*n+hi]
		switch beta {
		case 0:
			copy(row, src)
		case 1:
			for j := range row {
				row[j] += src[j]
			}
		default:
			for j := range row {
				row[j] = row[j]*beta + src[j]
			}
		}
	}
}
