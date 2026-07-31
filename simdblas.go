// Package simdblas is a BLAS implementation for gonum, backed by
// github.com/sebishogun/simd.
//
// Gonum's own BLAS is pure Go with SSE2-only assembly on amd64 and none at all
// on arm64. This replaces the routines where a vector unit helps, reaching
// AVX-512, NEON, SVE2, RVV, VSX, VX and LASX — with no cgo, because the kernels
// are compiled ahead of time and committed as assembly.
//
// Install it once, at startup:
//
//	import (
//		"gonum.org/v1/gonum/blas/blas64"
//		"github.com/sebishogun/simdblas"
//	)
//
//	func init() {
//		blas64.Use(simdblas.Implementation{})
//		blas32.Use(simdblas.Implementation{})
//	}
//
// Everything above BLAS — gonum's mat, stat and optimize — then runs on it
// without further change.
//
// # What is accelerated
//
// The routines where whole-slice vector work pays: the Level 1 set, Dgemv and
// Dgemm, in both precisions. Everything else is inherited from
// gonum.Implementation and behaves exactly as it did.
//
// Acceleration also requires unit strides and, for the matrix routines, no
// transpose, alpha of 1, beta of 0 and natural leading dimensions. Anything
// else falls through to gonum, so the answer is always a correct BLAS answer
// and never a wrong fast one. See [Implementation] for why that is the whole
// design.
//
// # Results differ from gonum's, and that is the point
//
// BLAS does not specify bit-exact results, and no two implementations agree.
// Reductions here use a fixed sixteen-accumulator tree, gonum's use a
// four-way unrolled loop, so a dot product of 4096 elements typically differs
// in the last unit in the last place.
//
// The difference worth knowing runs the other way. Gonum's answer depends on
// what it was compiled for: its amd64 assembly and its pure-Go fallback do not
// agree with each other, so the same program gives different results on x86 and
// on arm64. This implementation gives the same bits on every architecture and
// every instruction set, because the accumulation order is fixed rather than
// following the vector width. If you need a result that reproduces across
// machines, that is the reason to use this.
package simdblas

import "gonum.org/v1/gonum/blas/gonum"

// Implementation is a drop-in for gonum's BLAS.
//
// It embeds gonum.Implementation rather than reimplementing the interface, so
// every routine exists and is correct from the first line, and each accelerated
// one is an override of a working method rather than a new one that has to be
// got right. The 60-odd routines nobody has measured a win for are gonum's,
// unchanged.
//
// Each override checks whether its fast path applies and delegates to the
// embedded method when it does not — including for invalid arguments, so the
// panics a caller relies on come from gonum and read exactly as they always
// have.
type Implementation struct{ gonum.Implementation }
